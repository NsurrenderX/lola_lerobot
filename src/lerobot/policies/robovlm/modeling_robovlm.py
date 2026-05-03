import copy
import logging
from collections import deque
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.robovlm.configuration_robovlm import RoboVLMConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helper: parameter initialization
# ----------------------------------------------------------------------
def initialize_param(model):
    with torch.no_grad():
        for m in model.children():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.fill_(0)
            else:
                initialize_param(m)


# ----------------------------------------------------------------------
# 1. Policy Head Components (adapted from base_policy.py)
# ----------------------------------------------------------------------
class MLPTanhHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSigmoidHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.mlp(x)


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        in_features,
        action_dim,
        down_sample,
        latent,
        fwd_pred_next_n,
        window_size,
        hidden_size=1024,
        num_layers=4,
        policy_rnn_dropout_p=0.0,
    ):
        super().__init__()
        self.down_sample = down_sample
        self.latent = latent
        self.window_size = window_size
        self.history_len = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.history_memory = []

        self.rnn = nn.LSTM(
            input_size=in_features * latent,
            hidden_size=hidden_size * latent,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=policy_rnn_dropout_p,
        )
        self.actions = MLPTanhHead(
            hidden_size * latent, fwd_pred_next_n * (action_dim - 1)
        )
        self.gripper = MLPSigmoidHead(hidden_size * latent, fwd_pred_next_n)
        self.hidden_state = None

        if self.down_sample == "pooling":
            self.global_1d_pool = nn.AdaptiveMaxPool1d(latent)
        elif self.down_sample == "none":
            pass
        else:
            raise NotImplementedError(f"down_sample={down_sample} not supported")

        initialize_param(self)

    def reset(self):
        self.hidden_state = None
        self.history_memory = []

    def forward(self, tok_seq, h_0=None):
        if self.down_sample == "pooling":
            bs, seq_len = tok_seq.shape[:2]
            tok_seq = rearrange(tok_seq, "b l n d-> (b l) n d")
            tok_seq = self.global_1d_pool(tok_seq.permute(0, 2, 1))
            tok_seq = rearrange(tok_seq, "(b l) d n -> b l (n d)", b=bs, l=seq_len)
        elif self.down_sample == "none":
            tok_seq = rearrange(tok_seq, "b l n d-> b l (n d)")

        # cuDNN LSTM does not support bfloat16 — use native PyTorch implementation
        with torch.backends.cudnn.flags(enabled=False):
            tok_seq = tok_seq.contiguous()
            if tok_seq.shape[1] == 1:
                # Inference mode: single-step processing with history tracking
                self.history_memory.append(tok_seq)
                if len(self.history_memory) <= self.history_len:
                    x, h_n = self.rnn(tok_seq, self.hidden_state)
                    self.hidden_state = h_n
                    x = x[:, -1].unsqueeze(1)
                else:
                    for _ in range(len(self.history_memory) - self.history_len):
                        self.history_memory.pop(0)
                    hist_feature = torch.cat(self.history_memory, dim=1)
                    self.hidden_state = None
                    x, h_n = self.rnn(hist_feature, self.hidden_state)
                    x = x[:, -1].unsqueeze(1)
            else:
                # Training mode: full window processing
                self.hidden_state = h_0
                x, h_n = self.rnn(tok_seq, self.hidden_state)
                self.hidden_state = h_n

        actions = self.actions(x)
        gripper = self.gripper(x)

        actions = rearrange(actions, "b l (n d) -> b l n d", n=self.fwd_pred_next_n)
        gripper = rearrange(gripper, "b l (n d) -> b l n d", n=self.fwd_pred_next_n)

        return actions, gripper


# ----------------------------------------------------------------------
# 2. Loss computation (adapted from BasePolicyHead.loss)
# ----------------------------------------------------------------------
def compute_action_loss(pred_actions, arm_labels, gripper_labels, attention_mask=None):
    pred_arm = pred_actions[..., :6]
    pred_gripper = pred_actions[..., -1]

    if attention_mask is None:
        pose_loss = F.huber_loss(pred_arm, arm_labels)
        gripper_loss = F.binary_cross_entropy_with_logits(pred_gripper, gripper_labels)
        acc_gripper = (F.sigmoid(pred_gripper) > 0.5).eq(gripper_labels).float().mean()
    else:
        mask = attention_mask.bool()
        pose_loss = F.huber_loss(pred_arm, arm_labels, reduction="none")[mask].mean()
        gripper_loss = F.binary_cross_entropy_with_logits(
            pred_gripper, gripper_labels, reduction="none"
        )[mask].mean()
        acc_gripper = (F.sigmoid(pred_gripper) > 0.5).eq(gripper_labels).float()[mask].mean()

    return pose_loss, gripper_loss, acc_gripper


# ----------------------------------------------------------------------
# 3. Core Model: RoboVLMModel (adapted from BaseRoboVLM + RoboKosMos)
# ----------------------------------------------------------------------
class RoboVLMModel(nn.Module):
    def __init__(self, config: RoboVLMConfig):
        super().__init__()
        self.config = config
        self.window_size = config.window_size
        self.fwd_pred_next_n = config.fwd_pred_next_n
        self.latent_num = config.lstm_latent
        self.use_state = config.use_state

        # Load Kosmos-2 VLM backbone
        import transformers
        model_cls = getattr(transformers, config.vlm_model_type)
        self.backbone = model_cls.from_pretrained(
            config.vlm_pretrained_path, trust_remote_code=True
        )

        # Load tokenizer/processor
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            config.vlm_pretrained_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer

        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone.config.use_cache = False

        # Action token (learned parameter for continuous action space)
        self.action_token = nn.Parameter(torch.zeros(config.hidden_size))
        self.action_token.requires_grad_(True)

        # State embeddings
        if self.use_state:
            self.embed_arm_state = nn.Linear(config.state_dim - 1, config.hidden_size)
            self.embed_gripper_state = nn.Embedding(2, config.hidden_size)
            self.embed_state = nn.Linear(2 * config.hidden_size, config.hidden_size)

        # LSTMDecoder action head
        self.act_head = LSTMDecoder(
            in_features=config.hidden_size,
            action_dim=config.action_dim,
            down_sample=config.lstm_down_sample,
            latent=config.lstm_latent,
            fwd_pred_next_n=config.fwd_pred_next_n,
            window_size=config.window_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            policy_rnn_dropout_p=config.lstm_dropout_p,
        )

        # Setup trainable parameters
        self._trainable_params_setup()

    # -- VLM property accessors (from RoboKosMos) --
    @property
    def hidden_size(self):
        return self.backbone.config.text_config.embed_dim

    @property
    def word_embedding(self):
        return self.backbone.text_model.model.embed_tokens

    @property
    def text_tower(self):
        return self.backbone.text_model.model

    @property
    def vision_tower(self):
        return self.backbone.vision_model

    # -- Kosmos-specific image encoding (from RoboKosMos.model_encode_images) --
    def model_encode_images(self, images):
        vision_model_output = self.backbone.vision_model(
            pixel_values=images,
            output_attentions=self.backbone.config.output_attentions,
            output_hidden_states=self.backbone.config.output_hidden_states,
            return_dict=self.backbone.config.return_dict,
        )
        image_embeds = self.backbone.vision_model.model.post_layernorm(
            vision_model_output[0]
        )
        image_embeds = F.normalize(image_embeds, dim=-1)
        image_embeds, _ = self.backbone.image_to_text_projection(image_embeds)
        return image_embeds

    # -- Image encoding with batch/window handling (from BaseRoboVLM.encode_images) --
    def encode_images(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(1)
        bs, seq_len = images.shape[:2]
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.model_encode_images(concat_images)
        image_features = torch.stack(
            [image_features[i] for i in range(bs * seq_len)], dim=0
        ).view(bs, seq_len, -1, image_features.shape[-1])
        return image_features

    # -- State encoding (from BaseRoboVLM.encode_state) --
    def encode_state(self, state):
        arm_state_embeddings = self.embed_arm_state(state[..., :6])
        gripper_indices = state[..., -1].long()
        gripper_state_embeddings = self.embed_gripper_state(gripper_indices)
        state_embeddings = torch.cat(
            (arm_state_embeddings, gripper_state_embeddings), dim=-1
        )
        state_embeddings = self.embed_state(state_embeddings)
        return state_embeddings

    # -- Multi-modal input merging (from BaseRoboVLM.merge_multi_modal_input) --
    def merge_multi_modal_input(
        self,
        input_embeds,
        multimodal_feats=None,
        labels=None,
        attention_mask=None,
        is_image=True,
        insert_idx=1,
        fill_zero=False,
    ):
        bs = input_embeds.shape[0]

        if is_image:
            rgb_feats = self.encode_images(multimodal_feats)
            rgb_feats = rearrange(rgb_feats, "b l n d -> b (l n) d")
        else:
            rgb_feats = multimodal_feats

        added_seq_len = rgb_feats.shape[1]

        multimodal_embeds = torch.cat(
            [input_embeds[:, :insert_idx], rgb_feats, input_embeds[:, insert_idx:]],
            dim=1,
        )

        insert_mask = (
            torch.cat(
                [
                    torch.zeros(input_embeds[:, :insert_idx].shape[:2]),
                    torch.ones(rgb_feats.shape[:2]),
                    torch.zeros(input_embeds[:, insert_idx:].shape[:2]),
                ],
                dim=1,
            )
            .bool()
            .to(multimodal_embeds.device)
        )

        multimodal_labels = None
        if labels is not None:
            multimodal_labels = torch.full(
                (bs, added_seq_len), -100, dtype=labels.dtype, device=labels.device
            )
            multimodal_labels = self._cat_with_insert(
                labels, multimodal_labels, insert_idx, attention_mask
            )

        multimodal_attention_mask = None
        if attention_mask is not None:
            val = False if fill_zero else True
            multimodal_attention_mask = torch.full(
                (bs, added_seq_len),
                val,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            multimodal_attention_mask = self._cat_with_insert(
                attention_mask, multimodal_attention_mask, insert_idx, attention_mask
            )

        return (
            multimodal_embeds,
            multimodal_labels,
            multimodal_attention_mask,
            insert_mask,
        )

    def _cat_with_insert(self, base_tensor, insert_tensor, insert_idx, masks=None):
        if insert_idx >= 0:
            return torch.cat(
                (
                    base_tensor[:, :insert_idx],
                    insert_tensor,
                    base_tensor[:, insert_idx:],
                ),
                dim=1,
            )
        elif insert_idx == -1 and masks is not None:
            new_list = []
            for mask, base, insert in zip(masks, base_tensor, insert_tensor):
                indexs = (mask == False).nonzero()
                idx = indexs[0].item() if len(indexs) > 0 else len(mask)
                new_list.append(torch.cat((base[:idx], insert, base[idx:]), dim=0))
            return torch.stack(new_list, dim=0)
        else:
            raise ValueError("insert_idx should be -1 or >= 0")

    # -- Main forward for continuous action prediction (from BaseRoboVLM.forward_continuous) --
    def forward_continuous(
        self,
        vision_x,
        lang_x,
        attention_mask=None,
        action_labels=None,
        action_mask=None,
        rel_state=None,
        mode="train",
    ):
        bs, seq_len = vision_x.shape[:2]

        eos_offset = int(self.tokenizer.eos_token is not None)
        bos_offset = int(self.tokenizer.bos_token is not None)
        history_type = self.config.history_type

        if history_type in ["post", "pre"]:
            vision_x = vision_x.reshape(bs * seq_len, *vision_x.shape[2:]).unsqueeze(1)
            lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
            attention_mask = (
                attention_mask.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)
            )

        input_embeds = self.word_embedding(lang_x)

        (
            multimodal_embeds,
            multimodal_labels,
            multimodal_attention_mask,
            _,
        ) = self.merge_multi_modal_input(
            input_embeds,
            vision_x,
            labels=None,
            attention_mask=attention_mask,
            insert_idx=bos_offset,
        )

        # Insert state tokens
        if rel_state is not None and self.use_state:
            insert_idx = multimodal_embeds.shape[1] - eos_offset
            state_token = self.encode_state(rel_state)
            state_token = state_token.reshape(bs * seq_len, state_token.shape[-1]).unsqueeze(1)
            (
                multimodal_embeds,
                multimodal_labels,
                multimodal_attention_mask,
                action_token_mask,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                state_token,
                multimodal_labels,
                multimodal_attention_mask,
                is_image=False,
                insert_idx=insert_idx,
            )

        # Insert action tokens
        insert_idx = multimodal_embeds.shape[1] - eos_offset
        action_tokens = repeat(
            self.action_token, "d -> b n d", b=multimodal_embeds.shape[0], n=self.latent_num
        )
        (
            multimodal_embeds,
            multimodal_labels,
            multimodal_attention_mask,
            action_token_mask,
        ) = self.merge_multi_modal_input(
            multimodal_embeds,
            action_tokens,
            multimodal_labels,
            multimodal_attention_mask,
            is_image=False,
            insert_idx=insert_idx,
        )

        if history_type == "pre":
            multimodal_embeds = rearrange(
                multimodal_embeds, "(b l) n d -> b (l n) d", l=seq_len
            )
            if multimodal_attention_mask is not None:
                multimodal_attention_mask = rearrange(
                    multimodal_attention_mask, "(b l) n -> b (l n)", l=seq_len
                )

        # Run text decoder directly — we've already encoded images and merged
        # embeddings via merge_multi_modal_input, so we bypass
        # Kosmos2ForConditionalGeneration.forward which requires pixel_values
        # or image_embeds, and call the text transformer directly.
        output = self.backbone.text_model.model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            inputs_embeds=multimodal_embeds,
            image_embeds=None,
            image_embeds_position_mask=None,
            use_cache=False,
            output_hidden_states=True,
        )

        output_hs = output.hidden_states[-1].clone()

        if history_type == "pre":
            output_hs = rearrange(output_hs, "b (l n) d -> (b l) n d", l=seq_len)

        # Extract action hidden states
        action_hs = output_hs[action_token_mask].reshape(bs, seq_len, self.latent_num, -1)

        # Run LSTMDecoder
        pred_actions, pred_gripper = self.act_head(action_hs)

        if mode == "train":
            return pred_actions, pred_gripper, action_token_mask
        else:
            return pred_actions, pred_gripper

    # -- Trainable parameter setup (from BaseRoboVLM._trainable_params_setup) --
    def _trainable_params_setup(self):
        if self.config.freeze_backbone:
            self.backbone.requires_grad_(False)
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.word_embedding.register_forward_hook(make_inputs_require_grad)
        else:
            self.backbone.requires_grad_(True)

        if self.config.train_vision:
            self.vision_tower.requires_grad_(True)
        else:
            self.vision_tower.requires_grad_(False)

        if self.config.train_text_embedding:
            self.word_embedding.requires_grad_(True)
        else:
            self.word_embedding.requires_grad_(False)

        self.act_head.requires_grad_(True)

        if self.use_state:
            self.embed_arm_state.requires_grad_(True)
            self.embed_gripper_state.requires_grad_(True)
            self.embed_state.requires_grad_(True)

        trainable = {k for k, v in self.named_parameters() if v.requires_grad}
        logger.info(f"RoboVLM trainable parameters: {trainable}")


# ----------------------------------------------------------------------
# 4. Policy Wrapper: RoboVLMPolicy
# ----------------------------------------------------------------------
class RoboVLMPolicy(PreTrainedPolicy):
    config_class = RoboVLMConfig
    name = "robovlm"

    def __init__(self, config: RoboVLMConfig, **kwargs):
        super().__init__(config, **kwargs)

        self._action_queue = deque(maxlen=config.fwd_pred_next_n * 5)

        # Determine dtype
        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        self._dtype = dtype_map[config.dtype]

        # Build core model
        self.model = RoboVLMModel(config)
        self.model.to(self._dtype)

        logger.info(
            f"RoboVLM total parameters: "
            f"{sum(p.numel() for p in self.parameters()) / 1e6:.2f}M"
        )

    def _prepare_inputs(self, batch):
        """Extract inputs from RoboVLMDataset batch format.

        Expected batch keys:
          - rgb: [B, window_size, C, H, W] (already CLIP-normalized)
          - language: [B, text_len] (already tokenized)
          - text_mask: [B, text_len] (attention mask)
          - rel_state: [B, window_size, state_dim] (gripper already binarized)
          - action_chunck: [B, window_size, fwd_pred_next_n, action_dim] (gripper already binarized)
          - chunck_mask: [B, window_size, fwd_pred_next_n] (True=valid)
        """
        vision_x = batch["rgb"]
        lang_x = batch["language"]
        attention_mask = batch["text_mask"]
        rel_state = batch.get("rel_state", None)
        action_chunck = batch.get("action_chunck", None)
        chunk_mask = batch.get("chunck_mask", None)

        return vision_x, lang_x, attention_mask, action_chunck, chunk_mask, rel_state

    def forward(self, batch: dict[str, Any]) -> Tuple[torch.Tensor, dict | None]:
        config = self.config
        vision_x, lang_x, attention_mask, action_chunck, chunk_mask, rel_state = self._prepare_inputs(batch)

        with torch.amp.autocast("cuda", dtype=self._dtype):
            pred_actions, pred_gripper, _ = self.model.forward_continuous(
                vision_x=vision_x.to(self._dtype),
                lang_x=lang_x.to(self.model.word_embedding.weight.device),
                attention_mask=attention_mask.to(self.model.word_embedding.weight.device),
                rel_state=rel_state.to(self._dtype) if rel_state is not None else None,
                mode="train",
            )

        # pred_actions: [B, window_size, fwd_pred_next_n, 6]
        # pred_gripper: [B, window_size, fwd_pred_next_n, 1]
        pred_full = torch.cat([pred_actions, pred_gripper], dim=-1)  # [B, ws, fwd, 7]

        if action_chunck is not None:
            # Labels from dataset: gripper already binarized to {0, 1}
            arm_labels = action_chunck[..., :6]            # [B, ws, fwd, 6]
            gripper_labels = action_chunck[..., -1]         # [B, ws, fwd]

            pred_arm = pred_full[..., :6]                   # [B, ws, fwd, 6]
            pred_gripper_logit = pred_full[..., -1]         # [B, ws, fwd]

            if chunk_mask is not None and not chunk_mask.any():
                # All masked — return zero loss with grad for backward compatibility
                loss_arm = (pred_arm * 0).sum()
                loss_gripper = (pred_gripper_logit * 0).sum()
                acc_gripper = 0.0
            elif chunk_mask is not None and not chunk_mask.all():
                # Partial mask — compute masked loss
                mask = chunk_mask.bool()
                loss_arm = F.huber_loss(pred_arm, arm_labels, reduction="none")[mask].mean()
                loss_gripper = F.binary_cross_entropy_with_logits(
                    pred_gripper_logit, gripper_labels, reduction="none"
                )[mask].mean()
                acc_gripper = (
                    (F.sigmoid(pred_gripper_logit) > 0.5).eq(gripper_labels).float()[mask].mean().item()
                )
            else:
                # No mask (or all valid) — compute unmasked loss
                loss_arm = F.huber_loss(pred_arm, arm_labels)
                loss_gripper = F.binary_cross_entropy_with_logits(pred_gripper_logit, gripper_labels)
                acc_gripper = (
                    (F.sigmoid(pred_gripper_logit) > 0.5).eq(gripper_labels).float().mean().item()
                )

            total_loss = loss_arm + config.arm_gripper_loss_ratio * loss_gripper
        else:
            # No labels — return zero loss with grad
            total_loss = (pred_full * 0).sum()
            loss_arm = torch.tensor(0.0, device=pred_full.device)
            loss_gripper = torch.tensor(0.0, device=pred_full.device)
            acc_gripper = 0.0

        loss_dict = {
            "loss_arm": loss_arm.item(),
            "loss_gripper": loss_gripper.item(),
            "acc_gripper": acc_gripper,
        }

        return total_loss, loss_dict

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        vision_x, lang_x, attention_mask, _, _, rel_state = self._prepare_inputs(batch)

        with torch.amp.autocast("cuda", dtype=self._dtype):
            pred_actions, pred_gripper = self.model.forward_continuous(
                vision_x=vision_x.to(self._dtype),
                lang_x=lang_x.to(self.model.word_embedding.weight.device),
                attention_mask=attention_mask.to(self.model.word_embedding.weight.device),
                rel_state=rel_state.to(self._dtype) if rel_state is not None else None,
                mode="inference",
            )

        # [B, window_size, fwd_pred_next_n, action_dim]
        pred_full = torch.cat([pred_actions, pred_gripper], dim=-1)

        # Take last timestep's prediction -> [B, fwd_pred_next_n, action_dim]
        pred_chunk = pred_full[:, -1, :, :]

        # Convert gripper from logits to probability, then to [-1, 1]
        pred_chunk[..., -1] = 2 * F.sigmoid(pred_chunk[..., -1]) - 1

        return pred_chunk

    def select_action(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        if len(self._action_queue) == 0:
            action_chunk = self.predict_action_chunk(batch)
            for i in range(action_chunk.shape[1]):
                self._action_queue.append(action_chunk[:, i, :])
        return self._action_queue.popleft()

    def reset(self):
        self._action_queue.clear()
        self.model.act_head.reset()

    def get_optim_params(self):
        return self.model.parameters()