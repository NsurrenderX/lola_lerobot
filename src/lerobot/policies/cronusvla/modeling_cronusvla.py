import logging
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Image as Img
from transformers import LlamaTokenizerFast, Qwen2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from lerobot.policies.cronusvla.configuration_cronusvla import CronusVLAConfig
from lerobot.policies.cronusvla.decoder.decoder import ActionModel
from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class CronusVLAModel(nn.Module):
    """CronusVLA: PrismaticVLM backbone + DiT diffusion action decoder.

    Adapted from CronusVLA/vla/cronusvla.py, using PrismaticVLM imported
    directly from the prismatic package (installed from CronusVLA repo).
    """

    def __init__(self, config: CronusVLAConfig):
        super().__init__()
        self.config = config
        self.future_action_window_size = config.future_action_window_size
        self.past_action_window_size = config.past_action_window_size
        self.repeated_diffusion_steps = config.repeated_diffusion_steps

        # Determine token_size from config.hidden_size (must match LLM lm_head.in_features)
        self.token_size = config.hidden_size

        # Load PrismaticVLM via the prismatic package
        from prismatic.models.vlms.prismatic import PrismaticVLM

        # Build VLM from config
        # The vlm_base string determines which ModelConfig variant to use
        # (handled by prismatic/conf/models.py registry)
        vlm, _ = self._build_vlm(config)

        self.vlm = vlm
        self.norm_stats = None  # populated from dataset later

        # Determine token_size from actual VLM
        self.token_size = self.vlm.llm_backbone.llm.lm_head.in_features

        # DiT diffusion action decoder
        self.action_model = ActionModel(
            model_type=config.action_model_type,
            token_size=self.token_size,
            in_channels=config.action_dim,
            future_action_window_size=config.future_action_window_size,
            past_action_window_size=config.past_action_window_size,
            diffusion_steps=config.diffusion_steps,
            noise_schedule=config.noise_schedule,
        )

        # Load action_model weights from local checkpoint if available
        if config.local_vlm_path is not None:
            ckpt = torch.load(config.local_vlm_path, map_location="cpu")
            if "model" in ckpt:
                ckpt = ckpt["model"]
            if "action_model" in ckpt:
                self.action_model.load_state_dict(ckpt["action_model"])

        # EMA (optional)
        self.use_ema = False
        self._trainable_module_keys = ["action_model"]

        # Module keys for checkpoint saving
        self.all_module_keys = ["action_model"]
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)

        # Apply freeze settings
        self._apply_freeze_settings(config)

    @staticmethod
    def _build_vlm(config: CronusVLAConfig):
        """Construct a PrismaticVLM from config parameters.

        Uses prismatic's model config registry to resolve vlm_base,
        then constructs vision backbone, LLM backbone, and projector.

        When local_vlm_path is set:
        - Vision backbones are built with pretrained=False (random init, no timm download)
          since weights will be loaded from checkpoint
        - LLM is built in inference_mode (no HF weight download)
        - All VLM weights are loaded from the local .pt checkpoint
        - HF_HUB_OFFLINE=1 is set temporarily so LLM config/tokenizer use cache only
        """
        import os
        import timm as _timm
        from prismatic.conf.models import ModelConfig
        from prismatic.models.backbones.vision import get_vision_backbone_and_transform
        from prismatic.models.backbones.llm import get_llm_backbone_and_tokenizer
        from prismatic.models.vlms.prismatic import PrismaticVLM

        # Resolve model config from vlm_base identifier
        model_cfg = ModelConfig.get_choice_class(config.vlm_base)()

        # If local_vlm_path is set, build in offline mode (no downloads)
        offline_mode = config.local_vlm_path is not None

        if offline_mode:
            # Monkey-patch timm.create_model to use pretrained=False
            # so vision backbones don't download weights from HF/timm Hub
            _orig_create_model = _timm.create_model
            def _create_model_no_pretrained(*args, **kwargs):
                kwargs["pretrained"] = False
                return _orig_create_model(*args, **kwargs)
            _timm.create_model = _create_model_no_pretrained

            # Set HF_HUB_OFFLINE so LLM config/tokenizer use local cache only
            _prev_offline = os.environ.get("HF_HUB_OFFLINE", None)
            os.environ["HF_HUB_OFFLINE"] = "1"

        try:
            # Build vision backbone and image transform
            vision_backbone, image_transform = get_vision_backbone_and_transform(
                model_cfg.vision_backbone_id,
                image_resize_strategy=model_cfg.image_resize_strategy,
                view_sequence_len=config.view_sequence_len,
            )

            # Build LLM backbone and tokenizer
            # inference_mode=True => builds from config (no weight download)
            llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
                model_cfg.llm_backbone_id,
                llm_max_length=model_cfg.llm_max_length,
                hf_token=config.hf_token,
                inference_mode=offline_mode,
            )

            # Construct PrismaticVLM
            vlm = PrismaticVLM(
                model_cfg.model_id,
                vision_backbone,
                llm_backbone,
                enable_mixed_precision_training=not offline_mode,
                arch_specifier=model_cfg.arch_specifier,
            )
        finally:
            # Restore original timm.create_model and HF_HUB_OFFLINE
            if offline_mode:
                _timm.create_model = _orig_create_model
                if _prev_offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = _prev_offline
                else:
                    os.environ.pop("HF_HUB_OFFLINE", None)

        # Load VLM weights from local checkpoint
        if config.local_vlm_path is not None:
            state_dict = torch.load(config.local_vlm_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            assert "projector" in state_dict and "llm_backbone" in state_dict, (
                f"Checkpoint at {config.local_vlm_path} must contain 'projector' and 'llm_backbone' keys"
            )
            vlm.projector.load_state_dict(state_dict["projector"])
            vlm.llm_backbone.load_state_dict(state_dict["llm_backbone"])
            if "vision_backbone" in state_dict:
                vlm.vision_backbone.load_state_dict(state_dict["vision_backbone"])

        return vlm, (image_transform, tokenizer)

    def _apply_freeze_settings(self, config: CronusVLAConfig):
        """Apply freeze/unfreeze settings based on config flags."""
        if config.freeze_vision_backbone and config.freeze_llm_backbone and not config.unfreeze_last_llm_layer:
            # "finetune" / "vla-train" stage: vision frozen, LLM + projector trainable
            self.vlm.freeze_backbones("finetune")
        elif config.freeze_vision_backbone and config.freeze_llm_backbone and config.unfreeze_last_llm_layer:
            # "last-layer-finetune" / "vla-last-layer-train" stage
            self.vlm.freeze_backbones("vla-last-layer-train")
        elif not config.freeze_vision_backbone and not config.freeze_llm_backbone:
            # "full-finetune" / "vla-full-train" stage
            self.vlm.freeze_backbones("full-finetune")
        elif not config.freeze_vision_backbone and config.freeze_llm_backbone and config.unfreeze_last_llm_layer:
            # "vla-sandwich-train" stage
            self.vlm.freeze_backbones("vla-sandwich-train")
        else:
            # Default: "finetune"
            self.vlm.freeze_backbones("finetune")

        # Action model is always trainable
        self.action_model.requires_grad_(True)

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        return keys

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        per_device_batch_size: int = 16,
        action_masks=None,
    ) -> Tuple[torch.Tensor, CausalLMOutputWithPast]:
        """Run VLM forward, extract cognition features, compute diffusion loss.

        Reimplemented from CronusVLA/vla/cronusvla.py:94-166.
        """
        assert per_device_batch_size == actions.shape[0]
        assert input_ids.shape[0] == (per_device_batch_size * (self.past_action_window_size + 1))

        # Vision-Language Backbone forward
        output: CausalLMOutputWithPast = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract last hidden state
        last_hidden = output.hidden_states[-1]
        assert last_hidden.shape[0] == (per_device_batch_size * (self.past_action_window_size + 1))

        # Extract visual token number and skip visual patches
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, "siglip_featurizer") and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        num_patch = self.vlm.vision_backbone.view_sequence_len * num_patch
        last_hidden = last_hidden[:, num_patch:]

        # Extract cognition features: last non-padding token per sequence
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B*C, 1, D]
        cognition_features = cognition_features.view(
            per_device_batch_size, self.past_action_window_size + 1, 1, -1
        ).squeeze(2)  # [B, C, D]

        # Feature chunking: stop gradient on past features
        cognition_features_detached = cognition_features[:, :-1, :].detach()
        cognition_features = torch.cat(
            (cognition_features_detached, cognition_features[:, -1:, :]), dim=1
        )

        # Extract future actions
        actions_future = actions[:, -(self.future_action_window_size + 1):, :]

        # Repeat for diffusion training
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        cognition_features_repeated = cognition_features.repeat(repeated_diffusion_steps, 1, 1)

        # Diffusion loss
        loss = self.action_model.loss(actions_repeated, cognition_features_repeated)

        return loss, output

    def get_fsdp_wrapping_policy(self) -> callable:
        """Return an FSDP _or_policy over the VLM + DiT modules."""
        from functools import partial
        from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
        from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector
        from lerobot.policies.cronusvla.decoder.models import DiT

        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
        )

        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    @torch.inference_mode()
    def predict_action(
        self,
        image: Union[Img, List[Img]],
        instruction: str,
        unnorm_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        cognition_features_history=None,
        num_cognition_features_history=0,
        **kwargs,
    ) -> np.ndarray:
        """Core inference: map image+instruction to continuous action.

        Reimplemented from CronusVLA/vla/cronusvla.py:258-425.
        """
        # Get tokenizer and image transform
        image_transform = self.vlm.vision_backbone.image_transform
        tokenizer = self.vlm.llm_backbone.tokenizer

        # Build VLA prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?",
        )
        prompt_text = prompt_builder.get_prompt()

        # Tokenize
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)

        # Add tokenizer-specific special tokens
        if isinstance(tokenizer, LlamaTokenizerFast):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)),
                dim=1,
            )
        elif isinstance(tokenizer, Qwen2Tokenizer):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([220, 151645]).long(), dim=0).to(self.vlm.device)),
                dim=1,
            )
        else:
            raise ValueError(f"Unsupported tokenizer type = {type(tokenizer)}")

        # Preprocess image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vlm.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported pixel_values type = {type(pixel_values)}")

        # Vision-Language Backbone: extract cognition features
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype

        if isinstance(tokenizer, LlamaTokenizerFast):
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
                output = super(type(self.vlm), self.vlm).generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    **kwargs,
                )
            cognition_features = output.hidden_states[0][-1][:, -1, :]
            assert (cognition_features.shape[0], cognition_features.shape[1]) == (1, self.token_size)
        else:
            assert isinstance(tokenizer, Qwen2Tokenizer)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
                attention_mask = input_ids.ne(-10)
                labels = input_ids.clone()
                output: CausalLMOutputWithPast = self.vlm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    inputs_embeds=None,
                    past_key_values=None,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=True,
                    return_dict=None,
                )
            cognition_features = output.hidden_states[-1][0, -1:]
            assert (cognition_features.shape[0], cognition_features.shape[1]) == (1, self.token_size)

        using_cfg = cfg_scale > 1.0
        model_dtype = next(self.action_model.net.parameters()).dtype
        B = cognition_features.shape[0]

        # Feature chunking & queue updating
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]
        cognition_features_copy = cognition_features.clone()
        repeat_num = min(
            max(0, self.past_action_window_size - num_cognition_features_history),
            self.past_action_window_size,
        )
        cognition_features_history = list(cognition_features_history) + [cognition_features]
        cognition_features = torch.cat(
            [cognition_features_history[0]] * repeat_num + cognition_features_history, dim=1
        )

        # Diffusion sampling
        noise = torch.randn(
            B, self.future_action_window_size + 1, self.action_model.in_channels,
            device=cognition_features.device,
        ).to(model_dtype)

        # Classifier-free guidance setup
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0).expand(B, self.past_action_window_size + 1, -1)
            z = torch.cat([cognition_features, uncondition], 0)
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM or DDPM sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                sample_fn, noise.shape, noise,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=cognition_features.device, eta=0.0,
            )
        else:
            samples = self.action_model.diffusion.p_sample_loop(
                sample_fn, noise.shape, noise,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=False, device=cognition_features.device,
            )

        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize actions using quantile stats
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions, cognition_features_copy

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Model trained on more than one dataset; pass a `unnorm_key` from: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))
        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` is not in available statistics; choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_stats(self, unnorm_key=None):
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]


class CronusVLAPolicy(PreTrainedPolicy):
    config_class = CronusVLAConfig
    name = "cronusvla"

    def __init__(self, config: CronusVLAConfig, **kwargs):
        super().__init__(config, **kwargs)

        self._action_queue = deque(maxlen=config.future_action_window_size * 5)
        self._cognition_features_history = deque(maxlen=config.past_action_window_size)

        # Determine dtype
        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        self._dtype = dtype_map[config.dtype]

        # Build core model
        self.model = CronusVLAModel(config)
        self.model.to(self._dtype)

        logger.info(
            f"CronusVLA total parameters: "
            f"{sum(p.numel() for p in self.parameters()) / 1e6:.2f}M"
        )

    def forward(self, batch: dict[str, Any]) -> Tuple[torch.Tensor, dict | None]:
        """Compute diffusion loss from VLM cognition features + action targets."""
        with torch.amp.autocast("cuda", dtype=self._dtype):
            loss, _ = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch.get("labels"),
                actions=batch["actions"],
                repeated_diffusion_steps=self.config.repeated_diffusion_steps,
                per_device_batch_size=batch["actions"].shape[0],
                action_masks=batch.get("action_masks"),
                output_hidden_states=True,
            )

        loss_dict = {"diffusion_loss": loss.item()}
        return loss, loss_dict

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        """Predict a chunk of future actions using diffusion sampling."""
        # This method is for the standard lerobot pipeline (not used in CronusVLA's
        # standalone training). For inference, use select_action() which manages
        # the cognition feature history queue.
        raise NotImplementedError(
            "CronusVLA inference uses predict_action() with cognition feature history. "
            "Use select_action() instead."
        )

    def select_action(self, batch: dict[str, Any], **kwargs) -> torch.Tensor:
        """Pop action from queue; if empty, run full VLM+diffusion inference."""
        if len(self._action_queue) == 0:
            # Run full inference pipeline
            # Note: for CronusVLA, the batch should contain raw image + instruction
            # rather than pre-processed VLM inputs. The model.predict_action() handles
            # all preprocessing internally.
            actions, _, cognition_features_copy = self.model.predict_action(
                image=batch.get("image"),
                instruction=batch.get("instruction", ""),
                unnorm_key=batch.get("dataset_name"),
                cfg_scale=self.config.cfg_scale,
                use_ddim=self.config.use_ddim,
                num_ddim_steps=self.config.num_ddim_steps,
                cognition_features_history=list(self._cognition_features_history),
                num_cognition_features_history=len(self._cognition_features_history),
            )

            # Update cognition feature history
            self._cognition_features_history.append(cognition_features_copy)

            # Fill action queue
            for i in range(actions.shape[0]):
                self._action_queue.append(actions[i])

        return self._action_queue.popleft()

    def reset(self):
        """Clear action queue and cognition feature history."""
        self._action_queue.clear()
        self._cognition_features_history.clear()

    def get_optim_params(self):
        return self.model.parameters()