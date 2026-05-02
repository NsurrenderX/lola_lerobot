import logging
import os
from typing import Any, TYPE_CHECKING

import torch
import torchvision.transforms as T

from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.policies.robovlm.configuration_robovlm import RoboVLMConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None

logger = logging.getLogger(__name__)


@ProcessorStepRegistry.register(name="robovlm_image_processor")
class RoboVLMImageProcessor(ObservationProcessorStep):
    """
    Preprocesses images with CLIP-style transforms for Kosmos-2 vision encoder.

    Applies: Resize(224x224, bicubic), Normalize with CLIP mean/std.
    Unlike LoLA which uses Qwen's apply_chat_template, RoboVLM directly
    processes images as tensors through Kosmos-2's vision tower.
    """

    def __init__(
        self,
        image_size: int = 224,
        image_mean: tuple = (0.48145466, 0.4578275, 0.40821073),
        image_std: tuple = (0.26862954, 0.26130258, 0.27577711),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(image_mean, image_std),
        ])

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        new_observation = dict(observation)

        camera_keys = [k for k in observation.keys() if k.startswith("observation.images.")]
        for cam_key in camera_keys:
            img_data = observation[cam_key]
            if isinstance(img_data, torch.Tensor):
                # Already a tensor [C, H, W] or [B, C, H, W]
                if img_data.ndim == 3:
                    new_observation[cam_key] = self._transform_tensor(img_data)
                elif img_data.ndim == 4:
                    new_observation[cam_key] = torch.stack([
                        self._transform_tensor(img_data[i]) for i in range(img_data.shape[0])
                    ])
                # Higher dims (with history) left as-is for model to handle
            elif hasattr(img_data, 'convert'):
                # PIL Image
                new_observation[cam_key] = self.transform(img_data).unsqueeze(0)

        return new_observation

    def _transform_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] == 3 and tensor.dtype in [torch.float32, torch.uint8]:
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            return T.Normalize(self.image_mean, self.image_std)(
                T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC)(tensor)
            )
        return tensor

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="robovlm_text_processor")
class RoboVLMTextProcessor(ObservationProcessorStep):
    """
    Tokenizes task descriptions using Kosmos-2's tokenizer.

    Formats each task as: "<grounding>An image of a robot {task}"
    This is the standard Kosmos-2 text template from RoboVLMs.
    """

    def __init__(
        self,
        processor_path: str = ".vlms/kosmos-2-patch14-224",
        max_length: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.processor_path = processor_path
        self.max_length = max_length

        if not _transformers_available:
            raise ImportError("transformers is required for RoboVLM text processing")

        if os.path.isdir(processor_path):
            self.kosmos_processor = AutoProcessor.from_pretrained(
                processor_path, local_files_only=True, trust_remote_code=True
            )
        else:
            self.kosmos_processor = AutoProcessor.from_pretrained(
                processor_path, trust_remote_code=True
            )
        self.kosmos_tokenizer = self.kosmos_processor.tokenizer
        if self.kosmos_tokenizer.pad_token is None:
            self.kosmos_tokenizer.pad_token = self.kosmos_tokenizer.eos_token

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        new_observation = dict(observation)

        task = observation.get("task", "Perform the robot task.")
        if hasattr(self, 'transition') and self.transition is not None:
            from lerobot.processor.core import TransitionKey
            comp_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            task = comp_data.get("task", task)

        template = "<grounding>An image of a robot {}"
        if isinstance(task, list):
            texts = [template.format(t.strip()) for t in task]
        else:
            texts = [template.format(task.strip())]

        self.kosmos_tokenizer.padding_side = "right"
        encoded = self.kosmos_tokenizer(
            texts,
            truncation="only_first",
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
        )

        new_observation[OBS_LANGUAGE_TOKENS] = encoded["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = encoded["attention_mask"]

        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        return features


def make_robovlm_pre_post_processors(
    config: RoboVLMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    camera_keys: list[str] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Construct pre/post processor pipelines for the RoboVLM policy.
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        RoboVLMImageProcessor(
            image_size=config.image_size,
            image_mean=config.image_mean,
            image_std=config.image_std,
        ),
        RoboVLMTextProcessor(
            processor_path=config.vlm_pretrained_path,
            max_length=512,
        ),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )