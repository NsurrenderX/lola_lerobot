#!/usr/bin/env python

# Copyright 2025 Lola Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processor pipeline for LoLA (Vision-Language-Action) policy.

This module provides preprocessing and postprocessing pipelines for the LoLA policy,
which uses Qwen3.5 as the VLM backbone. The processor handles:
1. Image preprocessing for Qwen3.5 vision encoder (multi-camera support)
2. Text tokenization with Qwen3.5 chat template
3. Empty token appending for global intent control
4. Normalization of states and actions

Data flow:
1. LeRobotDataset returns data with keys:
   - observation.state: Robot state
   - observation.images.*: Image/video data (multiple cameras supported)
   - action: Actions
   - task: Task description string
   
2. Processor transforms to LoLA expected format:
   - input_ids: Tokenized text for VLM (with chat template)
   - pixel_values: Preprocessed images for vision encoder
   - image_grid_thw: Image grid information for Qwen3.5
   - attention_mask: Attention mask for text tokens
   - observation.state: Normalized state (for history actions)
   - action: Normalized target actions

Qwen3.5 Input Format:
    Qwen3.5 uses a chat template format with support for multi-image input.
    The processor uses `apply_chat_template` to format the input correctly.
    
    Example message format:
    ```python
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": "Task description"},
            ],
        }
    ]
    ```
"""

import os
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.lola.configuration_lola import LoLAConfig
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
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.import_utils import _transformers_available

# Lazy import for type checking
if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None


@ProcessorStepRegistry.register(name="lola_empty_token_processor")
class LolaEmptyTokenProcessor(ObservationProcessorStep):
    """
    Appends an empty token to the tokenized sequence for global intent control.
    
    According to the LoLA architecture:
    - An empty_token is appended at the end of the VLM sequence
    - This token is responsible for aggregating global task intent in self-attention
    - After extraction, it fuses with the diffusion timestep to generate modulation signals
    - These signals control DiT feature scaling and shifting through AdaLN
    """
    
    def __init__(self, empty_token_id: int, **kwargs):
        """
        Args:
            empty_token_id: The token ID to append as the empty token (default: Qwen3.5 eos_token)
        """
        super().__init__(**kwargs)
        self.empty_token_id = empty_token_id
    
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Appends the empty token to the language tokens sequence.
        
        Args:
            observation: The observation dictionary containing language tokens.
            
        Returns:
            Updated observation with empty token appended.
        """
        if OBS_LANGUAGE_TOKENS not in observation:
            return observation
        
        new_observation = dict(observation)
        
        # Get current tokens
        language_tokens = observation[OBS_LANGUAGE_TOKENS]  # [B, seq_len]
        batch_size = language_tokens.shape[0]
        
        # Create empty token tensor
        empty_token = torch.full(
            (batch_size, 1), 
            self.empty_token_id, 
            dtype=language_tokens.dtype, 
            device=language_tokens.device
        )
        
        # Append empty token to sequence
        new_observation[OBS_LANGUAGE_TOKENS] = torch.cat([language_tokens, empty_token], dim=1)
        
        # Extend attention mask accordingly
        if OBS_LANGUAGE_ATTENTION_MASK in observation:
            attention_mask = observation[OBS_LANGUAGE_ATTENTION_MASK]  # [B, seq_len]
            new_attention = torch.ones(
                (batch_size, 1), 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )
            new_observation[OBS_LANGUAGE_ATTENTION_MASK] = torch.cat([attention_mask, new_attention], dim=1)
        
        return new_observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates feature shapes to account for the additional empty token.
        
        Args:
            features: The input feature dictionary.
            
        Returns:
            Updated feature dictionary with adjusted sequence lengths.
        """
        # The sequence length increases by 1 due to the empty token
        # This is handled dynamically, so we just pass through the features
        return features


@ProcessorStepRegistry.register(name="lola_image_processor")
class LolaImageProcessor(ObservationProcessorStep):
    """
    Processes multi-camera images for Qwen3.5 vision encoder.

    This processor:
    1. Extracts images from observation.images.* keys
    2. Skips invalid cameras based on camera_valid_mask
    3. Collects PIL Images for Qwen3.5 apply_chat_template

    Supports two input formats:
    - Pretrain mode: camera values are PIL Image (valid) or None (invalid),
      passed as lists from the DataLoader collate (dynamic resolution).
    - Standard mode: camera values are tensors [C, H, W] (legacy fallback).

    Qwen3.5 supports multiple images in a single conversation turn with
    dynamic resolution, so different camera resolutions are handled natively.
    """

    def __init__(self, camera_keys: list[str] | None = None, **kwargs):
        """
        Args:
            camera_keys: List of camera keys to process (e.g., ['observation.images.left', 'observation.images.right']).
                        If None, automatically detects camera keys from observation.
        """
        super().__init__(**kwargs)
        self.camera_keys = camera_keys

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Extracts and prepares images for Qwen3.5 vision encoder.

        Supports both single-item inference and batched training:
        - Batch mode: observation.images.* values are list[PIL.Image | None] of length B.
          Produces '_lola_images_per_item': list[list[PIL.Image]] of length B.
        - Single-item mode: values are single PIL Images or tensors.
          Produces '_lola_images': list[PIL.Image].
        """
        new_observation = dict(observation)

        # Determine camera keys
        camera_keys = self.camera_keys
        if camera_keys is None:
            camera_keys = [k for k in observation.keys() if k.startswith('observation.images.')]

        if not camera_keys:
            return observation

        # Get camera validity mask
        # In pretrain mode, camera_valid_mask is routed to complementary_data by
        # batch_to_transition (it lacks the "observation." prefix). Read it from
        # self.transition, mirroring how LolaQwenProcessor reads "task".
        camera_valid_mask = observation.get('camera_valid_mask', {})
        if not camera_valid_mask:
            if hasattr(self, 'transition') and self.transition is not None:
                from lerobot.processor.core import TransitionKey
                comp_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
                camera_valid_mask = comp_data.get('camera_valid_mask', {})

        # Detect batch mode: camera values are lists of B items (from collate)
        first_cam_data = None
        for cam_key in camera_keys:
            if cam_key in observation:
                first_cam_data = observation[cam_key]
                break

        is_batch = (
            isinstance(first_cam_data, list)
            and len(first_cam_data) > 0
            and isinstance(first_cam_data[0], (Image.Image, type(None), torch.Tensor))
        )

        if is_batch:
            batch_size = len(first_cam_data)

            # camera_valid_mask: list[dict] of length B, or single dict (broadcast)
            if isinstance(camera_valid_mask, list):
                per_item_masks = camera_valid_mask
            else:
                per_item_masks = [camera_valid_mask] * batch_size

            # Build per-item image lists
            images_per_item = [[] for _ in range(batch_size)]
            for cam_key in camera_keys:
                if cam_key not in observation:
                    continue
                cam_images = observation[cam_key]  # list[PIL | None] of length B
                for i in range(batch_size):
                    cam_valid = (
                        per_item_masks[i].get(cam_key, True)
                        if isinstance(per_item_masks[i], dict)
                        else True
                    )
                    if not cam_valid:
                        continue
                    img = cam_images[i]
                    if img is not None:
                        if isinstance(img, Image.Image):
                            images_per_item[i].append(img)
                        elif isinstance(img, torch.Tensor):
                            if img.dim() == 3:
                                images_per_item[i].append(self._tensor_to_pil(img))
                            elif img.dim() == 4:
                                images_per_item[i].append(self._tensor_to_pil(img[-1]))

            new_observation['_lola_images_per_item'] = images_per_item
        else:
            # Single-item mode (inference): flatten as before
            images = []
            for cam_key in camera_keys:
                if cam_key not in observation:
                    continue
                cam_valid = (
                    camera_valid_mask.get(cam_key, True)
                    if isinstance(camera_valid_mask, dict)
                    else True
                )
                if not cam_valid:
                    continue
                img_data = observation[cam_key]
                if isinstance(img_data, Image.Image):
                    images.append(img_data)
                elif isinstance(img_data, torch.Tensor):
                    if img_data.dim() == 3:
                        images.append(self._tensor_to_pil(img_data))
                    elif img_data.dim() == 4:
                        images.append(self._tensor_to_pil(img_data[-1]))
                elif isinstance(img_data, dict) and 'image' in img_data:
                    images.append(img_data['image'])

            if images:
                new_observation['_lola_images'] = images

        return new_observation

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] tensor to PIL Image."""
        img = tensor.permute(1, 2, 0)  # [C, H, W] → [H, W, C]
        if img.dtype in [torch.float32, torch.float64, torch.bfloat16, torch.float16]:
            img = img.float()
            img = (img * 255).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(img.cpu().numpy())
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="lola_qwen_processor")
class LolaQwenProcessor(ObservationProcessorStep):
    """
    Processes images and text using Qwen3.5's apply_chat_template.
    
    This is the main processor for LoLA that:
    1. Uses Qwen3.5's AutoProcessor to handle multi-modal input
    2. Formats the conversation using Qwen3.5's chat template
    3. Generates input_ids, attention_mask, pixel_values, and image_grid_thw
    
    The processor uses the apply_chat_template method which is the recommended
    way to prepare inputs for Qwen3.5 models.
    """
    
    def __init__(
        self,
        processor_name: str = "Qwen/Qwen3.5-4B",
        max_length: int = 512,
        task_key: str = "task",
        completed_tasks_key: str = "completed_tasks",
        completed_tasks_ann_key: str = "completed_tasks_ann",
        task_text_template_version: str = "raw",
        max_image_pixels: int = 230400,
        min_image_pixels: int = 65536,
        static_vlm_padding: bool = False,
        vlm_max_length: int | None = None,
        **kwargs
    ):
        """
        Args:
            processor_name: The HuggingFace model name for the processor.
            max_length: Maximum sequence length for tokenization.
            task_key: Key in complementary_data containing the task description.
            completed_tasks_key: Key for completed task labels (list[str]).
            completed_tasks_ann_key: Key for completed task annotation texts (list[str]).
            task_text_template_version: "raw" = old behavior (just task string),
                "v1_with_completed" = new template with completed tasks.
            max_image_pixels: Maximum pixels per image for Qwen3.5 smart_resize.
            min_image_pixels: Minimum pixels per image for Qwen3.5 smart_resize.
            static_vlm_padding: Pad VLM tokens to fixed vlm_max_length instead of dynamic.
            vlm_max_length: Override tokenizer max_length for static padding.
        """
        super().__init__(**kwargs)
        self.processor_name = processor_name
        self.max_length = max_length
        self.task_key = task_key
        self.completed_tasks_key = completed_tasks_key
        self.completed_tasks_ann_key = completed_tasks_ann_key
        self.task_text_template_version = task_text_template_version
        self.static_vlm_padding = static_vlm_padding
        self.vlm_max_length = vlm_max_length

        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install transformers`."
            )

        if os.path.isdir(processor_name):
            self.qwen_processor = AutoProcessor.from_pretrained(processor_name, local_files_only=True)
        else:
            self.qwen_processor = AutoProcessor.from_pretrained(processor_name)
        self.qwen_processor.image_processor.max_pixels = max_image_pixels
        self.qwen_processor.image_processor.min_pixels = min_image_pixels
    
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Processes images and text using Qwen3.5's chat template.

        Supports batch mode (_lola_images_per_item) and single-item mode (_lola_images).
        Batch mode uses apply_chat_template with padding=True for proper [B, seq_len] output.
        """
        new_observation = dict(observation)

        # Get task + completed_tasks from complementary_data
        task = None
        completed_tasks_ann = None
        if hasattr(self, 'transition') and self.transition is not None:
            from lerobot.processor.core import TransitionKey
            complementary_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            task = complementary_data.get(self.task_key, "Perform the robot task.")
            completed_tasks_ann = complementary_data.get(self.completed_tasks_ann_key, None)

        if task is None:
            task = "Perform the robot task."

        # Format text based on template version
        if self.task_text_template_version == "v1_with_completed" and completed_tasks_ann:
            if isinstance(completed_tasks_ann, list) and len(completed_tasks_ann) > 0:
                numbered = [f"{i+1}. {t}" for i, t in enumerate(completed_tasks_ann)]
                tasks_str = ", ".join(numbered)
                if isinstance(task, list):
                    text_content = [f"Perform task: {t}. Completed: {tasks_str}" for t in task]
                else:
                    text_content = f"Perform task: {task}. Completed: {tasks_str}"
            else:
                if isinstance(task, list):
                    text_content = [f"Perform task: {t}" for t in task]
                else:
                    text_content = f"Perform task: {task}"
        else:
            text_content = task  # raw mode (backward compatible)

        # Check for batch mode
        images_per_item = observation.get('_lola_images_per_item', None)

        if images_per_item is not None:
            # Batch mode: one conversation per item
            batch_size = len(images_per_item)

            # Handle text_content: list[str] or single string
            if isinstance(text_content, list):
                per_item_texts = text_content
            else:
                per_item_texts = [text_content] * batch_size

            messages = []
            for i in range(batch_size):
                content = []
                for img in images_per_item[i]:
                    content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": per_item_texts[i]})
                messages.append([{"role": "user", "content": content}])

            self.qwen_processor.tokenizer.padding_side = 'left'
            inputs = self.qwen_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length" if self.static_vlm_padding else True,
                max_length=self.vlm_max_length if self.static_vlm_padding else None,
            )
        else:
            # Single-item mode (inference)
            images = observation.get('_lola_images', [])
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            text_str = text_content if isinstance(text_content, str) else str(text_content)
            content.append({"type": "text", "text": text_str})
            messages = [[{"role": "user", "content": content}]]

            self.qwen_processor.tokenizer.padding_side = 'left'
            inputs = self.qwen_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length" if self.static_vlm_padding else True,
                max_length=self.vlm_max_length if self.static_vlm_padding else None,
            )

        # Extract outputs
        new_observation[OBS_LANGUAGE_TOKENS] = inputs["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = inputs["attention_mask"]

        # Add visual features if present
        if "pixel_values" in inputs:
            new_observation["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            new_observation["image_grid_thw"] = inputs["image_grid_thw"]

        # Clean up temporary image storage
        for key in ['_lola_images', '_lola_images_per_item', 'camera_valid_mask']:
            if key in new_observation:
                del new_observation[key]
        # Remove camera key observations (PIL Image / None / tensor) — already processed into pixel_values
        for key in list(new_observation.keys()):
            if key.startswith('observation.images.'):
                del new_observation[key]

        return new_observation
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Adds feature definitions for Qwen3.5 outputs.
        
        Args:
            features: The input feature dictionary.
            
        Returns:
            Updated feature dictionary with Qwen3.5 output features.
        """
        # Add language tokens feature
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        
        # Add attention mask feature
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        
        return features


def make_lola_pre_post_processors(
    config: LoLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    camera_keys: list[str] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the LoLA policy.
    
    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Processing multi-camera images for Qwen3.5 vision encoder.
    3. Tokenizing text and images using Qwen3.5's apply_chat_template.
    4. Appending an empty token for global intent control.
    5. Normalizing input and output features based on dataset statistics.
    6. Moving all data to the specified device.
    
    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.
    
    Args:
        config: The configuration object for the LoLA policy.
        dataset_stats: A dictionary of statistics for normalization.
        camera_keys: List of camera keys to process (e.g., ['observation.images.left']).
                    If None, automatically detects camera keys from observation.
        
    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    
    Example:
        ```python
        from lerobot.policies.lola import LoLAConfig, make_lola_pre_post_processors
        
        config = LoLAConfig(vlm_model_name="Qwen/Qwen3.5-4B")
        preprocessor, postprocessor = make_lola_pre_post_processors(
            config,
            dataset_stats=dataset_stats,
            camera_keys=['observation.images.left', 'observation.images.right'],
        )
        ```
    """
    
    # Determine processor settings from config
    # Use local vlm_path for processor when available, fall back to vlm_model_name
    processor_name = config.vlm_path if config.vlm_path is not None else config.vlm_model_name
    max_length = getattr(config, 'tokenizer_max_length', 512)
    
    # Pre-processor steps
    # The pipeline processes data in the following order:
    # 1. Rename features (compatibility with pretrained format)
    # 2. Process images (extract and convert to PIL)
    # 3. Process with Qwen3.5 (apply_chat_template for text + images)
    # 4. Append empty token (global intent control for LoLA)
    # 5. Add batch dimension
    # 6. Move to device
    # 7. Normalize features
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To maintain compatibility with pretrained format
        LolaImageProcessor(camera_keys=camera_keys),  # Extract and prepare images for Qwen3.5
        LolaQwenProcessor(  # Process text + images with Qwen3.5's apply_chat_template
            processor_name=processor_name,
            max_length=max_length,
            max_image_pixels=config.max_image_pixels,
            min_image_pixels=config.min_image_pixels,
            static_vlm_padding=config.static_vlm_padding,
            vlm_max_length=config.vlm_max_length,
            task_text_template_version=config.task_text_template_version,
            completed_tasks_key="completed_tasks",
            completed_tasks_ann_key="completed_tasks_ann",
        ),
        LolaEmptyTokenProcessor(empty_token_id=config.empty_token_id),  # Append empty token for LoLA
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    
    # Post-processor steps
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, 
            norm_map=config.normalization_mapping, 
            stats=dataset_stats
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
