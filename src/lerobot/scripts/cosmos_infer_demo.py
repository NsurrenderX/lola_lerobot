from pathlib import Path

import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

model_path = "/data_6t_1/cosmos3/Cosmos3-Nano/"
model_id  = model_path
# model_id = "nvidia/Cosmos3-Nano"
image_path = Path("/datassd_1T/pizza_data/image_web_0_test_0.jpg").resolve()

processor = AutoProcessor.from_pretrained(model_id)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": str(image_path)},
            {"type": "text", "text": "Caption the image in detail."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output[0])