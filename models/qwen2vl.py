"""
Qwen2-VL inference wrapper.

Tested with: Qwen/Qwen2-VL-7B-Instruct
HuggingFace: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

Usage:
    model = Qwen2VL(model_name="qwen2vl_7b")
    summary = model.evaluate_file(
        input_path="data/processed/xpope_existence.jsonl",
        image_dir="data/raw/coco/val2014",
        output_path="results/predictions/qwen2vl_7b_xpope_existence.jsonl",
    )
"""

from pathlib import Path
import torch
from PIL import Image

from .base import BaseVLM


class Qwen2VL(BaseVLM):

    def __init__(
        self,
        model_name: str = "qwen2vl_7b",
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,
    ):
        super().__init__(model_name=model_name, device=device,
                         model_id=model_id, load_in_4bit=load_in_4bit)
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit

    def load_model(self, model_id: str, load_in_4bit: bool, **kwargs):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from transformers import BitsAndBytesConfig

        self.logger.info(f"Loading {model_id} (4bit={load_in_4bit})")

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant_config,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_id)

    def predict(self, image_path: str | Path, question: str) -> str:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=5,      # only need "Yes" or "No"
                do_sample=False,       # greedy — deterministic
                temperature=1.0,
                top_p=1.0,
            )

        # Slice off the input tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated[0], skip_special_tokens=True).strip()
