"""
Llama-3.2-Vision inference wrapper.

Tested with: meta-llama/Llama-3.2-11B-Vision-Instruct
"""

from pathlib import Path
import torch
from PIL import Image

from .base import BaseVLM


class Llama32Vision(BaseVLM):

    def __init__(
        self,
        model_name: str = "llama32v_11b",
        model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        device: str = "cuda",
    ):
        super().__init__(model_name=model_name, device=device, model_id=model_id)
        self.model_id = model_id

    def load_model(self, model_id: str, **kwargs):
        from transformers import MllamaForConditionalGeneration, AutoProcessor

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def predict(self, image_path: str | Path, question: str) -> str:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            image, text, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )

        decoded = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return decoded.strip()
