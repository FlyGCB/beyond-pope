"""
Idefics3 inference wrapper.

Tested with: HuggingFaceM4/Idefics3-8B-Llama3
"""

from pathlib import Path
import torch
from PIL import Image

from .base import BaseVLM


class Idefics3(BaseVLM):

    def __init__(
        self,
        model_name: str = "idefics3_8b",
        model_id: str = "HuggingFaceM4/Idefics3-8B-Llama3",
        device: str = "cuda",
    ):
        super().__init__(model_name=model_name, device=device, model_id=model_id)
        self.model_id = model_id

    def load_model(self, model_id: str, **kwargs):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def predict(self, image_path: str | Path, question: str) -> str:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
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
