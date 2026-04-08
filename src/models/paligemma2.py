"""
PaliGemma2 inference wrapper.

Tested with: google/paligemma2-3b-pt-448
"""

from pathlib import Path
import torch
from PIL import Image

from .base import BaseVLM


class PaliGemma2(BaseVLM):

    def __init__(
        self,
        model_name: str = "paligemma2_3b",
        model_id: str = "google/paligemma2-3b-pt-448",
        device: str = "cuda",
    ):
        super().__init__(model_name=model_name, device=device, model_id=model_id)
        self.model_id = model_id

    def load_model(self, model_id: str, **kwargs):
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def predict(self, image_path: str | Path, question: str) -> str:
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
            padding=True,
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
