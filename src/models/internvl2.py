"""
InternVL2.5 inference wrapper.

Tested with: OpenGVLab/InternVL2_5-8B
HuggingFace: https://huggingface.co/OpenGVLab/InternVL2_5-8B
"""

from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from .base import BaseVLM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class InternVL2(BaseVLM):

    def __init__(
        self,
        model_name: str = "internvl2_8b",
        model_id: str = "OpenGVLab/InternVL2_5-8B",
        device: str = "cuda",
        load_in_8bit: bool = False,
    ):
        super().__init__(model_name=model_name, device=device,
                         model_id=model_id, load_in_8bit=load_in_8bit)
        self.model_id = model_id
        self.load_in_8bit = load_in_8bit

    def load_model(self, model_id: str, load_in_8bit: bool, **kwargs):
        from transformers import AutoTokenizer, AutoModel

        self.logger.info(f"Loading {model_id} (8bit={load_in_8bit})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.transform = build_transform(input_size=448)

    def predict(self, image_path: str | Path, question: str) -> str:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image).unsqueeze(0).to(
            device=self.device, dtype=torch.bfloat16
        )

        generation_config = dict(
            max_new_tokens=5,
            do_sample=False,
        )

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
        )
        return response.strip()
