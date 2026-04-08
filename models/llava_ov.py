"""
LLaVA-OneVision inference wrapper.

Tested with: lmms-lab/llava-onevision-qwen2-7b-ov
"""

from pathlib import Path
import torch
from PIL import Image

from .base import BaseVLM


class LLaVAOneVision(BaseVLM):

    def __init__(
        self,
        model_name: str = "llava_ov_7b",
        model_id: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        device: str = "cuda",
    ):
        super().__init__(model_name=model_name, device=device, model_id=model_id)
        self.model_id = model_id

    def load_model(self, model_id: str, **kwargs):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_name = get_model_name_from_path(model_id)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(model_id, None, model_name)
        )
        self.model.eval()
        self.model_name_str = model_name

    def predict(self, image_path: str | Path, question: str) -> str:
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        import copy

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images(
            [image], self.image_processor, self.model.config
        )[0].to(self.device, dtype=torch.float16)

        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(
            conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question
        )
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0),
                max_new_tokens=5,
                do_sample=False,
            )

        output = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        return output.strip()
