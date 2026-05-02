"""
Phi-3.5-Vision inference wrapper.

Tested with: microsoft/Phi-3.5-vision-instruct
HuggingFace: https://huggingface.co/microsoft/Phi-3.5-vision-instruct

Usage:
    model = Phi35Vision(model_name="phi35v_4b")
    summary = model.evaluate_file(
        input_path="data/raw/benchmarks/pope/pope_adversarial.jsonl",
        image_dir="data/raw/coco/val2014",
        output_path="results/predictions/phi35v_4b_pope_adversarial.jsonl",
    )
"""

from pathlib import Path
import torch
from PIL import Image

from .base import BaseVLM


class Phi35Vision(BaseVLM):

    def __init__(
        self,
        model_name: str = "phi35v_4b",
        model_id: str = "microsoft/Phi-3.5-vision-instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,
    ):
        super().__init__(model_name=model_name, device=device,
                         model_id=model_id, load_in_4bit=load_in_4bit)
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit

    def load_model(self, model_id: str, load_in_4bit: bool = False, **kwargs):
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers import BitsAndBytesConfig

        self.logger.info(f"Loading {model_id} (4bit={load_in_4bit})")

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,         # required for Phi-3.5-Vision
            quantization_config=quant_config,
            _attn_implementation="eager",
        )
        self.model.eval()

        # num_crops=1: single crop, sufficient for POPE-style single-object questions
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=1,
        )

    def predict(self, image_path: str | Path, question: str) -> str:
        image = Image.open(image_path).convert("RGB")

        # Phi-3.5-Vision uses a chat template with <|image_1|> placeholder
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{question}"},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        decoded = self.processor.tokenizer.decode(
            output[0][input_len:],
            skip_special_tokens=True,
        )
        return decoded.strip()
