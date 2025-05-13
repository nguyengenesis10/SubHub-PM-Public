

import torch


from transformers import MllamaForConditionalGeneration,AutoProcessor


def setup_model():
    model = MllamaForConditionalGeneration.from_pretrained(
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model


def setup_processor():
    processor = AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
    )
    return processor