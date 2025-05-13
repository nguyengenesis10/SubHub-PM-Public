

import anthropic


from PIL.PngImagePlugin import PngImageFile
from dataclasses import dataclass


from . import api_keys, ANTHROPIC
from .utils import anthropic_encode_pil_image


@dataclass 
class AnthropicConfig:
    model:str
    max_tokens:int
    temperature:float
    system_prompts:list[str]
    user_prompts:list[str]


class AnthropicGenerator:
    def __init__(
        self,
        cfg:AnthropicConfig,
    ):
        self.client=anthropic.Anthropic(
            api_key=api_keys[ANTHROPIC]
        )
        self.model=cfg.model
        self.max_tokens=cfg.max_tokens
        self.system_prompts=cfg.system_prompts
        self.user_prompts=cfg.user_prompts
        

    def _encode_image(
        self,
        image:PngImageFile
    ):
        return anthropic_encode_pil_image(image)


