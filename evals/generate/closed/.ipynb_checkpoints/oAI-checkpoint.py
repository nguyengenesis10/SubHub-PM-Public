

import ast


from typing import Iterable
from openai import OpenAI
from pydantic import BaseModel
from dataclasses import dataclass
from PIL.PngImagePlugin import PngImageFile


from . import OPENAI, GOOGLE, api_keys
from .utils import base64_encode_pil_image


@dataclass
class OpenAIConfig:
    model_type:str=GOOGLE
    model:str="gemini-2.5-pro-preview-05-06"
    temperature:float=1
    response_format:BaseModel=None

    def __post_init__(
        self
    ):
        if self.model_type not in (OPENAI,GOOGLE):
            raise ValueError(
                f"Invalid model type {self.model_type} should be either {GOOGLE} or {OPENAI}!"
            )
        if not self.response_format :
            raise ValueError(
                f"Response format is set to None! Please define one."
            )


class OpenAIGenerator:
    def __init__(
        self,
        cfg:OpenAIConfig,
    ):
        
        if cfg.model_type == OPENAI:
            self.client=OpenAI(
                api_key=api_keys[OPENAI]
            )
        elif cfg.model_type == GOOGLE:
            self.client=OpenAI(
                api_key=api_keys[GOOGLE],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        self.model=cfg.model
        self.temperature=cfg.temperature
        self.response_format=cfg.response_format


    def _encode_image(
        self,
        image:PngImageFile,
    ):
        return f"data:image/jpeg;base64,{base64_encode_pil_image(image)}"


    def _build_messages(
        self,
        system_prompt:str,
        prompt:str,
        images:list[PngImageFile],
    )->list[dict]:
        messages:list[dict]=[
            {"role" : "system" , "content" : system_prompt},
            {
                "role" : "user" , "content" : [
                    {"type": "text","text": prompt} , 
                ]
            },
        ] 
        for image in images:
            messages[1]["content"].append({
                "type": "image_url" , "image_url": {
                    "url": self._encode_image(image) 
                },
            })
        return messages


    def __call__(
        self,
        system_prompt:str,
        prompt:str,
        images:Iterable[PngImageFile],
    )->list[str]:
        '''
        Args:
            prompt: Assume all variables required for the prompt have already been formatted into the string 
        '''
        messages=self._build_messages(
            system_prompt=system_prompt,
            prompt=prompt,
            images=images,
        )
        
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=self.response_format,
            temperature=self.temperature,
        )
        
        resp_dict=ast.literal_eval(response.choices[0].message.content)

        resps:list[str]=[]
        for k,v in resp_dict.items():
            resps.append(
                getattr(
                    self.response_format(**resp_dict), k, v
                )
            )
        return resps















#
        
        
        