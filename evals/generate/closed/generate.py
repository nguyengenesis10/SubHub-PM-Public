

from typing import Union


from ..configs import GenerationConfig
from ..format import DynamicPrompt
from .anthropic import AnthropicGenerator
from .oAI import OpenAIGenerator


class ClosedSrcGenerator:
    def __init__(
        self,
        cfg:GenerationConfig,
        generator:Union[OpenAIGenerator,AnthropicGenerator]
    ):
        
        if len(cfg.system_prompts) != len(cfg.prompts):
            raise ValueError(
                f"Length of system prompts and prompts should be the same and zippable!"
            )
        
        self.batch_images , self.page_ids=cfg._setup_images()
        self.system_prompts=cfg.system_prompts
        self.prompts=cfg.prompts
        self.generator=generator


    def _build_prompts(
        self,
        prompt:str,
        *args,
    ):
        if "{}" in prompt:
            return prompt.format(*args)
        return prompt


    def __call__(
        self,
    ):
        '''
        Iterate through the batch of pages 

        make all your prompt calls 

        and then add to dataframe 
        '''

        all_resps=[]
        for batch, page_ids in zip(self.batch_images,self.page_ids):
            batch_resps=[]
            for idx, (system_prompt, prompt) in enumerate(
                zip(self.system_prompts,self.prompts)
            ):
                if idx == 0:
                    resps=self.generator(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        images=batch,
                    )
                else:
                    resps=self.generator(
                        system_prompt=system_prompt,
                        prompt=self._build_prompts(prompt,*page_ids+resps),
                        images=batch,
                    )
                batch_resps.extend(resps)
            all_resps.append(batch_resps)
        return all_resps



#
