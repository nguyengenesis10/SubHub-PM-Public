

from dataclasses import dataclass 


@dataclass
class DynamicPrompt:
    system_prompt:str
    prompt:str
    formatted_prompt:str=None
    page_id:str=None


    def __call__(
        self,
        *args,
    ):
        if "{page_id}" in self.prompt:
            formatted_prompt=self.prompt.format(
                *args,
                page_id=self.page_id,
            )

        return formatted_prompt