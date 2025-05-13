

from dataclasses import dataclass
from datasets import Dataset


from .format import DynamicPrompt


class GenerationConfig:
    def __init__(
        self,
        prompts:list[tuple[str]],
        plans:Dataset,
        image_header:str,
        page_id_header:str,
        window_size:int,
        window_type:str="consecutive",
    ):
        '''
        Args:
            window_type - Essentially the given pages must be right next to the chosen page i.e. if page 1 
                          if page 1 is chosen pages 2 and 3 can be chosen, but not pages 3 and 4 w/o 2.
            prompts - A list of tuples in which each tuple contains a system prompt <> prompt - pair, in that order.
        '''
        self.window_type=window_type
        self.window_size=window_size
        
        self.system_prompts, self.prompts = map(
            list, zip(*prompts)
        ) if prompts else (
            [], []
        )
        
        if (
            len(self.system_prompts) != len(self.prompts) or 
            len(self.system_prompts) == 0 or 
            len(self.prompts) == 0 
        ):
            raise ValueError(
                f"Number of system prompts and prompts are supposed ot be the same and non-zero!"
            )
        
        if image_header not in plans.features.keys():
            raise KeyError(
                f"Image header {image_header} is not in dataset keys!"
            )
        if page_id_header not in plans.features.keys():
            raise KeyError(
                f"Page id header {page_id_header} is not in dataset keys!"
            )
        self.image_header=image_header
        self.page_id_header=page_id_header
        self.plans=plans


    def _setup_image_ranges(
        self
    ):
        CONSECUTIVE="consecutive"
        ranges_to_pull=[]
        if self.window_type == CONSECUTIVE and self.window_size != 1:
            iterator = list(
                range(
                    0,len(self.plans),self.window_size
                )
            )
            if iterator[-1] > len(self.plans)-1:
                iterator[-1] = len(self.plans)-1
            elif iterator[-1] < len(self.plans)-1:
                iterator.append(
                    len(self.plans)-1
                )
            else:
                iterator=iterator
            for working_idx, num in enumerate(iterator):
                if working_idx != len(iterator) - 1:
                    range_to_pull=(
                        num,iterator[working_idx+1]-1
                    ) if working_idx != len(iterator) - 2 else (
                        num,iterator[working_idx+1]
                    )
                    ranges_to_pull.append(range_to_pull)
                else:
                    break
        elif self.window_size == 1:
            iterator=list(
                range(0,len(self.plans)+1,self.window_size)
            )
            for working_idx, num in enumerate(iterator):
                if working_idx != len(iterator) - 1:
                    ranges_to_pull.append(
                        (num, iterator[working_idx])
                    )
        else:
            raise ValueError(
                f"Batching strategy: {self.window_type} is invalid!"
            )
        # Add as attribute for debugging purposes
        self.ranges_to_pull = ranges_to_pull 
        
        return ranges_to_pull


    def _setup_images(
        self
    ):
        ranges_to_pull = self._setup_image_ranges()
        
        image_samples=[]
        page_ids=[]
        for range_to_pull in ranges_to_pull:
            print(
                range_to_pull
            )
            samples=self.plans.select(
                range(range_to_pull[0] , range_to_pull[1]+1)
            )
            image_samples.append(
                samples[self.image_header]
            )
            page_ids.append(
                samples[self.page_id_header]
            )
        print(
            f"Number of batches: {len(image_samples)}"
        )
        return image_samples,page_ids








#