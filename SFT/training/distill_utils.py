
import os 
import sys 

# Add root directory, for higher level imports.
sys.path.append(os.path.abspath(__file__)[
    :os.path.abspath(__file__).find("SubHubProjectManager")+len("SubHubProjectManager")
    ]
)

import torch

from llama_recipes.utils.dataset_utils import (
    get_preprocessed_dataset,
)
from llama_recipes.utils.config_utils import (
    get_dataloader_kwargs,
)
from transformers.models.mllama.image_processing_mllama import (
    MllamaImageProcessor
)
from transformers.models.mllama.processing_mllama import (
    MllamaProcessor
)
from transformers.models.mllama.configuration_mllama import (
    MllamaConfig,
    MllamaVisionConfig,
)
from transformers.models.mllama.modeling_mllama import (
    MllamaVisionModel,
    MllamaForConditionalGeneration
)
from backend.modeling_mllama_vision import (
    MllamaProjectManagerForConditionalGeneration,
    MllamaSuperVisionModel
)
from transformers import (
    AutoConfig,
    AutoProcessor,
)

'''
Determines which initialization class to use for 
processors, configurations, and models.
'''
model_init_dict = {
    "teacher" : {
        "vision" : MllamaVisionModel,
        "multimodal" : MllamaForConditionalGeneration,
    }, 
    "student" : {
        "vision" : MllamaSuperVisionModel,
        "multimodal" : MllamaProjectManagerForConditionalGeneration,
    },
}
config_init_dict = {
    "vision" : MllamaVisionConfig,
    "multimodal" : AutoConfig,
}
processor_init_dict = {
    "vision" : MllamaImageProcessor,
    "multimodal" : AutoProcessor,
}


'''
Sets embedding layers for full parameter finetuning , if 'reverse==True' 
then will revert all params that are not in the embedding layers to 
'requires_grad=False'. 

The targeted layers are much more strict b/c fsdp creates wrapping complications.
'''
def prepare_vision_embedding_layers(model, reverse=False):
    embeddings = [
        "gated_positional_embedding.embedding",
        "gated_positional_embedding.tile_embedding",
        "patch_embedding",
        "post_tile_positional_embedding.embedding",
        "pre_tile_positional_embedding.embedding",
        "gated_positional_embedding._fsdp_wrapped_module.embedding",
        "gated_positional_embedding._fsdp_wrapped_module.tile_embedding",
        "pre_tile_positional_embedding._fsdp_wrapped_module.embedding",
        "post_tile_positional_embedding._fsdp_wrapped_module.embedding",
    ]

    for name, param in model.named_parameters():
        match = any(embed in name for embed in embeddings)
        param.requires_grad = match if not reverse else not match


'''
Creates dataloaders for a given processor config, mimics 
progressively increasing resolution of images.
'''
def create_dataloaders(
    teacher,
    teacher_processor,
    teacher_custom_data_collator,
    student,
    student_processor,
    student_custom_data_collator,
    distill_train_config,
    dataset_config,
    split:str,
):
    teacher_dl_kwargs = get_dataloader_kwargs(
        distill_train_config, 
        teacher, 
        teacher_processor, 
        split,
    )
    
    student_dl_kwargs = get_dataloader_kwargs(
        distill_train_config,
        student,
        student_processor, 
        split,
    )
    
    if student_custom_data_collator and teacher_custom_data_collator:
        print("Custom_data_collator is used.")
        teacher_dl_kwargs["collate_fn"] = teacher_custom_data_collator
        student_dl_kwargs["collate_fn"] = student_custom_data_collator
    
    teacher_dataloader = torch.utils.data.DataLoader(
        teacher,
        num_workers=distill_train_config.num_workers_dataloader,
        pin_memory=True,
        **teacher_dl_kwargs,
    )
    
    student_dataloader = torch.utils.data.DataLoader(
        student,
        num_workers=distill_train_config.num_workers_dataloader,
        pin_memory=True,
        **student_dl_kwargs,
    )
    
    return teacher_dataloader,student_dataloader


'''
Fetches all the datasets for distillation assumes that there is a 
"train" and "test" split for each dataset.
'''
def fetch_all_datasets(
    teacher_processor,
    student_processor,
    dataset_config
):
    processors=[teacher_processor,student_processor]
    splits=["train","test"]
    datasets = []
    for processor in processors:
        for split in splits:
            datasets.append(
                get_preprocessed_dataset(
                    processor,
                    dataset_config,
                    split=split,
                    )
                )
    # teacher_train , teacher_val , student_train , student_val - in that order
    return tuple(datasets)


# Check processor configuration and compares it w/ the student model config.  
def check_processor(
    processor:MllamaImageProcessor, 
    student_cfg:MllamaVisionConfig,
):
    if (
        not isinstance(
            processor,
            MllamaImageProcessor
        )
        and isinstance(
            config,
            MllamaVisionConfig
        )
    ):
        raise ValueError(
    f"Procesor is of type {type(processor)} and config is of type {type(config)}, must be of type MllamaImageProcessor and MllamaVisionConfig!"
            )
    if (
        processor.size["height"] != student_cfg.image_size or 
        processor.size["width"] != student_cfg.image_size or 
        processor.max_image_tiles != student_cfg.max_num_tiles 
    ):
        return True

    else:
        return False


# maintain homogeniety w/ processor and model settings 
def prep_processor(
    processor,
    config
):
    if (
        isinstance(processor,MllamaImageProcessor) and
        isinstance(config,MllamaVisionConfig)
    ):
        processor.size = {
            'height': config.image_size , 
            'width': config.image_size
        }
        processor.max_image_tiles = config.max_num_tiles
        
    elif (
        isinstance(processor,MllamaProcessor) and
        isinstance(config,MllamaConfig)
    ):
        processor.image_processor.size = {
            'height': config.vision_config.image_size ,
            'width': config.vision_config.image_size
        }
        processor.image_processor.max_image_tiles = config.vision_config.max_num_tiles
        processor.tokenizer.padding_side = "right"

    elif (
        isinstance(config,dict)
    ):
        # check for precense of required keys
        for k in ["image_size" , "max_num_tiles"]:
            if k not in list(config.keys()):
                raise KeyError(
                    f"Key {k} is not in the processor configuration: {config}!"
                )
        
        if isinstance(processor, MllamaProcessor):
                    processor.image_processor.size = {
                        'height': config["image_size"],
                        'width' : config["image_size"]
                    }
                    processor.image_processor.max_image_tiles = config["max_num_tiles"]
                    processor.tokenizer.padding_side = "right"
        
        elif isinstance(processor, MllamaImageProcessor):
                processor.size = {
                    'height': config["image_size"],
                    'width' : config["image_size"]
                }
                processor.max_image_tiles = config["max_num_tiles"]
        
    else:
        raise ValueError(
            f"Procesor is of type {type(processor)}. Config is of type {type(config)}!"
        )
    return processor


if __name__ == "__main__":
    
    from llama_recipes.utils.dataset_utils import (
        get_preprocessed_dataset,
    )






#
