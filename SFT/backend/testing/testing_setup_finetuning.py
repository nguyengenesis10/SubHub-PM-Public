
import copy
import itertools

import torch
import ast

from .modeling_mllama_vision import MllamaSuperVisionModel, MllamaProjectManagerForConditionalGeneration

from datasets import load_dataset

from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaVisionConfig, MllamaTextConfig
from transformers.models.mllama.image_processing_mllama import get_all_supported_aspect_ratios
from transformers import MllamaForConditionalGeneration, AutoProcessor


# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] in targets:
            return True
    return False

def replace_target(target,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] == target:
            seq[i],seq[i+1],seq[i+2] = -100,-100,-100
    return seq


def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt,padding = True, return_tensors="pt")
    label_list = []
    # loop through each sample and mask "user, system, assistant,image, and pad tokens to be -100"
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i,n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs,current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq,labels)
        # Mask the padding token and image token 128256 
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256: #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


# fetch dataset 
def get_custom_dataset(repo_id, processor,split):
    dataset = load_dataset(repo_id)
    
    dialogs_train = [ast.literal_eval(i) for i in dataset["train"]["texts"]]
    images_train  = dataset["train"]["images"]
    
    dialogs_val = [ast.literal_eval(i) for i in dataset["validation"]["texts"]]
    images_train  = dataset["validation"]["images"]
    
    batch_train = tokenize_dialogs(dialogs_train, images_train, processor)
    batch_val = tokenize_dialogs(dialogs_train, images_train, processor)

    if split == "train":
        return batch_train
    elif split == "test":
        return batch_val
    elif not split:
        quit("split is unspecified!")


def conver_to_nested_PIL(images):
    for idx, i in enumerate(images):
        if isinstance(i, list):
            bytes_dict = i[0]
            if isinstance(bytes_dict.get("bytes") , bytes):
                from PIL import Image
                import io
                image_bytes = bytes_dict.get("bytes")
                image = Image.open(io.BytesIO(image_bytes))
                images[idx] = [image]
                print(f"Identified type bytes & converted --> {type(image)}")
    return images


class CustomDataCollator:
    def __init__(self,processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
    
    def __call__(self,dataset):

        dialogs = []
        images = []
        
        for sample in dataset:
            if isinstance(sample, dict):
                images.append(sample["images"])
                # converts to list 
                dialogs.append(ast.literal_eval(sample["texts"]))
        
        
        return tokenize_dialogs(dialogs,images, self.processor)

def get_data_collator(processor):
    return CustomDataCollator(processor)

'''
import sys ; sys.path.append("/Users/genesisnguyen/SaaS_dev") 

from backend.PM.finetuning_new import init_model

from transformers.models.mllama.modeling_mllama_vision import MllamaSuperVisionModel, MllamaVisionModel, SuperEncoders
from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaVisionConfig, MllamaTextConfig
from transformers import MllamaForConditionalGeneration

model_path = "/Users/genesisnguyen/llama3.2-11B-vision-instruct"
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

config = MllamaConfig.from_pretrained(model_path)

vision_config = MllamaVisionConfig.from_pretrained(
    model_path, 
)

vision_config.use_acronym_encoder = True
vision_config.use_measurement_encoder = True
vision_config.use_drawing_encoder = True

vision_config.image_size = 3000

super_vision_path_or_id = model_path
use_super_vision_model = True

model = MllamaForConditionalGeneration(config, vision_config, super_vision_path_or_id, use_super_vision_model)

'''

def init_model(
        model_id_or_path,
        use_acronym_encoder,
        use_measurement_encoder,
        use_drawing_encoder,
        bnb_config = None,
        # defaults pulled from mllama3.2-11b
        image_size = 560,
        max_num_tiles = 4,
        patch_size = 14,
    ):

    # vision-model configuration w/ added attributes and custom image_size (rly tile size)
    config = MllamaVisionConfig.from_pretrained(pretrained_model_name_or_path = model_id_or_path)

    config.image_size = image_size
    config.use_acronym_encoder = use_acronym_encoder
    config.use_drawing_encoder = use_drawing_encoder
    config.use_measurement_encoder = use_measurement_encoder
    config.pretrained_model_name_or_path = model_id_or_path

    if max_num_tiles != 4:
        config.max_num_tiles = max_num_tiles
        config.supported_aspect_ratios = get_all_supported_aspect_ratios(max_num_tiles)
    if patch_size != 14:
        config.patch_size = patch_size
    
    # load whole multi-modal model 
    model = MllamaForConditionalGeneration.from_pretrained(
                                                           model_id_or_path,
                                                           quantization_config = bnb_config,
                                                           ignore_mismatched_sizes = True
                                                          )

    # load vision head seperately 
    vision_model = MllamaSuperVisionModel.from_pretrained(
                                                          model_id_or_path,
                                                          config = config,
                                                          ignore_mismatched_sizes = True,
                                                          quantization_config = bnb_config
                                                         )

    # switch to different vision head
    model.vision_model = vision_model
    model.config.vision_config = vision_model.config
    
    return model


