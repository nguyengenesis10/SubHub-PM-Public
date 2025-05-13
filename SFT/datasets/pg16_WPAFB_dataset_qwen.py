
import copy
import itertools

import torch
import ast

import io
from PIL import Image, JpegImagePlugin

from datasets import load_dataset

'''
from transformers.models.mllama.configuration_mllama import (
    MllamaConfig, 
    MllamaVisionConfig, 
    MllamaTextConfig
)
from transformers import (
    MllamaForConditionalGeneration, 
    AutoProcessor,
)
'''


# Adapted to Qwen level processors, finds the assistant prompt and masks everything but the expected response.
def tokenize_dialogs(
    dialogs:list[dict],
    images:list[JpegImagePlugin],
    processor
):
    text_prompt = processor.apply_chat_template(
        dialogs,
        tokenize=False,
        add_generation_prompt=False # dialogs should be setup to already have assistant in them
    )
    batch=processor(
        text=[text_prompt],
        images=images,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    batch = {k: (v.to(torch.bfloat16) if k == "pixel_values" else v) for k, v in batch.items()}

    label_list=[]
    for input_ids in batch["input_ids"]:
        if len(input_ids.shape) != 1:
            raise ValueError(
                f"Shape of 'input_ids' should be only one dimension, but is shape {input_ids.shape}!"
            )
        dialog_tokens=input_ids.tolist()
        labels=copy.copy(dialog_tokens)
        
        # Qwen assistant prompt, <|im_start|>assistant. Essentially anything after should be masked.
        start=151644
        end=151645
        assistant=77091
        for idx,input_id in enumerate(dialog_tokens):
            if idx == len(dialog_tokens)-1:
                break
            if input_id == start and dialog_tokens[idx+1] == assistant:
                #Mask everything up to the start of the assistant prompt.
                labels[:idx+2]=[-100 for i in range(idx+1)]
                #Mask the final end token.
                labels[labels[idx+2:].index(end)]=-100
        
        label_list.append(labels)
    batch["labels"]=torch.tensor(label_list)

    return batch
        


'''
Fetch dataset , properly formated each sample, such that each image 
is in nested list
'''
def get_custom_dataset(dataset_config, processor,split,split_ratio=0.9):
    dataset = load_dataset("genesis1SubHub/WP_AFB_PG16")
    
    def wrap_image_in_list(row):
        row["images"] = [row["images"]]  # Wrap image in a list
        return row
    
    if split == "train":
        return dataset["train"].map(wrap_image_in_list)
    elif split == "test":
        return dataset["validation"].map(wrap_image_in_list)
    elif not split:
        quit("split is unspecified!")


# used to convert images back to type PIL.JpegImagePlugin.JpegImageFile, so processor can handle
def bytes_to_image(image_bytes):
    
    
    # Image byte data (replace this with your provided byte data)
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    jpeg_image_stream = io.BytesIO()
    # Convert to RGB as JPEG doesn't support transparency
    image = image.convert("RGB")
    image.save(jpeg_image_stream, format="JPEG")
    
    jpeg_image_stream.seek(0)  # Reset stream position to the beginning
    jpeg_image = JpegImagePlugin.JpegImageFile(jpeg_image_stream)

    return jpeg_image


class CustomDataCollator:
    def __init__(self,processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
    
    def __call__(self,dataset):
        dialogs = []
        images = []
        for sample in dataset:
            if isinstance(sample, dict):
                image_bytes = sample["images"][0]["bytes"]
                jpeg_image = bytes_to_image(image_bytes)
                images.append([jpeg_image])
                # converts to list 
                dialogs.append(ast.literal_eval(sample["texts"]))
        return tokenize_dialogs(dialogs,images, self.processor)


def get_data_collator(processor):
    return CustomDataCollator(processor)







