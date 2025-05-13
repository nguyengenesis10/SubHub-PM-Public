
import copy
import itertools

import torch
import ast


from datasets import load_dataset
from transformers.models.mllama.configuration_mllama import MllamaConfig, MllamaVisionConfig, MllamaTextConfig
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

    #TEMP UNTIL WE FIND BETTER FIX, convert pixel values to bfloat16 as model weights are in bf16
    batch = {k: (v.to(torch.bfloat16) if k == "pixel_values" else v) for k, v in batch.items()}
    # [print(k, v.dtype) for k,v in batch.items()]
    #
    
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
'''
fetch dataset , properly formated each sample, such that each image 
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
    from PIL import Image, JpegImagePlugin
    import io
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







