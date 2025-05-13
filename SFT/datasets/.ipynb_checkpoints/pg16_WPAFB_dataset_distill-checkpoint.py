
import ast
import copy
import torch
import itertools


from tqdm import tqdm
from datasets import load_dataset
from transformers.models.mllama.configuration_mllama import (
    MllamaConfig, 
    MllamaVisionConfig, 
    MllamaTextConfig,
)
from transformers import (
    MllamaForConditionalGeneration, 
    AutoProcessor,
)

'''
Under the assumption that only images are being processed and that processor 
is of type MllamaImageProcessor.
'''

def tokenize_dialogs(images, processor):
    batch = processor(
        images=images,
        return_tensors="pt"
    )
    
    #TEMP UNTIL WE FIND BETTER FIX, convert pixel values to bfloat16 as model weights are in bf16
    batch = {k: (v.to(torch.bfloat16) if k == "pixel_values" else v) for k, v in batch.items()}
    # [print(k, v.dtype) for k,v in batch.items()]
    #
    
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
    
    def __call__(self,dataset):
        images = []
        for sample in dataset:
            if isinstance(sample, dict):
                # only grab the first image
                image_bytes = sample["images"][0]["bytes"]
                jpeg_image = bytes_to_image(image_bytes)
                images.append([jpeg_image])
                # converts to list 
        return tokenize_dialogs(images, self.processor)

def get_data_collator(processor):
    return CustomDataCollator(processor)







