
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
    # Convert pixels values to bf16, leave everything else int64.
    return {
        k: (v.to(torch.bfloat16) if k == "pixel_values" else v) for k, v in batch.items()
    }    


def bytes_to_image(image_bytes):
    
    from PIL import (
        Image, 
        JpegImagePlugin
    )
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


def get_custom_dataset(
    dataset_config,
    processor,
    split:str,
    split_ratio=0.9
):

    if (
        split != "train" and 
        split != "test"
    ):
        raise ValueError(f"Split is {split}, should only be 'train' or 'test'!")
    
    def wrap_image_in_list(row):
        row["image_path"] = [row["image_path"]]  # Wrap image in a list
        return row

    # For testing purposes only load 2 datasets of each type
    obelics_num_sub_datasets = 2  
    laion_coco_num_sub_datasets = 2 

    datasets = []
    for i in range(obelics_num_sub_datasets):
        dataset = load_dataset(
                    f"genesis1SubHub/OBELICS-1M-IMGs_PT{i}", 
                    split="train",
                ) if split == "train" else load_dataset(
                    f"genesis1SubHub/OBELICS-1M-IMGs_PT{i}", 
                    split="test",
                )
        datasets.append(dataset)

    for i in range(laion_coco_num_sub_datasets):
        dataset = load_dataset(
                    f"genesis1SubHub/Laion-Coco-1M-IMGs_PT{i}", 
                    split="train",
                ) if split == "train" else load_dataset(
                    f"genesis1SubHub/Laion-Coco-1M-IMGs_PT{i}", 
                    split="test",
                )
        # standardize to int64
        dataset = dataset.cast_column("WIDTH", Value("int64"))
        dataset = dataset.cast_column("HEIGHT", Value("int64"))
        
        datasets.append(dataset)

    if len(datasets) == 1:
        final_dataset = datasets[0]
    elif len(datasets) > 1:
        final_dataset = concatenate_datasets(datasets)
    else:
        raise ValueError(f"List 'datasets' has length of {len(datasets)}!")

    # took 5:50 minutes to map 36K samples 
    return final_dataset.map(wrap_image_in_list)


class CustomDataCollator:
    def __init__(self,processor):
        self.processor = processor
    
    def __call__(self,dataset):
        images = []
        pbar = tqdm(
            colour="blue",
            desc="Processing ",
            total=len(dataset),
            dynamic_ncols=True
        )
        for idx,sample in enumerate(dataset):
            if isinstance(sample, dict):
                # only grab the first image
                image_bytes = sample["image_path"][0]["bytes"]
                jpeg_image = bytes_to_image(image_bytes)
                images.append([jpeg_image])
                # converts to list 
            pbar.set_description(
                f"Sample {idx + 1}/{len(dataset)}"
            )
            pbar.update(1)
        return tokenize_dialogs(images, self.processor)


def get_data_collator(processor):
    return CustomDataCollator(processor)







