
import sys ; sys.path.append("/home/ubuntu/SubHub-South-TX")

from huggingface_hub import login,HfApi ; login(token="[REMOVED_SECRET]")

from datasets import ( 
    load_dataset,
    concatenate_datasets,
    Dataset,
    Value,
)

from backend.s1_training_data_utils import DataAnalyzer

from transformers import AutoProcessor
from transformers.models.mllama.image_processing_mllama import get_all_supported_aspect_ratios

from tqdm import tqdm

import time 
import torch 
import os
import numpy as np 
import pandas as pd


def fetch_weight_and_height(
    root_name:str, 
    num_datasets:int,
    custom_iterator:list= None,
    random_sample_cfg:dict={
        'random_sample' : False,
        'num_samples' : 10,
    }
):
    width = []
    height = [] 

    if not isinstance(random_sample_cfg,dict):
        raise TypeError(f"The config 'random_sample_cfg' shouldn't be of type {type(random_sample_cfg)}!")
    else:
        for req_key in ['random_sample' , 'num_samples']:
            if req_key not in random_sample_cfg.keys():
                raise KeyError(f"Required key {req_key} not in 'random_sample_cfg'!")
        random_sample = random_sample_cfg['random_sample']
        num_samples = random_sample_cfg['num_samples']

    if not random_sample:
        iterator = range(num_datasets) if not custom_iterator else custom_iterator
        pbar_total = 20_000 * len(iterator)
    else:  
        iterator = np.random.randint(0, num_datasets, size=num_samples).tolist() if not custom_iterator else custom_iterator
        pbar_total = 20_000 * len(iterator)
    
    pbar = tqdm(
            colour="blue",
            desc=f"Processing dataset {root_name}",
            total=pbar_total,
            dynamic_ncols=True
        )
    
    for i in iterator:
        for split in ["test", "train"]:
            dataset_iter = load_dataset(
                f"{root_name}{i}", 
                split=split,
            )
            for sample in dataset_iter:
                width.append(sample["WIDTH"])
                height.append(sample["HEIGHT"])
                pbar.update(1)
    
    return width,height 


if __name__ == "__main__":
    
    export_fn = "/home/ubuntu/SubHub-South-TX/datasets/stage1_dataset_analytics_LaionCoco.csv"

    num_laion_coco_datasets = 39 + 1 
    iterator = range(num_laion_coco_datasets,69)

    for i in iterator:
        try:
            laion_coco_width , laion_coco_height = fetch_weight_and_height(
                root_name=f"genesis1SubHub/Laion-Coco-1M-IMGs_PT",
                num_datasets=num_laion_coco_datasets,
                custom_iterator=range(i,i+1),
            )
        
            df = pd.DataFrame(
                {"LaionCoco Height": laion_coco_width,"LaionCoco Width" : laion_coco_height,}
            )

            if os.path.isfile(export_fn):
                df_1 = pd.read_csv(export_fn)
                df_2 = pd.concat([df,df_1]) 
                df_2.to_csv(export_fn , index=False)
                print(
                    f"Final dataframe length:{len(df_2)}"
                )
            else:
                df.to_csv(export_fn,index=False)
            
        except Exception as E:
            raise Exception (f"Failed on dataset w/ index {i}. Error: {E}")


    