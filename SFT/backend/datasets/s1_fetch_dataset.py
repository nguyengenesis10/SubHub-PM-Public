
import os 
import requests
import shutil
import asyncio 
import aiohttp
import random
import string
import time
import pandas as pd 

from tqdm import tqdm 
from PIL import Image as PIL_Image
from io import BytesIO

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    Image
)

from huggingface_hub import (
    login,
    HfApi
) 


login(token="[REMOVED_SECRET]")


def load_bar(
    sleep_time:int, 
    iterations:int
):
    '''
    Args:
        sleep_time: Time in seconds to sleep per iteration.
        iterations: Number of times to loop over.
    '''
    pbar = tqdm(
        colour="blue",
        desc="Processing Batches",
        total=iterations,
        dynamic_ncols=True
    )
    
    for _ in range(iterations):
        time.sleep(sleep_time)
        pbar.update(1)

    return None 


def generate_random_id(char_nums = 10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k = char_nums))


async def check_url(
    session, 
    url: str,
    image_save_path: str,
):
    """Asynchronously checks if a URL is valid (status code 200) and saves the image."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                image_path = os.path.join(
                    image_save_path,
                    f"image_{generate_random_id()}.jpg"
                )  # Fixed filename formatting
                img = PIL_Image.open(
                    BytesIO(
                        await response.read()
                    )
                )
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(
                    image_path,
                    format = "JPEG"
                )
                return (True, image_path)  # Return success flag & image path
    except Exception as E:
        return (False, None)  # Return False and None if error occurs


async def process_dataset(
    df: pd.DataFrame,
    export_fn: str,
    image_save_path: str,
    batch_size: int,
    max_imgs: int,
    custom_df_batch_indices: list=None,
    custom_batch_idxs: list=None,
    custom_max_imgs_idx: int=None,
):
    """
    Processes the dataset with async URL checking, saves images, and updates CSV.
    Args:
        df: The Pandas DataFrame which contains header 'URL' to fetch/validate the urls. 
        export_fn: A temporary csv to hold 'max_imgs' number of valid data-samples before pushing to the Hub.
        image_save_path: A temporary directory to hold 'max_imgs' number of images before casting and then pushing to the Hub.
        batch_size: The number of urls to fetch at one time. 
        
        In the event the loop fails:
            custom_df_batch_indices: A custom set of tuple-pairs which contains the specific indices with the dataframe to process e.g.
                                     (392_475 , 392_500) -> df.iloc[392_475:392_500]
            custom_batch_idxs: A custom set of batch indexes, refers to what number's batch urls are being processed.
            custom_max_imgs_idx: The HF repo id index that the next dataset pushed was suppoed to be.
    """
    # Ensure image save directory exists
    os.makedirs(image_save_path, exist_ok=True)
    
    # Load previously exported URLs to avoid duplicates
    seen_urls = set()
    if os.path.isfile(export_fn):
        seen_urls |= set(pd.read_csv(export_fn)["URL"].tolist())  # Avoid reprocessing URLs 
        
    # Define batch ranges
    if custom_df_batch_indices:
        batch_indices = custom_df_batch_indices
    elif not custom_df_batch_indices:
        batch_indices = [
            (idx, min(idx + batch_size, len(df))) for idx in range(0, len(df), batch_size)
        ]
    
    if custom_batch_idxs:
        iterate = zip(
            custom_batch_idxs,
            batch_indices
        )
    elif not custom_batch_idxs:
        iterate = zip( 
            list(range(0, len(batch_indices))), 
            batch_indices
        )
    # Initialize progress bar
    og_length = len(list(range(0, len(df), batch_size)))
    pbar = tqdm(
        colour="blue",
        desc="Processing Batches",
        total=og_length,
        dynamic_ncols=True
    )
    if (
        custom_max_imgs_idx and 
        custom_batch_idxs and 
        custom_df_batch_indices
    ):
        # adjust the starting point of the progress bar
        pbar.update(custom_batch_idxs[0])
    
    if custom_max_imgs_idx:
        max_imgs_idx=custom_max_imgs_idx
    else: 
        max_imgs_idx = 0 
    async with aiohttp.ClientSession() as session:
        for batch_idx, (start, end) in iterate:
            tasks = []
            batch_rows = df.iloc[start:end].copy()  # Copy to avoid modifying the iterator

            # Collect tasks for async URL checking
            for idx, row in batch_rows.iterrows():
                url = row.get("URL", None)
                if url and url not in seen_urls:
                    tasks.append(check_url(session, url, image_save_path))

            # Run async checks
            results = await asyncio.gather(*tasks)

            # Collect valid rows & update image paths
            valid_rows = []
            for res , (_, row) in zip(results, batch_rows.iterrows()):
                '''
                res[0] (bool): boolean flag 
                res[1] (str) : image_path if applicable 
                '''
                if isinstance(res,tuple):
                    if res[0] and row["URL"] not in seen_urls:
                        row["image_path"] = res[1]  # Assign image path to row
                        valid_rows.append(row)

            # Update seen URLs
            seen_urls.update(row["URL"] for row in valid_rows)

            # Append only new valid rows to CSV (efficient writing)
            if valid_rows:
                pd.DataFrame(valid_rows).to_csv(
                    export_fn,
                    mode='a', 
                    header=not os.path.exists(export_fn),
                    index=False
                )

            if os.path.isfile(export_fn):
                if len(pd.read_csv(export_fn)) >= max_imgs:
                    
                    dataset = Dataset.from_pandas( 
                        pd.read_csv(export_fn) 
                    ).shuffle(seed=42)
                    
                    dataset = DatasetDict(dataset.train_test_split(test_size=0.1)) 
                    
                    for split in ["train", "test"]:
                        dataset[split] = dataset[split].cast_column("image_path", Image()) 
                    
                    dataset.push_to_hub(
                        repo_id = f"genesis1SubHub/OBELICS-1M-IMGs_PT{max_imgs_idx}",
                        private = True,
                    )
                    max_imgs_idx +=1
                    
                    # delete and reset 
                    if os.path.exists(export_fn):
                        os.remove(export_fn)
                    
                    if os.path.exists(image_save_path):
                        shutil.rmtree(image_save_path)
                        os.makedirs(image_save_path, exist_ok=True)
                    
            # Update progress bar
            pbar.update(1)
            pbar.set_description(
                f"Batch {batch_idx + 1}/{og_length} | Total Valid: {len(seen_urls)}"
            )

    pbar.close()
    print(f"✅ Finished. Total valid URLs: {len(seen_urls)}")



if __name__ == "__main__":

    df = pd.read_csv("/home/ubuntu/SubHub-South-TX/datasets/obelics_1M_IMGs_PT1.csv")
    image_save_path = "/home/ubuntu/SubHub-South-TX/datasets/obelics_imgs"
    export_fn = "/home/ubuntu/SubHub-South-TX/datasets/obelics_1M_IMGs_CLEAN_PT1.csv"
    batch_size = 25
    max_imgs = 20_000
    
    start = 392_475 # the slice to start in the dataframe
    custom_df_batch_indices = [
        (idx, min(idx + batch_size, len(df))) for idx in range(start, len(df), batch_size)
    ]
    start_max_imgs_idx = 15699
    custom_batch_idxs=list(
        range(start_max_imgs_idx,int(len(df)/batch_size))
    )
    
    custom_max_imgs_idx = 39
    
    # if no custom values are required, just set all values to None
    custom_df_batch_indices,custom_batch_idxs,custom_max_imgs_idx = None,None,custom_max_imgs_idx

    export_rows = asyncio.run(
        process_dataset(
            df=df,
            export_fn=export_fn,
            image_save_path=image_save_path,
            batch_size=batch_size,
            max_imgs=max_imgs,
            custom_batch_idxs=custom_batch_idxs,
            custom_df_batch_indices=custom_df_batch_indices,
            custom_max_imgs_idx=custom_max_imgs_idx,
        )
    )
    print(f"\nFinal batch number: {len(df) / max_imgs}.")






#