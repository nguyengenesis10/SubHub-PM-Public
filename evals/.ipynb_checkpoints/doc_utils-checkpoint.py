import os
import shutil
import pandas as pd

from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from huggingface_hub import login, delete_repo

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    ClassLabel,
    Image,
    load_dataset,
    concatenate_datasets
)

class PDF_document:
    
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(file_path, "rb") as file:
            self.reader = PdfReader(file)
            self.num_pages = len(self.reader.pages)
    
    
    def split_to_img(
        self, 
        chunk_size:int=20, 
        dpi:int=200
    ):
        if self.num_pages >= chunk_size:
            max_idx = self.num_pages // chunk_size
            images = []
            exit=False
            for idx, i in enumerate(range(0, self.num_pages, chunk_size)):
                start = i + 1
                if idx != max_idx - 1:
                    end = i + chunk_size 
                else:
                    end=self.num_pages
                    exit=True
                chunk = convert_from_path(
                    self.file_path,
                    dpi=dpi,
                    first_page=start,
                    last_page=end
                )
                images.extend(chunk)
                print(f"Number of images: {len(images)}")
                if exit:
                    break
            self.images = images
        else:
            self.images = convert_from_path(self.file_path, dpi=dpi)
        
        temp_path=os.path.join(
            os.path.dirname(self.file_path),"temp"
        )
        os.makedirs(
            temp_path, exist_ok=True
        )
        self.temp_path=temp_path
        image_paths=[]
        for idx,image in enumerate(self.images):
            path=os.path.join(
                temp_path,f"page_{idx}.png"
            )
            image.save(path)
            image_paths.append(path)
        self.image_paths=image_paths
        
    
    
    def push_to_hub(
        self, 
        repo_id: str, 
        testing:bool=True
    ):
        df = pd.DataFrame(
            {
                "image": self.image_paths,
                "page_number":[i+1 for i in range(len(self.image_paths))]
            },
        )
        train_dataset = Dataset.from_pandas(df)

        dataset_dict = DatasetDict({
            "train": train_dataset.cast_column("image", Image())
        })

        # Push to hub
        dataset_dict.push_to_hub(
            repo_id,
            private=True,
        ) if not testing else None
        
        return dataset_dict
    
    
    def cleanup(self):
        approve=input(f"Approve deleting temp_path:{self.temp_path} select 'y' or 'n' ")
        if approve=="y":
            shutil.rmtree(self.temp_path)





if __name__ == "__main__":
    pdf=PDF_document(file_path="/home/ubuntu/SubHubProjectManager/evals/Drawings/Civil Set.pdf")
    pdf.split_to_img()
    pdf.push_to_hub(
        repo_id="genesis1SubHub/NHA-Civil-set",
        testing=False
    )
    pdf.cleanup()


'''
Example usage:

pdf=PDF_document(file_path="some/file/path.pdf")
pdf.split_to_img()
pdf.push_to_hub(repo_id="some/repo-id")

'''

#