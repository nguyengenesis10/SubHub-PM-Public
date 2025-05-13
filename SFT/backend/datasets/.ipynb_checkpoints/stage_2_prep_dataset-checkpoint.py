
from datasets import load_dataset
from dataclasses import dataclass
from typing import Tuple

import random
import fitz  
import io

from io import BytesIO
from PIL import (
    Image,
    ImageDraw, 
    ImageFont,
)


@dataclass
class BaseBoundingBoxes:
    text:list[str]
    bbox:list[list[float]]
    score:list[float]


@dataclass
class WordsBoundingBoxes(BaseBoundingBoxes):
    line_pos:list[list[int,int]]


@dataclass
class LinesBoundingBoxes(BaseBoundingBoxes):
    word_slice:list[list[int,int]]


@dataclass
class Page:
    lines:LinesBoundingBoxes
    words:WordsBoundingBoxes
    images_bbox:list[list[int]]
    images_bbox_no_text_overlap:list[list[int]]

    def __post_init__(
        self
    ):
        if isinstance(self.lines,dict):
            self.lines=LinesBoundingBoxes(**self.lines)
        if isinstance(self.words,dict):
            self.words=WordsBoundingBoxes(**self.words)


@dataclass
class Sample:
    pages:list[Page]
    pdf_bytes:str
    images:list[Image]=None
    images_bbox:list[Image]=None


class Stage2DatasetHandler:
    def __init__(
        self,
    ):
        self.iter_dataset=load_dataset(
            "pixparse/pdfa-eng-wds",
            split="train",
            streaming=True,
        )
        self.root_name_to_push="genesis1SubHub/PDFA"
    
    
    def pdf_bytes_to_images_pymupdf(
        self,
        pdf_bytes,
    )->list[Image]:
        
        """
        Convert PDF bytes to images using PyMuPDF (fitz).
        """
        
        pdf_document = fitz.open(
            stream=pdf_bytes, 
            filetype="pdf"
        )
        images = []
        for page in pdf_document:
            pix = page.get_pixmap()  # Render page as an image
            img = Image.frombytes(
                "RGB", 
                [pix.width, pix.height], 
                pix.samples
            )
            images.append(img)
        return images
    
    
    def extract_sample(
        self,
        sample_pages:list[dict],
    )->list[Page]:
        
        if not isinstance(sample_pages,list):
            raise TypeError(
                f"'sample_pages' should be of type list[dict], is of type {type(sample_pages)}!"
            )
        req_keys=['words', 'lines', 'images_bbox', 'images_bbox_no_text_overlap']
        pages=[]
        for page in sample_pages:
            master_kwargs={}
            for req_key in req_keys:
                if req_key not in page.keys():
                    raise KeyError(
                        f"Required key: {req_key} not in 'page.keys()'-{page.keys()}."
                    )
                if (
                    req_key == "words" or 
                    req_key == "lines"
                ):
                    sub_req_keys = ['text', 'bbox', 'score']
                    kwargs = {}
                    for sub_req_key in sub_req_keys:
                        if sub_req_key not in page[req_key].keys():
                            raise KeyError(
                                f"Required key: '{sub_req_key}' not in 'page[{req_key}]': {page[req_key].keys()}!"
                            )
                        kwargs[sub_req_key]=page[req_key][sub_req_key]
                    if req_key == "words":
                        special_key = "line_pos"
                    elif req_key == "lines":
                        special_key = "word_slice"
                    kwargs[special_key]= page[req_key][special_key]
                    master_kwargs[req_key]=kwargs
                else:
                    master_kwargs[req_key]=page[req_key]
            pages.append(
                Page(**master_kwargs)
            )
        return pages 
    
    
    def generate_samples(
        self,
        target_samples:int,
        add_factor:float,
        print_interval:int=50_000,
    )->list[Sample]:
        
        def fill_with_none(keep_idxs):
            
            max_val = max(keep_idxs)
            results = [None] * (max_val+1)
            for idx in keep_idxs:
                results[idx]=idx
            return results 
        
        
        keep_idxs=sorted(
            list(
                random.sample(
                    range(int(target_samples*add_factor)), 
                    target_samples
                )
            )
        )
        iterator = zip(
            fill_with_none(keep_idxs),
            self.iter_dataset,
        )
        samples=[]
        for (keep_idx,sample) in iterator:
            if keep_idx is not None:
                samples.append(
                    Sample(
                        self.extract_sample(
                            sample_pages=sample["json"]["pages"]
                        ),
                        sample["pdf"]
                    )
                )
        self.samples = samples
        if len(self.samples) != target_samples:
            raise ValueError(
                f"Sampling has failed there are {len(self.samples)} and there should be {target_samples}! Keep idxs: {fill_with_none(keep_idxs)}"
            )
        return samples
    
    
    def map_bounding_boxes(
        self,
    ):
        
        def convert_list_of_bboxs(
            bboxes:list[list],
            height:int,
            width:int,
        )->list[list[int]]:
            
            if not isinstance(bboxes,list):
                raise TypeError(
                    f"bboxes should be of type list, not of type: {type(bboxes)}!"
                )
            def scale_bounding_box(
                norm_bbox:list[int,int,int,int],
                height:int,
                width:int,
            )->list[int,int,int,int]:
                
                # assume norm_bbox has schema - [left, top, width, height]
                if len(norm_bbox) != 4:
                    raise ValueError(
                        f"Bounding box has length: {len(norm_bbox)}, should have length 4!"
                    )                
                return [
                    norm_bbox[0] * width,
                    norm_bbox[1] * height,
                    norm_bbox[2] * width,
                    norm_bbox[3] * height,
                ]
            if len(bboxes) > 0:
                return [
                    scale_bounding_box(
                        norm_bbox,
                        height,
                        width
                    ) for norm_bbox in bboxes
                ]
            else:
                return bboxes
        
        
        def draw_bounding_box_pil(
            image: Image.Image, 
            bboxs:list[list[int,int,int,int]], 
            color="red", 
            thickness=3, 
            label=None
        )->Image.Image:
            
            image_copy = image.copy()
            if len(bboxs)==0:
                return image_copy
            draw = ImageDraw.Draw(image_copy)
            for bbox in bboxs:
                # bbox has schema - [left, top, width, height]
                left, top, width, height = bbox
                x1 = left
                x2 = left + width # Convert width/height to bottom-right coordinates
                y1 = top
                y2 = top + height                
                # Draw the rectangle
                for i in range(thickness):  # Thickness effect by drawing multiple rectangles
                    rect = [x1, y1, x2, y2]
                    if rect[0] < rect[2] and rect[1] < rect[3]:
                        draw.rectangle(
                            rect, 
                            outline=color
                        )
            return image_copy
            
            
        if not hasattr(self,"samples"):
            raise AttributeError(
                f"'samples' isn't an attribute, call 'generate_samples()' first!"
            )
        for sample in self.samples:
            if not isinstance(sample, Sample):
                raise TypeError(
                    f"sample is not of type 'Sample', it's of type {sample}!"
                )
            sample.images = self.pdf_bytes_to_images_pymupdf(
                pdf_bytes=sample.pdf_bytes
            )
            if len(sample.pages) != len(sample.images):
                raise ValueError(
                    f"'pages' has length: {len(pages)} and 'images' has length: {len(images)}!"
                )
            images_bbox=[]
            for page,image in zip(sample.pages,sample.images):
                height=image.size[1]
                width=image.size[0]
                
                images_bboxs=convert_list_of_bboxs(
                    bboxes=page.images_bbox,
                    height=height,
                    width=width,
                )
                images_bboxs_no_text_overlap=convert_list_of_bboxs(
                    bboxes=page.images_bbox_no_text_overlap,
                    height=height,
                    width=width,
                )
                lines_bbox=convert_list_of_bboxs(
                    bboxes=page.lines.bbox,
                    height=height,
                    width=width,
                )
                words_bbox=convert_list_of_bboxs(
                    bboxes=page.words.bbox,
                    height=height,
                    width=width,
                )
                
                color_plot_dict = {
                    "Red":lines_bbox,
                    "Blue":words_bbox,
                    "Green":images_bboxs,
                    "Yellow":images_bboxs_no_text_overlap,
                }
                image_bbox=image.copy()
                for color,bboxs in color_plot_dict.items():
                    image_bbox=draw_bounding_box_pil(
                        image=image_bbox, 
                        bboxs=bboxs, 
                        color=color, 
                        thickness=1, 
                    )
                images_bbox.append(image_bbox)
            sample.images_bbox = images_bbox
            
            
def fetch_dataset():
    
    dataset=load_dataset(
        "pixparse/pdfa-eng-wds",
        split="train",
        streaming=True,
    )
    
    return dataset


if __name__ == "__main__":


    stage2_dataset_handler=Stage2DatasetHandler()
    samples = stage2_dataset_handler.generate_samples(
        target_samples=2,
        add_factor=2,
        print_interval=50_000,
    )
    
    stage2_dataset_handler.map_bounding_boxes()
    
    '''
    samples[0] -> 
        pdf: bytes 
        json: dict - dict w/ 1 key 'pages'
            
            ["json"]["pages"]: list[dict] - dict w/ keys 'words', 'lines', 'images_bbox', 'images_bbox_no_text_overlap'

                Iterate through pages and then access relevant sub-attributes.
                
                'words' SubClass
                
                ["json"]["pages"][n]["words"] - dict w/ keys 'text', 'bbox', 'score', 'line_pos'
                    ["json"]["pages"][n]["words"]["text"]: list[str]
                    
                    ["json"]["pages"][n]["words"]["bbox"]: list 
                        ["json"]["pages"][n]["words"]["bbox"][n]: list[float]
                        
                    ["json"]["pages"][n]["words"]["score"]: list[float]
                        
                    ["json"]["pages"][n]["words"]["line_pos"]: list - 
                        ["json"]["pages"][n]["words"]["line_pos"][n]: list[int,int]
                
                'line' SubClass - follows same schema as 'words' SubClass
                
                ["json"]["pages"][n]["lines"] - dict w/ keys 'text', 'bbox', 'score', 'line_pos'
                        ["json"]["pages"][n]["lines"]["text"]: list[str]

                'images_bbox' SubClass
                
                ["json"]["pages"][n]["images_bbox"]: list[list[int]]
                
                ["json"]["pages"][n]["images_bbox_no_text_overlap"]: list[list[int]]
                
    '''
    
    
#