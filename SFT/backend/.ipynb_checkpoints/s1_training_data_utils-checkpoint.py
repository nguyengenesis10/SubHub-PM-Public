import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter, defaultdict
from typing import Tuple

from transformers.models.mllama.image_processing_mllama import (
    get_optimal_tiled_canvas,
    get_image_size_fit_to_canvas,
    get_all_supported_aspect_ratios,
)

class DataAnalyzer:
    def __init__(
        self, 
        width:list[int],
        height:list[int],
    ):
        if len(width) != len(height):
            raise ValueError(
                f"Length of list 'width' {len(width)} and of list 'height' {len(height)} are not the same!"
            )
        
        self.width = width
        self.height = height 

    def compute_aspect_ratios(
        self,
        num_tiles:int,
        tile_res:int
    )->list[int,int]:

        self.aspect_ratios = []
        supported_aspect_ratios = get_all_supported_aspect_ratios(num_tiles)
        for w,h in zip(self.width,self.height):
            aspect_ratio = [
                int(i) for i in list(
                    get_optimal_tiled_canvas(
                        image_height=h,
                        image_width=w,
                        max_image_tiles=num_tiles,
                        tile_size=tile_res
                    ) / tile_res
                )
            ]
            self.aspect_ratios.append(aspect_ratio)
        
        return self.aspect_ratios
    
    def compute_areas(self):
        self.areas = [
            w*h for w,h in zip(self.width,self.height)
        ]
        return self.areas

    def compute_area_distribution(
        self, 
        bins:list[Tuple]
    )->dict:
        # Create defaultdict with list as default factory
        binned = defaultdict(list)
        values = self.areas
        for value in values:
            matched = False
            for bin_range in bins:
                min_val, max_val = bin_range
                if min_val <= value <= max_val:
                    binned[bin_range].append(value)
                    matched = True
                    break
            # Collect out-of-range values if needed
            if not matched:
                binned[None].append(value) 
        # get the length of each bin 
        final_binned = {}
        for k,v in binned.items():
            if k != None:
                final_binned[
                    k[0]**0.5
                ] = len(v)        
        return  dict(sorted(final_binned.items()))

    def plot_hist_aspect_ratios(self):
        if not hasattr(self, 'aspect_ratios'):
            raise AttributeError(
                "Aspect Ratios have not been computed. Call 'compute_aspect_ratios()' first."
            )

        counter = Counter(tuple(sublist) for sublist in self.aspect_ratios)
        df = pd.DataFrame(
            counter.items(), 
            columns=["Sublist", "Frequency"]
        )
        print(df)
        plt.figure(figsize=(16, 12), dpi=300)  # Adjust size & resolution
        
        x = [coord[0] for coord in df["Sublist"]]
        y = [coord[1] for coord in df["Sublist"]]
        weights = [count for count in df["Frequency"]]
        
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[max(x), max(y)], weights=weights)
        heatmap = np.flipud(heatmap.T)
        
        # heatmap = gaussian_filter(heatmap, sigma=1)
        
        print(heatmap) ; input("Approval:")
        
        ax = sns.heatmap(
            heatmap,
            annot=True,
            cmap="viridis",
            fmt=".0f",
            xticklabels=range(1, max(x) + 1), 
            yticklabels=range(max(y), 0, -1), 
        )

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Heatmap of (x, y) Frequency Counts")
        plt.savefig("test.png")
        
        
        # plt.bar(
        #     range(len(df)),
        #     df["Frequency"], 
        #     tick_label=[str(list(k)) for k in df["Sublist"]]
        # )
        # plt.xlabel("Unique Sublists")
        # plt.ylabel("Frequency")
        # plt.title("Histogram of Sublist Frequencies")
        # plt.xticks(rotation=45, ha="right")
        # plt.show()
        


if __name__ == "__main__":

    import sys;sys.path.append("/home/ubuntu/llama-cookbook-main_20250116/src")
    import pandas as pd

    def drop_na(
        height:list[int],
        width:list[int]
    ):
        cleaned_height = []
        cleaned_width = []
        if len(height) == len(width):
            for h,w in zip(height,width):
                if (
                    pd.notna(h) and 
                    pd.notna(w) 
                ):
                    cleaned_height.append(h)
                    cleaned_width.append(w)
            return cleaned_height,cleaned_width
        else:
            raise ValueError(
                f"Height has length: {len(height)} and Width has length: {len(width)}!"
            )
        
    
    df_obelics = pd.read_csv(
        "/home/ubuntu/SubHub-South-TX/datasets/stage1_dataset_analytics.csv"
    )
    
    obelics_width = df_obelics["Obelics Width"].tolist()
    obelics_height = df_obelics["Obelics Height"].tolist()
    
    df_laion_coco = pd.read_csv(
        "/home/ubuntu/SubHub-South-TX/datasets/stage1_dataset_analytics_LaionCoco.csv"
    )
    
    laion_coco_height = df_laion_coco["LaionCoco Height"].tolist()
    laion_coco_width = df_laion_coco['LaionCoco Width'].tolist()
    
    laion_coco_height,laion_coco_width=drop_na(
        height=laion_coco_height,
        width=laion_coco_width,
    )
    
    student_processor_cfg = [
        {"image_size" : 560, "max_num_tiles" : 4},
        {"image_size" : 800, "max_num_tiles" : 9},
        {"image_size" : 1200, "max_num_tiles" : 16},
    ]
    data_analyzer = DataAnalyzer(
        width=obelics_width + laion_coco_width,
        height=obelics_height + laion_coco_height,
    )
    data_analyzer.compute_aspect_ratios(
        num_tiles = student_processor_cfg[0]["max_num_tiles"],
        tile_res = student_processor_cfg[0]["image_size"],
    )

    data_analyzer.plot_hist_aspect_ratios()




