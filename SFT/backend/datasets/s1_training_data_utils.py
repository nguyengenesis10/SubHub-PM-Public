
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.special import softmax

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
        
        self.width , self.height = drop_na(
            height=height,
            width=width
        )
    
    
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
    
    
    def compute_areas(self)->list[int]:
        self.areas = [
            w*h for w,h in zip(self.width,self.height)
        ]
        return self.areas
    
    
    def compute_area_distribution(
        self, 
        bins:list[Tuple[int]]
    )->dict:
        
        if not hasattr(self,"areas"):
            self.compute_areas()
        
        # Create defaultdict with list as default factory
        binned = defaultdict(int)
        for area in self.areas:
            matched = False
            for bin_range in bins:
                min_val, max_val = bin_range
                if min_val**2 <= area <= max_val**2:
                    binned[bin_range]+=1
                    matched = True
                    break
            # Collect out-of-range values if needed
            if not matched:
                binned[bin_range]+=0
        return {
            k: binned[k] for k in sorted(
                binned.keys()
            )
        }
    
    
    def plot_hist_aspect_ratios(
        self,
        export_path:str,
    ):
        if not hasattr(self, 'aspect_ratios'):
            raise AttributeError(
                "Aspect Ratios have not been computed. Call 'compute_aspect_ratios()' first."
            )

        binned_aspect_ratios=defaultdict(int)
        for sublist in self.aspect_ratios:
            binned_aspect_ratios[tuple(sublist)]+=1
        df=pd.DataFrame(
            list(binned_aspect_ratios.items()) , columns=["Sublist", "Frequency"]
        )
        print(df)
        
        plt.figure(figsize=(12, 7))  # Adjust size & resolution
        
        x = [coord[0] for coord in df["Sublist"]]
        y = [coord[1] for coord in df["Sublist"]]
        weights = [count for count in df["Frequency"]]
        
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[max(x), max(y)], weights=weights)
        heatmap = np.flipud(heatmap.T)
        
        # heatmap = gaussian_filter(heatmap, sigma=1)
        # print(heatmap) ; input("Approval:")
        
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
        
        plt.savefig(export_path) if export_path else None
        
        plt.show()
        
        return plt
    
    
    def plot_hist_area_distributions(
        self,
        increments:int,
        start:int,
        end:int,
        export_path:str=None,
    )->plt:
        
        
        area_dist=self.compute_area_distribution(
            bins=[
                (i,i+increments) for i in range(start,end,increments)
            ]
        )
        bins = [
            key[0] for key in area_dist.keys()
        ] + [
            list(area_dist.keys())[-1][1]
        ]
        frequencies = list(
            area_dist.values()
        )
        soft_frequencies = np.sqrt(frequencies)
        
        fig, ax = plt.subplots(
            figsize=(12,6)
        )
        bars = ax.bar(
            bins[:-1],
            soft_frequencies, 
            width=increments, 
            align='edge', 
            edgecolor='black'
        )
        for bar, freq in zip(bars, frequencies):
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height(), 
                f"{int(freq):,}",
                ha='center', 
                va='bottom', 
                fontsize=10
            )
        
        '''
        ax.set_xticks(
            cont_x * self.plt_config.spacing_factor + bar_width * (len(categories) / 2)
        )
        ax.set_xticklabels(x, rotation=45, wrap=label_config.wrap)
        ax.set_yticklabels([]) if not y_labels_on else None
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plt_title)
        ax.legend()
        
        if grid_config.on:
            ax.grid(
                grid_config.on,
                linestyle=grid_config.grid_style,
                alpha=grid_config.grid_alpha,
            )
        '''
        
        
        ax.set_xticks(bins)
        ax.set_xticklabels(bins)
        ax.set_yticklabels([])
        ax.set_xlabel(f"Range (increments of {increments} pxs)")
        ax.set_ylabel("Sqrt-Transformed Frequency (sqrt)")
        ax.set_title(
            f"Histogram with Sqrt-Transformed Frequencies | Total: {int(sum(frequencies)):,}"
        )
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        plt.show()
        
        return plt
    
    
    def plot_3d_hist_area_distributions(
        self,
        increments:int,
        start:int,
        end:int,
        export_path:str=None,
    )->plt:
        
        iterator = range(start,end+increments,increments)
        
        area_matrix = np.zeros(
            (len(iterator), len(iterator)), 
            dtype=int
        )
        for h,w in zip(self.height,self.width):
            for i_x,x in enumerate(iterator):
                if  x <= w < x + increments:
                    for i_y,y in enumerate(iterator):
                        if y <= h < y + increments:
                            area_matrix[(i_x,i_y)]+=1
        
        # soften matrix values
        soft_area_matrix = np.power(area_matrix , 1/2)
        
        fig = plt.figure(
            figsize=(16, 10)
        )
        ax = fig.add_subplot(
            111, projection='3d'
        )
        ax.view_init(
            elev=30, azim=-315
        )
        xpos, ypos = np.meshgrid(
            np.arange(len(iterator)) * increments, 
            np.arange(len(iterator)) * increments, 
            indexing="ij"
        )
        zpos = np.zeros_like(xpos)
        dx,dy = increments,increments 
        dz = soft_area_matrix.flatten()
        bars = ax.bar3d(
            xpos.flatten(), 
            ypos.flatten(), 
            zpos.flatten(), 
            dx, 
            dy, 
            dz, 
            shade=True
        )
        for i,freq in enumerate(area_matrix.flatten()):
            if dz[i] > 0:  
                ax.text(
                    xpos.flatten()[i] + dx / 2, 
                    ypos.flatten()[i] + dy / 2, 
                    dz[i],
                    f"{freq:,}",
                    ha='center', 
                    va='bottom', 
                    fontsize=7, 
                    fontweight="bold",
                    color='black'
                )
                
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # X pane transparent
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # X pane transparent
        
        ax.set_xlabel(f"Width (bins of {increments})")
        ax.set_xticks(list(iterator))
        ax.set_xticklabels(list(iterator))
        
        ax.set_ylabel(f"Height (bins of {increments})")
        ax.set_yticks(list(iterator))
        ax.set_yticklabels(list(iterator))
        
        ax.set_zlabel("Frequency")
        ax.set_zticklabels([])
        
        ax.set_title("3D Bar Plot: Height, Width, Frequency")
        plt.show()
        
        return plt


if __name__ == "__main__":

    import sys;sys.path.append("/home/ubuntu/llama-cookbook-main_20250116/src")
    import pandas as pd
    import os 
    
    # root = "/home/ubuntu/SubHub-South-TX/datasets"
    root = "/Users/genesisnguyen/SaaS_dev/SubHub-South-TX/datasets"
    
    df_obelics = pd.read_csv(
        os.path.join(root,"stage1_dataset_analytics.csv")
    )
    
    obelics_width = df_obelics["Obelics Width"].tolist()
    obelics_height = df_obelics["Obelics Height"].tolist()
    
    df_laion_coco = pd.read_csv(
        os.path.join(root, "stage1_dataset_analytics_LaionCoco.csv")
    )
    
    laion_coco_height = df_laion_coco["LaionCoco Height"].tolist()
    laion_coco_width = df_laion_coco['LaionCoco Width'].tolist()
    
    
    student_processor_cfg = [
        {"image_size" : 560, "max_num_tiles" : 4},
        {"image_size" : 800, "max_num_tiles" : 9},
        {"image_size" : 1200, "max_num_tiles" : 16},
    ]
    data_analyzer = DataAnalyzer(
        width=obelics_width + laion_coco_width,
        height=obelics_height + laion_coco_height,
    )
    increments=560
#     data_analyzer.plot_3d_hist_area_distributions(
#         increments=increments,
#         start=0,
#         end=increments*5,
#     )
    data_analyzer.plot_hist_area_distributions(
        increments=increments,
        start=0,
        end=increments*12,
    )
    data_analyzer.compute_aspect_ratios(
        num_tiles = student_processor_cfg[1]["max_num_tiles"],
        tile_res = student_processor_cfg[1]["image_size"],
    )
    data_analyzer.plot_hist_aspect_ratios(
        export_path=None
    )




