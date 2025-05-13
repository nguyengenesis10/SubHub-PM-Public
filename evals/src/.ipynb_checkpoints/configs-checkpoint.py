from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Any


from .system_prompts import (
    merge_system_prompt_new,
    phrase_normalize_system_prompt,
    gray_area_analysis_system_prompt,
)


import warnings


frozen_status=False


@dataclass(frozen=frozen_status)
class EmbedConfig:
    model_name: str = "sentence-t5-xl"


@dataclass(frozen=frozen_status)
class ClusterConfig:
    linkage: str = "average"
    affinity: str = "precomputed"


@dataclass(frozen=frozen_status)
class LLMConfig:
    client: Any = None
    model_name: str = "gpt-4o-2024-08-06"
    temperature: float = 0.25


    def __post_init__(self):
        if not self.client:
            print("No evaluating ground truth client has been set!")


@dataclass(frozen=frozen_status)
class GroupPredsConfig:
    threshold: float = 0.1
    n_clusters: int = None
    cluster: ClusterConfig = ClusterConfig


    def __post_init__(self):
        if self.n_clusters and self.threshold:
            raise ValueError(
                f"Cannot have both n_clusters and threshold defined!"
            )


@dataclass(frozen=frozen_status)
class MergePredsConfig(LLMConfig):
    display_pbar:bool=True
    system_prompt:str=merge_system_prompt_new


@dataclass(frozen=frozen_status)
class NormalizeConfig(LLMConfig):
    print_log:bool=False
    display_pbar:bool=True
    system_prompt:str=phrase_normalize_system_prompt


@dataclass(frozen=frozen_status)
class ComparePredsAndLabelsConfig:
    threshold: float = 0.16
    n_clusters: int = None 
    scs_alpha: float = 0.5
    cluster: ClusterConfig = ClusterConfig

    def __post_init__(self):
        if self.n_clusters and self.threshold:
            raise ValueError(
                f"Cannot have both n_clusters and threshold defined!"
            )


@dataclass(frozen=frozen_status)
class GrayAreaAnalyzerConfig(LLMConfig):
    hybrid_dist_alpha:float=0.5
    system_prompt:str=gray_area_analysis_system_prompt


class ExperimentConfig:
    def __init__(
        self,
        **kwargs,
    ):
        self.norm=NormalizeConfig
        self.embed=EmbedConfig
        self.group=GroupPredsConfig
        self.merge=MergePredsConfig
        self.compare=ComparePredsAndLabelsConfig
        self.gray=GrayAreaAnalyzerConfig

        DOT="."
        grouped_kwargs = defaultdict(list)
        for k,v in kwargs.items():
            for attr,base_class in self.__dict__.items():
                #Instances checking - norm in norm.client 
                if attr in k:
                    if k.count(DOT) == 1:
                        config_name, attr_name = k.split(DOT)
                        # print(config_name,attr_name)
                        grouped_kwargs[attr].append(
                            (attr_name,v)
                        )
                    elif k.count(DOT) == 2 :
                        _, high_attr, sub_attr = k.split(DOT)
                        if hasattr(
                            getattr(base_class,high_attr),sub_attr
                        ):
                            grouped_kwargs[attr].append(
                                {high_attr : (sub_attr,v)}
                            )
                    else:
                        raise ValueError(
                            f"{k} is an invalid parameter."
                        )
                else:
                    if k in dir(base_class):
                        grouped_kwargs[attr].append(
                            (k,v)
                        )
        for k,v in grouped_kwargs.items():
            kwargs={}
            nested_kwargs=defaultdict(list)
            for attr_pair in v:
                if isinstance(attr_pair,tuple):
                    kwargs[attr_pair[0]]=attr_pair[1]
                elif isinstance(attr_pair,dict):
                    for a,b in attr_pair.items():
                        if a not in nested_kwargs.keys():
                            nested_kwargs[a]={b[0]:b[1]}
                        nested_kwargs[a].update({b[0]:b[1]})
            
            setattr(self,k,getattr(self,k)(**kwargs))

            if nested_kwargs:
                for sub_k,kwargs in nested_kwargs.items():
                    setattr(
                        getattr(self,k),
                        sub_k,
                        getattr(getattr(self,k),sub_k)(**kwargs)
                    )
            
        for k,v in self.__dict__.items():
            if k not in grouped_kwargs.keys():
                setattr(
                    self,
                    k,
                    getattr(self,k)()
                )


    def __str__(self):
        to_print = "Experiment Config Schema:\n"
        for k, v in self.__dict__.items():
            to_print += f"\n  {k.upper()} config:\n"
            for a, b in v.__dict__.items():
                if a != "system_prompt":
                    to_print += f"    {a}: {b}\n"
        return to_print








#
