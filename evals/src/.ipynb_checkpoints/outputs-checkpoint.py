

from pydantic import BaseModel
from dataclasses import dataclass,field
from collections import defaultdict


import numpy as np


class NormPhrase(BaseModel):
    phrase:str


@dataclass
class GrouperOutput:
    cluster_map:defaultdict
    dist_matrix:np.ndarray
    labels:np.ndarray


@dataclass
class MergeOutput:
    singleton_pred_groups:list[tuple]
    merged_pred_groups:defaultdict[list[tuple]]
    merged_norm_preds:list[str]


class MergedScopes(BaseModel):
    scopes:list[str] 


@dataclass
class PredGroup:
    pred:str
    dist_score:float


@dataclass
class BiparteCluster:
    label:str
    preds:list[PredGroup]


class GrayAreaBin(BaseModel):
    group:str


@dataclass
class SCSMetrics:
    scs_alpha:float
    
    scs:float=field(init=False)
    
    precision_count:float
    pred_count:float
    
    precision:float=field(init=False)
    
    recall_count:float
    label_count:float
    
    recall:float=field(init=False)


    def __post_init__(self):
        self.recall=self.recall_count/self.label_count
        self.precision=self.precision_count/self.pred_count
        self.scs=self.scs_alpha * self.recall - (1-self.scs_alpha) * (1-self.precision)


    def __str__(self):
        return (
            f"Recall: {self.recall_count:.2f}/{self.label_count:.0f} = {self.recall:.2%}\n"
            f"Precision: {self.precision_count:.2f}/{self.pred_count:.0f} = {self.precision:.2%}\n"
            f"SCSₐ={self.scs_alpha:.2f}: {self.scs:.2%}"
        )


@dataclass
class PageMetrics:
    page_id:str
    page_num:int
    matches:int
    misses:int
    extra:int


@dataclass
class MatchStats:
    total: int = 0
    in_singleton: int = 0
    in_merged: int = 0


@dataclass
class PageMetricsSummary:
    recall: float = 0.0
    precision: float = 0.0
    jaccard: float = 0.0
    total_label_pages: int = 0
    total_predicted_pages: int = 0
    total_union_pages: int = 0


@dataclass
class DocStats:
    matches: MatchStats = field(default_factory=MatchStats)
    no_matches: MatchStats = field(default_factory=MatchStats)
    gray_matches: MatchStats = field(default_factory=MatchStats)
    
    page_stats_match: PageMetricsSummary = field(default_factory=PageMetricsSummary)
    page_stats_no_match: PageMetricsSummary = field(default_factory=PageMetricsSummary)
    
    # Tuple Order (TP, FN, FP) 
    page_schema_stats_match: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0]))
    page_schema_stats_no_match: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0]))
    page_schema_stats_combined_match: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0]))


    def _create_combined_page_schema_stats(self):
        for key in set(self.page_schema_stats_match) | set(self.page_schema_stats_no_match):
            v1 = self.page_schema_stats_match[key]
            v2 = self.page_schema_stats_no_match[key]
            self.page_schema_stats_combined_match[key] = [a + b for a, b in zip(v1, v2)]


    def safe_div(
        self,
        num:float,
        denom:float
    )->float:
        
        return num / denom if denom else 0.0


    def compute_document_metrics(
        self,
        doc_alpha:float=0.5
    )->dict:
        
        avg_jaccard = self.safe_div(
            self.page_stats_match.jaccard + self.page_stats_no_match.jaccard,
            self.page_stats_match.total_union_pages + self.page_stats_no_match.total_union_pages,
        )

        avg_recall = self.safe_div(
            self.page_stats_match.recall + self.page_stats_no_match.recall,
            self.page_stats_match.total_label_pages + self.page_stats_no_match.total_label_pages,
        )

        avg_precision = self.safe_div(
            self.page_stats_match.precision + self.page_stats_no_match.precision,
            self.page_stats_match.total_predicted_pages + self.page_stats_no_match.total_predicted_pages,
        )

        doc_score = doc_alpha * avg_recall + (1-doc_alpha) * avg_precision 

        return {
            "avg_jaccard": avg_jaccard,
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
            "doc_score": doc_score,
        }


    def __str__(self):
        
        mpp = 100 * self.safe_div(
            self.page_stats_match.precision,
            self.page_stats_match.total_predicted_pages
        )
        mrp = 100 * self.safe_div(
            self.page_stats_match.recall,
            self.page_stats_match.total_label_pages
        )
        mjp = 100 * self.safe_div(
            self.page_stats_match.jaccard,
            self.page_stats_match.total_union_pages
        )
        mtpp = self.page_stats_match.total_predicted_pages
        mtlp = self.page_stats_match.total_label_pages
        mtup = self.page_stats_match.total_union_pages

        nmtpp = self.page_stats_no_match.total_predicted_pages
        nmtlp = self.page_stats_no_match.total_label_pages
        nmtup = self.page_stats_no_match.total_union_pages
        
        return (
            f"No Matches Schema\n"
            f"  Total No Matches: {self.no_matches.total}\n"
            f"  in_singleton: {self.no_matches.in_singleton}\n"
            f"  in_merged: {self.no_matches.in_merged}\n\n"
            f"Matches Schema\n"
            f"  Total Matches: {self.matches.total}\n"
            f"  in_singleton: {self.matches.in_singleton}\n"
            f"  in_merged: {self.matches.in_merged}\n\n"
            f"Gray Matches Schema\n"
            f"  Total Matches: {self.gray_matches.total}\n"
            f"  in_singleton: {self.gray_matches.in_singleton}\n"
            f"  in_merged: {self.gray_matches.in_merged}\n\n"
            f"Match Page Accuracy:\n"
            f"  Precision: {self.page_stats_match.precision:.2f}/{mtpp} - {mpp:.2f}%\n"
            f"  Recall:  {self.page_stats_match.recall:.2f}/{mtlp} - {mrp:.2f}%\n"
            f"  Jaccard: {self.page_stats_match.jaccard:.2f}/{mtup} - {mjp:.2f}%\n\n"
            f"No Match Page Accuracy:\n"
            f"  Precision: {self.page_stats_no_match.precision:.2f}/{nmtpp}\n"
            f"  Recall:  {self.page_stats_no_match.recall:.2f}/{nmtlp}\n"
            f"  Jaccard: {self.page_stats_no_match.jaccard:.2f}/{nmtup}\n\n"
        )


@dataclass
class PageEvaluationResults:
    stats:DocStats
    no_match_page_metrics:list[PageMetrics]
    match_page_metrics:list[PageMetrics]
    combined_page_metrics:list[PageMetrics]




#