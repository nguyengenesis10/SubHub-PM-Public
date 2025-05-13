

from datasets import Dataset
from collections import defaultdict
from typing import Union, List, Tuple, Set 


from ..configs import (
    ComparePredsAndLabelsConfig,
)
from ..outputs import (
    DocStats,
    PageEvaluationResults,
    BiparteCluster,
    PredGroup,
    PageMetrics,
)


class PageAccuracyEvaluator:
    def __init__(
        self,
        plans:Dataset,
        scope_breakout:Dataset,
        singleton_pred_groups:list[tuple],
        merged_pred_groups:defaultdict[list[tuple]],
        labels:list[str],
        classifications:defaultdict[list],
        compare_preds_and_labels_cfg:ComparePredsAndLabelsConfig
    ):
        '''
        Purpose:
            Evaluate the model's document accuracy based on coverage of labels and predictions.
        
        Args:
            labels - List of normalized labels, need to be the same as the one's in the cluster_objs.
        '''
        
        
        PAGE_ID = "page_id"
        if PAGE_ID not in plans.features.keys():
            raise KeyError(
                f"'{PAGE_ID}' is missing from plans!"
            )
        PAGE_IDS = "page_ids"
        if PAGE_IDS not in scope_breakout.features.keys():
            raise KeyError(
                f"'{PAGE_IDS}' is missing from scope_breakout!"
            )
        
        # Create Lookups for O(1) access
        self.singleton_lookup = {
            p: pages for ps, pages in singleton_pred_groups for p in ps
        }
        self.merged_lookup = {
            p: pages for mpg in merged_pred_groups.values() for ps, pages in mpg for p in ps
        }
        self.scope_breakout_loopkup={
            a:b  for a,b in zip(labels , scope_breakout[PAGE_IDS])
        }
        self.plan_page_ids = plans[
            PAGE_ID
        ]
        
        MATCH = "Match"
        self.gray_area_matches = [
            i.pred for i in classifications[MATCH]
        ] if MATCH in classifications.keys() else []

        SUBSCOPE = "Subscope"
        self.gray_area_subscopes = [
            i.pred for i in classifications[SUBSCOPE]
        ] if SUBSCOPE in classifications.keys() else []

        IRRELEVANT = "Irrelevant"
        
        self.threshold=compare_preds_and_labels_cfg.threshold


    def _jaccard_score(
        self,
        label_pages: List[str], 
        pred_pages: List[str],
    ) -> Tuple[float, Set[str], Set[str], Set[str]]:
        """
        Returns:
            jaccard_score : float
            matched_pages : set of TP pages (label ∩ pred)
            missed_pages  : set of FN pages (label - pred) 
            extra_pages   : set of FP pages (pred - label) -> False Positives
        """
        L, P = set(label_pages), set(pred_pages)
        intersection = L & P
        union = L | P
        missed = L - P
        extra = P - L

        score = len(intersection) / len(union) if union else 0.0
        return (
            score, intersection, missed, extra
        )


    def _page_recall_and_precision(
        self,
        label_pages:list[str], 
        pred_pages:list[str],
    )->tuple[int]:
        
        '''
        Purpose:
            Helper function that computes the recall and precision for a given label-prediction grouping. 
        '''
        
        recall = 0
        if len(label_pages) != 0:
            recall  = len(set(label_pages) & set(pred_pages)) / len(set(label_pages))

        precision = 0
        if len(pred_pages) != 0:
            precision = len(set(label_pages) & set(pred_pages)) / len(set(pred_pages))

        return recall, precision


    def _jaccard_recall_and_precision(
        self,
        label_pages:list[str], 
        pred_pages:list[str],
    )->Tuple[float, Set[str], Set[str], Set[str],int,int]:
        
        '''
        Purpose:
             Computes jaccard score, page matches, page misses(label page not in prediciton pages), extra(prediction page not in label pages), recall and precision.
        '''
        
        score, intersection, missed, extra = self._jaccard_score(
            label_pages,pred_pages
        )
        recall, precision = self._page_recall_and_precision(
            label_pages,pred_pages
        )
        return score, intersection, missed, extra, recall, precision


    def _update_page_schema_stats(
        self,
        page_schema_stats: dict,
        intersection: set[str],
        missed: set[str],
        extra: set[str]
    ) -> None:
        
        
        for page_id in intersection:
            if page_id in self.plan_page_ids:
                page_schema_stats[page_id][0] += 1  # matches
        for page_id in missed:
            if page_id in self.plan_page_ids:
                page_schema_stats[page_id][1] += 1  # misses
        for page_id in extra:
            if page_id in self.plan_page_ids:
                page_schema_stats[page_id][2] += 1  # extras


    def _get_pred_page_ids(
        self, 
        pred_str: str
    )->tuple[list[str]]:
        
        singleton_pages = self.singleton_lookup.get(pred_str, [])
        merged_pages = self.merged_lookup.get(pred_str, [])
        
        return singleton_pages , merged_pages


    def _update_counts(
        self,
        singleton_page_ids:list[str],
        merge_page_ids:list[str],
    )->tuple[int]:
        if merge_page_ids and singleton_page_ids:
            return 1,1 
        elif singleton_page_ids:
            return 1,0
        elif merge_page_ids:
            return 0,1


    def _classify_pred(
        self,
        pred:PredGroup,
    )->str:
        if pred.dist_score < self.threshold or pred.pred in self.gray_area_matches:
            return "match"
        elif pred.pred in self.gray_area_subscopes:
            return "gray"
        else:
            return "none"


    def eval(
        self,
        cluster_objs:list[BiparteCluster],
    )->PageEvaluationResults:
        
        '''
        Purpose:
            Main method for this class. 
            
            Computes the doc accuracy taking into account gray area predictions that could've matched with labels and computes a series of doc metrics to be used for plotting.
        '''
        
        # Nested dataclass that counts everything.
        stats=DocStats()
        
        MATCH="match"
        GRAY="gray"
        NONE="none"
        HYBRID_DIST="hybrid_dist"
        
        for cluster_obj in cluster_objs:
            page_ids_labels = self.scope_breakout_loopkup[cluster_obj.label] 
            # page_ids_preds=[]
            
            for pred in cluster_obj.preds:
                pred_to_retr=pred.pred
                
                singleton_page_ids, merge_page_ids = self._get_pred_page_ids(
                        pred_str=pred_to_retr
                    )
                merge_count, single_count = self._update_counts(
                        singleton_page_ids=singleton_page_ids,
                        merge_page_ids=merge_page_ids,
                    )
                category=self._classify_pred(
                    pred=pred
                )
                
                if category == MATCH:
                    stats.matches.in_merged+=merge_count
                    stats.matches.in_singleton+=single_count
                    stats.matches.total+=1
                elif category == GRAY:
                    stats.gray_matches.in_merged+=merge_count
                    stats.gray_matches.in_singleton+=single_count
                    stats.gray_matches.total+=1
                else:
                    stats.no_matches.in_merged+=merge_count
                    stats.no_matches.in_singleton+=single_count
                    stats.no_matches.total+=1
                    
                if category in (MATCH, GRAY):
                    page_ids_preds=set(
                        singleton_page_ids+merge_page_ids
                    )
                    j, inter, miss, ext, r, p = self._jaccard_recall_and_precision(
                        page_ids_preds,
                        page_ids_labels
                    )
                    # Weigh down gray area values
                    if hasattr(pred,HYBRID_DIST) and category==GRAY:
                        j*=pred.hybrid_dist
                        r*=pred.hybrid_dist
                        p*=pred.hybrid_dist
                    
                    self._update_page_schema_stats(
                        stats.page_schema_stats_match,
                        intersection=inter,
                        missed=miss,
                        extra=ext,
                    )
                    stats.page_stats_match.recall += r
                    stats.page_stats_match.precision += p
                    stats.page_stats_match.jaccard += j
                    stats.page_stats_match.total_label_pages += len(inter) + len(miss)
                    stats.page_stats_match.total_predicted_pages += len(inter) + len(ext)
                    stats.page_stats_match.total_union_pages += len(inter) + len(miss) + len(ext)
                    
                else:
                    pred_only_pages = set(
                        singleton_page_ids + merge_page_ids
                    )
                    label_pages_empty = []
                    
                    j, inter, miss, ext, r, p = self._jaccard_recall_and_precision(
                        label_pages_empty, 
                        pred_only_pages,
                    )
                    self._update_page_schema_stats(
                        stats.page_schema_stats_no_match,
                        intersection=inter,
                        missed=miss,
                        extra=ext,
                    )
                    stats.page_stats_no_match.recall += r
                    stats.page_stats_no_match.precision += p
                    stats.page_stats_no_match.jaccard += j
                    stats.page_stats_no_match.total_label_pages += len(inter) + len(miss)
                    stats.page_stats_no_match.total_predicted_pages += len(inter) + len(ext)
                    stats.page_stats_no_match.total_union_pages += len(inter) + len(miss) + len(ext)
        
        print(stats)
        
        self.match_page_metrics = [
            PageMetrics(
                page_id=page_id,
                page_num=self.plan_page_ids.index(page_id)+1,
                matches=metrics[0],
                misses=metrics[1],
                extra=metrics[2],
            ) for page_id , metrics in stats.page_schema_stats_match.items()
        ]
        
        
        self.no_match_page_metrics = [
            PageMetrics(
                page_id=page_id,
                page_num=self.plan_page_ids.index(page_id)+1,
                matches=metrics[0],
                misses=metrics[1],
                extra=metrics[2],
            ) for page_id , metrics in stats.page_schema_stats_no_match.items()
        ]
        
        
        stats._create_combined_page_schema_stats()
        self.combined_page_metrics = [
            PageMetrics(
                page_id=page_id,
                page_num=self.plan_page_ids.index(page_id)+1,
                matches=metrics[0],
                misses=metrics[1],
                extra=metrics[2],
            ) for page_id , metrics in stats.page_schema_stats_combined_match.items()
        ]
        
        
        return PageEvaluationResults(
            stats=stats,
            match_page_metrics=self.match_page_metrics, 
            no_match_page_metrics=self.no_match_page_metrics,
            combined_page_metrics=self.combined_page_metrics,
        )















#