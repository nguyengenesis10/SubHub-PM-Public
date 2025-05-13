
import ast
import numpy as np
import Levenshtein as lev


from datasets import Dataset
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from typing import Union, List, Tuple, Set 
from tqdm import tqdm


from .system_prompts import (
    gray_area_analysis_system_prompt
)
from .configs import (
    EmbedConfig,
    ComparePredsAndLabelsConfig,
    GrayAreaAnalyzerConfig,
)
from .outputs import (
    BiparteCluster,
    SCSMetrics,
    PredGroup,
    PageMetrics,
    GrayAreaBin,
    DocStats,
    MasterPageStats,
)


class ScopeAlignmentEvaluator:
    def __init__(
        self,
        embed_cfg:EmbedConfig,
        compare_preds_and_labels_cfg:ComparePredsAndLabelsConfig,
        device:str,
    ):
        self.threshold=compare_preds_and_labels_cfg.threshold
        self.scs_alpha=compare_preds_and_labels_cfg.scs_alpha
        self.model=SentenceTransformer(
            embed_cfg.model_name,
            device=device,
        )
        self.linkage=compare_preds_and_labels_cfg.cluster_cfg.linkage
        self.affinity=compare_preds_and_labels_cfg.cluster_cfg.affinity
        self.n_clusters=compare_preds_and_labels_cfg.n_clusters


    def _embed_strs(
        self,
        strs:list[str],
    )->np.ndarray:
        
        return self.model.encode(strs,convert_to_tensor=True)


    def _cluster(
        self,
        preds:list[str],
        labels:list[str],
    )->tuple[np.ndarray,np.ndarray]:
    
        pred_embeds = self._embed_strs(strs=preds)
        label_embeds = self._embed_strs(strs=labels)
        
        dist_matrix = cdist(
            XA=pred_embeds.cpu().numpy(), 
            XB=label_embeds.cpu().numpy(), 
            metric='cosine'
        )
        clustering = AgglomerativeClustering(
            linkage=self.linkage,
            distance_threshold=self.threshold, # Similarity = 1-distance_threshold
            n_clusters=self.n_clusters
        )
        cluster_labels=clustering.fit_predict(
            dist_matrix
        )
        return dist_matrix,cluster_labels


    def _generate_cluster_objs(
        self,
        preds:list[str],
        labels:list[str],
    )->list[BiparteCluster]:

        dist_matrix,cluster_labels=self._cluster(
            preds=preds,
            labels=labels
        )
        highest_labels_idxs=np.argmin(
            dist_matrix,axis=1
        )
        highest_label_sims=np.min(
            dist_matrix,axis=1
        )
        label_to_preds_mapping=defaultdict(list)
        for preds_idx, labels_idx in enumerate(highest_labels_idxs):
            label_to_preds_mapping[
                labels[labels_idx]
            ].append(
                (preds[preds_idx], highest_label_sims[preds_idx])
            )
        
        cluster_objs:list[BiparteCluster] = []
        for label, tupled_preds in label_to_preds_mapping.items():
            pred_groups=[]
            for pred, dist_score in tupled_preds:
                pred_groups.append(
                    PredGroup(
                        pred=pred,
                        dist_score=dist_score
                    )
                )
            cluster_objs.append(
                BiparteCluster(
                    label=label,
                    preds=pred_groups,
                )
            )
        return cluster_objs


    def eval(
        self,
        labels:list[str],
        preds:list[str],
        print_labels_and_preds:bool=False,
    )->SCSMetrics:
        
        
        if not self.threshold:
            raise ValueError(
                f"Analysis not supported with k-clustering, please define a float threshold in the configuration!"
            )
        
        cluster_objs = self._generate_cluster_objs(
            preds=preds,
            labels=labels
        )
        
        self.cluster_objs = cluster_objs
        
        passed_preds_count=0
        passed_labels_count=0
        for cluster in self.cluster_objs:
            label_used=False
            if print_labels_and_preds:
                print(f"\nLabel: {cluster.label}") 
                print(f"  Preds:")
            for pred in cluster.preds:
                if pred.dist_score <= self.threshold:
                    if print_labels_and_preds:
                        print(f"    {pred.dist_score:0.4f} - {pred.pred}")
                    passed_preds_count+=1
                    label_used=True
            passed_labels_count+=1 if label_used else 0

        # Convert to logging, instead of printing in production.
        print(f"\nResults given threshold: {self.threshold:0.4f}")
        print(f"  Used Preds | Precision: {passed_preds_count}/{len(preds)} - {(passed_preds_count/len(preds))*100:2.4f}%")
        print(f"  Used Labels | Recall: {passed_labels_count}/{len(labels)} - {(passed_labels_count/len(labels))*100:2.4f}%")

        self.recall=passed_labels_count/len(labels)
        self.precision=passed_preds_count/len(preds)

        return SCSMetrics(
            scs=self._compute_semantic_coverage(),
            scs_alpha=self.scs_alpha,
            
            precision_count=passed_preds_count,
            pred_count=len(preds),
            precision=self.precision,
            
            recall_count=passed_labels_count,
            label_count=len(labels),
            recall=self.recall,
        )


    def _compute_semantic_coverage(
        self
    )->float:

        if not hasattr(self,"recall") and not hasattr(self,"precision"):
            raise AttributeError(
                f"Please call 'eval' method first!"
            )
        return self.scs_alpha*self.recall - (1-self.scs_alpha)*(1-self.precision)


    # Considering putting the function as a seperate utility function. 
    def _bin_labels(
        self,
        labels:list[str],
        label_pages:list[list[str]],
        pred_pages:list[list[str]],
        plans:Dataset,
    )->tuple[list[str]]:
        
        
        if "page_id" not in plans.features.keys():
            raise KeyError(
                "Plan should have key `page_id`!"
            )
        
        
        used_labels=[]
        for cluster_obj in self.cluster_objs:
            if cluster_obj.label in labels:
                used_labels.append(
                    cluster_obj.label
                )
        unused_labels=[]
        for label in labels:
            if label not in used_labels:
                unused_labels.append(label)
        
        print(f"Label Breakdown Count: ")
        print(f"  Unused Label Count: {len(unused_labels)}")
        print(f"  Used Label Count: {len(used_labels)}\n")
        print(f"Unused Labels:")
        
        for label in sorted(unused_labels):
            page_nums=[]
            # Only retrieve the first page in which the label was seen. 
            for page_id in label_pages[labels.index(label)][:1]:
                try:
                    page_num = plans["page_id"].index(page_id) + 1 
                except ValueError:
                    page_num = page_id
                page_nums.append(
                    page_num
                )
            if len(page_nums) == 1:
                print(f"  Pg {page_nums[0]:02.0f} - {label}")
            else:
                print(f"  Pg {page_nums} - {label}")
        
        return used_labels,unused_labels


class GrayAreaAnalyzer:
    def __init__(
        self,
        gray_area_analyzer_cfg:GrayAreaAnalyzerConfig,
        compare_preds_and_labels_cfg:ComparePredsAndLabelsConfig,
    ):
        self.client=gray_area_analyzer_cfg.client
        self.model_name=gray_area_analyzer_cfg.model_name
        self.temperature=gray_area_analyzer_cfg.temperature
        self.hybrid_dist_alpha=gray_area_analyzer_cfg.hybrid_dist_alpha
        
        self.threshold=compare_preds_and_labels_cfg.threshold
        self.scs_alpha=compare_preds_and_labels_cfg.scs_alpha


    def _classify(
        self,
        label:str,
        pred:str,
        system_prompt:str=gray_area_analysis_system_prompt,
    )->str:
        
        user_prompt_template = f"""
            Given the following label and model prediction, classify their relationship.
            Label: {label}
            
            Prediction: {pred}
        """.strip()
        
        messages = [
            { "role": "system" , "content": system_prompt},
            { "role": "user" , "content": user_prompt_template}
        ]
    
        resp=self.client.beta.chat.completions.parse(
            model = self.model_name, 
            messages = messages, 
            temperature = self.temperature, 
            response_format = GrayAreaBin
        )
        classification=GrayAreaBin(
            **ast.literal_eval(resp.choices[0].message.content)
        ).group
        
        return classification


    def _compute_hybrid_distance(
        self,
        pred:PredGroup,
        label:str,
    )->float:
        norm_lev_dist=lev.ratio(pred.pred, label)
        norm_cos_dist=pred.dist_score
        
        return self.hybrid_dist_alpha*norm_lev_dist + (1-self.hybrid_dist_alpha)*norm_cos_dist


    def analyze(
        self,
        cluster_objs:list[BiparteCluster],
        preds:list[str],
        labels:list[str],
        print_and_log:bool=False,
        display_pbar:bool=True,
    ):
        
        if display_pbar:
            pbar = tqdm(
                colour="blue",
                desc="Analyzing Gray Area ",
                total=len(cluster_objs),
                dynamic_ncols=True
            )
        
        gray_area_count=0
        hard_pred_count=0
        used_label_count=0
        classifications=defaultdict(list)
        
        for working_idx,cluster_obj in enumerate(cluster_objs):
            if print_and_log:
                print(f"\nLabel: {cluster_obj.label}")
            used_labels=False
            for pred in cluster_obj.preds:
                if print_and_log:
                    print(f"  {pred.dist_score:0.4f} - {pred.pred}")
                if pred.dist_score <= self.threshold:
                    hard_pred_count+=1
                    used_labels=True
                elif pred.dist_score > self.threshold:
                    gray_area_count+=1
                    classification=self._classify(
                        label=cluster_obj.label,
                        pred=pred.pred,
                    )
                    if print_and_log:
                        print(f"  {pred.dist_score:0.4f} - {classification} - {pred.pred}")
                    # Compute hybrid distance and add as attribute
                    pred.hybrid_dist=self._compute_hybrid_distance(
                        pred=pred,
                        label=cluster_obj.label,
                    )
                    classifications[classification].append(pred)
            if used_labels:
                used_label_count+=1
            if display_pbar:
                pbar.update(1)
                pbar.set_description(
                    f"Cluster Objs Processed: {working_idx+1}/{len(cluster_objs)}"
                )

        # Sanity Check print
        print(f"\nHard pred count: {hard_pred_count}/{len(preds)} - {(hard_pred_count/len(preds))*100:2.4f}%")
        print(f"Gray Area count: {gray_area_count}/{len(preds)} - {(gray_area_count/len(preds))*100:2.4f}%")
        print(f"Used Label count: {used_label_count}/{len(labels)} - {(used_label_count/len(labels))*100:2.4f}%")

        # Print classifications schema
        print("\nGray Area Breakdown:")
        for cls, pred_list in classifications.items():
            print(
                f"  {cls} : {len(pred_list)} - {(len(pred_list)/gray_area_count)*100:2.4f}%"
            )

        self._classifications = classifications
        self._hard_pred_count = hard_pred_count
        self._used_label_count = used_label_count


    def update_scs(
        self,
        labels:list[str],
        preds:list[str],
    )->float:
        
        MATCH="Match"
        SUBSCOPE="Subscope"
        
        recall=self._used_label_count
        precision=self._hard_pred_count
        for key in (MATCH, SUBSCOPE):
            for m_p in self._classifications.get(key, []):
                if key == MATCH:
                    weight = 1.0 
                elif key == SUBSCOPE:
                    weight = m_p.hybrid_dist
                recall += weight
                precision += weight
        
        recall_float=recall/len(labels)
        precision_float=precision/len(preds)
        updated_scs = self.scs_alpha * recall_float - (1-self.scs_alpha) * (1-precision_float)
        
        print(f"Updated Recall: {recall:0.0f}/{len(labels)} - {recall_float*100:.4f}%")
        print(f"Updated Precision: {precision:0.0f}/{len(preds)} - {precision_float*100:.4f}%")
        print(f"Updated SCS with T={self.threshold} : {updated_scs*100:.4f}%")

        
        return SCSMetrics(
            scs=updated_scs,
            scs_alpha=self.scs_alpha,
            
            precision_count=precision,
            pred_count=len(preds),
            precision=precision_float,
            
            recall_count=recall,
            label_count=len(labels),
            recall=recall_float,
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
    )->MasterPageStats:

        # Nested dataclass that counts, everything
        stats=DocStats()
        
        MATCH="match"
        GRAY="gray"
        NONE="none"
        
        for cluster_obj in cluster_objs:
            page_ids_labels = self.scope_breakout_loopkup[cluster_obj.label] 
            page_ids_preds=[]
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
                    page_ids_preds.extend(
                        singleton_page_ids+merge_page_ids
                    )
                    j, inter, miss, ext, r, p = self._jaccard_recall_and_precision(
                        page_ids_preds,
                        page_ids_labels
                    )
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
                    # Add to docstring - Singleton and merged page groups be default cover all model predictions, clusters 
                    # form that meet the threshold defined in the grouper. Hence this condition, should never fail. 
                    pred_only_pages = singleton_page_ids + merge_page_ids
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
        
        
        return MasterPageStats(
            stats=stats,
            match_page_metrics=self.match_page_metrics, 
            no_match_page_metrics=self.no_match_page_metrics,
            combined_page_metrics=self.combined_page_metrics,
        )




#