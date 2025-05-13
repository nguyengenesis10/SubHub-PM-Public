

import numpy as np


from collections import defaultdict
from typing import Iterable
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances 
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist


from ..utils import build_embed_model
from ..processing import Merger
from ..configs import ClusterConfig,EmbedConfig
from ..outputs import MergeOutput, SCSMetrics
from ..evaluate import ScopeAlignmentEvaluator


class MergeParamEngine:
    def __init__(
        self,
        cluster_cfg:ClusterConfig,
        embed_cfg:EmbedConfig,
        device:str,
        env:dict,
        merger:Merger,
        Ts:Iterable[float]=None,
        Ns:Iterable[int]=None,
    ):

        self.model = build_embed_model(env, embed_cfg, device)
        self.linkage=cluster_cfg.linkage
        self.affinity=cluster_cfg.affinity
        self.merger=merger
        
        self.THRESHOLD="threshold"
        self.KMEANS="kmeans"
        if Ts and Ns:
            raise ValueError(
                f"can only execute param sweep with thresholds or n-clusters, not both!"
            )
        elif Ts:
            self.Ts=Ts
            self.run_type=self.THRESHOLD
        else:
            self.Ns=Ns
            self.run_type=self.KMEANS


    def _group_preds(
        self,
        preds:list[str],
        dist_matrix:np.ndarray,
        t:float=None,
        n:int=None,
    )->defaultdict:
        if not t and not n:
            raise ValueError(
                f"cannot perform clustering with no threshold or k-means value!"
            )
        
        clustering = AgglomerativeClustering(
            affinity=self.affinity,
            linkage=self.linkage,
            distance_threshold=t, 
            n_clusters=n
        )
        labels = clustering.fit_predict(dist_matrix)

        cluster_map = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_map[label].append(preds[idx])

        return cluster_map


    def _create_cluster_maps(
        self,
        preds:list[str],
    )->list[defaultdict]:
        pred_embeds = self.model.encode(
            preds, 
            convert_to_tensor=True
        )
        dist_matrix = cosine_distances(
            pred_embeds.cpu().numpy()
        )
        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            n = dist_matrix.shape[0]
            m = dist_matrix.shape[1]
            raise ValueError(
                f"Dist matrix has shape {n}x{m} is suppose to be square!"
            )
        if self.run_type == self.THRESHOLD:
            iterator = self.Ts
        elif self.run_type == self.KMEANS:
            iterator = self.Ns
        else:
            raise ValueError(
                f"run type: {self.run_type} is an invalid option!"
            )
        cluster_maps = []
        for val in iterator:
            if self.run_type == self.THRESHOLD:
                cluster_map=self._group_preds(
                    dist_matrix=dist_matrix,
                    preds=preds,
                    t=val,
                )
            elif self.run_type == self.KMEANS:
                cluster_map=self._group_preds(
                    dist_matrix=dist_matrix,
                    preds=preds,
                    n=val,
                )
            # Intentionally throw error if cluster_map goes undefined
            cluster_maps.append(cluster_map)
        
        return cluster_maps


    def _extract_metrics(
        self,
        merge_output:MergeOutput,
    ):
        greater_than_one = 0 
        print(f"\nCombined Pred Schema")
        for k,v in merge_output.merged_pred_groups.items():
            print(f"  Lenth of Combined Pred: {k} Count: {len(v)}")
            if k > 1:
                greater_than_one += k * len(v)
        return greater_than_one


    def run(
        self,
        preds:list[str],
        pred_pages:list[list[str]],
        testing:bool=True,
    )->list[float]:
        cluster_maps = self._create_cluster_maps(
            preds=preds
        )
        if testing:
            cluster_maps=cluster_maps[:2]
        
        merge_outputs=[]
        for cluster_map in cluster_maps:
            merge_outputs.append(
                self.merger.merge(
                    cluster_map=cluster_map,
                    pred_pages=pred_pages,
                    norm_preds=preds,
                    visualize_results=False,
                )
            )
        greater_than_ones=[
            self._extract_metrics(i) for i in merge_outputs
        ]
        
        return greater_than_ones


# Sloppy job. Will deprecate in the future. 
class CompareParamsEngine(MergeParamEngine):
    def __init__(
        self,
        cluster_cfg:ClusterConfig,
        embed_cfg:EmbedConfig,
        device:str,
        env:dict,
        alignment_evaluator:ScopeAlignmentEvaluator,
        scs_alpha:float,
        Ts:Iterable[float]=None,
        Ns:Iterable[int]=None,
    ):
        super().__init__(
            cluster_cfg,
            embed_cfg,
            device,
            env,
            merger=None,
            Ts=Ts,
            Ns=Ns,
        )
        self.alignment_evaluator=alignment_evaluator
        self.scs_alpha=scs_alpha
        

    def _cluster(
        self,
        dist_matrix:np.ndarray,
        t:float=None,
        n:int=None,
    ):
        clustering = AgglomerativeClustering(
            linkage=self.linkage,
            affinity=self.affinity,
            distance_threshold=t, # Similarity = 1-distance_threshold
            n_clusters=n,
        )
        cluster_labels=clustering.fit_predict(
            dist_matrix
        )
        return cluster_labels


    def _create_cluster_map(
        self,
        cluster_labels:np.ndarray,
        combined:list[str],
    )->defaultdict:
        
        cluster_map=defaultdict(list)
        for cluster_idx,ele in zip(cluster_labels,combined):
            cluster_map[cluster_idx].append(ele)
        return cluster_map


    def _get_cluster_dict(
        self,
        cluster_map:defaultdict,
        labels:list[str],
        preds:list[str],
    )->tuple[list[dict],defaultdict]:

        cluster_schema=defaultdict(int)
        cluster_dicts=[]
        for cluster_idx, cluster_elements in cluster_map.items():
            cluster_labels=[]
            cluster_preds=[]
            for ele in cluster_elements:
                if ele in labels:
                    cluster_labels.append(ele)
                elif ele in preds:
                    cluster_preds.append(ele)
            
            cluster_dicts.append(
                (cluster_labels,cluster_preds)
            )
            cluster_schema[
                ( len(cluster_labels) , len(cluster_preds) )
            ]+=1

        print(f"\nCluster Schema - {len(cluster_map.keys())}")
        for k,v in cluster_schema.items():
            print(
                f"  Label, Pred - {k[0] ,k[1]} : Count {v}"
            )
        
        return cluster_dicts,cluster_schema


    def _compute_recall_and_precision(
        self,
        cluster_schema:defaultdict,
        labels:list[str],
        preds:list[str],
    )->SCSMetrics:
        
        recall=0
        precision=0
        for k,v in cluster_schema.items():
            label_count=k[0]
            pred_count=k[1]
            if label_count > 0 and pred_count >0:
                recall+=label_count*v
                precision+=pred_count*v
        return SCSMetrics(
            scs_alpha=self.scs_alpha,
            precision_count=precision,
            pred_count=len(preds),
            recall_count=recall,
            label_count=len(labels),
        )


    def KMeans_compare(
        self,
        labels:list[str],
        preds:list[str],
    )->list[SCSMetrics]:

        combined = preds+labels
        combined_embeds=self.alignment_evaluator._embed_strs(combined)
        
        dist_matrix = cdist(
            XA=combined_embeds.cpu().numpy(), 
            XB=combined_embeds.cpu().numpy(), 
            metric='cosine'
        )
        print(
            f"Dist matrix shape: {dist_matrix.shape}"
        )
        if self.run_type == self.THRESHOLD:
            iterator = self.Ts
        elif self.run_type == self.KMEANS:
            iterator = self.Ns[self.Ns < len(preds)]
        else:
            raise ValueError(
                f"run type: {self.run_type} is an invalid option!"
            )

        scs_metrics = []
        for val in iterator:
            if self.run_type == self.THRESHOLD:
                cluster_labels=self._cluster(
                    dist_matrix=dist_matrix,
                    t=val,
                )
            elif self.run_type == self.KMEANS:
                cluster_labels=self._cluster(
                    dist_matrix=dist_matrix,
                    n=val,
                )
            cluster_map=self._create_cluster_map(
                cluster_labels=cluster_labels,
                combined=combined,
            )
            cluster_dict, cluster_schema = self._get_cluster_dict(
                cluster_map=cluster_map,
                labels=labels,
                preds=preds,
            )
            scs_metrics.append(
                self._compute_recall_and_precision(
                    cluster_schema=cluster_schema,
                    labels=labels,
                    preds=preds
                )
            )
        return scs_metrics,iterator
        








#

