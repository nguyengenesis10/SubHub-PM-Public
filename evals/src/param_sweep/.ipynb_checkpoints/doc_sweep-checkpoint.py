
import ast
import Levenshtein as lev


from tqdm import tqdm
from typing import Iterable
from collections import defaultdict
from datasets import Dataset


from ..evaluate import PageAccuracyEvaluator
from ..outputs import BiparteCluster, MergeOutput, GrayAreaBin, PredGroup, PageEvaluationResults
from ..configs import GrayAreaAnalyzerConfig, ComparePredsAndLabelsConfig


class DocSweepEngine:
    def __init__(
        self,
        cluster_objs:list[BiparteCluster],
        merge_output:MergeOutput,
        gray_area_analyzer_cfg:GrayAreaAnalyzerConfig
    ):
        self.cluster_objs=cluster_objs
        self.singleton_pred_groups = merge_output.singleton_pred_groups
        self.merged_pred_groups = merge_output.merged_pred_groups

        self.client=gray_area_analyzer_cfg.client
        self.model_name=gray_area_analyzer_cfg.model_name
        self.temperature=gray_area_analyzer_cfg.temperature
        self.hybrid_dist_alpha=gray_area_analyzer_cfg.hybrid_dist_alpha
        self.system_prompt=gray_area_analyzer_cfg.system_prompt


    def _setup_pbar(
        self
    ):
        total=0
        for cluster_obj in self.cluster_objs:
            for pred_group in cluster_obj.preds:
                total+=1
        
        pbar = tqdm(
            colour="blue",
            desc="Creating Lookup ",
            total=total,
            dynamic_ncols=True
        )
        return pbar


    # Copied and pasted from src.evaluate.gray
    def _classify(
        self,
        label:str,
        pred:str,
    )->str:
        
        user_prompt_template = f"""
            Given the following label and model prediction, classify their relationship.
            Label: {label}
            
            Prediction: {pred}
        """.strip()
        
        messages = [
            { "role": "system" , "content": self.system_prompt},
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


    def _create_lookup(
        self,
    )->dict:

        pbar = self._setup_pbar()
        lookup={}
        for cluster in self.cluster_objs:
            for pred_group in cluster.preds:
                classification=self._classify(
                    label=cluster.label,
                    pred=pred_group.pred
                )
                lookup[pred_group.pred]=classification
                pbar.update(1)
        
        return lookup


    # Copied and pasted from src.evaluate.gray
    def _compute_hybrid_distance(
        self,
        pred:PredGroup,
        label:str,
    )->float:
        norm_lev_dist=lev.ratio(pred.pred, label)
        norm_cos_dist=pred.dist_score
        
        return self.hybrid_dist_alpha*norm_lev_dist + (1-self.hybrid_dist_alpha)*norm_cos_dist


    def _updated_bins(
        self,
        t:float,
        lookup:dict,
    ):
        
        classifications=defaultdict(list)
        for cluster_obj in self.cluster_objs:
            used_labels=False
            for pred_group in cluster_obj.preds:
                if pred_group.dist_score > t:
                    classifications[
                        lookup[pred_group.pred]
                    ].append(pred_group)
        return classifications


    def run(
        self,
        Ts:Iterable[float],
        plans:Dataset,
        scope_breakout:Dataset,
        labels:list[str],
        compare_preds_and_labels_cfg:ComparePredsAndLabelsConfig,
    )->list[dict]:

        '''
        Args:
            labels: Use norm_labels generate from corpus.norm_labels as PageAccuracyEvaluator creates a lookup by zipping normalized labels with scope breakout page ids to compute Jaccard, precision, and recall.  
        '''

        if not hasattr(self, "lookup"):
            self.lookup=self._create_lookup()
        
        doc_param_sweep=[]
        doc_stats=[]
        for t in Ts:
            classifications=self._updated_bins(
                lookup=self.lookup,
                t=t,
            )
            
            # Update threshold 
            compare_preds_and_labels_cfg.threshold=t
            
            doc_evaluator=PageAccuracyEvaluator(
                classifications=classifications,
                plans=plans,
                scope_breakout=scope_breakout,
                singleton_pred_groups=self.singleton_pred_groups,
                merged_pred_groups=self.merged_pred_groups,
                labels=labels,
                compare_preds_and_labels_cfg=compare_preds_and_labels_cfg
            )
            doc_metrics=doc_evaluator.eval(
                cluster_objs=self.cluster_objs
            )
            doc_param_sweep.append(
                doc_metrics.stats.compute_document_metrics()
            )
            doc_stats.append(
                doc_metrics.stats
            )
        return doc_param_sweep, doc_stats






#