

from typing import Iterable
from sentence_transformers import SentenceTransformer


from ..outputs import BiparteCluster, SCSMetrics
from ..configs import ComparePredsAndLabelsConfig


class SCSParamSearchEngine:
    def __init__(
        self,
        cluster_objs:list[BiparteCluster],
        compare_preds_and_labels_cfg:ComparePredsAndLabelsConfig,
    ):
        self.cluster_objs = cluster_objs
        self.scs_alpha = compare_preds_and_labels_cfg.scs_alpha


    # Deprecated SCSMetrics, handles upon init. Copied and pasted from evaluate.alignment.py, with minor modifications. 
    def _compute_semantic_coverage(
        self,
        recall:float,
        precision:float
    )->float:

        return self.scs_alpha*recall - (1-self.scs_alpha)*(1-precision)


    def _log_recall_and_precision(
        self,
        t:float,
        pred_count:int,
        passed_preds_count:int,
        label_count:int,
        passed_labels_count:int,
    )->None:
        
        print(f"\nResults given threshold: {t:0.4f}")
        print(f"  Used Preds | Precision: {passed_preds_count}/{pred_count} - {(passed_preds_count/pred_count)*100:2.4f}%")
        print(f"  Used Labels | Recall: {passed_labels_count}/{label_count} - {(passed_labels_count/label_count)*100:2.4f}%")


    def _compute_recall_and_precision(
        self,
        t:float,
        labels:list[str],
        preds:list[str],
    )->SCSMetrics:

        passed_preds_count=0
        passed_labels_count=0
        for cluster_obj in self.cluster_objs:
            label_used=False
            for pred in cluster_obj.preds:
                if pred.dist_score <= t:
                    passed_preds_count+=1
                    label_used=True
            passed_labels_count+=1 if label_used else 0

        self._log_recall_and_precision(
            t=t,
            pred_count=len(preds),
            passed_preds_count=passed_preds_count, # precision count
            label_count=len(labels),
            passed_labels_count=passed_labels_count, # recall count 
        )
        
        
        return SCSMetrics(
            scs_alpha=self.scs_alpha,
            precision_count=passed_preds_count,
            pred_count=len(preds),
            recall_count=passed_labels_count,
            label_count=len(labels),
        )


    def run(
        self,
        Ts:Iterable[float],
        labels:list[str],
        preds:list[str],
    )->list[SCSMetrics]:

        param_sweep_results=[]
        for t in Ts:
            param_sweep_results.append(
                self._compute_recall_and_precision(
                    t=t,
                    labels=labels,
                    preds=preds,
                )
            )
        return param_sweep_results






#
