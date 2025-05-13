

import ast
import Levenshtein as lev


from tqdm import tqdm
from collections import defaultdict


from ..configs import (
    ComparePredsAndLabelsConfig,
    GrayAreaAnalyzerConfig,
)
from ..outputs import (
    BiparteCluster,
    SCSMetrics,
    PredGroup,
    GrayAreaBin,
)


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
        self.system_prompt=gray_area_analyzer_cfg.system_prompt
        
        self.threshold=compare_preds_and_labels_cfg.threshold
        self.scs_alpha=compare_preds_and_labels_cfg.scs_alpha


    def _classify(
        self,
        label:str,
        pred:str,
    )->str:
        
        '''
        Purpose:
            Given a single label-prediction possible pair classifies the prediction with the chosen ground truth model.
        '''
        
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


    def _compute_hybrid_distance(
        self,
        pred:PredGroup,
        label:str,
    )->float:
        norm_lev_dist=lev.ratio(pred.pred, label)
        norm_cos_dist=pred.dist_score
        
        '''
        Purpose:
            Computes a custom weight for gray area matches based on the literal and semantic difference. 

            Literal refering to the levenshtein distance and cosine similarity for the difference.
        '''
        
        return self.hybrid_dist_alpha*norm_lev_dist + (1-self.hybrid_dist_alpha)*norm_cos_dist


    def analyze(
        self,
        cluster_objs:list[BiparteCluster],
        preds:list[str],
        labels:list[str],
        print_and_log:bool=False,
        display_pbar:bool=True,
    ):
        '''
        Purpose:
            Main method for this class bins a series of cluster objects (each cluster has a label and N# of predictions) into hard matches or gray area predictions which 
            then get classified as matches, subscopes, or irrelevant.

        Returns: 
            self._classifications - A defaultdict with keys that are classifications and values lists of prediction objects updated with a hybrid weight value. 
        '''
        
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

        '''
        Purpose:
            Updates semantic coverage taking into account predictions that are classified as subscopes.
        '''
        
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
        
        return SCSMetrics(
            scs_alpha=self.scs_alpha,
            precision_count=precision,
            pred_count=len(preds),
            recall_count=recall,
            label_count=len(labels),
        )
