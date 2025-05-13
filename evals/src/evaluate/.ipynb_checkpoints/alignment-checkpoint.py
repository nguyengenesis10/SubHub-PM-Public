

import numpy as np


from datasets import Dataset
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist


from ..configs import (
    EmbedConfig,
    ComparePredsAndLabelsConfig,
)
from ..outputs import (
    SCSMetrics,
    PredGroup,
    BiparteCluster,
)


class ScopeAlignmentEvaluator:
    def __init__(
        self,
        compare_preds_and_labels_cfg:ComparePredsAndLabelsConfig,
        emb_model:SentenceTransformer
    ):
        '''
        Purpose:
            Evaluates normalized predictions and labels at a text level. Prediction-label matches are based on a strict distance threshold--in short computes hard semantic coverage does not 
            accounting for predictions that could match labels i.e. the gray area. 

        Args:
            emb_model - A sentence transformer which embeds text in order to compute cosine distance.
            compare_preds_and_labels_cfg - Class configuration.  
        '''
        
        self.emb_model=emb_model
        self.threshold=compare_preds_and_labels_cfg.threshold
        self.scs_alpha=compare_preds_and_labels_cfg.scs_alpha
        self.linkage=compare_preds_and_labels_cfg.cluster.linkage
        self.affinity=compare_preds_and_labels_cfg.cluster.affinity
        self.n_clusters=compare_preds_and_labels_cfg.n_clusters


    def _embed_strs(
        self,
        strs:list[str],
    )->np.ndarray:
        
        '''
        Purpose:
            Helper function which embeds a list of strings and converts into tensor. 
        '''
        
        return self.emb_model.encode(strs,convert_to_tensor=True)


    def _cluster(
        self,
        preds:list[str],
        labels:list[str],
    )->tuple[np.ndarray,np.ndarray]:
        
        '''
        Purpose:
            Given a list of normalized labels and predictions uses Agglomerative clusterinig to generate clusters of labels and predictions.
        
        Returns:
            dist_matrix - An m x n distance matrix shape based on the length of labels and predictions. 
        '''
        
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
        
        '''
        Purpose:
            Genertes a list of cluster objects by using top-1 sampling from the given distance matrix. 
            Essentially, for each prediction we sample the highest label that matched. 
            We essentially top-1 sample and this allows us to gauge how close the model got to generating a label. 

        Returns:
            cluster_objs - N number of labels that matched with predictions in a list of cluster objects 
        '''
        
        dist_matrix,_=self._cluster(
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


    # Considering putting the function as a seperate utility function. 
    def _bin_labels(
        self,
        labels:list[str],
        label_pages:list[list[str]],
        pred_pages:list[list[str]],
        plans:Dataset,
    )->tuple[list[str]]:

        '''
        Purpose:
            Seperate utility function for binning used and unused labels, too complicated.
        '''
        
        
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


    def eval(
        self,
        labels:list[str],
        preds:list[str],
        print_labels_and_preds:bool=False,
    )->SCSMetrics:
        
        '''
        Purpose:
            Main method for this class. Returns hard semantic coverage. 
        
        Args:
            labels: normalized list of labels 
            preds: normalized list of predictions
            print_labels_and_preds: Boolean value to visual label and prediction matched groups. 
        '''
        
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


        return SCSMetrics(
            scs_alpha=self.scs_alpha,
            precision_count=passed_preds_count,
            pred_count=len(preds),
            recall_count=passed_labels_count,
            label_count=len(labels),
        )







#