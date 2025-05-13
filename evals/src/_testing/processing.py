

import ast 
import numpy as np


from typing import Union
from tqdm import tqdm 
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer


from .outputs import (
    NormPhrase,
    GrouperOutput,
    MergeOutput,
    MergedScopes
)
from .configs import (
    NormalizeConfig,
    GroupPredsConfig,
    EmbedConfig,
    MergePredsConfig,
)
from .system_prompts import (
    phrase_normalize_system_prompt,
    merge_system_prompt_new,
)


class PhraseNormalizer:
    def __init__(
        self,
        norm_cfg:NormalizeConfig,
    ):
        self.client=norm_cfg.client
        self.model_name=norm_cfg.model_name
        self.temperature=norm_cfg.temperature
        self.print_log=norm_cfg.print_log
        self.display_pbar=norm_cfg.display_pbar


    def normalize(
        self,
        phrases:list[str],
    ):
        
        
        print_log=self.print_log
        display_pbar=self.display_pbar
        if print_log and display_pbar:
            print(
                f"print_log and display_pbar cannot be same value, setting print_log=False and display_pbar=True"
            )
            print_log=False
            display_pbar=True
        
        pbar=None
        if display_pbar:
            pbar = tqdm(
                colour="blue",
                desc="Normalizing Phrases",
                total=len(phrases),
                dynamic_ncols=True
            )
        cleaned_phrases=[]
        for phrase in phrases:
            messages = [
                { "role": "system" , "content": phrase_normalize_system_prompt},
                { "role": "user" , "content": f"Normalize the following scope:\n{phrase}"}
            ]
    
            resp=self.client.beta.chat.completions.parse(
                model = self.model_name, 
                messages = messages, 
                temperature = self.temperature, 
                response_format = NormPhrase
            )
            cleaned_phrase=NormPhrase(
                **ast.literal_eval(resp.choices[0].message.content)
            ).phrase
            if print_log:
                print(
                    f"Original: {phrase}\nCleaned: {cleaned_phrase}\n"
                )
            cleaned_phrases.append(cleaned_phrase)
            if pbar:
                pbar.update(1)
                pbar.set_description(
                    f"Phrases Processed: {len(cleaned_phrases)}/{len(phrases)}"
                )
        return cleaned_phrases


class Grouper:
    def __init__(
        self,
        group_cfg:GroupPredsConfig,
        embed_cfg:EmbedConfig,
        device:str
    ):
        self.model=SentenceTransformer(
            embed_cfg.model_name,
            device=device,
        )
        self.threshold=group_cfg.threshold
        self.n_clusters=group_cfg.n_clusters
        self.linkage=group_cfg.cluster.linkage
        self.affinity=group_cfg.cluster.affinity
    
    
    def group_preds(
        self,
        preds:list[str],
        output_cluster_results:bool=False,
    )->Union[defaultdict, GrouperOutput]:
        
        pred_embeds = self.model.encode(
            preds, 
            convert_to_tensor=True
        )
        dist_matrix = cosine_distances(
            pred_embeds.cpu().numpy()
        )
        print(
            f"Dist-matrix shape: {dist_matrix.shape}"
        )
        clustering = AgglomerativeClustering(
            affinity=self.affinity,
            linkage=self.linkage,
            distance_threshold=self.threshold, 
            n_clusters=self.n_clusters
        )
        labels = clustering.fit_predict(dist_matrix)
        
        cluster_map = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_map[label].append(preds[idx])
        
        self.cluster_map=cluster_map
        self.dist_matrix=dist_matrix
        
        if output_cluster_results:
            return GrouperOutput(
                labels=labels,
                cluster_map=cluster_map,
                dist_matrix=dist_matrix
            )
        return cluster_map


    def _visualize_cluster_schema(
        self,
        print_clusters:bool=False,
        return_cluster_schema:bool=False,
    ):
        cluster_schema=defaultdict(int)
        nonsingleton_cluser_count=0
        singleton_cluser_count=0
        for k,v in self.cluster_map.items():
            cluster_schema[len(v)]+=1
            if len(v) > 1:
                nonsingleton_cluser_count+=1
                if print_clusters:
                    print(f"\nCluster {k}:")
                for pred in v:
                    if print_clusters:
                        print(f"  {pred}")
            else:
                singleton_cluser_count+=1

        print(f"\nClusters Breakdown:")
        total_preds=0
        for k,v in dict(sorted(cluster_schema.items())).items():
            print(f"  Node Count {k} : Cluster Count {v}")
            total_preds+= k*v
        
        print(f"\nCluster Schema:")
        print(f"  Nonsingleton Cluster Count: {nonsingleton_cluser_count}")
        print(f"  Singleton Cluster Count: {singleton_cluser_count}")
        print(f"  Total Clusters: {singleton_cluser_count+nonsingleton_cluser_count}")
        
        print(f"\nNum preds | Total Nodes: {total_preds}")
        
        return cluster_schema if return_cluster_schema else None


    def _visualize_avg_cluster_dist(
        self,
        preds:list[str],
        return_cluster_schema:bool=False,
    ):
        
        cluster_dist_check=[]
        cluster_schema_avg_dist=defaultdict(list)
        for cluster_idx , sim_preds in self.cluster_map.items():
            if len(sim_preds) > 1:
                sim_preds_idxs=[]
                for p in sim_preds:
                    sim_preds_idxs.append(
                        preds.index(p)
                    )
                # fetch the distance between each node using an upper triangular matrix
                sim_preds_dist=self.dist_matrix[np.ix_(
                    sim_preds_idxs, sim_preds_idxs
                )][
                    np.triu_indices(
                        len(sim_preds_idxs),
                        k=1
                    )
                ]
        
                # compute mean edge distance within cluster 
                cluster_dist_check.append(
                    np.mean(sim_preds_dist)
                )
                cluster_schema_avg_dist[len(sim_preds)].append(
                    sim_preds_dist
                )
        print(f"Cluster Avg Dist Schema")
        for node_count, avgs in dict(sorted(cluster_schema_avg_dist.items())).items():
            display_avg = np.mean(np.concatenate(avgs))
            display_std = np.std(np.concatenate(avgs), ddof=1)
            print(f"  Num Nodes: {node_count} Avg Dist: {display_avg:0.4f} Dist Stdev: {display_std:0.4f}")

        if self.threshold:
            fail_dist=False
            for avg_cluster_dist in cluster_dist_check:
                if avg_cluster_dist > self.threshold:
                    fail_dist=True
                    break 
            if not fail_dist:
                print(f"\nAll clusters have intra distance less than {self.threshold}")
            else:
                print(f"\nSome clusters have intra distance greater than {self.threshold}")

        return cluster_schema_avg_dist if return_cluster_schema else None 


class Merger:
    def __init__(
        self,
        merge_cfg:MergePredsConfig,
    ):

        self.client=merge_cfg.client
        self.model_name=merge_cfg.model_name
        self.temperature=merge_cfg.temperature
        self.display_pbar=merge_cfg.display_pbar


    def _merge(
        self,
        scopes:list[str],
        system_prompt:str,
    )->list[str]:
    
        if system_prompt:
            messages=[
                { "role": "system" , "content": system_prompt},
                { "role": "user" , "content": f"Combine the lists of scopes in a concise way:\n{scopes}"}
            ]
        else:
            messages=[
                { "role": "user" , "content": f"Combine the lists of scopes in a concise way:\n{scopes}"}
            ]
        resp=self.client.beta.chat.completions.parse(
            model = self.model_name,
            messages = messages, 
            temperature = self.temperature, 
            response_format = MergedScopes
        )
        merged_scope=MergedScopes(
             **ast.literal_eval(resp.choices[0].message.content)
        ).scopes

        return merged_scope


    def merge(
        self,
        cluster_map:defaultdict,
        pred_pages:list[str],
        norm_preds:list[str],
        system_prompt:str=merge_system_prompt_new,
        visualize_results:bool=True,
    )->MergeOutput:
            
        display_pbar=self.display_pbar
        if display_pbar:
            pbar = tqdm(
            colour="blue",
            desc="Merging Phrases",
            total=len(cluster_map.keys()),
            dynamic_ncols=True
        )
            
        merged_pred_schema=defaultdict(int)
        merged_pred_groups=defaultdict(list)
        already_in_merged=[]
        singleton_pred_groups=[]

        for working_idx, (luster_idx, sim_preds) in enumerate(cluster_map.items()):
            retrieved_pages=set()
            for p in sim_preds:
                retr_page = pred_pages[
                    norm_preds.index(p)
                ]
                if isinstance(retr_page, str):
                    pred_page=ast.literal_eval(
                        retr_page
                )
                elif isinstance(retr_page, list):
                    pred_page=retr_page
                else:
                    raise TypeError(
                        f"'{retr_page}' is suppose to be of type list or string, is of type: {type(retr_page)} "
                    )
                # Current method of generation is setup such that every model prediction has exactly one page.
                if len(pred_page) == 1:
                    retrieved_pages.add(
                        pred_page[0]
                    )
                else:
                    raise ValueError(
                        f"Error when extracting page from '{pred_pages[norm_preds.index(p)]}', expecting 'pred_page' to have length of 1!"
                    )
            retrieved_pages = list(retrieved_pages)
            if len(sim_preds) > 1:
                combined_preds=self._merge(
                    scopes=sim_preds,
                    system_prompt=system_prompt
                )
                if combined_preds not in already_in_merged:
                    merged_pred_schema[
                        ( len(combined_preds) , len(sim_preds) )
                    ]+=1
                    merged_pred_groups[len(combined_preds)].append(
                        (combined_preds , retrieved_pages)
                    )
                    already_in_merged.append(
                        combined_preds
                    ) 
            else:
                singleton_pred_groups.append(
                    (sim_preds , retrieved_pages)
                )
            
            if display_pbar:
                pbar.update(1)
                pbar.set_description(
                    f"Clusters Processed: {working_idx+1}/{len(cluster_map.keys())}"
                )
        
        if visualize_results:
            print(f"Combined Pred Schema Before & After")
            for (combined_pred_cnt,sim_pred_cnt) , count in merged_pred_schema.items():
                print(
                    f"  Before: {sim_pred_cnt} Lenth of Combined Pred: {combined_pred_cnt} Count: {count}"
                )
            print(f"\nCombined Pred Schema")
            for len_combined_pred, sim_preds in merged_pred_groups.items():
                print(
                    f"  Lenth of Combined Pred: {len_combined_pred} Count: {len(sim_preds)}"
                )
            print(f"\nSingleton Pred Groups Count: {len(singleton_pred_groups)}")
        
        
        merged_norm_preds=[]
        for merged_pred_count, pred_groups in merged_pred_groups.items():
            for preds,page_ids in pred_groups:
                merged_norm_preds.extend(preds)
        for pred, page_ids in singleton_pred_groups:
            merged_norm_preds.extend(pred)
        
        
        return MergeOutput(
            singleton_pred_groups=singleton_pred_groups,
            merged_pred_groups=merged_pred_groups,
            merged_norm_preds=merged_norm_preds,
        )



#