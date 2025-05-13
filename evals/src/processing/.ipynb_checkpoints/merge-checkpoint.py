

import ast


from tqdm import tqdm
from collections import defaultdict


from ..outputs import (
    MergeOutput,
    MergedScopes
)
from ..configs import (
    MergePredsConfig,
)


class Merger:
    """
    Merges groups of similar textual predictions (scopes of work) using a language model.

    This class takes a map of clustered predictions, iterates through clusters
    containing multiple predictions, and uses an external API (e.g., OpenAI)
    to combine these predictions into a more concise representation.
    Singleton clusters (containing only one prediction) are passed through without merging.

    Attributes:
        client (Any): The API client (e.g., OpenAI client) used for merging.
        model_name (str): The name of the language model to use.
        temperature (float): The sampling temperature for the language model.
        display_pbar (bool): Flag to control the display of a progress bar.
        system_prompt (str): The system prompt used to instruct the language model.
    """
    
    def __init__(
        self,
        merge_cfg:MergePredsConfig,
    ):
        """
        Initializes the Merger with configuration for the merging process.

        Args:
            merge_cfg: An instance of MergePredsConfig containing the API client,
                       model name, temperature, progress bar display flag, and
                       system prompt.
        """
        self.client=merge_cfg.client
        self.model_name=merge_cfg.model_name
        self.temperature=merge_cfg.temperature
        self.display_pbar=merge_cfg.display_pbar
        self.system_prompt=merge_cfg.system_prompt


    def _merge(
        self,
        scopes:list[str],
    )->list[str]:
        """
        Internal method to merge a list of scope strings using the configured language model.

        It sends the scopes to the model with a specific instruction to combine them
        concisely and parses the structured response.

        Args:
            scopes: A list of scope strings to be merged.

        Returns:
            A list of merged scope strings as returned by the language model.
        """
        messages=[
            { "role": "system" , "content": self.system_prompt},
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
        visualize_results:bool=True,
        return_stats:bool=False,
    )->MergeOutput:
        """
        Merges predictions within each cluster provided in the cluster_map.

        Iterates through each cluster of predictions. If a cluster contains more
        than one prediction, it calls the `_merge` method to combine them.
        Singleton clusters are kept as is. It collects all merged and singleton
        predictions along with their associated page information.

        Args:
            cluster_map: A defaultdict where keys are cluster IDs and values are
                         lists of (normalized) prediction strings belonging to that cluster.
            pred_pages: A list of page identifiers (or lists of page identifiers)
                        corresponding to each prediction in `norm_preds`. The order
                        must match `norm_preds`.
            norm_preds: The list of all normalized predictions, used to find the
                        original index and thus the page information for predictions
                        in the `cluster_map`.
            visualize_results: If True, prints statistics about the merging process,
                               such as the schema of merged predictions and counts.

        Returns:
            A MergeOutput object containing:
            - `singleton_pred_groups`: Predictions that were not merged.
            - `merged_pred_groups`: Predictions that were merged by the LLM.
            - `merged_norm_preds`: A flat list of all predictions after merging.
        """
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
                    pred_page=ast.literal_eval(retr_page)
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
                print(f"  Before: {sim_pred_cnt} Lenth of Combined Pred: {combined_pred_cnt} Count: {count}")
            
            print(f"\nCombined Pred Schema")
            for len_combined_pred, sim_preds in merged_pred_groups.items():
                print(f"  Lenth of Combined Pred: {len_combined_pred} Count: {len(sim_preds)}")
            
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