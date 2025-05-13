

import numpy as np


from typing import Union
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering


from ..configs import (
    GroupPredsConfig,
    EmbedConfig,
)
from ..outputs import (
    GrouperOutput
)


class Grouper:
    def __init__(
        self,
        group_cfg:GroupPredsConfig,
        emb_model:SentenceTransformer,
    ):
        """
        Groups the input predictions based on semantic similarity.

        This method first encodes the predictions into embeddings, then computes a
        pairwise distance matrix (cosine distance). Agglomerative clustering is
        then applied to this distance matrix to assign cluster labels to predictions.

        The results (cluster_map and dist_matrix) are stored as instance attributes.

        Args:
            preds: A list of string predictions to be grouped.
            output_cluster_results: If True, returns a GrouperOutput object containing
                                    detailed clustering results (labels, cluster_map, dist_matrix).
                                    If False (default), returns only the cluster_map.

        Returns:
            If `output_cluster_results` is True, a GrouperOutput object.
            Otherwise, a defaultdict mapping cluster labels (int) to lists of
            predictions (str) belonging to that cluster.
        """

        self.emb_model=emb_model
        self.threshold=group_cfg.threshold
        self.n_clusters=group_cfg.n_clusters
        self.linkage=group_cfg.cluster.linkage
        self.affinity=group_cfg.cluster.affinity
    
    
    def group_preds(
        self,
        preds:list[str],
        output_cluster_results:bool=False,
    )->Union[defaultdict, GrouperOutput]:

        """
        Visualizes and optionally returns the schema of the generated clusters.

        This method prints a breakdown of clusters by the number of predictions
        (nodes) they contain. It also counts singleton and non-singleton clusters.
        It relies on `self.cluster_map` being populated by `group_preds`.

        Args:
            print_clusters: If True, prints the individual predictions within each
                            non-singleton cluster.
            return_cluster_schema: If True, returns the cluster schema (a defaultdict
                                   mapping node count to cluster count).

        Returns:
            If `return_cluster_schema` is True, a defaultdict mapping the number
            of nodes in a cluster (int) to the count of such clusters (int).
            Otherwise, returns None.
        
        """
        
        pred_embeds = self.emb_model.encode(
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
        """
        Visualizes and optionally returns the schema of the generated clusters.

        This method prints a breakdown of clusters by the number of predictions
        (nodes) they contain. It also counts singleton and non-singleton clusters.
        It relies on `self.cluster_map` being populated by `group_preds`.

        Args:
            print_clusters: If True, prints the individual predictions within each
                            non-singleton cluster.
            return_cluster_schema: If True, returns the cluster schema (a defaultdict
                                   mapping node count to cluster count).

        Returns:
            If `return_cluster_schema` is True, a defaultdict mapping the number
            of nodes in a cluster (int) to the count of such clusters (int).
            Otherwise, returns None.
        """

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
        """
        Calculates and visualizes the average intra-cluster distances.

        For each non-singleton cluster, this method computes the average pairwise
        distance between its member predictions. It then prints these average
        distances grouped by cluster size (number of nodes).
        It relies on `self.cluster_map` and `self.dist_matrix` being populated
        by `group_preds`.

        Args:
            preds: The original list of predictions. This is used to map predictions
                   back to their original indices to extract distances from `self.dist_matrix`.
            return_cluster_schema: If True, returns a defaultdict mapping cluster size
                                   to a list of distance arrays (for each cluster of that size).

        Returns:
            If `return_cluster_schema` is True, a defaultdict where keys are
            cluster sizes (int) and values are lists of numpy arrays, each array
            containing the pairwise distances within a cluster of that size.
            Otherwise, returns None.

        Raises:
            AttributeError: If `self.cluster_map` or `self.dist_matrix` have not been
                            populated by calling `group_preds` first.
            ValueError: If a prediction in `self.cluster_map` is not found in the input `preds` list.
        """

        
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






#