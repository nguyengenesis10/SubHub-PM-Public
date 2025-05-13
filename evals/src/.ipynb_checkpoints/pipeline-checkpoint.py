

from datasets import Dataset
from collections import defaultdict


from .configs import (
    EmbedConfig,
    GroupPredsConfig,
    MergePredsConfig,
    NormalizeConfig,
    ComparePredsAndLabelsConfig,
    GrayAreaAnalyzerConfig,
    ExperimentConfig,
)
from .data import (
    Corpus
)
from .outputs import (
    GrouperOutput,
    MergeOutput,
    SCSMetrics,
    PageEvaluationResults,
)
from .processing import (
    PhraseNormalizer, 
    Grouper,
    Merger,
)
from .evaluate import (
    ScopeAlignmentEvaluator,
    PageAccuracyEvaluator,
    GrayAreaAnalyzer,
)
from .utils import build_embed_model


class PipelineOrchestrator:
    """
    Orchestrates the entire evaluation pipeline for comparing model predictions
    against ground truth labels.

    This class initializes and manages various components involved in the
    evaluation process, including phrase normalization, prediction grouping,
    merging of similar predictions, scope alignment, gray area analysis,
    and document-level accuracy assessment.

    It takes an ExperimentConfig object which encapsulates all the necessary
    sub-configurations for each step of the pipeline.

    Attributes:
        experiment_config (ExperimentConfig): The overall configuration for the pipeline.
        emb_model (MockSentenceTransformer): The sentence embedding model.
        normalizer (PhraseNormalizer): Component for normalizing text phrases.
        grouper (Grouper): Component for grouping similar predictions.
        merger (Merger): Component for merging grouped predictions.
        alignment_evaluator (ScopeAlignmentEvaluator): Component for evaluating scope alignment.
        gray_area_analyzer (GrayAreaAnalyzer): Component for analyzing ambiguous predictions.
        corpus (Optional[Corpus]): Holds the input predictions and labels. Set by `setup_corpus`.
        doc_evaluator (Optional[PageAccuracyEvaluator]): Component for document-level evaluation. Initialized in `doc_evals`.
    """
    def __init__(
        self,
        device:str,
        env:dict,
        **kwargs,
    ):
        """
        Initializes the PipelineOrchestrator and its components.

        Args:
            device: The device to run models on (e.g., "cpu", "cuda").
            env: An environment dictionary, potentially for caching models or other resources.
            **kwargs: Arbitrary keyword arguments that are unpacked to initialize
                      the ExperimentConfig. This allows for flexible configuration
                      passing (e.g., experiment_config=ExperimentConfig(embed=..., norm=...))
                      or directly passing sub-config kwargs if ExperimentConfig handles it.
        """
        self.experiement_config=ExperimentConfig(**kwargs)
        self.emb_model=build_embed_model(
            env=env, 
            embed_cfg=self.experiement_config.embed,
            device=device
        )
        self.normalizer=PhraseNormalizer(
            norm_cfg=self.experiement_config.norm
        )
        self.grouper=Grouper(
            group_cfg=self.experiement_config.group,
            emb_model=self.emb_model
        )
        self.merger = Merger(
            merge_cfg=self.experiement_config.merge
        )
        self.alignment_evaluator=ScopeAlignmentEvaluator(
            compare_preds_and_labels_cfg=self.experiement_config.compare,
            emb_model=self.emb_model
        )
        self.gray_area_analyzer = GrayAreaAnalyzer(
            compare_preds_and_labels_cfg=self.experiement_config.compare,
            gray_area_analyzer_cfg=self.experiement_config.gray
        )


    def setup_corpus(
        self,
        preds:list[str],
        pred_pages:list[list[str]],
        labels:list[str],
        label_pages:list[list[str]],
    ):
        """
        Sets up the corpus with predictions and labels for the evaluation pipeline.

        Initializes a Corpus object and triggers the normalization of predictions
        and labels, caching them within the corpus.

        Args:
            preds: A list of raw model prediction strings.
            pred_pages: A list where each item corresponds to a prediction and
                        contains its associated page identifier(s).
                        (Type changed to list[Any] to match Corpus mock).
            labels: A list of raw ground truth label strings.
            label_pages: A list where each item corresponds to a label and
                         contains its associated page identifier(s).
                         (Type changed to list[Any] to match Corpus mock).
        """
        self.corpus = Corpus(
            preds=preds,
            pred_pages=pred_pages,
            labels=labels,
            label_pages=label_pages,
            normalizer=self.normalizer,
        )
        self.corpus.norm_preds
        self.corpus.norm_labels


    def _group_preds(
        self,
    )->GrouperOutput:
        """
        Internal method to group normalized predictions using the configured Grouper.

        It visualizes the cluster schema and average cluster distances after grouping.

        Returns:
            A GrouperOutput object containing the cluster labels, the map of
            clusters to predictions, and the distance matrix.
        """
        grouper_output=self.grouper.group_preds(
            preds=self.corpus.norm_preds,
            output_cluster_results=True,
        )
        self.grouper._visualize_cluster_schema(
            print_clusters=False,
            return_cluster_schema=True
        )
        self.grouper._visualize_avg_cluster_dist(
            preds=self.corpus.norm_preds,
            return_cluster_schema=True
        )
        return grouper_output


    def _merge(
        self,
        grouper_output:GrouperOutput,
    )->MergeOutput:
        """
        Internal method to merge grouped predictions using the configured Merger.

        Args:
            grouper_output: The output from the `_group_preds` method, containing
                            the `cluster_map`.

        Returns:
            A MergeOutput object containing singleton groups, merged groups, and
            a flat list of all normalized predictions post-merging.
        """
        merge_output=self.merger.merge(
            cluster_map=grouper_output.cluster_map,
            pred_pages=self.corpus.pred_pages,
            norm_preds=self.corpus.norm_preds,
            visualize_results=True,
        )
        return merge_output


    def _update_scs(
        self,
        preds:list[str],
    )->SCSMetrics:
        """
        Internal method to perform gray area analysis and update Semantic Coverage Score (SCS) metrics.

        It uses the `GrayAreaAnalyzer` to analyze predictions that were not clear matches
        and then updates the SCS metrics based on this analysis.

        Args:
            preds: The list of predictions (typically merged and normalized) to be analyzed.

        Returns:
            An SCSMetrics object representing the "soft" SCS after gray area analysis.
        """
        self.gray_area_analyzer.analyze(
            cluster_objs=self.alignment_evaluator.cluster_objs,
            preds=preds,
            labels=self.corpus.norm_labels,
            print_and_log=False,
        )
        soft_scs_metrics = self.gray_area_analyzer.update_scs(
            labels=self.corpus.norm_labels,
            preds=preds,
        )
        return soft_scs_metrics


    def scs_evals(
        self,
    ):
        """
        Performs Scope Alignment and Semantic Coverage Score (SCS) evaluations.

        This involves grouping predictions, merging them, performing an initial
        "hard" scope alignment, and then a "soft" alignment after gray area analysis.

        Returns:
            A tuple containing:
            - hard_scs_metrics (SCSMetrics): Metrics from the initial alignment.
            - soft_scs_metrics (SCSMetrics): Metrics after gray area analysis.
            - merge_output (MergeOutput): The output from the merging step.
        """
        merge_output=self._merge(
            grouper_output=self._group_preds()
        )
        hard_scs_metrics=self.alignment_evaluator.eval(
            labels=self.corpus.norm_labels,
            preds=merge_output.merged_norm_preds,
            print_labels_and_preds=False,
        )
        soft_scs_metrics=self._update_scs(
            preds=merge_output.merged_norm_preds
        )
        return hard_scs_metrics , soft_scs_metrics, merge_output


    def doc_evals(
        self,
        plans:Dataset,
        scope_breakout:Dataset,
        singleton_pred_groups:defaultdict,
        merged_pred_groups:defaultdict,
    )->PageEvaluationResults:
        """
        Performs document-level page accuracy evaluations.

        Initializes a `PageAccuracyEvaluator` and uses it to evaluate predictions
        against the provided plans and scope breakout data.

        Args:
            plans: A Dataset object representing the plan documents.
            scope_breakout: A Dataset object representing the ground truth scope breakouts.
            singleton_pred_groups: A list of singleton prediction groups from `MergeOutput`.
            merged_pred_groups: A defaultdict of merged prediction groups from `MergeOutput`.

        Returns:
            A PageEvaluationResults object containing detailed page-level accuracy metrics.
        """
        self.doc_evaluator=PageAccuracyEvaluator(
            plans=plans,
            scope_breakout=scope_breakout,
            singleton_pred_groups=singleton_pred_groups,
            merged_pred_groups=merged_pred_groups,
            labels=self.corpus.norm_labels,
            classifications=self.gray_area_analyzer._classifications,
            compare_preds_and_labels_cfg=self.experiement_config.compare,
        )
        master_page_results = self.doc_evaluator.eval(
            cluster_objs=self.alignment_evaluator.cluster_objs,
        )
        return master_page_results


    def eval(
        self,
        plans:Dataset,
        scope_breakout:Dataset,
    ):
        """
        Executes the full evaluation pipeline.

        This method orchestrates the Semantic Coverage Score (SCS) evaluations
        and the document-level page accuracy evaluations. It requires the corpus
        to be set up first via `setup_corpus`.

        Args:
            plans: A Dataset object for plan documents (for `doc_evals`).
            scope_breakout: A Dataset object for scope breakout data (for `doc_evals`).

        Returns:
            A tuple containing:
            - hard_scs_metrics (SCSMetrics): SCS metrics before gray area analysis.
            - soft_scs_metrics (SCSMetrics): SCS metrics after gray area analysis.
            - doc_results (PageEvaluationResults): Document-level page accuracy results.
            - merge_output (MergeOutput): Output from the prediction merging step.
        """
        hard_scs_metrics , soft_scs_metrics, merge_output = self.scs_evals()
        
        singleton_pred_groups = merge_output.singleton_pred_groups
        merged_pred_groups = merge_output.merged_pred_groups
        
        doc_results = self.doc_evals(
            plans=plans,
            scope_breakout=scope_breakout,
            singleton_pred_groups=singleton_pred_groups,
            merged_pred_groups=merged_pred_groups,
        )

        return hard_scs_metrics , soft_scs_metrics, doc_results, merge_output


if __name__ == "__main__":
    import time
    from openai import OpenAI
    
    ground_truth_model="gpt-4o-2024-08-06"
    client = OpenAI(api_key="[REMOVED_SECRET]")

    kwargs = {
        "display_pbar":True,
        "print_log":False,
        "n_clusters":None,
        "client":client,
    
        "norm.temperature":0.25,

        "group.threshold":0.1,
        "group.cluster.linkage":"average",
        "group.cluster.affinity":"precomputed",

        "merge.model_name": ground_truth_model,
        "merge.temperature": 0.25,

        "compare.threshold":0.16,
        "compare.scs_alpha":0.7,
        "compare.cluster.linkage":"average",
        "compare.cluster.affinity":None,
    
        "gray.model_name":ground_truth_model,
        "gray.temperature" :0.25,
        "gray.hybrid_dist_alpha":0.4,
    }

    pipeline_orchestrator=PipelineOrchestrator(
        device="cuda:0",
        env=globals(),
        **kwargs
    )
    time.sleep(60)







#