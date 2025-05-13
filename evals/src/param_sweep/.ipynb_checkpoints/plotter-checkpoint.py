

import matplotlib.pyplot as plt


from typing import Iterable, Union


from ..outputs import SCSMetrics,MergeOutput


class SCSPlotter:
    def __init__(
        self,
        metrics:list[SCSMetrics],
        Ts:Iterable[float],
    ):
        if not isinstance(metrics,list):
            raise TypeError(
                f"metrics needs to be a list!"
            )
        if not isinstance(metrics[0],SCSMetrics):
            raise TypeError(
                f"metric elements to be of type SCSMetrics."
            )
        
        self.metrics = metrics
        self.scs = [i.scs for i in self.metrics]
        self.ps = [i.precision for i in self.metrics]
        self.rs = [i.recall for i in self.metrics]
        
        self.Ts = Ts

    def plot_scs_and_ts(
        self,
        figsize:tuple[int]=(10, 6),
        x_label:str="Threshold",
        y_label:str="SCS Score",
        title:str="SCS Score vs. Threshold",
    ):

        fig, ax = plt.subplots(
            figsize=figsize
        )

        # Scatter and line plot
        ax.scatter(
            self.Ts, 
            self.scs, 
            color='blue', 
            label='SCS Score (Scatter)'
        )
        ax.plot(
            self.Ts, 
            self.scs,
            color='orange', 
            linestyle='--', 
            label='SCS Score (Line)'
        )

        # Labels and formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        return fig, ax


class KMeansPlotter:
    def __init__(
        self,
        Xs:Iterable[Union[float,int]],
        metrics:Iterable[Union[float ,int]],
        merge_outputs:list[MergeOutput]=None,
    ):

        if len(metrics) != len(Xs):
            raise ValueError(
                f"Unable to plot metrics has length: {len(metrics)} and Xs has length {len(Xs)}. Suppose to be of equal length."
            )
        
        if metrics:
            self.metrics=metrics
        
        self.Xs=Xs


    def plot(
        self,
        figsize:tuple[int]=(10,6),
        x_label:str="k-clusters",
        y_label:str="Num Greater Than 1 Merged Pred Groups",
        title:str="Num Greater Than 1 Merged Pred Groups vs. k-clusters",
    ):
        fig, ax = plt.subplots(
            figsize=figsize
        )

        # Scatter and line plot
        ax.scatter(
            self.Xs, 
            self.metrics, 
            color='blue', 
            label='Num Greater Than 1 Merged Pred Groups (Scatter)'
        )
        ax.plot(
            self.Xs, 
            self.metrics,
            color='orange', 
            linestyle='--', 
            label='Num Greater Than 1 Merged Pred Groups (Line)'
        )
        # Labels and formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        return fig, ax


class PageEvalPlotter:
    def __init__(
        self,
        metrics:list[dict],
        Ts:Iterable[float],
    ):
      
        JACCARD="avg_jaccard"
        RECALL="avg_recall"
        PRECISION="avg_precision"
        DOC_SCORE="doc_score"
        
        for metric in metrics:
            for k in [JACCARD, RECALL, PRECISION, DOC_SCORE]:
                if k not in metric.keys():
                    raise KeyError(
                        f"required key '{k}' is missing!"
                    )
        self.jaccard=[
            i[JACCARD] for i in metrics
        ]
        self.recall=[
            i[RECALL] for i in metrics
        ]
        self.precision=[
            i[PRECISION] for i in metrics
        ]
        self.doc_score=[
            i[DOC_SCORE] for i in metrics
        ]
        self.Ts=Ts

    def plot_doc_scores(
        self,
        figsize=(10,6),
        x_label:str="Threshold",
        y_label:str="Doc Score",
        title:str="Distance Threshold vs. Doc Score",
        
    ):
        fig, ax = plt.subplots(
            figsize=figsize
        )

        # Scatter and line plot
        ax.scatter(
            self.Ts, 
            self.doc_score, 
            color='blue', 
            label='Doc Score (Scatter)'
        )
        ax.plot(
            self.Ts, 
            self.doc_score,
            color='orange', 
            linestyle='--', 
            label='Doc Score (Line)'
        )
        # Labels and formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        return fig, ax


    def plot_all(
        self,
        figsize=(12, 8),
        title:str="Evaluation Metrics vs. Threshold"
    ):
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self.Ts, self.jaccard, label='Jaccard', marker='o')
        ax.plot(self.Ts, self.recall, label='Recall', marker='s')
        ax.plot(self.Ts, self.precision, label='Precision', marker='^')
        ax.plot(self.Ts, self.doc_score, label='Doc Score', marker='x')

        ax.set_title(title)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        return fig, ax








#