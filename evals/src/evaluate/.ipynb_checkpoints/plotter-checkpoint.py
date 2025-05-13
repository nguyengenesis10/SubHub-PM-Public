

import matplotlib.pyplot as plt


from ..outputs import PageMetrics


class DocPlotter():
    def __init__(
        self,
        match_page_metrics:list[PageMetrics],
    ):
        self.match_page_metrics=sorted(match_page_metrics, key=lambda x: x.page_num)
        
        self.page_nums = [pm.page_num for pm in self.match_page_metrics]
        self.matches = [pm.matches for pm in self.match_page_metrics]
        self.misses  = [pm.misses for pm in self.match_page_metrics]  
        self.extras  = [pm.extra for pm in self.match_page_metrics]    


    def plot_match_and_misses(
        self,
    ):

        fig, ax = plt.subplots(
            figsize=(20, 10)
        )

        # Bar plots
        ax.bar(
            self.page_nums, 
            self.matches, 
            color='seagreen', 
            label='Matches'
        )
        
        # Apply negative transformation
        ax.bar(
            self.page_nums, 
            [-i for i in self.misses], 
            color='Orange', 
            label='Misses'
        )

        # Axis and Labels
        ax.axhline(0, color='black', linewidth=1.2)
        ax.set_xlabel("Page Number")
        ax.set_xticks(
            range(min(self.page_nums)-1,max(self.page_nums)+1)
        )
        ax.set_ylabel("Count")
        ax.set_title("Page-Level Match vs. Miss (Vertical Split)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        return fig


    def plot_all(
        self,
    ):
        # Plot
        fig, ax = plt.subplots(
            figsize=(20,10)
        )

        # Positive: matches
        ax.bar(self.page_nums, self.matches, label="Matches", color="seagreen")

        # Negative: misses + extras (stacked)
        ax.bar(
            self.page_nums, 
            [-i for i in self.misses], 
            label="Misses", 
            color="tomato"
        )
        ax.bar(
            self.page_nums, 
            [-i for i in self.extras], 
            bottom=[-i for i in self.misses], 
            label="Extras", 
            color="Orange"
        )

        # Aesthetics
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel("Page Number")
        ax.set_xticks(
            range(min(self.page_nums)-1,max(self.page_nums)+1)
        )
        ax.set_ylabel("Count")
        ax.set_title("Page-Level Alignment: Matches vs. Misses and Extras")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

        return fig


