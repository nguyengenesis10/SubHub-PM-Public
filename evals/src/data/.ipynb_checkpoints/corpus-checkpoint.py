

from ..processing import PhraseNormalizer


class Corpus:
    def __init__(
        self,
        preds: list[str],
        pred_pages: list[str],
        labels: list[str],
        label_pages: list[str],
        normalizer: PhraseNormalizer,
    ):
        '''
        Purpose:
            Represents a collection of the raw human generated labels and model predictions along with their corresponding page ids. 
            Handles phrase normalization to increase semantic coverage and document accuracy by maintaining core semantic and literal meaning. 
        
        Args: 
            normalizer: An instance of the phrase normalizer object which handles phrase normalization, lazily.
        '''
        
        if not isinstance(preds,list):
            raise TypeError(
                f"'preds' is supposed to be of type list!"
            )
        if not isinstance(labels,list):
            raise TypeError(
                f"'labels' is supposed to be of type list!"
            )
        if len(pred_pages) != len(preds):
            raise ValueError(
                f"'pred_pages' has length: {len(pred_pages)} and 'preds' has length: {len(preds)} are suppose to be the same length!"
            )
        if len(label_pages) != len(labels):
            raise ValueError(
                f"'labels' has length: {len(labels)} and 'label_pages' has length: {len(label_pages)} are suppose to be the same length!"
            )
        
        self.preds  = preds
        self.pred_pages = pred_pages
        self.labels = labels
        self.label_pages = label_pages
        self._norm  = normalizer
        self._norm_preds  = None
        self._norm_labels = None


    # Lazy, cached normalisation
    @property
    def norm_preds(self) -> list[str]:
        if self._norm_preds is None:
            self._norm_preds = self._norm.normalize(self.preds)
        return self._norm_preds


    @property
    def norm_labels(self) -> list[str]:
        if self._norm_labels is None:
            self._norm_labels = self._norm.normalize(self.labels)
        return self._norm_labels


#