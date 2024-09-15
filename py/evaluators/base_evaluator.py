from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import CustomDataset

class BaseEvaluator:
    """
    Base class for all evaluators.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize the BaseEvaluator.

        Args:
            model: The model to be evaluated.
            tokenizer: The tokenizer associated with the model.
        """
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, *args, **kwargs):
        """
        Base evaluate method. Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def generate(self, *args, **kwargs):
        """
        Base generate method. Should be implemented by subclasses if needed.
        """
        raise NotImplementedError("Subclasses should implement this method if generation is supported.")
    
    def prepare_dataset(self, data, max_length=128):
        """
        Prepare a CustomDataset from the input data.

        Args:
            data: The input data (can be a DataFrame, list of texts, or list of tuples).
            max_length (int): Maximum sequence length for tokenization.

        Returns:
            CustomDataset: A dataset ready for evaluation.
        """
        return CustomDataset(data, self.tokenizer, max_length)

    def compute_metrics(self, eval_pred):
        """
        Base compute_metrics method. Should be implemented by subclasses.

        Args:
            eval_pred: The evaluation predictions and labels.

        Returns:
            dict: A dictionary of computed metrics.
        """
        raise NotImplementedError("Subclasses should implement this method.")