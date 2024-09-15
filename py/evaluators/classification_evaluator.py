from .base_evaluator import BaseEvaluator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import logging

class ClassificationEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer):
        """
        Initialize the ClassificationEvaluator.
        
        Args:
            model: The model to evaluate.
            tokenizer: The tokenizer for the model.
        """
        super().__init__(model, tokenizer)

    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics for classification tasks.
        
        Args:
            eval_pred (tuple): A tuple containing predictions and labels.
        
        Returns:
            dict: A dictionary containing computed metrics.
        """
        try:
            predictions, labels = eval_pred
            
            # Ensure predictions and labels are numpy arrays
            predictions = np.array(predictions)
            labels = np.array(labels)
            
            # If predictions are logits, convert to class predictions
            if predictions.ndim > 1 and predictions.shape[-1] > 1:
                predictions = np.argmax(predictions, axis=-1)
            
            # Compute precision, recall, and F1 score with zero_division=1
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=1)
            # Compute accuracy
            acc = accuracy_score(labels, predictions)
            
            return {
                'accuracy': float(acc),
                'f1': float(f1),
                'precision': float(precision),
                'recall': float(recall)
            }
        except Exception as e:
            logging.error(f"Error in compute_metrics: {str(e)}")
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
    
    def evaluate(self, test_dataset):
        """
        Evaluate the model on classification tasks.
        
        Args:
            test_dataset: Dataset for testing (DataFrame or list).
        
        Returns:
            dict: A dictionary containing evaluation metrics and plots.
        """
        custom_dataset = self.prepare_dataset(test_dataset)
        predictions, true_labels = self.get_predictions(custom_dataset)
        
        # Use zero_division=1 in classification_report
        report = classification_report(true_labels, predictions, output_dict=True, zero_division=1)
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'confusion_matrix_plot': plt.gcf()
        }

    def get_predictions(self, dataset):
        """
        Get model predictions for a dataset.
        
        Args:
            dataset: The dataset to get predictions for.
        
        Returns:
            tuple: Predicted labels and true labels.
        """
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(dataset, batch_size=8):
                inputs = {k: v.to(self.model.device) for k, v in batch.items() if k != 'labels'}
                outputs = self.model(**inputs)
                predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
        
        return np.array(predictions), np.array(true_labels)