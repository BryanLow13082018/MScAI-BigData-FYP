from .base_evaluator import BaseEvaluator
from comet import download_model, load_from_checkpoint
import numpy as np
import logging

class AfriCOMETEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer):
        """
        Initialize the AfriCOMETEvaluator.
        
        Args:
            model: The model to evaluate (not used directly in this evaluator).
            tokenizer: The tokenizer for the model.
        """
        super().__init__(model, tokenizer)
        self.africomet_model = load_from_checkpoint(download_model("masakhane/africomet-mtl"))

    def prepare_africomet_data(self, source_texts, translated_texts, reference_texts):
        """
        Prepare data for AfriCOMET evaluation.

        Args:
            source_texts (list): List of source language texts.
            translated_texts (list): List of translated texts.
            reference_texts (list): List of reference translations.

        Returns:
            list: A list of dictionaries in the format required by AfriCOMET.
        """
        return [
            {"src": src, "mt": trans, "ref": ref}
            for src, trans, ref in zip(source_texts, translated_texts, reference_texts)
        ]

    def evaluate(self, source_texts, translated_texts, reference_texts):
        """
        Evaluate translations using AfriCOMET.
        
        Args:
            source_texts (list): List of source language texts.
            translated_texts (list): List of translated texts.
            reference_texts (list): List of reference translations.
        
        Returns:
            dict: A dictionary containing AfriCOMET scores and the average score.
        """
        data = self.prepare_africomet_data(source_texts, translated_texts, reference_texts)
        
        predictions = self.africomet_model.predict(data, batch_size=8, gpus=1)
        
        scores = []
        for pred in predictions:
            try:
                if isinstance(pred, dict) and 'score' in pred:
                    scores.append(float(pred['score']))
                elif isinstance(pred, (int, float)):
                    scores.append(float(pred))
                else:
                    logging.warning(f"Unexpected prediction format: {pred}. Skipping.")
            except ValueError:
                logging.warning(f"Could not convert prediction to float: {pred}. Skipping.")
        
        return {
            'africomet_scores': scores,
            'average_score': np.mean(scores) if scores else 0.0
        }

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for the Trainer class.

        This method is called by the Trainer during evaluation. It decodes the
        model predictions and labels, and then evaluates them using AfriCOMET.

        Args:
            eval_pred (tuple): A tuple containing predictions and labels.

        Returns:
            dict: A dictionary containing the average AfriCOMET score.
        """
        predictions, labels = eval_pred
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Safely extract source texts
        source_texts = []
        for pred in decoded_preds:
            tokens = pred.split()
            source_texts.append(tokens[0] if tokens else "")
        
        # Evaluate using AfriCOMET
        results = self.evaluate(source_texts, decoded_preds, decoded_labels)
        
        return {
            "africomet_score": results["average_score"]
        }