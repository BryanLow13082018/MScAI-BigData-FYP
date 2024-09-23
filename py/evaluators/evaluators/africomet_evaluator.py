from comet import download_model, load_from_checkpoint
from .base_evaluator import BaseEvaluator
import torch
import numpy as np

class AfriCOMETEvaluator(BaseEvaluator):
    def __init__(self, model, tokenizer):
        """
        Initialize the AfriCOMETEvaluator.
        
        Args:
            model: The model to evaluate (not used directly in this evaluator).
            tokenizer: The tokenizer for the model (not used directly in this evaluator).
        """
        super().__init__(model, tokenizer)
        model_path = download_model("masakhane/africomet-mtl")
        self.africomet_model = load_from_checkpoint(model_path)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.africomet_model.to(self.device)

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
        try:
            self.prepare_model()  # Ensure model is on GPU if available
            
            print(f"Number of texts to evaluate: {len(source_texts)}")
            
            data = [
                {"src": src, "mt": mt, "ref": ref}
                for src, mt, ref in zip(source_texts, translated_texts, reference_texts)
            ]
            
            print("Starting AfriCOMET prediction")
            model_output = self.africomet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
            print("AfriCOMET prediction completed")
            
            print(f"Type of model_output: {type(model_output)}")
            
            # Handle comet.models.utils.Prediction object
            if hasattr(model_output, 'scores'):
                scores = model_output.scores
            elif hasattr(model_output, 'system_score'):
                scores = [model_output.system_score]
            else:
                print(f"Unexpected model_output structure. Available attributes: {dir(model_output)}")
                return {'africomet_scores': [], 'average_score': 0.0}
            
            # Ensure all scores are numeric and within the expected range [0, 1]
            valid_scores = [score for score in scores if isinstance(score, (int, float)) and 0 <= score <= 1]
            
            if not valid_scores:
                print("No valid scores found in model_output")
                return {'africomet_scores': [], 'average_score': 0.0}
            
            average_score = np.mean(valid_scores)
            min_score = np.min(valid_scores)
            max_score = np.max(valid_scores)
            median_score = np.median(valid_scores)
            
            print(f"Number of valid scores: {len(valid_scores)}")
            print(f"Score statistics: Min: {min_score:.4f}, Max: {max_score:.4f}, Median: {median_score:.4f}, Average: {average_score:.4f}")
        
            return {
                'africomet_scores': valid_scores,
                'average_score': average_score,
                'min_score': min_score,
                'max_score': max_score,
                'median_score': median_score
            }
            
        except Exception as e:
            print(f"Error in evaluate method: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {'africomet_scores': [], 'average_score': 0.0}
        finally:
            self.clear_cuda_memory()  # Clear memory after evaluation

    def prepare_model(self):
        """Move the model to GPU if available."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.africomet_model.to(self.device)
            print("AfriCOMET model moved to GPU.")
        else:
            print("CUDA is not available. Using CPU.")

    def clear_cuda_memory(self):
        """Clear CUDA memory and move model to CPU if it was on GPU."""
        if hasattr(self, 'device') and self.device.type == "cuda":
            torch.cuda.empty_cache()
            self.africomet_model.to('cpu')
            torch.cuda.empty_cache()
            print("CUDA memory cleared and AfriCOMET model moved to CPU.")

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
        try:
            predictions, labels = eval_pred
            # Decode predictions and labels
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Assuming the source texts are available. If not, you'll need to modify this.
            source_texts = ["" for _ in range(len(decoded_preds))]  # placeholder
            
            # Evaluate using AfriCOMET
            results = self.evaluate(source_texts, decoded_preds, decoded_labels)
            
            return {
                "africomet_score": results["average_score"]
            }
        finally:
            self.clear_cuda_memory()  # Ensure memory is cleared after computation