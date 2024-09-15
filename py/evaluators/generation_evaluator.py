import torch
import numpy as np
from .base_evaluator import BaseEvaluator

class GenerationEvaluator(BaseEvaluator):
    """
    A class for evaluating and generating text using language models.
    """

    def evaluate(self, prompt_texts, generated_texts):
        """
        Evaluate the quality of generated texts using perplexity.

        Args:
            prompt_texts (list): List of input prompts.
            generated_texts (list): List of generated texts corresponding to the prompts.

        Returns:
            dict: A dictionary containing perplexities for each text and the average perplexity.
        """
        perplexities = []

        for prompt, generated in zip(prompt_texts, generated_texts):
            inputs = self.tokenizer(prompt + generated, return_tensors="pt").to(self.model.device)
               
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)

        average_perplexity = np.mean(perplexities)

        return {
            'perplexities': perplexities,
            'average_perplexity': average_perplexity
        }

    def generate(self, prompt_texts):
        """
        Generate text based on input prompts.

        Args:
            prompt_texts (list): List of input prompts to generate text from.

        Returns:
            list: List of generated texts corresponding to the input prompts.
        """
        generated_texts = []

        for prompt in prompt_texts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output_sequences = self.model.generate(
                    **inputs,
                    max_length=100,  # Adjust as needed
                    num_return_sequences=1,
                    # Add any other generation parameters here
                )
            
            generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            generated_texts.append(generated_text)

        return generated_texts

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for the Trainer class.

        This method is called by the Trainer during evaluation. It generates text
        based on the input sequences and then evaluates the generated text using perplexity.

        Args:
            eval_pred (tuple): A tuple containing input_ids and labels.

        Returns:
            dict: A dictionary containing the average perplexity.
        """
        input_ids, _ = eval_pred
        
        # Decode input sequences
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Generate text based on input sequences
        generated_texts = self.generate(input_texts)
        
        # Evaluate the generated texts
        results = self.evaluate(input_texts, generated_texts)
        
        return {
            "perplexity": results["average_perplexity"]
        }