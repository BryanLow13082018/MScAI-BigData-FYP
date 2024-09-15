from .base_trainer import BaseTrainer
import torch
import logging

class DecoderTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, config):
        """
        Initialize the DecoderTrainer.
        
        Args:
            model: The decoder model (e.g., LLaMA) to train.
            tokenizer: The tokenizer for the model.
            config (dict): Configuration parameters for training.
        """
        super().__init__(model, tokenizer, config)
        logging.info(f"Initialized DecoderTrainer with config: {config}")

    def train_step(self, batch):
        """
        Perform a single training step for the decoder model.
        
        This method processes a batch of data, computes the loss, and returns it.
        
        Args:
            batch: A batch of training data.
        
        Returns:
            torch.Tensor: The loss for this training step.
        """
        logging.debug(f"Train step - Batch keys: {batch.keys()}")
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device)
        
        inputs, labels = self._ensure_tensor_shapes(inputs, labels)
        
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        logging.debug(f"Train step - Loss: {loss.item()}")
        return loss

    def evaluation_step(self, batch):
        """
        Perform a single evaluation step for the decoder model.
        
        This method processes a batch of data, computes the loss without gradient computation, and returns it.
        
        Args:
            batch: A batch of evaluation data.
        
        Returns:
            torch.Tensor: The loss for this evaluation step.
        """
        logging.debug(f"Evaluation step - Batch keys: {batch.keys()}")
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device)
        
        inputs, labels = self._ensure_tensor_shapes(inputs, labels)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        logging.debug(f"Evaluation step - Loss: {loss.item()}")
        return loss

    def generate(self, texts, max_new_tokens=50, num_beams=1, **kwargs):
        """
        Generate text using the decoder model.
        
        This method takes input texts, processes them through the model, and generates new text.
        
        Args:
            texts: A list of input texts for text generation.
            max_new_tokens (int, optional): The maximum number of new tokens to generate.
            num_beams (int, optional): Number of beams for beam search. 1 means no beam search.
            **kwargs: Additional keyword arguments for the generate method.
        
        Returns:
            list: Generated text sequences.
        """
        logging.info(f"Generating text for {len(texts)} inputs")
        self.model.eval()
        with torch.no_grad():
            try:
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                logging.debug(f"Tokenized inputs shape: {inputs['input_ids'].shape}")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "num_beams": num_beams,
                    **kwargs
                }
                generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}