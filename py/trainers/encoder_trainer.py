from .base_trainer import BaseTrainer
import torch
import logging

class EncoderTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, config):
        """
        Initialize the EncoderTrainer.
        
        Args:
            model: The encoder model to train.
            tokenizer: The tokenizer for the model.
            config (dict): Configuration parameters for training.
        """
        super().__init__(model, tokenizer, config)
        logging.info(f"Initialized EncoderTrainer with config: {config}")

    def train_step(self, batch):
        """
        Perform a single training step for the encoder model.
        
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
        Perform a single evaluation step for the encoder model.
        
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

    def inference(self, dataset):
        """
        Perform inference using the encoder model on the entire dataset.
        
        This method runs the model in evaluation mode and returns predictions for all samples in the dataset.
        
        Args:
            dataset: The dataset to perform inference on.
        
        Returns:
            list: The model's predictions for the entire dataset.
        """
        self.model.eval()
        all_predictions = []
        dataloader = self._get_dataloader(dataset, is_train=False)
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                inputs, _ = self._ensure_tensor_shapes(inputs)
                outputs = self.model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
        
        return all_predictions

    def train_combined(self, encoder, decoder, train_dataset, eval_dataset):
        """
        Train the encoder as part of a combined encoder-decoder model.
        
        This method sets the encoder as the current model and runs the training process.
        
        Args:
            encoder: The encoder model.
            decoder: The decoder model (not used in this method).
            train_dataset: Dataset for training.
            eval_dataset: Dataset for evaluation.
        """
        self.model = encoder
        self.train(train_dataset, eval_dataset)

    def evaluate_combined(self, encoder, decoder, eval_dataset):
        """
        Evaluate the encoder as part of a combined encoder-decoder model.
        
        This method sets the encoder as the current model and runs the evaluation process.
        
        Args:
            encoder: The encoder model.
            decoder: The decoder model (not used in this method).
            eval_dataset: Dataset for evaluation.
        
        Returns:
            float: The average evaluation loss.
        """
        self.model = encoder
        return self.evaluate(self._get_dataloader(eval_dataset, is_train=False))

    def _ensure_tensor_shapes(self, inputs, labels=None):
        """
        Ensure that input tensors have the correct shape.
        
        This method adds batch dimensions to inputs and labels if necessary, and ensures
        that the batch sizes match.
        
        Args:
            inputs (dict): The input tensors.
            labels (torch.Tensor, optional): The label tensor.
        
        Returns:
            tuple: The processed inputs and labels.
        """
        if inputs['input_ids'].dim() == 1:
            inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)
        
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            
            batch_size = inputs['input_ids'].size(0)
            if labels.size(0) != batch_size:
                labels = labels.repeat(batch_size, 1)
        
        return inputs, labels