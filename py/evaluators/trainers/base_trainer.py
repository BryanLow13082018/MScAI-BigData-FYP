import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging

class BaseTrainer:
    def __init__(self, model, tokenizer, config):
        """
        Initialize the BaseTrainer.
        
        Args:
            model: The model to train.
            tokenizer: The tokenizer for the model.
            config (dict): Configuration parameters for training.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Set up the device
        if hasattr(self.model, 'hf_device_map'):
            logging.info("Model is already distributed across devices. Skipping device movement.")
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        
        logging.info(f"Using device: {self.device}")

    def train(self, train_dataset, val_dataset, optimizer):
        """
        Train the model.
        
        This method handles the entire training loop, including evaluation at each epoch.
        It now better handles variable batch sizes.
        
        Args:
            train_dataset: Dataset or DataLoader for training.
            val_dataset: Dataset or DataLoader for evaluation.
            optimizer: The optimizer to use for training.
        
        Returns:
            float: The final evaluation loss.
        """
        self.optimizer = optimizer  # Set the optimizer here

        # Prepare data loaders
        train_dataloader = self._get_dataloader(train_dataset, is_train=True)
        eval_dataloader = self._get_dataloader(val_dataset, is_train=False)

        num_training_steps = len(train_dataloader) * self.config['num_train_epochs']
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=self.config['warmup_steps'], 
            num_training_steps=num_training_steps
        )

        for epoch in range(self.config['num_train_epochs']):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_train_epochs']}")):
                try:
                    # Handle variable batch sizes
                    batch = self._prepare_batch(batch)
                    loss = self.train_step(batch)
                    total_loss += loss.item()
                    
                    loss.backward()
                    
                    if self.config.get('max_grad_norm'):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                    
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()

                except RuntimeError as e:
                    if "CUDA error: device-side assert triggered" in str(e):
                        logging.error(f"CUDA assert error in batch {step}: {str(e)}")
                        logging.error(f"Batch contents: {batch}")
                        logging.error(f"Device: {self.device}")
                        logging.error(f"CUDA available: {torch.cuda.is_available()}")
                        logging.error(f"Current CUDA device: {torch.cuda.current_device()}")
                        continue  # Skip this batch and continue with the next one
                    else:
                        raise

            avg_train_loss = total_loss / len(train_dataloader)
            logging.info(f"Average train loss: {avg_train_loss:.4f}")

            eval_loss = self.evaluate(eval_dataloader)
            logging.info(f"Evaluation loss: {eval_loss:.4f}")

        return eval_loss  # Return the final evaluation loss

    def evaluate(self, eval_dataloader):
        """
        Evaluate the model.
        
        This method runs the model on the evaluation dataset and computes the average loss.
        It now handles variable batch sizes.
        
        Args:
            eval_dataloader: DataLoader for evaluation data.
        
        Returns:
            float: The average evaluation loss.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                try:
                    # Handle variable batch sizes
                    batch = self._prepare_batch(batch)
                    loss = self.evaluation_step(batch)
                    total_loss += loss.item()
                except RuntimeError as e:
                    if "CUDA error: device-side assert triggered" in str(e):
                        logging.error(f"CUDA assert error in evaluation batch: {str(e)}")
                        logging.error(f"Batch contents: {batch}")
                        continue  # Skip this batch and continue with the next one
                    else:
                        raise
        return total_loss / len(eval_dataloader)

    def train_step(self, batch):
        """
        Perform a single training step.
        
        This method should be implemented by subclasses to define the specific training logic.
        
        Args:
            batch: A batch of training data.
        
        Returns:
            torch.Tensor: The loss for this training step.
        """
        raise NotImplementedError("Subclasses must implement train_step method")

    def evaluation_step(self, batch):
        """
        Perform a single evaluation step.
        
        This method should be implemented by subclasses to define the specific evaluation logic.
        
        Args:
            batch: A batch of evaluation data.
        
        Returns:
            torch.Tensor: The loss for this evaluation step.
        """
        raise NotImplementedError("Subclasses must implement evaluation_step method")

    def _get_dataloader(self, dataset, is_train=True):
        """
        Get a DataLoader for the given dataset.
        
        Args:
            dataset: The dataset to create a DataLoader for.
            is_train (bool): Whether this is for training data.
        
        Returns:
            DataLoader: The created DataLoader.
        """
        if isinstance(dataset, DataLoader):
            return dataset
        elif isinstance(dataset, Dataset):
            return DataLoader(
                dataset, 
                batch_size=self.config['per_device_train_batch_size'] if is_train else self.config['per_device_eval_batch_size'], 
                shuffle=is_train,
                drop_last=False  # Keep the last batch even if it's smaller
            )
        else:
            raise ValueError("dataset must be either a Dataset or a DataLoader")

    def _prepare_batch(self, batch):
        """
        Prepare a batch for processing.
        
        This method moves the batch to the correct device and handles variable batch sizes.
        
        Args:
            batch: A batch of data.
        
        Returns:
            dict: The prepared batch.
        """
        prepared_batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Log batch information
        logging.debug(f"Batch keys: {prepared_batch.keys()}")
        for key, value in prepared_batch.items():
            logging.debug(f"{key} shape: {value.shape}, dtype: {value.dtype}")
        
        return prepared_batch

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