import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable CUDA device-side assertions

from .base_trainer import BaseTrainer
import torch
import logging
import torch.amp as amp
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

class EncoderDecoderTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, config):
        """
        Initialize the EncoderDecoderTrainer.
        
        This trainer is designed for models that are already integrated encoder-decoder architectures (e.g., ERNIE-M).
        
        Args:
            model: The encoder-decoder model to train.
            tokenizer: The tokenizer for the model.
            config (dict): Configuration parameters for training.
        """
        super().__init__(model, tokenizer, config)
        logging.info(f"Initialized EncoderDecoderTrainer with config: {config}")
        
        # Initialize the optimizer with AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay'])
        )
        
        # Set up the device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Initialize the gradient scaler for mixed precision training
        self.scaler = amp.GradScaler()

    def train(self, train_dataset, eval_dataset):
        """
        Train the encoder-decoder model.
        
        Args:
            train_dataset: Dataset for training.
            eval_dataset: Dataset for evaluation.
        
        Returns:
            tuple: Final training loss and evaluation loss.
        """
        try:
            # Prepare data loaders
            train_dataloader = self._get_dataloader(train_dataset, is_train=True)
            eval_dataloader = self._get_dataloader(eval_dataset, is_train=False)

            # Set up learning rate scheduler
            num_training_steps = len(train_dataloader) * self.config['num_train_epochs']
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.config['warmup_steps'], 
                num_training_steps=num_training_steps
            )

            total_train_loss = 0
            total_eval_loss = 0

            # Training loop
            for epoch in range(self.config['num_train_epochs']):
                self.model.train()
                epoch_loss = 0
                for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_train_epochs']}")):
                    try:
                        # Use mixed precision training
                        with amp.autocast(device_type='cuda', dtype=torch.float16):
                            loss = self.train_step(batch)
                        
                        epoch_loss += loss.item()
                        
                        # Scale the loss and compute gradients
                        self.scaler.scale(loss).backward()
                        
                        # Gradient accumulation
                        if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                            # Unscale the gradients
                            self.scaler.unscale_(self.optimizer)
                            
                            # Clip gradients to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                            
                            # Optimizer and scheduler step
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scheduler.step()
                            self.optimizer.zero_grad()
                    except RuntimeError as e:
                        # Handle CUDA out of memory error
                        if "CUDA out of memory" in str(e):
                            logging.error(f"CUDA OOM error in batch {step}. Trying to recover...")
                            torch.cuda.empty_cache()
                            self.optimizer.zero_grad()
                            if self.config['per_device_train_batch_size'] > 1:
                                self.config['per_device_train_batch_size'] //= 2
                                logging.info(f"Reduced batch size to {self.config['per_device_train_batch_size']}")
                                train_dataloader = self._get_dataloader(train_dataset, is_train=True)
                            else:
                                logging.error("Cannot reduce batch size further. Skipping this batch.")
                            continue
                        # Handle CUDA device-side assert error
                        elif "CUDA error: device-side assert triggered" in str(e):
                            logging.error(f"CUDA assert error in batch {step}: {str(e)}")
                            logging.error(f"Batch contents: {batch}")
                            continue  # Skip this batch and continue with the next one
                        else:
                            raise

                # Compute average training loss for the epoch
                avg_train_loss = epoch_loss / len(train_dataloader)
                total_train_loss += avg_train_loss
                logging.info(f"Epoch {epoch+1}/{self.config['num_train_epochs']} - Average train loss: {avg_train_loss:.4f}")

                # Evaluate the model
                eval_loss = self.evaluate(eval_dataloader)
                total_eval_loss += eval_loss
                logging.info(f"Epoch {epoch+1}/{self.config['num_train_epochs']} - Evaluation loss: {eval_loss:.4f}")

            # Return average training and evaluation loss across all epochs
            return total_train_loss / self.config['num_train_epochs'], total_eval_loss / self.config['num_train_epochs']

        except Exception as e:
            logging.error(f"Unexpected error during training: {str(e)}")
            raise

        finally:
            # Clear CUDA cache
            torch.cuda.empty_cache()

    def train_step(self, batch):
        """
        Perform a single training step for the encoder-decoder model.
        
        This method processes a batch of data, computes the loss, and returns it.
        
        Args:
            batch: A batch of training data.
        
        Returns:
            torch.Tensor: The loss for this training step.
        """
        try:
            # Move input tensors to the correct device
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)
            
            # Ensure all tensors have the same shape
            inputs, labels = self._ensure_tensor_shapes(inputs, labels)
            
            # Forward pass
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            
            return loss

        except Exception as e:
            logging.error(f"Error in train_step: {str(e)}")
            logging.error(f"Inputs shapes: {[(k, v.shape) for k, v in inputs.items()]}")
            logging.error(f"Labels shape: {labels.shape}")
            raise

    def evaluate(self, eval_dataloader):
        """
        Evaluate the model on the evaluation dataset.
        
        Args:
            eval_dataloader: DataLoader for the evaluation dataset.
        
        Returns:
            float: The average evaluation loss.
        """
        self.model.eval()
        total_eval_loss = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                loss = self.evaluation_step(batch)
                total_eval_loss += loss.item()

        return total_eval_loss / len(eval_dataloader)

    def evaluation_step(self, batch):
        """
        Perform a single evaluation step for the encoder-decoder model.
        
        This method processes a batch of data, computes the loss without gradient computation, and returns it.
        
        Args:
            batch: A batch of evaluation data.
        
        Returns:
            torch.Tensor: The loss for this evaluation step.
        """
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device)
        
        inputs, labels = self._ensure_tensor_shapes(inputs, labels)
        
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        return loss

    def _ensure_tensor_shapes(self, inputs, labels):
        """
        Ensure that all input tensors have the same shape along the sequence dimension.
        
        Args:
            inputs (dict): Dictionary containing input tensors.
            labels (torch.Tensor): Labels tensor.
        
        Returns:
            tuple: Tuple containing adjusted inputs dictionary and labels tensor.
        """
        max_length = max(inputs['input_ids'].size(1), labels.size(1))
        
        inputs['input_ids'] = self._adjust_tensor_size(inputs['input_ids'], max_length, dim=1)
        inputs['attention_mask'] = self._adjust_tensor_size(inputs['attention_mask'], max_length, dim=1)
        labels = self._adjust_tensor_size(labels, max_length, dim=1)
        
        return inputs, labels

    def _adjust_tensor_size(self, tensor, target_size, dim=1):
        """
        Adjust the size of a tensor to match the target size along the specified dimension.
        
        Args:
            tensor (torch.Tensor): Input tensor to adjust.
            target_size (int): Target size for the specified dimension.
            dim (int): Dimension along which to adjust the tensor size.
        
        Returns:
            torch.Tensor: Adjusted tensor.
        """
        current_size = tensor.size(dim)
        
        if current_size < target_size:
            padding = torch.zeros((*tensor.shape[:dim], target_size - current_size, *tensor.shape[dim+1:]), 
                                  dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=dim)
        elif current_size > target_size:
            return tensor.narrow(dim, 0, target_size)
        else:
            return tensor

    def validate_dataset(self, dataset):
        """
        Validate a dataset by running it through multiple evaluation steps.
        
        This method checks the dataset for potential issues by processing
        a few batches through the model without computing gradients.
        
        Args:
            dataset: Dataset to validate.
        """
        # Create a dataloader for the dataset
        dataloader = self._get_dataloader(dataset, is_train=False)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Validate up to 10 batches or the entire dataset, whichever is smaller
        num_batches_to_validate = min(10, len(dataloader))
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches_to_validate:
                    break
                
                try:
                    logging.debug(f"Validating batch {i+1}/{num_batches_to_validate}")
                    
                    # Log shape and dtype information for each tensor in the batch
                    for key, value in batch.items():
                        logging.debug(f"{key} shape: {value.shape}, dtype: {value.dtype}")

                    # Move batch to the appropriate device (GPU if available)
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # Check if the batch size matches the expected size
                    actual_batch_size = batch['input_ids'].size(0)
                    expected_batch_size = self.config['per_device_eval_batch_size']
                    if actual_batch_size != expected_batch_size:
                        logging.warning(f"Batch size mismatch. Expected: {expected_batch_size}, Actual: {actual_batch_size}")
                        # Adjust batch size if necessary
                        if actual_batch_size < expected_batch_size:
                            # Pad the batch to match the expected size
                            padding_size = expected_batch_size - actual_batch_size
                            for key in batch:
                                padding = torch.zeros(padding_size, *batch[key].shape[1:], dtype=batch[key].dtype, device=self.device)
                                batch[key] = torch.cat([batch[key], padding], dim=0)
                        else:
                            # Truncate the batch to match the expected size
                            for key in batch:
                                batch[key] = batch[key][:expected_batch_size]

                    # Ensure tensor shapes are consistent
                    inputs = {k: v for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels']
                    inputs, labels = self._ensure_tensor_shapes(inputs, labels)

                    # Process the batch through the model
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss

                    logging.debug(f"Model output keys: {outputs.keys() if isinstance(outputs, dict) else 'not a dict'}")
                    logging.debug(f"Loss: {loss.item()}")

                except RuntimeError as e:
                    logging.error(f"RuntimeError in batch {i+1}: {str(e)}")
                    logging.error("Detailed batch information:")
                    for key, value in batch.items():
                        logging.error(f"{key} shape: {value.shape}, dtype: {value.dtype}")
                        if value.numel() > 0:
                            logging.error(f"{key} min: {value.min().item()}, max: {value.max().item()}")
                    logging.error("Skipping this batch and continuing with the next one.")
                    continue
                except Exception as e:
                    logging.error(f"Error validating batch {i+1}: {str(e)}")
                    logging.error(f"Problematic batch: {batch}")
                    for key, value in batch.items():
                        logging.error(f"{key} shape: {value.shape}, dtype: {value.dtype}")
                    raise

        logging.info(f"Dataset validation completed. Validated {num_batches_to_validate} batches.")