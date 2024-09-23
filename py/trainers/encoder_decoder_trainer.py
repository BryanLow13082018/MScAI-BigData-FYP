import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable CUDA device-side assertions

from .base_trainer import BaseTrainer
import torch
import torch.nn as nn
import logging
import traceback
import time
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

class EncoderDecoderTrainer(BaseTrainer):
    """
    Trainer class for encoder-decoder models like ERNIE-M.
    
    This class handles the training, evaluation, and generation processes
    for encoder-decoder models, with a focus on memory efficiency and
    error handling.
    """

    def __init__(self, model, tokenizer, config):
        """
        Initialize the EncoderDecoderTrainer.
    
        Args:
            model (nn.Module): The encoder-decoder model (e.g., ERNIE-M).
            tokenizer (PreTrainedTokenizer): The tokenizer for the model.
            config (dict): Configuration parameters for training.
        """
        super().__init__(model, tokenizer, config)
        logging.info(f"Initialized EncoderDecoderTrainer with config: {config}")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay'])
        )
        
        # Set up device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Set context window and max new tokens from config or use defaults
        self.context_window = config.get('context_window', 2048)
        self.max_new_tokens = config.get('max_new_tokens', 128)
        
        # Use CrossEntropyLoss for language modeling tasks
        self.loss_fn = nn.CrossEntropyLoss()
        
        logging.info(f"Model vocab size: {len(self.tokenizer)}")

    def train(self, train_dataset, eval_dataset):
        """
        Train the encoder-decoder model.
        
        Args:
            train_dataset: Dataset for training.
            eval_dataset: Dataset for evaluation.
        
        Returns:
            dict: Training results including losses.
        """
        try:
            # Validate datasets before training
            self.validate_dataset(train_dataset, "Training")
            self.validate_dataset(eval_dataset, "Evaluation")
            
            train_dataloader = self._get_dataloader(train_dataset, is_train=True)
            eval_dataloader = self._get_dataloader(eval_dataset, is_train=False)

            num_training_steps = len(train_dataloader) * self.config['num_train_epochs']
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.config['warmup_steps'], 
                num_training_steps=num_training_steps
            )

            total_train_loss = 0
            total_eval_loss = 0

            for epoch in range(self.config['num_train_epochs']):
                self.model.train()
                epoch_loss = 0

                for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.config['num_train_epochs']}")):
                    try:
                        # Log batch information
                        logging.info(f"Processing batch {step}")
                        for k, v in batch.items():
                            logging.info(f"{k} shape: {v.shape}, dtype: {v.dtype}, max: {v.max().item()}, min: {v.min().item()}")

                        # Move batch to device safely
                        device_batch = {}
                        for k, v in batch.items():
                            try:
                                device_batch[k] = v.to(self.device)
                            except RuntimeError as e:
                                logging.error(f"Error moving '{k}' to device: {str(e)}")
                                logging.error(f"{k} shape: {v.shape}, dtype: {v.dtype}")
                                raise

                        # Compute loss
                        loss = self.train_step(device_batch)

                        if loss is None or torch.isnan(loss) or torch.isinf(loss):
                            logging.warning(f"Skipping batch {step} due to invalid loss.")
                            continue

                        logging.info(f"Step {step}, Loss: {loss.item()}")

                        loss = loss / self.config['gradient_accumulation_steps']
                        epoch_loss += loss.item()

                        # Backward pass
                        loss.backward()

                        if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                            # Clip gradients to prevent exploding gradients
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                            logging.info(f"Gradient norm: {grad_norm}")

                            # Check for Inf values before stepping
                            if grad_norm.isinf():
                                logging.warning(f"Skipping optimizer step due to inf gradients.")
                                continue

                            # Step and update
                            self.optimizer.step()
                            scheduler.step()

                            # Zero gradients
                            self.optimizer.zero_grad(set_to_none=True)

                    except RuntimeError as e:
                        logging.error(f"RuntimeError in batch {step}: {str(e)}")
                        logging.error(f"Error type: {type(e).__name__}")
                        logging.error(f"Error traceback:\n{traceback.format_exc()}")
                        
                        # Handle CUDA out of memory error
                        if "CUDA out of memory" in str(e):
                            logging.error("CUDA out of memory error. Trying to free up GPU memory.")
                            torch.cuda.empty_cache()
                        continue  # Skip this batch and continue with the next one

                avg_train_loss = epoch_loss / len(train_dataloader)
                total_train_loss += avg_train_loss
                logging.info(f"Epoch {epoch + 1} - Average train loss: {avg_train_loss:.4f}")

                eval_loss = self.evaluate(eval_dataloader)
                total_eval_loss += eval_loss
                logging.info(f"Epoch {epoch + 1} - Evaluation loss: {eval_loss:.4f}")

                # Clear CUDA memory after each epoch's training to help with memory management
                torch.cuda.empty_cache()

            return {
                "train_loss": total_train_loss / self.config['num_train_epochs'],
                "eval_loss": total_eval_loss / self.config['num_train_epochs']
            }

        except Exception as e:
            logging.error(f"Unexpected error during training: {str(e)}")
            raise

    def train_step(self, batch):
        """
        Perform a single training step.
        
        Args:
            batch (dict): A batch of training data containing 'input_ids', 
                          'attention_mask', and 'labels'.
        
        Returns:
            torch.Tensor: The computed loss for this training step, or None if an error occurred.
        """
        try:
            # Move input tensors to the correct device and apply torch.nan_to_num
            inputs = {k: torch.nan_to_num(v.to(self.device)) for k, v in batch.items()}
            
            # Log input shapes for debugging
            logging.debug(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Compute loss
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), inputs['labels'].view(-1))
            
            # Log the loss
            logging.debug(f"Step loss: {loss.item()}")
            
            return loss
    
        except Exception as e:
            logging.error(f"Error in train_step: {str(e)}")
            logging.error(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
            return None

    def evaluate(self, eval_dataloader, max_batches=None, timeout=300):
        """
        Evaluate the model on the evaluation dataset.
        
        Args:
            eval_dataloader: DataLoader for the evaluation dataset.
            max_batches (int, optional): Maximum number of batches to evaluate.
            timeout (int, optional): Maximum time (in seconds) for evaluation.
        
        Returns:
            float: The average evaluation loss.
        """
        self.model.eval()
        total_eval_loss = 0
        num_batches = 0
        start_time = time.time()

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if max_batches is not None and i >= max_batches:
                    logging.info(f"Reached max_batches limit of {max_batches}")
                    break
                
                if time.time() - start_time > timeout:
                    logging.warning(f"Evaluation timed out after {timeout} seconds")
                    break

                logging.info(f"Evaluating batch {i+1}")
                loss = self.evaluate_step(batch)
                if loss is not None:
                    total_eval_loss += loss.item()
                    num_batches += 1
                logging.info(f"Batch {i+1} loss: {loss.item() if loss is not None else 'None'}")

                if torch.cuda.is_available():
                    logging.info(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

        avg_loss = total_eval_loss / num_batches if num_batches > 0 else 0
        logging.info(f"Total evaluation batches: {num_batches}")
        logging.info(f"Average evaluation loss: {avg_loss:.6f}")
        return avg_loss

    def evaluate_step(self, batch):
        """
        Perform a single evaluation step.
        
        Args:
            batch (dict): A batch of evaluation data.
        
        Returns:
            torch.Tensor: The computed loss for this evaluation step, or None if an error occurred.
        """
        try:
            inputs = {k: torch.nan_to_num(v.to(self.device)) for k, v in batch.items()}
            
            # Ensure all tensors have the same sequence length
            min_length = min(v.size(1) for v in inputs.values() if v.dim() > 1)
            inputs = {k: v[:, :min_length] if v.dim() > 1 else v for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            loss = outputs.loss
            return loss
        except Exception as e:
            logging.error(f"Error in evaluate_step: {str(e)}")
            logging.error(f"Inputs shapes: {[(k, v.shape) for k, v in inputs.items()]}")
            return None

    def _get_dataloader(self, dataset, is_train=True):
        """
        Create a DataLoader for the given dataset.

        Args:
            dataset: The dataset to create a DataLoader for.
            is_train (bool): Whether this is for training data or not.

        Returns:
            torch.utils.data.DataLoader: The created DataLoader.
        """
        # Determine the appropriate batch size
        batch_size = self.config['per_device_train_batch_size'] if is_train else self.config['per_device_eval_batch_size']
        
        # Create and return the DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=True
        )

    def validate_dataset(self, dataset, dataset_type=""):
        """
        Validate the dataset by checking the shapes of the first few batches.

        Args:
            dataset: The dataset to validate.
            dataset_type (str): A string to identify the type of dataset (e.g., "Training" or "Evaluation").
        """
        dataloader = self._get_dataloader(dataset, is_train=False)
        logging.info(f"Validating {dataset_type} dataset:")
        
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Check first 5 batches
                break
            
            logging.info(f"Batch {i} shapes:")
            for k, v in batch.items():
                logging.info(f"  {k}: {v.shape}")
            
            required_keys = ['input_ids', 'attention_mask', 'labels']
            for key in required_keys:
                if key not in batch:
                    logging.warning(f"  Missing required key: {key}")
            
            if 'input_ids' in batch and 'attention_mask' in batch:
                if batch['input_ids'].shape != batch['attention_mask'].shape:
                    logging.warning(f"  Shape mismatch: input_ids {batch['input_ids'].shape} vs attention_mask {batch['attention_mask'].shape}")
            
            if 'input_ids' in batch and 'labels' in batch:
                if batch['input_ids'].shape != batch['labels'].shape:
                    logging.warning(f"  Shape mismatch: input_ids {batch['input_ids'].shape} vs labels {batch['labels'].shape}")
            
            unexpected_keys = set(batch.keys()) - set(required_keys)
            if unexpected_keys:
                logging.info(f"  Unexpected keys in batch: {unexpected_keys}")

        logging.info(f"Finished validating {dataset_type} dataset")

    def generate(self, input_text):
        """
        Generate text using the encoder-decoder model.

        Args:
            input_text (str): The input text to start generation from.

        Returns:
            str: The generated text.
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=self.context_window)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text
    
    def generate_batch(self, dataset):
        """
        Generate text for a batch of inputs using the encoder-decoder model.
        
        Args:
            dataset (CustomDataset): The dataset containing input texts and other information.
        
        Returns:
            list: A list of generated texts.
        """
        self.model.eval()
        generated_texts = []
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                
                # Decode input_ids to text
                input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
                # Generate text for each input in the batch
                for input_text in input_texts:
                    generated_text = self.generate(input_text)
                    generated_texts.append(generated_text)
        
        return generated_texts

    def collate_fn(self, batch):
        """
        Custom collate function to handle CustomDataset batches.
        """
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch])
        }