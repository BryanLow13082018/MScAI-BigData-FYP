import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable CUDA device-side assertions

import torch
import torch.nn as nn
import logging
import traceback
from torch.cuda.amp import autocast
from torch.amp import GradScaler 
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput

class CombinedEncoderDecoderTrainer:
    def __init__(self, encoder, decoder, tokenizer, config):
        """
        Initialize the CombinedEncoderDecoderTrainer.

        Args:
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model.
            tokenizer: The tokenizer for processing input text.
            config (dict): Configuration parameters for training.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.config = config
        
        # Set up the device
        if hasattr(self.encoder, 'hf_device_map') or hasattr(self.decoder, 'hf_device_map'):
            logging.info("Models are already distributed across devices. Skipping device movement.")
            self.device = next(self.encoder.parameters()).device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.encoder = self.encoder.to(self.device)
            self.decoder = self.decoder.to(self.device)

        # Initialize optimizers with separate learning rates for encoder and decoder
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=float(config['encoder_lr']),
            weight_decay=float(config['weight_decay'])
        )
        self.decoder_optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=float(config['decoder_lr']),
            weight_decay=float(config['weight_decay'])
        )

        # Modify the combine layer to match the number of classes
        num_classes = len(tokenizer.vocab)  # Assuming tokenizer vocab size is the number of classes
        self.linear_layer = nn.Linear(self.encoder.config.hidden_size + self.decoder.config.hidden_size, num_classes)  # Store reference to Linear layer
        self.combine_layer = nn.Sequential(
            self.linear_layer,
            nn.LogSoftmax(dim=-1)
        ).to(self.device)

        # Use BCEWithLogitsLoss instead of CrossEntropyLoss for binary classification
        self.loss_fn = nn.BCEWithLogitsLoss() 

        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler()

        logging.info(f"Initialized CombinedEncoderDecoderTrainer with config: {config}")

        # Log the structure of encoder and decoder outputs
        logging.info(f"Encoder output structure: {self.encoder(torch.zeros((1, 128), dtype=torch.long).to(self.device), attention_mask=torch.ones((1, 128), dtype=torch.long).to(self.device))}")

    def train(self, train_dataset, eval_dataset):
        """
        Train the combined encoder-decoder model.
        
        Args:
            train_dataset: Dataset for training.
            eval_dataset: Dataset for evaluation.
        
        Returns:
            dict: Training results including losses.
        """
        # Create DataLoaders for training and evaluation
        train_dataloader = self._get_dataloader(train_dataset, is_train=True)
        eval_dataloader = self._get_dataloader(eval_dataset, is_train=False)

        num_training_steps = len(train_dataloader) * self.config['num_train_epochs']
        
        # Initialize learning rate schedulers
        encoder_scheduler = get_linear_schedule_with_warmup(
            self.encoder_optimizer, 
            num_warmup_steps=self.config['warmup_steps'], 
            num_training_steps=num_training_steps
        )
        decoder_scheduler = get_linear_schedule_with_warmup(
            self.decoder_optimizer, 
            num_warmup_steps=self.config['warmup_steps'], 
            num_training_steps=num_training_steps
        )

        total_train_loss = 0
        total_eval_loss = 0

        for epoch in range(self.config['num_train_epochs']):
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_train_epochs']}")):
                try:
                    # Enable mixed precision training
                    with autocast():
                        loss = self.train_step(batch)
                    
                    if loss is None:
                        logging.warning(f"Skipping batch {step} due to error")
                        continue

                    loss = loss / self.config['gradient_accumulation_steps']
                    epoch_loss += loss.item()

                    # Scale gradients
                    self.scaler.scale(loss).backward()

                    if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                        # Unscale gradients and update model parameters
                        self.scaler.unscale_(self.encoder_optimizer)
                        self.scaler.unscale_(self.decoder_optimizer)
                        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config['max_grad_norm'])
                        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config['max_grad_norm'])

                        # Update model parameters
                        self.scaler.step(self.encoder_optimizer)
                        self.scaler.step(self.decoder_optimizer)
                        self.scaler.update()
                        encoder_scheduler.step()
                        decoder_scheduler.step()
                        self.encoder_optimizer.zero_grad()
                        self.decoder_optimizer.zero_grad()

                except RuntimeError as e:
                    logging.error(f"RuntimeError in batch {step}: {str(e)}")
                    continue  # Skip this batch and continue with the next one

            avg_train_loss = epoch_loss / len(train_dataloader)
            total_train_loss += avg_train_loss
            logging.info(f"Epoch {epoch+1} - Average train loss: {avg_train_loss:.4f}")

            eval_loss = self.evaluate(eval_dataloader)
            total_eval_loss += eval_loss
            logging.info(f"Epoch {epoch+1} - Evaluation loss: {eval_loss:.4f}")

        return {
            "train_loss": total_train_loss / self.config['num_train_epochs'],
            "eval_loss": total_eval_loss / self.config['num_train_epochs']
        }

    def train_step(self, batch):
        """
        Perform a single training step for the combined encoder-decoder model.
        
        Args:
            batch: A batch of training data.
        
        Returns:
            torch.Tensor: The loss for this training step, or None if an error occurred.
        """
        input_ids = None
        attention_mask = None
        labels = None

        try:
            # Move input IDs, attention masks, and labels to the specified device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logging.debug(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")

            # Validate the labels
            if labels.max() >= self.linear_layer.out_features:
                logging.error(f"Label values are out of bounds: {labels.max().item()} >= {self.linear_layer.out_features}")
                return None
            
            # Check input IDs are within tokenizer vocabulary
            if input_ids.max() >= self.tokenizer.vocab_size:
                logging.error(f"Input IDs contain invalid values: max(input_ids) = {input_ids.max().item()}, vocab_size = {self.tokenizer.vocab_size}")
                return None

            # Get encoder outputs
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

            # Handle SequenceClassifierOutput
            if isinstance(encoder_outputs, SequenceClassifierOutput):
                encoder_hidden_states = encoder_outputs.logits
            elif hasattr(encoder_outputs, 'last_hidden_state'):
                encoder_hidden_states = encoder_outputs.last_hidden_state
            elif isinstance(encoder_outputs, tuple) and len(encoder_outputs) > 0:
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs

            logging.debug(f"Encoder hidden states shape: {encoder_hidden_states.shape}")

            # Check and truncate input_ids for decoder if necessary
            decoder_vocab_size = self.decoder.config.vocab_size
            if input_ids.max() >= decoder_vocab_size:
                logging.warning(f"Truncating input_ids to fit decoder vocab size. Max value: {input_ids.max().item()}, Decoder vocab size: {decoder_vocab_size}")
                input_ids = torch.clamp(input_ids, max=decoder_vocab_size - 1)

            # Get decoder outputs
            decoder_outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask)

            # Handle different output structures for decoder
            if hasattr(decoder_outputs, 'last_hidden_state'):
                decoder_hidden_states = decoder_outputs.last_hidden_state
            elif isinstance(decoder_outputs, tuple) and len(decoder_outputs) > 0:
                decoder_hidden_states = decoder_outputs[0]
            else:
                decoder_hidden_states = decoder_outputs

            logging.debug(f"Decoder hidden states shape: {decoder_hidden_states.shape}")

            # Combine the encoder and decoder outputs
            combined_hidden_states = torch.cat([encoder_hidden_states, decoder_hidden_states], dim=-1)
            logging.debug(f"Combined hidden states shape: {combined_hidden_states.shape}")

            # Pass through the combination layer
            logits = self.combine_layer(combined_hidden_states)
            logging.debug(f"Logits shape: {logits.shape}")

            # Compute the loss using BCEWithLogitsLoss
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1).float())
            logging.debug(f"Loss: {loss.item()}")

            return loss

        except Exception as e:
            logging.error(f"Unexpected error in train_step: {str(e)}")
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error traceback:\n{traceback.format_exc()}")
            if input_ids is not None and attention_mask is not None and labels is not None:
                logging.error(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")
            return None  # Return None to indicate that this batch should be skipped

    def evaluate(self, eval_dataloader):
        """
        Evaluate the model on the evaluation dataset.
        
        Args:
            eval_dataloader: DataLoader for the evaluation dataset.
        
        Returns:
            float: The average evaluation loss.
        """
        self.encoder.eval()
        self.decoder.eval()
        total_eval_loss = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                loss = self.evaluation_step(batch)
                if loss is not None:
                    total_eval_loss += loss.item()

        return total_eval_loss / len(eval_dataloader)

    def evaluation_step(self, batch):
        """
        Perform a single evaluation step for the combined encoder-decoder model.
        
        Args:
            batch: A batch of evaluation data.
        
        Returns:
            torch.Tensor: The loss for this evaluation step, or None if an error occurred.
        """
        try:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Get encoder outputs
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle SequenceClassifierOutput
            if isinstance(encoder_outputs, SequenceClassifierOutput):
                encoder_hidden_states = encoder_outputs.logits
            elif hasattr(encoder_outputs, 'last_hidden_state'):
                encoder_hidden_states = encoder_outputs.last_hidden_state
            elif isinstance(encoder_outputs, tuple) and len(encoder_outputs) > 0:
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs

            # Get decoder outputs
            decoder_outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different output structures for decoder
            if hasattr(decoder_outputs, 'last_hidden_state'):
                decoder_hidden_states = decoder_outputs.last_hidden_state
            elif isinstance(decoder_outputs, tuple) and len(decoder_outputs) > 0:
                decoder_hidden_states = decoder_outputs[0]
            else:
                decoder_hidden_states = decoder_outputs

            combined_hidden_states = torch.cat([encoder_hidden_states, decoder_hidden_states], dim=-1)
            logits = self.combine_layer(combined_hidden_states)
            
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            return loss
        except Exception as e:
            logging.error(f"Error in evaluation_step: {str(e)}")
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error traceback:\n{traceback.format_exc()}")
            logging.error(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")
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
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['per_device_train_batch_size'] if is_train else self.config['per_device_eval_batch_size'],
            shuffle=is_train,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=True
        )

    def validate_dataset(self, dataset):
        """
        Validate the dataset by checking the shapes of the first few batches.

        Args:
            dataset: The dataset to validate.
        """
        dataloader = self._get_dataloader(dataset, is_train=False)
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Check first 5 batches
                break
            logging.info(f"Batch {i} shapes: {[(k, v.shape) for k, v in batch.items()]}")