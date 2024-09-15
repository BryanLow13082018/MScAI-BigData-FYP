import os
import torch
import torch.nn as nn
import logging
import traceback
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput
from accelerate import Accelerator
import torch_xla.core.xla_model as xm  # Importing TPU model functionalities

# Set environment variables for CUDA device-side assertions
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable CUDA device-side assertions

class CombinedEncoderDecoderTrainer:
    def __init__(self, encoder, decoder, tokenizer, config, num_labels):
        """
        Initialize the CombinedEncoderDecoderTrainer.

        Args:
            encoder (nn.Module): The encoder model.
            decoder (nn.Module): The decoder model.
            tokenizer: The tokenizer for processing input text.
            config (dict): Configuration parameters for training.
            num_labels (int): Number of labels for the classification task.
        """
        print("Initializing CombinedEncoderDecoderTrainer")
        self.accelerator = Accelerator()  # Initialize the accelerator
        self.encoder = encoder  # Do not move to TPU here
        self.decoder = decoder  # Do not move to TPU here
        self.tokenizer = tokenizer
        self.config = config
        self.num_labels = num_labels
        
        # Initialize optimizers
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

        # Use BCEWithLogitsLoss for multi-label classification
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler()

        # Prepare models and optimizers with the accelerator
        self.encoder, self.decoder, self.encoder_optimizer, self.decoder_optimizer = self.accelerator.prepare(
            self.encoder, self.decoder, self.encoder_optimizer, self.decoder_optimizer
        )

        logging.info(f"Initialized CombinedEncoderDecoderTrainer with config: {config}")

    def train(self, train_dataset, eval_dataset):
        """
        Train the combined encoder-decoder model.
        
        Args:
            train_dataset: Dataset for training.
            eval_dataset: Dataset for evaluation.
        
        Returns:
            dict: Training results including losses.
        """
        train_dataloader = self._get_dataloader(train_dataset, is_train=True)
        eval_dataloader = self._get_dataloader(eval_dataset, is_train=False)

        num_training_steps = len(train_dataloader) * self.config['num_train_epochs']
        
        # Set up learning rate schedulers for both encoder and decoder
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

        for epoch in range(self.config['num_train_epochs']):
            self.encoder.train()  # Set encoder to training mode
            self.decoder.train()  # Set decoder to training mode
            epoch_loss = 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_train_epochs']}")):
                try:
                    # Use mixed precision autocast
                    with autocast():
                        loss = self.train_step(batch)

                    if loss is None:
                        logging.warning(f"Skipping batch {step} due to error")
                        continue

                    # Scale down loss for gradient accumulation
                    loss = loss / self.config['gradient_accumulation_steps']
                    epoch_loss += loss.item()

                    # Scale the loss and perform backpropagation
                    self.scaler.scale(loss).backward()

                    # Update weights every `gradient_accumulation_steps` batches
                    if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                        self.scaler.unscale_(self.encoder_optimizer)
                        self.scaler.unscale_(self.decoder_optimizer)
                        nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config['max_grad_norm'])
                        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config['max_grad_norm'])

                        self.scaler.step(self.encoder_optimizer)
                        self.scaler.step(self.decoder_optimizer)
                        self.scaler.update()
                        encoder_scheduler.step()
                        decoder_scheduler.step()
                        self.encoder_optimizer.zero_grad()
                        self.decoder_optimizer.zero_grad()

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logging.error(f"CUDA OOM in batch {step}: {str(e)}")
                        xm.rendezvous('oom')  # Synchronize across TPUs
                    else:
                        logging.error(f"RuntimeError in batch {step}: {str(e)}")
                        logging.error(traceback.format_exc())

            # Calculate average training loss for the epoch
            avg_train_loss = epoch_loss / len(train_dataloader)
            total_train_loss += avg_train_loss
            logging.info(f"Epoch {epoch+1} - Average train loss: {avg_train_loss:.4f}")

            eval_loss = self.evaluate(eval_dataloader)
            logging.info(f"Epoch {epoch+1} - Evaluation loss: {eval_loss:.4f}")

            # Clear TPU memory
            xm.rendezvous('clear_memory')

        return {
            "train_loss": total_train_loss / self.config['num_train_epochs'],
            "eval_loss": eval_loss
        }

    def train_step(self, batch):
        """
        Perform a single training step.

        Args:
            batch (dict): A dictionary containing the input batch data.

        Returns:
            torch.Tensor: The computed loss for this batch, or None if an error occurred.
        """
        try:
            input_ids = batch['input_ids'].to(self.accelerator.device)  # Use xla device
            attention_mask = batch['attention_mask'].to(self.accelerator.device)  # Use xla device
            labels = batch['labels'].to(self.accelerator.device)  # Use xla device

            if input_ids.numel() == 0:
                logging.warning("Skipping empty batch")
                return None

            # Get encoder and decoder outputs
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            decoder_outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # Extract hidden states
            encoder_hidden_states = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs[0]
            decoder_hidden_states = decoder_outputs.hidden_states[-1] if hasattr(decoder_outputs, 'hidden_states') else decoder_outputs[0]

            # Ensure both hidden states are 3D and have the same sequence length
            if encoder_hidden_states.dim() == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            if decoder_hidden_states.dim() == 2:
                decoder_hidden_states = decoder_hidden_states.unsqueeze(1)

            # Align sequence lengths
            min_length = min(encoder_hidden_states.shape[1], decoder_hidden_states.shape[1])
            encoder_hidden_states = encoder_hidden_states[:, :min_length, :]
            decoder_hidden_states = decoder_hidden_states[:, :min_length, :]

            combined_hidden_states = torch.cat([encoder_hidden_states, decoder_hidden_states], dim=-1)

            # Pass through combine layer
            logits = self.combine_layer(combined_hidden_states)  # Ensure this matches the number of labels

            # Calculate loss
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.float())  # Reshape as needed
            return loss

        except Exception as e:
            logging.error(f"Unexpected error in train_step: {str(e)}")
            logging.error(traceback.format_exc())
            return None

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
        Perform a single evaluation step.

        Args:
            batch (dict): A dictionary containing the input batch data.

        Returns:
            torch.Tensor: The computed loss for this batch, or None if an error occurred.
        """
        try:
            input_ids = batch['input_ids'].to(self.accelerator.device)  # Use xla device
            attention_mask = batch['attention_mask'].to(self.accelerator.device)  # Use xla device
            labels = batch['labels'].to(self.accelerator.device)  # Use xla device

            if input_ids.numel() == 0:
                logging.warning("Skipping empty batch in evaluation")
                return None

            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            decoder_outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            encoder_hidden_states = encoder_outputs.last_hidden_state
