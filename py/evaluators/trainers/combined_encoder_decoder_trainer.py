import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable CUDA device-side assertions

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import traceback
import time
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput, CausalLMOutput, CausalLMOutputWithPast
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

class CombinedEncoderDecoderTrainer:
    """
    Trainer class for encoder-decoder models like ERNIE-M.
    
    This class handles the training, evaluation, and generation processes
    for encoder-decoder models, with a focus on memory efficiency and
    error handling.
    """
    
    def __init__(self, encoder, decoder, encoder_tokenizer, decoder_tokenizer, config, accelerator, batch_size=64):
        """
        Initialize the CombinedEncoderDecoderTrainer.

        Args:
            encoder (nn.Module): The encoder model (AfroXLMR).
            decoder (nn.Module): The decoder model (LLaMA 2).
            encoder_tokenizer (PreTrainedTokenizer): The tokenizer for the encoder model.
            decoder_tokenizer (PreTrainedTokenizer): The tokenizer for the decoder model.
            config (dict): Configuration parameters for training.
            accelerator (Accelerator): The Hugging Face Accelerator object.
            batch_size (int, optional): Batch size for training. Defaults to 64.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.config = config
        self.batch_size = batch_size
        self.accelerator = accelerator

        # Set the dtype to bfloat16
        self.dtype = torch.bfloat16

        # Convert models to bfloat16
        self.encoder = self.encoder.to(dtype=self.dtype)
        self.decoder = self.decoder.to(dtype=self.dtype)

        # Set padding token for decoder tokenizer if not already set
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
            self.decoder.config.pad_token_id = self.decoder_tokenizer.eos_token_id
        
        # Set context window and max new tokens from config or use defaults
        self.context_window = config.get('context_window', 2048)
        self.max_new_tokens = config.get('max_new_tokens', 128)

        # Enable gradient checkpointing for memory efficiency
        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()
        
        # Store the decoder's dtype
        self.decoder_dtype = next(self.decoder.parameters()).dtype

        # Set up the architecture for combining encoder and decoder inputs
        self.encoder_hidden_size = self.encoder.config.hidden_size
        self.decoder_hidden_size = self.decoder.config.hidden_size
        combined_hidden_size = self.encoder_hidden_size + self.decoder_hidden_size
            
        # Create a projection layer to match dimensions
        self.projection = nn.Linear(combined_hidden_size, self.decoder_hidden_size)

        # Initialize the projection layer weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Apply init_weights to the projection layer
        self.projection.apply(init_weights)

        # Convert projection layer to bfloat16
        self.projection = self.projection.to(dtype=self.dtype)

        # Initialize optimizers with separate learning rates for encoder and decoder
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=float(config['encoder_lr']),
            weight_decay=float(config['weight_decay']),
            eps=1e-4  # Increased epsilon for bfloat16 stability
        )
        self.decoder_optimizer = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=float(config['decoder_lr']),
            weight_decay=float(config['weight_decay']),
            eps=1e-4  # Increased epsilon for bfloat16 stability
        )

        # Add learning rate schedulers
        self.encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        self.decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.decoder_optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # Prepare models, optimizers, projection layer, and schedulers with Accelerator
        self.encoder, self.decoder, self.projection, self.encoder_optimizer, self.decoder_optimizer, self.encoder_scheduler, self.decoder_scheduler = self.accelerator.prepare(
            self.encoder, self.decoder, self.projection, self.encoder_optimizer, self.decoder_optimizer, self.encoder_scheduler, self.decoder_scheduler
        )

        # Initialize embed_tokens weights
        self.init_embed_tokens()

        # Check decoder embedding layer
        embed_tokens = self.decoder.model.embed_tokens
        if torch.isnan(embed_tokens.weight).any() or torch.isinf(embed_tokens.weight).any():
            logging.error("NaN or Inf values detected in decoder embedding layer")
            logging.error(f"Embed tokens weight stats: min={embed_tokens.weight.min().item():.4f}, max={embed_tokens.weight.max().item():.4f}, mean={embed_tokens.weight.mean().item():.4f}")
            raise ValueError("Decoder embedding layer contains NaN or Inf values")

        # Ensure decoder tokenizer and model have matching vocabulary sizes
        assert len(self.decoder_tokenizer) == self.decoder.model.embed_tokens.weight.shape[0], "Decoder tokenizer and model vocabulary sizes do not match"

        # Use CrossEntropyLoss for language modeling tasks
        self.loss_fn = nn.CrossEntropyLoss()

        # Early stopping parameters
        self.patience = config.get('early_stopping_patience', 3)  # Number of epochs to wait before stopping
        self.min_delta = config.get('early_stopping_min_delta', 0.001)  # Minimum change to qualify as an improvement

        encoder_hidden_size = self.encoder.config.hidden_size
        decoder_hidden_size = self.decoder.config.hidden_size

        # Extract num_intent_classes from the config
        num_intent_classes = config.get('num_intent_classes', None)
        if num_intent_classes is None:
            raise ValueError("num_intent_classes must be provided in the config")

        num_slot_classes = config.get('num_slot_classes', None)
        if num_slot_classes is None:
            raise ValueError("num_slot_classes must be provided in the config")
            
        # Define the intent classifier
        self.intent_classifier = torch.nn.Linear(in_features=encoder_hidden_size, out_features=num_intent_classes)
        
        # Define the slot classifier
        self.slot_classifier = torch.nn.Linear(in_features=decoder_hidden_size, out_features=num_slot_classes)

        # Initiate intent_labels and slot_labels
        self.intent_preds = None
        self.intent_labels = None
        self.slot_preds = None
        self.slot_labels = None        

        logging.info(f"Initialized CombinedEncoderDecoderTrainer with config: {config}")
        logging.info(f"Using dtype: {self.dtype}")

    def log_cuda_memory(self, stage):
        """
        Log the current CUDA memory usage statistics.
    
        This function reports the current and peak memory allocation and reservation
        for the CUDA device. It's useful for monitoring memory usage during different
        stages of model training or inference, helping to identify potential memory
        leaks or inefficient memory usage.
    
        Args:
            stage (str): A string describing the current stage of execution
                         (e.g., "Start of epoch 1", "After batch 100").
    
        Prints:
            A formatted string containing:
            - The current stage
            - Currently allocated memory and peak allocated memory
            - Currently reserved memory and peak reserved memory
    
        Note:
            - Memory values are converted to gigabytes (GB) for readability.
            - This function only works if CUDA is available; otherwise, it prints
              a message indicating that CUDA is not available.
            - Allocated memory refers to the memory actively used by tensors.
            - Reserved memory is the total memory managed by the caching allocator,
              which may be larger than the allocated memory.
    
        Example output:
            CUDA Memory (Start of epoch 1) - Allocated: 1.23GB (Max: 2.34GB),
            Reserved: 3.45GB (Max: 4.56GB)
        """
        if torch.cuda.is_available():
            # Get current memory allocated and convert to gigabytes
            allocated = torch.cuda.memory_allocated() / 1024**3
            # Get maximum memory allocated so far and convert to gigabytes
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            # Get current memory reserved by cache allocator and convert to gigabytes
            reserved = torch.cuda.memory_reserved() / 1024**3
            # Get maximum memory reserved so far and convert to gigabytes
            max_reserved = torch.cuda.max_memory_reserved() / 1024**3
    
            # Print formatted memory usage statistics
            print(f"CUDA Memory ({stage}) - "
                  f"Allocated: {allocated:.2f}GB (Max: {max_allocated:.2f}GB), "
                  f"Reserved: {reserved:.2f}GB (Max: {max_reserved:.2f}GB)")
        else:
            # Print a message if CUDA is not available
            print("CUDA is not available")   
    
    def train(self, train_dataset, eval_dataset):
        """
        Train the combined encoder-decoder model.
        
        Args:
            train_dataset: Dataset for training.
            eval_dataset: Dataset for evaluation.
        
        Returns:
            dict: Training results including losses.
        """
        print("Starting training process")
        self.print_model_size()
        self.log_every_n_steps = 50
        self.current_step = 0
        self.current_epoch = 0

        self.log_cuda_memory("Start of training")
    
        # Validate datasets before training
        self.validate_dataset(train_dataset, "Training")
        self.validate_dataset(eval_dataset, "Evaluation")
        
        # Create DataLoaders for training and evaluation
        train_dataloader = self._get_dataloader(train_dataset, is_train=True)
        eval_dataloader = self._get_dataloader(eval_dataset, is_train=False)
    
        # Prepare dataloaders with Accelerator
        train_dataloader, eval_dataloader = self.accelerator.prepare(train_dataloader, eval_dataloader)
    
        num_training_steps = len(train_dataloader) * self.config['num_train_epochs']
        print(f"Total training steps: {num_training_steps}")
        
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
    
        # Prepare schedulers with Accelerator
        encoder_scheduler, decoder_scheduler = self.accelerator.prepare(encoder_scheduler, decoder_scheduler)
    
        total_train_loss = 0
        total_eval_loss = 0
    
        best_metric = float('inf')  # Initialize for early stopping
        patience_counter = 0
        best_model_state = None  # To save the best model state
    
        # Loop over epochs
        for epoch in range(self.config['num_train_epochs']):
            self.current_epoch = epoch + 1
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0

            for step, batch in enumerate(train_dataloader):
                self.current_step = step + 1
                try:
                    # Check embed_tokens at the start of each step
                    if not self.check_embed_tokens(self.current_step):
                        logging.warning(f"Skipping batch {step} due to issues in embed_tokens.")
                        continue

                    # Move batch to the correct device and dtype
                    batch = {k: v.to(device=self.accelerator.device, dtype=self.dtype) for k, v in batch.items()}

                    # Perform a training step
                    with self.accelerator.autocast():
                        loss = self.train_step(batch)
                        
                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        logging.warning(f"Skipping batch {step} due to invalid loss.")
                        continue
    
                    loss = loss / self.config['gradient_accumulation_steps']
                    epoch_loss += loss.item()
    
                    self.log_gradients()
    
                    if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                        # Clip gradients
                        self.accelerator.clip_grad_norm_(self.encoder.parameters(), self.config['max_grad_norm'])
                        self.accelerator.clip_grad_norm_(self.decoder.parameters(), self.config['max_grad_norm'])
    
                        # Optimizer steps
                        self.encoder_optimizer.step()
                        self.decoder_optimizer.step()
    
                        # Scheduler steps
                        encoder_scheduler.step()
                        decoder_scheduler.step()
    
                        # Zero gradients
                        self.encoder_optimizer.zero_grad()
                        self.decoder_optimizer.zero_grad()
    
                    if self.current_step % self.log_every_n_steps == 0:
                        logging.info(f"Epoch {self.current_epoch}, Step {self.current_step}/{len(train_dataloader)}")
    
                except RuntimeError as e:
                    logging.error(f"RuntimeError in batch {step}: {str(e)}")
                    if "CUDA out of memory" in str(e):
                        logging.error("CUDA out of memory error. Trying to free up GPU memory.")
                        torch.cuda.empty_cache()
                    continue
    
            avg_train_loss = epoch_loss / len(train_dataloader)
            total_train_loss += avg_train_loss
            logging.info(f"Epoch {epoch + 1} - Average train loss: {avg_train_loss:.4f}")
            
            torch.cuda.empty_cache()
    
        self.log_cuda_memory("End of training")
        print("Training completed")
    
        return {
            "train_loss": total_train_loss / self.config['num_train_epochs'],
            "epochs_trained": self.config['num_train_epochs']
        }

    def log_gradients(self):
        """
        Log the gradients of the model parameters.
        """
        for name, param in self.encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logging.warning(f"NaN or Inf gradient detected in encoder {name}")
                logging.debug(f"Encoder gradient norm for {name}: {grad_norm}")
    
        for name, param in self.decoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logging.warning(f"NaN or Inf gradient detected in decoder {name}")
                logging.debug(f"Decoder gradient norm for {name}: {grad_norm}")

    
    def train_step(self, batch):
        """
        Perform a single training step for the combined encoder-decoder model, including intent recognition and slot filling.
    
        This method processes a batch of data through the encoder and decoder,
        computes the loss, and performs backpropagation. It includes extensive
        error checking and logging to help diagnose training issues.
    
        Args:
            batch (dict): A dictionary containing the input batch data.
    
        Returns:
            torch.Tensor: The computed loss for this step, or None if an error occurred.
        """
        try:
            logging.info(f"Starting train_step for batch {self.current_step}")
            
            # Clean input data to avoid NaN or Inf issues
            batch = self.clean_input_data(batch)
    
            # Check embed_tokens layer for NaN or Inf values at the start
            if not self.check_embed_tokens(self.current_step):
                logging.warning(f"Issues detected in embed_tokens. Reinitializing and skipping batch {self.current_step}")
                self.init_embed_tokens()
                return None
    
            # Prepare input tensors and replace NaN values
            input_ids = torch.nan_to_num(batch['input_ids'].to(dtype=self.dtype))
            attention_mask = torch.nan_to_num(batch['attention_mask'].to(dtype=self.dtype))
            labels = torch.nan_to_num(batch['labels'].to(dtype=self.dtype))
    
            # Convert input_ids and labels to Long for embedding layers, but keep original bf16 versions
            input_ids_long = input_ids.to(dtype=torch.long)
            labels_long = labels.to(dtype=torch.long)
    
            # Log input shapes and types
            logging.debug(f"Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
            logging.debug(f"Attention mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")
            logging.debug(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
    
            # Process input through the encoder (AfroXLMR)
            with self.accelerator.autocast():
                with torch.cuda.amp.autocast(enabled=True):
                    encoder_outputs = self.encoder(input_ids=input_ids_long, attention_mask=attention_mask)
                
                    # Extract the encoder's hidden states based on the output type
                    encoder_hidden_states = self.extract_encoder_hidden_states(encoder_outputs)
                    logging.debug(f"Encoder hidden states shape: {encoder_hidden_states.shape}")
            
                    # Handle case where encoder_hidden_states has 2 dimensions
                    if encoder_hidden_states.dim() == 2:
                        batch_size, hidden_size = encoder_hidden_states.shape
                        seq_length = input_ids.size(1)
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(batch_size, seq_length, hidden_size)
                        
                    logging.debug(f"Final encoder hidden states shape: {encoder_hidden_states.shape}")
                
                    # Convert encoder input_ids to decoder vocabulary
                    decoder_input_ids = self.convert_encoder_ids_to_decoder_ids(input_ids)
                    
                    # Ensure decoder_input_ids are Long
                    decoder_input_ids = decoder_input_ids.long()
        
                    # Check for invalid decoder input IDs
                    vocab_size = self.decoder.config.vocab_size
                    if (decoder_input_ids < 0).any() or (decoder_input_ids >= vocab_size).any():
                        logging.error(f"Invalid decoder input IDs detected. Valid range: [0, {vocab_size-1}]")
                        return None
        
                    # Get decoder input embeddings
                    decoder_inputs_embeds = self.decoder.model.embed_tokens(decoder_input_ids)
            
                    # Check for NaN or Inf in decoder_inputs_embeds
                    if torch.isnan(decoder_inputs_embeds).any() or torch.isinf(decoder_inputs_embeds).any():
                        logging.error("NaN or Inf values detected in decoder_inputs_embeds")
                        problematic_ids = decoder_input_ids[torch.isnan(decoder_inputs_embeds).any(dim=-1) | torch.isinf(decoder_inputs_embeds).any(dim=-1)]
                        logging.error(f"Decoder input IDs causing NaN or Inf: {problematic_ids}")
                        logging.error(f"Embed tokens weight stats for problematic IDs: min={embed_tokens.weight[problematic_ids].min().item():.4f}, max={embed_tokens.weight[problematic_ids].max().item():.4f}, mean={embed_tokens.weight[problematic_ids].mean().item():.4f}")
                        return None
            
                    logging.debug(f"Decoder inputs embeds shape: {decoder_inputs_embeds.shape}")
                    logging.debug(f"Decoder inputs embeds stats: min={decoder_inputs_embeds.min().item():.4f}, max={decoder_inputs_embeds.max().item():.4f}, mean={decoder_inputs_embeds.mean().item():.4f}")
            
                    # Ensure both tensors have the same batch and sequence dimensions
                    batch_size, encoder_seq_length, _ = encoder_hidden_states.shape
                    decoder_batch_size, decoder_seq_length, _ = decoder_inputs_embeds.shape
            
                    # Adjust sequence lengths to match
                    min_seq_length = min(encoder_seq_length, decoder_seq_length)
                    encoder_hidden_states = encoder_hidden_states[:batch_size, :min_seq_length, :]
                    decoder_inputs_embeds = decoder_inputs_embeds[:batch_size, :min_seq_length, :]
        
                    # Update attention mask to match the new sequence length
                    attention_mask = attention_mask[:batch_size, :min_seq_length]
        
                    logging.debug(f"Adjusted encoder hidden states shape: {encoder_hidden_states.shape}")
                    logging.debug(f"Adjusted decoder inputs embeds shape: {decoder_inputs_embeds.shape}")
                    logging.debug(f"Adjusted attention mask shape: {attention_mask.shape}")
        
                    # Combine encoder hidden states with decoder input embeddings
                    combined_input = torch.cat([encoder_hidden_states, decoder_inputs_embeds], dim=-1)
                    logging.debug(f"Combined input shape: {combined_input.shape}")
            
                    # Check for NaN in combined_input
                    if torch.isnan(combined_input).any():
                        logging.error("NaN values detected in combined_input")
                        return None
                    
                    # Check if we need to update the projection layer
                    if self.projection.in_features != combined_input.size(-1):
                        logging.info(f"Updating projection layer: {self.projection.in_features} -> {combined_input.size(-1)}")
                        self.projection = nn.Linear(combined_input.size(-1), self.decoder_hidden_size)
                        # Prepare the new projection layer with Accelerator
                        self.projection = self.accelerator.prepare(self.projection)
        
                    # Ensure projection layer parameters match the input dtype
                    self.projection.to(dtype=combined_input.dtype)
            
                    # Project combined input to decoder's hidden size
                    projected_input = self.projection(combined_input)
        
                    # Check for NaN in projected_input
                    if torch.isnan(projected_input).any():
                        logging.error("NaN values detected in projected_input")
                        logging.error(f"Projection layer weights stats: min={self.projection.weight.min().item():.4f}, max={self.projection.weight.max().item():.4f}, mean={self.projection.weight.mean().item():.4f}")
                        logging.error(f"Projection layer bias stats: min={self.projection.bias.min().item():.4f}, max={self.projection.bias.max().item():.4f}, mean={self.projection.bias.mean().item():.4f}")
                        return None
            
                    # Determine the dtype of the decoder
                    decoder_dtype = next(self.decoder.parameters()).dtype
        
                    # Convert projected_input to the decoder's dtype
                    projected_input = projected_input.to(dtype=decoder_dtype)
                    
                    # Add batch normalization if it doesn't exist
                    if not hasattr(self, 'bn'):
                        self.bn = nn.BatchNorm1d(self.decoder_hidden_size).to(projected_input.device)
                        self.bn = self.accelerator.prepare(self.bn)
                    projected_input = self.bn(projected_input.transpose(1, 2)).transpose(1, 2)
            
                    # Check for NaN or Inf values in projected_input after batch normalization
                    if torch.isnan(projected_input).any() or torch.isinf(projected_input).any():
                        logging.error("NaN or Inf values detected in projected_input after normalization")
                        return None
            
                    # Process through the decoder (LLaMA 2)
                    decoder_outputs = self.decoder(
                        inputs_embeds=projected_input,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False
                    )
        
                    # Extract logits from decoder outputs
                    if isinstance(decoder_outputs, CausalLMOutput):
                        logits = decoder_outputs.logits
                    elif isinstance(decoder_outputs, dict) and 'logits' in decoder_outputs:
                        logits = decoder_outputs['logits']
                    elif hasattr(decoder_outputs, 'logits'):
                        logits = decoder_outputs.logits
                    else:
                        logging.error(f"Unexpected decoder output format: {type(decoder_outputs)}")
                        return None
            
                    logging.debug(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
            
                    # Check for NaN or Inf values in logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logging.error("NaN or Inf values detected in logits")
                        return None
                        
                    # Convert labels to decoder vocabulary
                    decoder_labels = self.convert_encoder_ids_to_decoder_ids(labels)
            
                    # Adjust decoder labels to match the new sequence length
                    decoder_labels = decoder_labels[:input_ids.size(0), :logits.size(1)]
        
                    # Ensure logits and labels have the same sequence length
                    min_length = min(logits.size(1), decoder_labels.size(1))
                    logits = logits[:, :min_length, :]
                    decoder_labels = decoder_labels[:, :min_length]
            
                    logging.debug(f"Final logits shape: {logits.shape}")
                    logging.debug(f"Final decoder labels shape: {decoder_labels.shape}")
            
                    # Clip gradients before loss computation
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                
                    # Compute the loss using custom_loss function
                    loss = self.custom_loss(logits, decoder_labels)
    
            if torch.isnan(loss) or torch.isinf(loss):
                logging.error("NaN or Inf loss detected")
                return None
    
            # Scale the loss if using gradient accumulation
            if self.config['gradient_accumulation_steps'] > 1:
                loss = loss / self.config['gradient_accumulation_steps']
                
            # Perform backpropagation
            self.accelerator.backward(loss)
    
            # Clip gradients if needed
            if self.config['max_grad_norm'] is not None:
                self.accelerator.clip_grad_norm_(self.encoder.parameters(), self.config['max_grad_norm'])
                self.accelerator.clip_grad_norm_(self.decoder.parameters(), self.config['max_grad_norm'])
    
            # Optimization step
            if (self.current_step + 1) % self.config['gradient_accumulation_steps'] == 0:
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
    
            logging.debug(f"Loss: {loss.item()}")
            
            return loss
                
        except Exception as e:
            logging.error(f"Unexpected error in train_step: {str(e)}")
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error traceback:\n{traceback.format_exc()}")
            return None

    def init_embed_tokens(self):
        """
        Initialize the weights of the embed_tokens layer.

        This method applies a normal distribution initialization to the embed_tokens weights.
        """
        embed_tokens = self.decoder.model.embed_tokens
        nn.init.normal_(embed_tokens.weight, mean=0.0, std=0.02)

    def clean_input_data(self, batch):
        """
        Clean the input data by replacing NaN and Inf values.
    
        Args:
            batch (dict): The input batch data.
    
        Returns:
            dict: The cleaned batch data.
        """
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = torch.nan_to_num(batch[key], nan=0.0, posinf=1e6, neginf=-1e6)
        return batch

    def convert_encoder_ids_to_decoder_ids(self, encoder_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs from the encoder's vocabulary to the decoder's vocabulary.
        
        This method is necessary because the encoder (AfroXLMR) and decoder (LLaMA 2)
        may have different vocabularies. It first decodes the encoder's token IDs to text,
        then re-encodes this text using the decoder's tokenizer.
        
        Args:
            encoder_ids (torch.Tensor): Tensor of token IDs in the encoder's vocabulary.
        
        Returns:
            torch.Tensor: Tensor of token IDs in the decoder's vocabulary.
        """
        # Convert encoder_ids to long integers
        encoder_ids_long = encoder_ids.long()
    
        # Convert encoder token IDs to text
        # skip_special_tokens=True to avoid special tokens in the output text
        text = self.encoder_tokenizer.batch_decode(encoder_ids_long.cpu(), skip_special_tokens=True)
    
        # Convert text to decoder token IDs
        max_length = self.decoder.config.max_position_embeddings if hasattr(self.decoder.config, 'max_position_embeddings') else None
        decoder_inputs = self.decoder_tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length
        )
        
        decoder_ids = decoder_inputs['input_ids'].to(encoder_ids.device).long()
    
        # Clip the decoder_ids to be within the valid range
        vocab_size = len(self.decoder_tokenizer)
        decoder_ids = torch.clamp(decoder_ids, min=0, max=vocab_size - 1)
        
        logging.debug(f"Encoder IDs shape: {encoder_ids.shape}, Decoder IDs shape: {decoder_ids.shape}")
        logging.debug(f"Encoder IDs min: {encoder_ids.min()}, max: {encoder_ids.max()}")
        logging.debug(f"Decoder IDs min: {decoder_ids.min()}, max: {decoder_ids.max()}")
        logging.debug(f"Decoder vocabulary size: {vocab_size}")
        
        return decoder_ids

    def custom_loss(self, logits, labels):
        """
        Compute a custom loss for the model's output.

        This function calculates the loss while handling padding tokens and checking for
        out-of-range labels. It also includes checks for NaN and Inf values.

        Args:
            logits (torch.Tensor): The model's output logits.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss, or a zero tensor if no valid tokens are found.

        Raises:
            ValueError: If labels are out of range or if NaN/Inf values are detected.
        """
        # Flatten the inputs using reshape
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        
        # Create a mask for non-padding tokens
        pad_token_id = self.decoder_tokenizer.pad_token_id if self.decoder_tokenizer.pad_token_id is not None else -100
        mask = (labels != pad_token_id).to(dtype=self.dtype)
        
        # Check for any out-of-range labels
        if (labels >= logits.size(-1)).any() or (labels < 0).any():
            raise ValueError(f"Labels out of range. Labels range: [{labels.min()}, {labels.max()}], Logits size: {logits.size(-1)}")

        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Check for NaN or Inf in log_probs
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            raise ValueError("NaN or Inf values detected in log probabilities")
        
        # Gather the log probabilities for the target tokens
        target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
        
        # Compute the loss
        losses = -target_log_probs * mask
        
        # Sum the losses and divide by the number of non-padding tokens
        total_loss = losses.sum()
        num_tokens = mask.sum()
        
        if num_tokens > 0:
            loss = total_loss / num_tokens
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN or Inf loss computed: {loss.item()}")
            return loss
        else:
            logging.warning("No valid tokens found in batch")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

    def check_embed_tokens(self, step):
        """
        Check the embed_tokens layer for NaN or Inf values.

        This method inspects the weights of the embed_tokens layer and logs any issues found.

        Args:
            step (int): The current training step.

        Returns:
            bool: True if no issues were found, False otherwise.
        """
        embed_tokens = self.decoder.model.embed_tokens
        weight = embed_tokens.weight
        is_nan = torch.isnan(weight).any()
        is_inf = torch.isinf(weight).any()

        if is_nan or is_inf:
            logging.error(f"Step {step}: NaN or Inf detected in embed_tokens weight")
            logging.error(f"NaN count: {torch.isnan(weight).sum().item()}")
            logging.error(f"Inf count: {torch.isinf(weight).sum().item()}")
            logging.error(f"Min: {weight.min().item():.4f}, Max: {weight.max().item():.4f}, Mean: {weight.mean().item():.4f}")
            return False
        return True

    def extract_encoder_hidden_states(self, encoder_outputs):
        """
        Extract hidden states from the encoder's output.
    
        This method handles different types of encoder outputs and returns
        the appropriate hidden states.
    
        Args:
            encoder_outputs: The output from the encoder model.
    
        Returns:
            torch.Tensor: The extracted hidden states.
        """
        if isinstance(encoder_outputs, SequenceClassifierOutput):
            return encoder_outputs.logits
        elif hasattr(encoder_outputs, 'last_hidden_state'):
            return encoder_outputs.last_hidden_state
        elif isinstance(encoder_outputs, tuple) and len(encoder_outputs) > 0:
            return encoder_outputs[0]
        else:
            return encoder_outputs

        
    def evaluate(self, eval_dataloader, max_batches=None, timeout=300):  # 5-minute timeout
        """
        Evaluate the model on the evaluation dataset.
        
        Args:
            eval_dataloader: DataLoader for the evaluation dataset.
            max_batches: Maximum number of batches to evaluate (optional).
            timeout: Maximum time in seconds for evaluation (default: 300).
        
        Returns:
            float: The average evaluation loss.
        """
        self.encoder.eval()
        self.decoder.eval()
        total_eval_loss = 0
        num_batches = 0
        start_time = time.time()

        # Initialize lists to store predictions and true labels for intent and slot
        self.intent_preds = []
        self.intent_labels = []
        self.slot_preds = []
        self.slot_labels = []

        try:
            # Use Accelerator's autocast context manager for mixed precision
            with torch.no_grad(), self.accelerator.autocast():
                for i, batch in enumerate(eval_dataloader):
                    if max_batches is not None and i >= max_batches:
                        logging.info(f"Reached max_batches limit of {max_batches}")
                        break
                    
                    if time.time() - start_time > timeout:
                        logging.warning(f"Evaluation timed out after {timeout} seconds")
                        break
    
                    # Ensure batch tensors are in bfloat16
                    batch = {k: v.to(dtype=self.dtype) if torch.is_floating_point(v) else v 
                             for k, v in batch.items()}
                    
                    loss, lm_logits, intent_logits, slot_logits, intent_labels, slot_labels = self.evaluate_step(batch)

                    if all(v is not None for v in [loss, intent_logits, slot_logits, intent_labels, slot_labels]):
                        # Use Accelerator's gather method to collect loss from all processes
                        gathered_losses = self.accelerator.gather(loss.repeat(self.accelerator.num_processes))
                        total_eval_loss += gathered_losses.sum().item()
                        num_batches += self.accelerator.num_processes
    
                        # Process intent predictions
                        intent_preds = torch.argmax(intent_logits, dim=1)
                        self.intent_preds.extend(intent_preds.cpu().numpy())
                        self.intent_labels.extend(intent_labels.cpu().numpy())
    
                        # Process slot predictions
                        slot_preds = torch.argmax(slot_logits, dim=2)
                        self.slot_preds.extend(slot_preds.cpu().numpy())
                        self.slot_labels.extend(slot_labels.cpu().numpy())
                    else:
                        logging.warning(f"Skipping batch {i} due to None values returned from evaluate_step")
    
                    # Log only batch number
                    if i % 10 == 0:  # Adjust this number to control logging frequency
                        logging.info(f"Evaluating: Batch {i}/{len(eval_dataloader)}")
    
                    # Optionally clear GPU memory every N batches
                    if i % 50 == 0 and i > 0:  # Adjust frequency as needed
                        self.clear_gpu_memory()
    
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            logging.error(f"Error traceback:\n{traceback.format_exc()}")
            self.clear_gpu_memory()
            return None  # Return None instead of raising to allow for graceful handling
    
        finally:
            # Compute average loss
            avg_loss = total_eval_loss / num_batches if num_batches > 0 else 0
            avg_loss = self.accelerator.reduce(torch.tensor(avg_loss, device=self.accelerator.device)).item()
            logging.info(f"Average evaluation loss: {avg_loss:.6f}")
    
            # Compute metrics
            intent_accuracy = accuracy_score(self.intent_labels, self.intent_preds)
            logging.info(f"Intent recognition accuracy: {intent_accuracy:.4f}")
    
            try:
                slot_f1 = f1_score(
                    [label for seq in self.slot_labels for label in seq], 
                    [pred for seq in self.slot_preds for pred in seq], 
                    average='weighted'
                )
                logging.info(f"Slot filling F1 score: {slot_f1:.4f}")
            except ValueError as e:
                logging.error(f"Error computing slot F1 score: {str(e)}")
                slot_f1 = None
            
            # Log some examples for debugging
            num_examples = min(5, len(self.intent_preds))
            for i in range(num_examples):
                logging.info(f"Example {i+1}:")
                logging.info(f"  True Intent: {self.intent_labels[i]}, Predicted Intent: {self.intent_preds[i]}")
                logging.info(f"  True Slots: {self.slot_labels[i]}")
                logging.info(f"  Predicted Slots: {self.slot_preds[i]}")
    
            # Log unique values for debugging
            logging.info(f"Unique true intents: {set(self.intent_labels)}")
            logging.info(f"Unique predicted intents: {set(self.intent_preds)}")
            logging.info(f"Unique true slots: {set(label for seq in self.slot_labels for label in seq)}")
            logging.info(f"Unique predicted slots: {set(pred for seq in self.slot_preds for pred in seq)}")
            
            # Clear GPU memory at the end of evaluation
            self.clear_gpu_memory()
    
        # Return a dictionary with all evaluation metrics
        metrics = {
            'eval_loss': avg_loss,
            'intent_accuracy': intent_accuracy,
            'slot_f1': slot_f1
        }
    
        return metrics


    def evaluate_step(self, batch):
        """
        Perform a single evaluation step for the combined encoder-decoder model,  including intent recognition and slot filling.
        
        Args:
            batch: A batch of evaluation data.
        
        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The total loss for this evaluation step.
                - lm_logits (torch.Tensor): Language model logits.
                - intent_logits (torch.Tensor): Intent classification logits.
                - slot_logits (torch.Tensor): Slot filling logits.
        """
        try:
            # Accelerator handles device placement, so we don't need to move tensors manually
            # Convert input tensors to bfloat16 if they are floating point
            input_ids = torch.nan_to_num(batch['input_ids']).to(self.accelerator.device)
            attention_mask = torch.nan_to_num(batch['attention_mask']).to(self.accelerator.device)
            labels = torch.nan_to_num(batch['labels']).to(self.accelerator.device)

            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                labels = labels.unsqueeze(0)

            # Log shapes for debugging
            logging.debug(f"Input IDs shape: {input_ids.shape}")
            logging.debug(f"Attention mask shape: {attention_mask.shape}")
            logging.debug(f"Labels shape: {labels.shape}")
           
            with self.accelerator.autocast():
                # Get encoder outputs (AfroXLMR)
                encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

                device = self.accelerator.device

                # Handle different types of encoder outputs
                encoder_hidden_states = self.extract_encoder_hidden_states(encoder_outputs)

                # Replace NaN/Inf values with zeros
                encoder_hidden_states = torch.nan_to_num(encoder_hidden_states, nan=0.0, posinf=0.0, neginf=0.0)

                # Handle the case where encoder_hidden_states has 2 dimensions
                if encoder_hidden_states.dim() == 2:
                    batch_size, hidden_size = encoder_hidden_states.shape
                    seq_length = input_ids.size(1)
                    encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(batch_size, seq_length, hidden_size)

                # Dynamically initialize intent_classifier during the first forward pass
                if not hasattr(self, 'intent_classifier_initialized'):
                    # logging.info(f"Initializing intent_classifier with in_features={encoder_hidden_states.size(-1)}")
                    self.intent_classifier = torch.nn.Linear(in_features=encoder_hidden_states.size(-1), 
                                                             out_features=self.intent_classifier.out_features).to(self.accelerator.device)
                    self.intent_classifier_initialized = True
                
                # Intent classification using encoder_hidden_states[:, 0, :]
                encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float, device=self.intent_classifier.weight.device)
                intent_logits = self.intent_classifier(encoder_hidden_states[:, 0, :])  # Use [CLS] token for intent
                logging.debug(f"Intent logits shape: {intent_logits.shape}")

                # Dynamically generate intent labels (predictions)
                intent_labels = torch.argmax(intent_logits, dim=-1)
                logging.debug(f"Generated intent labels shape: {intent_labels.shape}")

                logging.debug(f"Final encoder hidden states shape: {encoder_hidden_states.shape}")

                # Convert encoder input_ids to decoder vocabulary
                decoder_input_ids = self.convert_encoder_ids_to_decoder_ids(input_ids)

                # Get decoder embeddings and apply torch.nan_to_num
                decoder_inputs_embeds = torch.nan_to_num(self.decoder.model.embed_tokens(decoder_input_ids))
                
                # Ensure consistent sequence lengths across all relevant tensors
                min_seq_length = min(encoder_hidden_states.size(1), decoder_inputs_embeds.size(1), attention_mask.size(1))
                encoder_hidden_states = encoder_hidden_states[:, :min_seq_length, :]
                decoder_inputs_embeds = decoder_inputs_embeds[:, :min_seq_length, :]
                attention_mask = attention_mask[:, :min_seq_length]
                
                # Combine encoder hidden states with decoder input embeddings and apply torch.nan_to_num
                combined_input = torch.nan_to_num(torch.cat([encoder_hidden_states, decoder_inputs_embeds], dim=-1))
                combined_input = combined_input.to(dtype=torch.float)
            
                # Project combined input to decoder's hidden size and apply torch.nan_to_num
                projected_input = torch.nan_to_num(self.projection(combined_input)).to(self.accelerator.device)

                # Convert projected_input to the same dtype as the decoder
                projected_input = projected_input.to(dtype=self.dtype)
            
                # Replace NaN/Inf values with zeros in projected input
                projected_input = torch.nan_to_num(projected_input, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Process through the decoder (LLaMA 2)
                try:
                    decoder_outputs = self.decoder(
                        inputs_embeds=projected_input, 
                        attention_mask=attention_mask, 
                        output_hidden_states=True,
                        use_cache=False
                    )
                except RuntimeError as e:
                    logging.error(f"Error during decoder forward pass: {str(e)}")
                    logging.error(f"Projected input shape: {projected_input.shape}")
                    logging.error(f"Attention mask shape: {attention_mask.shape}")
                    raise
                
                # Handle different types of decoder outputs
                if isinstance(decoder_outputs, (CausalLMOutput, CausalLMOutputWithPast)):
                    device = self.accelerator.device
                    lm_logits = torch.nan_to_num(decoder_outputs.logits).to(device )
                    # Ensure hidden_states from the decoder are on the same device and dtype as the slot classifier
                    hidden_states = decoder_outputs.hidden_states[-1].to(dtype=self.dtype, device=device)

                    if not hasattr(self, 'slot_classifier_initialized'):
                        # logging.info(f"Initializing slot_classifier with in_features={hidden_states.size(-1)}")
                        self.slot_classifier = torch.nn.Linear(in_features=hidden_states.size(-1), 
                                                               out_features=self.slot_classifier.out_features).to(self.accelerator.device)
                        self.slot_classifier_initialized = True
                    # Ensure the slot classifier is on the same device as hidden_states
                    self.slot_classifier = self.slot_classifier.to(dtype=self.dtype, device=device)
                    
                    # logging.info(f"Initializing slot_classifier with in_features={hidden_states}")
                    
                    # Add debug logging
                    # logging.debug(f"Device of hidden_states: {hidden_states.device}")
                    # logging.debug(f"Device of slot_classifier: {next(self.slot_classifier.parameters()).device}")
                    # logging.debug(f"Dtype of hidden_states: {hidden_states.dtype}")
                    # logging.debug(f"Dtype of slot_classifier: {next(self.slot_classifier.parameters()).dtype}")
                
                    # Slot filling
                    try:
                        slot_logits = self.slot_classifier(hidden_states)
                                            
                        # Generate slot labels
                        slot_labels = torch.argmax(slot_logits, dim=-1)

                        logging.debug(f"slot_logits shape before view: {slot_logits.shape}")
                        logging.debug(f"slot_labels shape before view: {slot_labels.shape}")
                        
                        logging.debug(f"Slot logits shape: {slot_logits.shape}")
                    except RuntimeError as e:
                        logging.error(f"Error during slot classification: {str(e)}")
                        logging.error(f"hidden_states shape: {hidden_states.shape}")
                        logging.error(f"slot_classifier input shape: {self.slot_classifier.weight.shape[1]}")
                        raise
                else:
                    logging.error(f"Unexpected decoder output format during evaluation: {type(decoder_outputs)}")
                    return None
                
                logging.debug(f"LM logits shape: {lm_logits.shape}")
                
                # Convert labels to decoder vocabulary
                decoder_labels = self.convert_encoder_ids_to_decoder_ids(labels).to(self.accelerator.device)
                logging.debug(f"Decoder labels shape: {decoder_labels.shape}")
                
                # Adjust logits to match the sequence length of labels
                min_length = min(lm_logits.size(1), decoder_labels.size(1))
                lm_logits = lm_logits[:, :min_length, :]
                decoder_labels = decoder_labels[:, :min_length]
                slot_logits = slot_logits[:, :min_length, :]
                slot_labels = slot_labels[:, :min_length]

                if lm_logits.size(1) != decoder_labels.size(1):
                    logging.warning(f"Mismatch in sequence length after adjustment. LM_logits: {lm_logits.size(1)}, Labels: {decoder_labels.size(1)}")

                # logging.debug(f"Adjusted LM_logits shape: {lm_logits.shape}")
                # logging.debug(f"Adjusted decoder labels shape: {decoder_labels.shape}")
                # logging.debug(f"Adjusted slot logits shape: {slot_logits.shape}")
                # logging.debug(f"Adjusted slot labels shape: {slot_labels.shape}")
                
                # Compute the loss using CrossEntropyLoss for language modeling
                try:
                    # Create a mask to ignore padding in the loss computation
                    pad_token_id = self.decoder_tokenizer.pad_token_id if self.decoder_tokenizer.pad_token_id is not None else -100
                    mask = (decoder_labels != pad_token_id).to(dtype=self.dtype)

                    # Compute the loss
                    loss_fct = nn.CrossEntropyLoss(reduction='none')

                    # Ensure tensors are contiguous and use reshape instead of view
                    logits_flat = lm_logits.contiguous().reshape(-1, lm_logits.size(-1))
                    labels_flat = decoder_labels.contiguous().reshape(-1)
                    
                    # Apply torch.nan_to_num to the loss
                    lm_loss = torch.nan_to_num(loss_fct(logits_flat, labels_flat))

                    # Apply the mask to the loss and use torch.nan_to_num
                    masked_loss = torch.nan_to_num(lm_loss * mask.reshape(-1))
                    lm_loss = torch.nan_to_num(masked_loss.sum() / mask.sum() if mask.sum() > 0 else masked_loss.sum())

                    # Compute intent classification loss
                    intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_labels)

                    # Compute slot filling loss
                    slot_loss = nn.CrossEntropyLoss()(slot_logits.reshape(-1, slot_logits.size(-1)), slot_labels.reshape(-1))

                    # Combine losses
                    total_loss = lm_loss + intent_loss + slot_loss

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        logging.error("NaN or Inf loss detected")
                        return None

                    logging.debug(f"Computed LM loss: {lm_loss.item():.4f}")
                    logging.debug(f"Computed intent loss: {intent_loss.item():.4f}")
                    logging.debug(f"Computed slot loss: {slot_loss.item():.4f}")
                    logging.debug(f"Computed total loss: {total_loss.item():.4f}")
                except ValueError as ve:
                    logging.error(f"ValueError in loss computation: {str(ve)}")
                    logging.error(f"LM Logits shape: {lm_logits.shape}, Decoder labels shape: {decoder_labels.shape}")
                    logging.error(f"Intent logits shape: {intent_logits.shape}, Intent labels shape: {intent_labels.shape}")
                    logging.error(f"Slot logits shape: {slot_logits.shape}, Slot labels shape: {slot_labels.shape}")
                    return None
                
                return total_loss, lm_logits, intent_logits, slot_logits, intent_labels, slot_labels
                
        except Exception as e:
            logging.error(f"Unexpected error in evaluate_step: {str(e)}")
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error traceback:\n{traceback.format_exc()}")
            return None, None, None, None, None, None


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

    def validate_dataset(self, dataset, dataset_type=""):
        """
        Validate the dataset by checking the shapes of the first few batches, including intent and slot labels.

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
            
            # Check if all necessary keys are present
            required_keys = ['input_ids', 'attention_mask', 'labels']
            for key in required_keys:
                if key not in batch:
                    logging.warning(f"  Missing required key: {key}")
            
            # Check if shapes are consistent
            if 'input_ids' in batch and 'attention_mask' in batch:
                if batch['input_ids'].shape != batch['attention_mask'].shape:
                    logging.warning(f"  Shape mismatch: input_ids {batch['input_ids'].shape} vs attention_mask {batch['attention_mask'].shape}")
            
            if 'input_ids' in batch and 'labels' in batch:
                if batch['input_ids'].shape != batch['labels'].shape:
                    logging.warning(f"  Shape mismatch: input_ids {batch['input_ids'].shape} vs labels {batch['labels'].shape}")
            
            # Check for any unexpected keys
            unexpected_keys = set(batch.keys()) - set(required_keys)
            if unexpected_keys:
                logging.info(f"  Unexpected keys in batch: {unexpected_keys}")
            
            # Check if floating point tensors are in bfloat16
            for k, v in batch.items():
                if torch.is_floating_point(v) and v.dtype != self.dtype:
                    logging.warning(f"  {k} is not in {self.dtype} dtype. Current dtype: {v.dtype}")

        logging.info(f"Finished validating {dataset_type} dataset")

    def get_encoder_hidden_states(self, encoder_outputs):
        if isinstance(encoder_outputs, SequenceClassifierOutput):
            hidden_states = encoder_outputs.logits
        elif hasattr(encoder_outputs, 'last_hidden_state'):
            hidden_states = encoder_outputs.last_hidden_state
        else:
            hidden_states = encoder_outputs
    
        # Ensure hidden_states is 3D
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
        
        return hidden_states
    
    def generate_batch(self, dataset):
        """
        Generate text for a batch of inputs using the combined encoder-decoder model.
        
        Args:
            dataset (CustomDataset): The dataset containing input texts and other information.
        
        Returns:
            list: A list of generated texts.
        """
        self.encoder.eval()
        self.decoder.eval()
        results = []
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        # Prepare the dataloader with Accelerator
        dataloader = self.accelerator.prepare(dataloader)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating batches"):
                input_ids = batch['input_ids']
                input_texts = self.encoder_tokenizer.batch_decode(input_ids.cpu(), skip_special_tokens=True)
    
                # Generate text and predict intent and slots for each input in the batch
                for input_text in input_texts:
                    generated_text, predicted_intent, predicted_slots = self.generate(input_text)
                    results.append((generated_text, predicted_intent, predicted_slots))
    
        # If using distributed training, gather results from all processes
        if self.accelerator.num_processes > 1:
            all_results = self.accelerator.gather(results)
            return [result for sublist in all_results for result in sublist]
        else:
            return results

    def generate(self, input_text):
        """
        Generate text using the combined encoder-decoder model, including intent recognition and slot filling.

        Args:
            input_text (str): The input text to start generation from.

        Returns:
            tuple: A tuple containing:
            - str: The generated text.
            - str: The predicted intent.
            - list: The predicted slot labels.
        """
        self.encoder.eval()
        self.decoder.eval()

        # Use self.dtype instead of decoder_dtype for consistency
        dtype = self.dtype

        def replace_nan_inf(tensor):
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
            return tensor

        with torch.no_grad():
            # Tokenize input text
            inputs = self.encoder_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=self.context_window)
            input_ids = inputs['input_ids'].to(self.accelerator.device)
            attention_mask = inputs['attention_mask'].to(self.accelerator.device)

            # Use Accelerator's autocast for mixed precision
            with self.accelerator.autocast():
                # Get encoder outputs
                encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                
                # Handle SequenceClassifierOutput
                if isinstance(encoder_outputs, SequenceClassifierOutput):
                    encoder_hidden_states = encoder_outputs.logits
                else:
                    encoder_hidden_states = encoder_outputs.last_hidden_state

                encoder_hidden_states = replace_nan_inf(encoder_hidden_states)

                # Ensure encoder_hidden_states is 3D and convert to bfloat16
                if encoder_hidden_states.dim() == 2:
                    encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                encoder_hidden_states = encoder_hidden_states.to(dtype)

                # Convert encoder input_ids to decoder vocabulary
                decoder_input_ids = self.convert_encoder_ids_to_decoder_ids(input_ids)
                
                # Get decoder embeddings
                decoder_inputs_embeds = self.decoder.model.embed_tokens(decoder_input_ids)
                decoder_inputs_embeds = replace_nan_inf(decoder_inputs_embeds)

                # Ensure both tensors have the same sequence length
                seq_length = min(encoder_hidden_states.size(1), decoder_inputs_embeds.size(1))
                encoder_hidden_states = encoder_hidden_states[:, :seq_length, :]
                decoder_inputs_embeds = decoder_inputs_embeds[:, :seq_length, :]

                # Ensure the hidden sizes match before concatenation
                if encoder_hidden_states.size(-1) != self.encoder_hidden_size:
                    encoder_hidden_states = nn.functional.linear(
                        encoder_hidden_states, 
                        torch.randn(self.encoder_hidden_size, encoder_hidden_states.size(-1), device=self.accelerator.device, dtype=dtype)
                    )
                
                if decoder_inputs_embeds.size(-1) != self.decoder_hidden_size:
                    decoder_inputs_embeds = nn.functional.linear(
                        decoder_inputs_embeds, 
                        torch.randn(self.decoder_hidden_size, decoder_inputs_embeds.size(-1), device=self.accelerator.device, dtype=dtype)
                    )

                # Combine encoder hidden states with decoder input embeddings
                combined_input = torch.cat([encoder_hidden_states, decoder_inputs_embeds], dim=-1)
                combined_input = replace_nan_inf(combined_input)

                # Ensure the projection layer uses the correct dtype
                if self.projection.weight.dtype != dtype:
                    self.projection = nn.Linear(self.projection.in_features, self.projection.out_features, dtype=dtype)
                    self.projection = self.accelerator.prepare(self.projection)

                # Ensure the projection layer input size matches the combined input size
                if self.projection.in_features != combined_input.size(-1):
                    self.projection = nn.Linear(combined_input.size(-1), self.decoder_hidden_size, dtype=dtype)
                    self.projection = self.accelerator.prepare(self.projection)

                # Project combined input to decoder's hidden size
                projected_input = self.projection(combined_input)
                projected_input = replace_nan_inf(projected_input)

                # Get hidden states from the decoder
                decoder_outputs = self.decoder(
                    inputs_embeds=projected_input,
                    attention_mask=attention_mask[:, :seq_length],
                    output_hidden_states=True,
                    use_cache=False
                )

                # Get the last hidden state from the decoder
                hidden_states = decoder_outputs.hidden_states[-1]

                # Predict slots
                slot_classifier_dtype = next(self.slot_classifier.parameters()).dtype
                slot_logits = self.slot_classifier(hidden_states.to(slot_classifier_dtype))
                predicted_slots = torch.argmax(slot_logits, dim=-1)

                # Ensure predicted_slots is a 1D tensor
                if predicted_slots.dim() == 2:
                    predicted_slots = predicted_slots.squeeze(0)
                elif predicted_slots.dim() > 2:
                    predicted_slots = predicted_slots.view(-1)

                # Convert to list and then to label strings
                predicted_slots = predicted_slots.tolist()
                predicted_slots = [self.slot_labels[slot] for slot in predicted_slots]  # Convert to label strings

                # Generate from the decoder
                try:
                    def logits_processor(input_ids, scores):
                        scores = replace_nan_inf(scores)
                        # Ensure no negative probabilities
                        scores = torch.clamp(scores, min=1e-9)
                        # Re-normalize
                        scores = scores / scores.sum(dim=-1, keepdim=True)
                        return scores

                    generated_ids = self.accelerator.unwrap_model(self.decoder).generate(
                        inputs_embeds=projected_input,
                        attention_mask=attention_mask[:, :seq_length],
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        logits_processor=[logits_processor],
                    )

                    

                except RuntimeError as e:
                    print(f"Error during generation: {str(e)}")
                    print(f"Projected input stats: min={projected_input.min().item()}, max={projected_input.max().item()}, mean={projected_input.mean().item()}")
                    return "", "", []
                
                generated_text = self.decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                # Predict intent
                intent_classifier_dtype = next(self.intent_classifier.parameters()).dtype
                expected_input_size = self.intent_classifier.weight.shape[1]

                # Resize the input if it doesn't match the expected size
                if encoder_hidden_states.size(-1) != expected_input_size:
                    # logging.warning(f"Resizing encoder hidden states from {encoder_hidden_states.size(-1)} to {expected_input_size}")
                    
                    # Use a linear layer to project the encoder_hidden_states to the correct size
                    linear_projection = nn.Linear(encoder_hidden_states.size(-1), expected_input_size).to(encoder_hidden_states.device)

                    # Ensure the encoder hidden states are in the same dtype as the linear layer
                    encoder_hidden_states = encoder_hidden_states.to(linear_projection.weight.dtype)

                    encoder_hidden_states_resized = linear_projection(encoder_hidden_states)
                else:
                    encoder_hidden_states_resized = encoder_hidden_states
                
                # Now, select the correct portion of the encoder hidden states
                if encoder_hidden_states_resized.size(1) > 0:
                    intent_input = encoder_hidden_states_resized[:, 0, :]
                else:
                    intent_input = encoder_hidden_states_resized.squeeze(1)
                
                # Logging for debugging
                logging.debug(f"Encoder hidden states shape: {encoder_hidden_states.shape}")
                logging.debug(f"Intent classifier input shape: {intent_input.shape}")
                logging.debug(f"Intent classifier weight shape: {self.intent_classifier.weight.shape}")
                
                # Pass the resized hidden states to the classifier
                intent_logits = self.intent_classifier(intent_input.to(intent_classifier_dtype)).to(self.accelerator.device)
                predicted_intent = torch.argmax(intent_logits, dim=-1).item()
                predicted_intent = self.intent_labels[predicted_intent]  # Convert to label string

        return generated_text, predicted_intent, predicted_slots


    def collate_fn(self, batch):
        """
        Custom collate function to handle CustomDataset batches.

        This function combines a list of samples into a batch, ensuring that
        floating-point tensors are converted to the appropriate dtype (bfloat16).

        Args:
            batch (list): A list of samples from the dataset.

        Returns:
            dict: A dictionary containing the batched tensors.
        """
        # Combine the list of dicts into a dict of lists
        batch_dict = {key: [sample[key] for sample in batch] for key in batch[0]}
        
        # Stack the tensors
        for key, value in batch_dict.items():
            if isinstance(value[0], torch.Tensor):
                stacked_tensor = torch.stack(value)
                # Convert floating-point tensors to self.dtype (bfloat16)
                if torch.is_floating_point(stacked_tensor):
                    stacked_tensor = stacked_tensor.to(dtype=self.dtype)
                batch_dict[key] = stacked_tensor
            else:
                # For non-tensor data (if any), keep as is
                batch_dict[key] = value
        
        return batch_dict


    def print_model_size(self):
        """
        Print the size of the encoder and decoder models, including the total number of parameters
        and information about the model's precision.
        """
        encoder_size = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_size = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total_size = encoder_size + decoder_size

        logging.info(f"Encoder parameters: {encoder_size:,}")
        logging.info(f"Decoder parameters: {decoder_size:,}")
        logging.info(f"Total parameters: {total_size:,}")
        
        # Print information about model precision
        encoder_dtype = next(self.encoder.parameters()).dtype
        decoder_dtype = next(self.decoder.parameters()).dtype
        logging.info(f"Encoder dtype: {encoder_dtype}")
        logging.info(f"Decoder dtype: {decoder_dtype}")
        logging.info(f"Training dtype: {self.dtype}")

        # Print memory usage if CUDA is available
        if torch.cuda.is_available():
            logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            logging.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        # Optionally, print the model architecture
        # logging.info(f"Encoder architecture:\n{self.encoder}")
        # logging.info(f"Decoder architecture:\n{self.decoder}")

    def clear_gpu_memory(self):
        """
        Clear unused memory to prevent out-of-memory errors.
        
        This function uses Python's garbage collector and PyTorch's CUDA memory 
        cache clearing (if available) to free up memory.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.debug("GPU memory cleared")
            logging.info(f"Cleared GPU memory. Current allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    