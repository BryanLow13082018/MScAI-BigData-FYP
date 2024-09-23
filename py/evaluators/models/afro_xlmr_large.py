import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig, AutoTokenizer

class AfroXLMRLarge(nn.Module):
    def __init__(self, config):
        """
        Initialize the AfroXLMRLarge model.
        
        This model is based on XLM-RoBERTa and is fine-tuned for African language tasks.
        
        Args:
            num_labels (int): Number of labels for the classification task.
        """
        super(AfroXLMRLarge, self).__init__()
        
        self.num_labels = config.num_labels
        
        # Load the pretrained XLM-RoBERTa large model
        self.model = XLMRobertaModel(config)
        
        # Add a dropout layer for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Add a linear layer for classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Load the tokenizer associated with the model
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type IDs.
            position_ids (torch.Tensor): Position IDs.
            head_mask (torch.Tensor): Head mask for transformer.
            inputs_embeds (torch.Tensor): Input embeddings.
            labels (torch.Tensor, optional): Ground truth labels.
        
        Returns:
            dict: Contains 'loss' and 'logits' if labels are provided, otherwise just 'logits'.
        """
        # Pass input through the base XLM-RoBERTa model
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        
        # Get the pooled output (CLS token representation)
        pooled_output = outputs[1]
        
        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)
        
        # Pass through the classification layer
        logits = self.classifier(pooled_output)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

    def get_tokenizer(self):
        """
        Get the tokenizer associated with this model.
        
        Returns:
            AutoTokenizer: The tokenizer instance.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pre-trained model.
        
        This method allows loading a pre-trained model with a custom classification head.
        
        Args:
            pretrained_model_name_or_path (str): Path or identifier of the pre-trained model.
            *model_args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            AfroXLMRLarge: An instance of the model with pre-trained weights.
        """
        config = kwargs.pop("config", None)
        if config is None:
            config = XLMRobertaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        model = cls(config.num_labels)
        model.model = XLMRobertaModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.classifier = nn.Linear(config.hidden_size, config.num_labels)
        return model

    def save_pretrained(self, save_directory):
        """
        Save the model to a directory.

        This method saves both the base model and the classification head.

        Args:
            save_directory (str): The directory to save the model to.
        """
        self.model.save_pretrained(save_directory)
        torch.save(self.classifier.state_dict(), f"{save_directory}/classifier.pt")
        self.tokenizer.save_pretrained(save_directory)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Prepare inputs for text generation tasks.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Prepared inputs for the model.
        """
        return {"input_ids": input_ids}

    def get_output_embeddings(self):
        """
        Get the output embeddings layer.

        Returns:
            torch.nn.Linear: The output embeddings layer (classifier in this case).
        """
        return self.classifier

    def resize_token_embeddings(self, new_num_tokens):
        """
        Resize token embeddings.
        
        This method is useful when adding new tokens to the vocabulary.

        Args:
            new_num_tokens (int): The new number of tokens in the embedding matrix.

        Returns:
            torch.nn.Embedding: The resized token embedding matrix.
        """
        self.model.resize_token_embeddings(new_num_tokens)
        return self.model.embeddings.word_embeddings