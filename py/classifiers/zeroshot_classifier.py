import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ZeroShotClassifier:
    """
    ZeroShotClassifier for classifying texts into categories without specific training.

    This class provides functionality to classify texts into arbitrary categories
    that the model hasn't been explicitly trained on.
    """

    def __init__(self, encoder, decoder, tokenizer, batch_size=8):
        """
        Initialize the ZeroShotClassifier.
        
        Args:
            model: The model to use for classification (combined encoder-decoder model).
            tokenizer: The tokenizer for the model.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.batch_size = batch_size  # Set the default batch size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def evaluate(self, texts, true_labels, candidate_labels):
        """
        Evaluate texts using zero-shot classification.
        
        Args:
            texts (list): List of texts to classify.
            true_labels (list): List of true labels for each text.
            candidate_labels (list): List of candidate labels for classification.
        
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        self.encoder.eval()  
        self.decoder.eval()
        results = self.classify(texts, candidate_labels)
        
        # Calculate accuracy
        correct = sum(result['labels'][0] == true_label 
                      for result, true_label in zip(results, true_labels))
        accuracy = correct / len(texts)

        return {
            'accuracy': accuracy,
            'results': results  # Include detailed results for further analysis if needed
        }

    def classify(self, texts, candidate_labels):
        """
        Classify the texts into the given candidate labels.

        Args:
            texts (list): List of texts to classify.
            candidate_labels (list): List of candidate labels for classification.

        Returns:
            list: List of dictionaries containing classification results for each text.
        """
        dataset = ZeroShotDataset(texts, candidate_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

        results = []
        with torch.no_grad():
            for batch in dataloader:
                batch_results = self._classify_batch(batch)
                results.extend(batch_results)

        return results

    def _classify_batch(self, batch):
        """
        Classify a batch of texts using the encoder-decoder combined model.
    
        Args:
            batch (dict): A dictionary containing the batch data.
    
        Returns:
            list: List of dictionaries containing classification results for the batch.
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        texts = batch['texts']
        candidate_labels = batch['candidate_labels']
    
        # Forward pass through the encoder to get hidden states or logits
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
        # Determine the structure of the encoder output
        if hasattr(encoder_outputs, 'hidden_states'):
            # Use the last hidden state if available
            encoder_hidden_states = encoder_outputs.hidden_states[-1]
        elif hasattr(encoder_outputs, 'logits'):
            # If `logits` are present, use them
            encoder_hidden_states = encoder_outputs.logits
        else:
            raise ValueError("Unexpected encoder output format. Ensure the encoder returns either `hidden_states` or `logits`.")
    
        batch_results = []
        for i, (text, labels) in enumerate(zip(texts, candidate_labels)):
            # Take the hidden states or logits corresponding to the current batch (shape: (1, sequence_length, hidden_size))
            text_hidden_states = encoder_hidden_states[i].unsqueeze(0)  
            
            # Compute label scores using the decoder
            label_scores = self._compute_label_scores(text_hidden_states, labels)
    
            # Sort the scores to get the top labels
            sorted_scores, sorted_indices = torch.sort(label_scores, descending=True)
            top_labels = [labels[idx] for idx in sorted_indices[:3]]
            top_scores = sorted_scores[:3].tolist()
    
            # Append results
            batch_results.append({
                'text': text,
                'labels': top_labels,
                'scores': top_scores
            })
    
        return batch_results


    def _compute_label_scores(self, text_hidden_states, candidate_labels):
        """
        Compute scores for each candidate label for the given text representation.
    
        Args:
            text_hidden_states (torch.Tensor): Hidden states of the input text.
            candidate_labels (list): List of candidate labels.
    
        Returns:
            torch.Tensor: A tensor of scores for each label.
        """
        label_scores = []
    
        for label in candidate_labels:
            # Encode the label using the tokenizer
            label_ids = self.tokenizer.encode(label, return_tensors='pt').to(self.device)  # Encode as plain text
            decoder_outputs = self.decoder(input_ids=label_ids, encoder_hidden_states=text_hidden_states)
            logits = decoder_outputs.logits
    
            # Use softmax and take the maximum score from the logits
            score = F.softmax(logits[:, -1, :], dim=-1).max().item()
            label_scores.append(score)
    
        return torch.tensor(label_scores, device=self.device)


    def collate_fn(self, batch):
        """
        Collate function for batching data.

        Args:
            batch (list): List of dictionaries containing text and label data.

        Returns:
            dict: A dictionary of batched data.
        """
        texts = [item['text'] for item in batch]
        candidate_labels = [item['candidate_labels'] for item in batch]
        
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'texts': texts,
            'candidate_labels': candidate_labels
        }

class ZeroShotDataset(Dataset):
    """
    Dataset class for zero-shot classification tasks.
    """

    def __init__(self, texts, candidate_labels):
        """
        Initialize the ZeroShotDataset.

        Args:
            texts (list): List of texts.
            candidate_labels (list): List of candidate labels for each text.
        """
        self.texts = texts
        self.candidate_labels = candidate_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'candidate_labels': self.candidate_labels
        }
