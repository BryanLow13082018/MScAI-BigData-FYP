import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CodeSwitchClassifier:
    """
    CodeSwitchClassifier for identifying languages in code-switched text.

    This class provides functionality to classify the language of text
    that may contain code-switching between multiple languages.
    """

    def __init__(self, encoder, decoder, tokenizer):
        """
        Initialize the CodeSwitchClassifier.
        
        Args:
            model: The model to use for classification (combined encoder-decoder model).
            tokenizer: The tokenizer for the model.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.languages = ['eng', 'swh', 'kin', 'lug']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def evaluate(self, texts, true_languages):
        """
        Evaluate code-switched texts.
        
        Args:
            texts (list): List of code-switched texts.
            true_languages (list): List of true languages for each text.
        
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        self.model.eval()
        results = self.classify(texts)
        
        # Calculate accuracy
        correct = sum(result['predicted_language'] == true_lang 
                      for result, true_lang in zip(results, true_languages))
        accuracy = correct / len(texts)

        return {
            'accuracy': accuracy,
            'results': results  # Include detailed results for further analysis if needed
        }

    def classify(self, texts):
        """
        Classify the language of each text in the input list.

        Args:
            texts (list): List of texts to classify.

        Returns:
            list: List of dictionaries containing classification results for each text.
        """
        dataset = CodeSwitchDataset(texts)
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=self.collate_fn)

        results = []
        with torch.no_grad():
            for batch in dataloader:
                batch_results = self._classify_batch(batch)
                results.extend(batch_results)

        return results

    def _classify_batch(self, batch):
        """
        Classify a batch of texts.

        Args:
            batch (dict): A dictionary containing the batch data.

        Returns:
            list: List of dictionaries containing classification results for the batch.
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        batch_results = []
        for i, text in enumerate(batch['texts']):
            text_hidden_states = encoder_hidden_states[i].unsqueeze(0)
            lang_scores = self._compute_language_scores(text_hidden_states)
            
            predicted_lang = max(lang_scores, key=lang_scores.get)
            
            batch_results.append({
                'text': text,
                'predicted_language': predicted_lang,
                'language_scores': lang_scores
            })

        return batch_results

    def _compute_language_scores(self, text_hidden_states):
        """
        Compute scores for each language for the given text representation.

        Args:
            text_hidden_states (torch.Tensor): Hidden states of the input text.

        Returns:
            dict: A dictionary of language scores.
        """
        language_scores = {}
        for lang in self.languages:
            lang_token = f"<{lang}>"
            lang_ids = self.tokenizer.encode(lang_token, return_tensors='pt').to(self.device)
            decoder_outputs = self.decoder(input_ids=lang_ids, encoder_hidden_states=text_hidden_states)
            logits = decoder_outputs.logits

            score = F.softmax(logits[:, -1, :], dim=-1).max().item()
            language_scores[lang] = score

        return language_scores

    def collate_fn(self, batch):
        """
        Collate function for batching data.

        Args:
            batch (list): List of dictionaries containing text data.

        Returns:
            dict: A dictionary of batched data.
        """
        texts = [item['text'] for item in batch]
        
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'texts': texts
        }

class CodeSwitchDataset(Dataset):
    """
    Dataset class for code-switched texts.
    """

    def __init__(self, texts):
        """
        Initialize the CodeSwitchDataset.

        Args:
            texts (list): List of texts.
        """
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx]}
