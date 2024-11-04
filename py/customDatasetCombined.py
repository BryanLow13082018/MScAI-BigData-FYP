import logging  # Library for logging messages
import torch  # PyTorch library for tensor computations and neural networks
from torch.utils.data import Dataset  # Base class for all datasets

class CustomDatasetCombined(Dataset):
    """
    A custom dataset class for handling text data with labels.

    This class processes text and label data, tokenizes the text, and prepares
    the data for model input. It handles potential mismatches between text and
    label lengths and provides a mapping between text labels and numerical IDs.

    Attributes:
        data (pd.DataFrame): The input data containing text and labels.
        tokenizer: The tokenizer used to process the text data.
        max_length (int): The maximum length for padding/truncating sequences.
        model_type (str): The type of model being used (e.g., 'encoder_decoder').
        label_to_id (dict): A mapping from text labels to numerical IDs.
        id_to_label (dict): A mapping from numerical IDs back to text labels.
    """

    def __init__(self, dataframe, tokenizer, model_type, max_length=128):
        """
        Initialize the CustomDataset.

        Args:
            dataframe (pd.DataFrame): The input dataframe containing 'text' and 'label' columns.
            tokenizer: The tokenizer to use for encoding the text.
            model_type (str): The type of model being used.
            max_length (int, optional): Maximum length for padding/truncating sequences. Defaults to 128.
        """
        self.data = dataframe.dropna()  # Remove rows with NaN values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type

        # Create a mapping from string labels to integer indices
        unique_labels = set(' '.join(self.data['label'].dropna()).split())
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        logging.info(f"Initialized CustomDataset with {len(self.data)} samples")
        logging.info(f"Number of unique labels: {len(self.label_to_id)}")
        logging.info(f"Model type: {self.model_type}")

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        This method retrieves the text and label for a given index, tokenizes the text,
        and prepares the label sequence. It ensures that the label sequence matches the
        length of the tokenized input.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            dict: A dictionary containing the tokenized inputs and labels.
        """
        # Retrieve the text and label
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']

        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'  # Return PyTorch tensors
        )

        # Convert labels to label IDs
        label_ids = [self.label_to_id.get(l, self.label_to_id['O']) for l in label.split()]
        
        # Ensure label_ids match the length of input_ids
        if len(label_ids) < self.max_length:
            label_ids = label_ids + [self.label_to_id['O']] * (self.max_length - len(label_ids))
        else:
            label_ids = label_ids[:self.max_length]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

    def validate_data(self, words, labels):
        """
        Validate and align the words and labels.

        This method checks if the number of words matches the number of labels.
        If there's a mismatch, it truncates to the shorter length.

        Args:
            words (list): List of words from the text.
            labels (list): List of labels corresponding to the words.

        Returns:
            tuple: A tuple containing the validated (and possibly truncated) words and labels.
        """
        if len(words) != len(labels):
            logging.warning(f"Mismatch in words and labels length: words={len(words)}, labels={len(labels)}")
            # Truncate to the shorter length
            min_length = min(len(words), len(labels))
            return words[:min_length], labels[:min_length]
        return words, labels

    def get_labels(self):
        """
        Get the list of unique labels in the dataset.

        Returns:
            list: A list of all unique labels.
        """
        return list(self.label_to_id.keys())

    def get_num_labels(self):
        """
        Get the number of unique labels in the dataset.

        Returns:
            int: The number of unique labels.
        """
        return len(self.label_to_id)

    def print_sample(self, idx=0):
        """
        Print a sample from the dataset for debugging purposes.

        Args:
            idx (int, optional): The index of the sample to print. Defaults to 0.
        """
        sample = self[idx]
        logging.info(f"Sample at index {idx}:")
        logging.info(f"Input IDs shape: {sample['input_ids'].shape}")
        logging.info(f"Attention mask shape: {sample['attention_mask'].shape}")
        logging.info(f"Labels shape: {sample['labels'].shape}")
        logging.info(f"First few tokens: {sample['input_ids'][:10]}")
        logging.info(f"First few labels: {sample['labels'][:10]}")