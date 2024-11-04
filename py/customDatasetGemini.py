import logging  # Library for logging messages
import torch  # PyTorch library for tensor computations and neural networks
from torch.utils.data import Dataset  # Base class for all datasets

class CustomDatasetGemini(Dataset):
    """
    A custom dataset class for preparing data for the Gemini model.

    This class processes input text and corresponding labels, and structures them
    as text-input and output pairs suitable for the Gemini model.
    
    Attributes:
        data (pd.DataFrame): The input data containing 'text' and 'label' columns.
    """

    def __init__(self, dataframe):
        """
        Initialize the CustomDatasetGemini.

        Args:
            dataframe (pd.DataFrame): The input dataframe containing 'text' and 'label' columns.
        """
        # Ensure that required columns are present and drop rows with NaN values in 'text' or 'label'
        self.data = dataframe.dropna(subset=['text', 'label'])

        if self.data.empty:
            raise ValueError("The provided DataFrame is empty after removing rows with NaN in 'text' or 'label' columns.")

        logging.info(f"Initialized CustomDatasetGemini with {len(self.data)} samples")

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

        This method retrieves the text and label for a given index and structures them
        as input-output pairs.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            dict: A dictionary containing 'text_input' and 'output'.
        """
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']

        return {
            'text_input': text,
            'output': label  # The label is the expected output for the input text
        }

    def print_sample(self, idx=0):
        """
        Print a sample from the dataset for debugging purposes.

        Args:
            idx (int, optional): The index of the sample to print. Defaults to 0.
        """
        sample = self[idx]
        logging.info(f"Sample at index {idx}:")
        logging.info(f"Text input: {sample['text_input'][:50]}...")  # Print the first 50 characters for brevity
        logging.info(f"Output: {sample['output'][:50]}...")  # Print the first 50 characters for brevity

