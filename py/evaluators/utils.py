import random  # Standard library for generating random numbers
import numpy as np  # Library for numerical computations
import yaml  # Library for reading YAML configuration files
import logging  # Library for logging messages
import torch  # PyTorch library for tensor computations and neural networks
from torch.utils.data import Dataset  # Base class for all datasets
from typing import Dict, Any, List  # Type hints for better code clarity
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across different libraries.
    
    This function sets the seed for the random, numpy, and PyTorch libraries
    to ensure consistent results across runs.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)  # Set seed for Python's random module
    np.random.seed(seed)  # Set seed for NumPy
    torch.manual_seed(seed)  # Set seed for PyTorch CPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs if available

def get_model(model_name: str, num_labels: int, auth_token: str = None, cache_dir: str = None):
    """
    Initialize and return a model and its tokenizer based on the given model name.
    
    Args:
        model_name (str): Name of the model to initialize.
        num_labels (int): Number of labels for classification tasks.
        auth_token (str, optional): Authentication token for accessing models.
        cache_dir (str, optional): Directory to use for caching model files.
    
    Returns:
        tuple: Initialized model and its tokenizer.
    """
    if model_name == "afro-xlmr-large":
        # Load Afro-XLMR model and tokenizer
        config = AutoConfig.from_pretrained("xlm-roberta-large", num_labels=num_labels, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large", config=config, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", cache_dir=cache_dir)
        return model.to('cuda'), tokenizer  # Move model to GPU if available
    elif model_name == "meta-llama/Llama-2-7b-hf":
        # Load LLaMA model and tokenizer
        config_file = f"models--{model_name.replace('/', '--')}/config.json"
        cache_path = os.path.join(cache_dir, config_file) if cache_dir else None
        is_cached = cache_path and os.path.exists(cache_path)

        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            device_map="auto",  # Automatically handle device placement
            torch_dtype=torch.bfloat16,  # Use bfloat16 for lower memory usage
            low_cpu_mem_usage=True,  # Optimize CPU memory usage
            local_files_only=is_cached,  # Use cached files if available
            token=auth_token,  # Use authentication token if provided
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=is_cached, token=auth_token)
        
        if not next(model.parameters()).is_cuda:  # Check if model is already on CUDA
            model = model.to('cuda')  # Move model to GPU if not already done
        return model, tokenizer
    elif model_name == "ernie-m-large":
        # Load ERNIE model and tokenizer
        config = AutoConfig.from_pretrained("ernie-m-large", num_labels=num_labels, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained("ernie-m-large", config=config, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained("ernie-m-large", cache_dir=cache_dir)
        return model.to('cuda'), tokenizer  # Move model to GPU if available
    else:
        raise ValueError(f"Unknown model type: {model_name}")  # Raise an error for unknown models

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.
    
    This function reads a YAML configuration file and returns its contents as a dictionary.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    
    Raises:
        IOError: If the config file cannot be read.
        yaml.YAMLError: If the config file is not valid YAML.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)  # Load the YAML file and return it as a dictionary
        return config
    except IOError as e:
        logging.error(f"Error reading config file: {str(e)}")  # Log file read errors
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {str(e)}")  # Log parsing errors
        raise

def get_device(device_preference: str = "auto") -> torch.device:
    """
    Get the appropriate device for torch operations.
    
    This function determines whether to use CPU or CUDA based on availability
    and user preference.
    
    Args:
        device_preference (str): Preferred device ('cpu', 'cuda', or 'auto').
    
    Returns:
        torch.device: The device to use for torch operations.
    """
    if device_preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Return 'cuda' if available, else 'cpu'
    elif device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")  # Return 'cuda' if requested and available
    else:
        return torch.device("cpu")  # Default to 'cpu'

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging based on the configuration.
    
    This function configures the logging module based on the settings
    provided in the configuration dictionary.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """
    logging.basicConfig(
        filename=config['logging']['log_file'],  # Set the log file
        level=config['logging']['log_level'],  # Set the logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define log message format
    )

def get_available_gpus() -> List[int]:
    """
    Get a list of available GPU indices.
    
    This function returns a list of indices for available CUDA GPUs.
    
    Returns:
        List[int]: List of available GPU indices.
    """
    return list(range(torch.cuda.device_count()))  # Return a list of GPU indices

def print_results_summary(results_summary, best_params):
    """
    Print a summary of the evaluation results and best hyperparameters.

    Args:
        results_summary (dict): A dictionary containing summarized results.
        best_params (dict): A dictionary of the best hyperparameters found during optimization.
    """
    print("\n===== EVALUATION RESULTS SUMMARY =====")

    if 'classification' in results_summary:
        print("\nClassification Results:")
        for dataset, metrics in results_summary['classification'].items():
            print(f"\n{dataset}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

    if 'translation' in results_summary:
        print("\nTranslation Results:")
        print(f"  FLORES-200 Average AfriCOMET Score (A to B): {results_summary['translation']['a_to_b']['average_score']:.4f}")
        print(f"  FLORES-200 Average AfriCOMET Score (B to A): {results_summary['translation']['b_to_a']['average_score']:.4f}")

    if 'generation' in results_summary:
        print("\nGeneration Results:")
        print(f"  FLORES-200 Average Perplexity: {results_summary['generation']['average_perplexity']:.4f}")

    if 'zero_shot' in results_summary:
        print("\nZero-shot Results:")
        print(f"  Accuracy: {results_summary['zero_shot']['accuracy']:.4f}")

    if 'code_switch' in results_summary:
        print("\nCode-switch Results:")
        print(f"  Accuracy: {results_summary['code_switch']['accuracy']:.4f}")

    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    print("\n======================================")

class CustomDataset(Dataset):
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
