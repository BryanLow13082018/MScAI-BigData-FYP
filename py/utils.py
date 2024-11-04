import random  # Standard library for generating random numbers
import numpy as np  # Library for numerical computations
import yaml  # Library for reading YAML configuration files
import logging  # Library for logging messages
import torch  # PyTorch library for tensor computations and neural networks
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






