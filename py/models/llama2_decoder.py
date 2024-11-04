import os
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

class Llama2Decoder:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", auth_token=None, cache_dir=None):
        """
        Initialize the Llama2Decoder for CPU usage.

        Args:
            model_name (str): The name of the LLaMA model to use.
            auth_token (str): The Hugging Face authentication token.
            cache_dir (str): The directory to use for caching model files.
        """
        print(f"Initializing Llama2Decoder with:")
        print(f"  model_name: {model_name}")
        print(f"  auth_token: {auth_token[:5]}...{auth_token[-5:] if auth_token else None}")
        print(f"  cache_dir: {cache_dir}")

        self.model_name = model_name
        self.auth_token = auth_token
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None

        # Set the cache directory in the environment if provided
        if self.cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = self.cache_dir
            print(f"Set cache directory to: {self.cache_dir}")

        try:
            # Construct the path to the model's config file in the cache
            config_file = f"models--{self.model_name.replace('/', '--')}/config.json"
            cache_path = os.path.join(self.cache_dir, config_file) if self.cache_dir else None
            
            # Check if the model is already in the cache
            if cache_path and os.path.exists(cache_path):
                print(f"Model {self.model_name} found in cache. Loading from cache.")
                is_cached = True
            else:
                print(f"Model {self.model_name} not found in cache. Will attempt to download.")
                is_cached = False

            # Initialize the tokenizer
            print(f"Initializing tokenizer from {self.model_name}...")
            print(f"Tokenizer parameters:")
            print(f"  pretrained_model_name_or_path: {self.model_name}")
            print(f"  use_auth_token: {self.auth_token is not None}")
            print(f"  cache_dir: {self.cache_dir}")
            print(f"  local_files_only: {is_cached}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                token=self.auth_token, 
                cache_dir=self.cache_dir, 
                local_files_only=is_cached
            )

            # Initialize the model
            print(f"Initializing model from {self.model_name}...")
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                token=self.auth_token,
                cache_dir=self.cache_dir,
                device_map="auto",  # Changed to "auto" for GPU usage
                torch_dtype=torch.float16,  # Changed to float16 for GPU
                low_cpu_mem_usage=True,
                local_files_only=is_cached
            )
            print(f"Successfully initialized {self.model_name}")
            print(f"Model device: {next(self.model.parameters()).device}")
        except Exception as e:
            print(f"Error initializing model from {self.model_name}: {str(e)}")
            print(f"Auth token used: {self.auth_token[:5]}...{self.auth_token[-5:] if self.auth_token else None}")
            print(f"Cache directory: {self.cache_dir}")
            raise

    def get_model(self):
        """Get the underlying LLaMA model."""
        return self.model

    def get_tokenizer(self):
        """Get the tokenizer for the LLaMA model."""
        return self.tokenizer