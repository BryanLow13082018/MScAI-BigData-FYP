import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

class AfroXLMRLarge:
    def __init__(self, model_name="Davlan/afro-xlmr-large", num_labels=2, cache_dir=None):
        """
        Initialize the AfroXLMRLarge model for sequence classification.

        Args:
            model_name (str): Pretrained model name or path.
            num_labels (int): Number of labels for the classification task.
            cache_dir (str, optional): Directory for caching model files.
        """
        print(f"Initializing AfroXLMRLarge with:")
        print(f"  model_name: {model_name}")
        print(f"  num_labels: {num_labels}")
        print(f"  cache_dir: {cache_dir}")

        self.model_name = model_name
        self.num_labels = num_labels
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
            print(f"  cache_dir: {self.cache_dir}")
            print(f"  local_files_only: {is_cached}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=is_cached
            )

            # Initialize the model
            print(f"Initializing model from {self.model_name}...")
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                cache_dir=self.cache_dir
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config,
                cache_dir=self.cache_dir,
                local_files_only=is_cached
            )

            print(f"Successfully initialized {self.model_name}")
            print(f"Model device: {next(self.model.parameters()).device}")

        except Exception as e:
            print(f"Error initializing model from {self.model_name}: {str(e)}")
            print(f"Cache directory: {self.cache_dir}")
            raise

    def get_model(self):
        """Get the underlying XLM-RoBERTa model with classification head."""
        return self.model

    def get_tokenizer(self):
        """Get the tokenizer for the XLM-RoBERTa model."""
        return self.tokenizer
