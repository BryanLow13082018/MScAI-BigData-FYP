import os
import traceback
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache
from sklearn.model_selection import train_test_split
from collections import Counter
from datasets import load_dataset

# Constants for dataset size limits
MASAKHANE_SAMPLE_SIZE = 500
FLORES_SAMPLE_SIZE = 200
EXPERIMENTAL_SAMPLE_SIZE = 50
ONTONOTES_SAMPLE_SIZE = 1000 

class DatasetLoader:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DatasetLoader.

        Args:
            config (Dict[str, Any]): Configuration parameters for data loading.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._configure_logging()

        # Set up directory paths
        self.masakhane_dir = os.path.abspath(config['data']['masakhane_dir'])
        self.flores_dir = os.path.abspath(config['data']['flores_dir'])
        self.ontonotes_dir = os.path.abspath(config['data']['ontonotes_dir'])

        self._check_directories()

    def _configure_logging(self):
        """ Configure the logging settings for the application. """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger.info("Logging is configured.")
        
    def _check_directories(self):
        """ Check if all required directories exist. """
        for dir_name, dir_path in [
            ("Masakhane", self.masakhane_dir),
            ("FLORES-200", self.flores_dir),
            ("OntoNotes", self.ontonotes_dir)
        ]:
            # Log directory paths
            if not os.path.exists(dir_path):
                self.logger.warning(f"{dir_name} directory does not exist: {dir_path}")
            else:
                self.logger.info(f"{dir_name} dir: {dir_path}")

    @lru_cache(maxsize=None)
    def load_datasets(self) -> Dict[str, Any]:
        """
        Load all datasets specified in the configuration.
    
        Returns:
            Dict[str, Any]: A dictionary containing all loaded datasets.
        """
        datasets = {}
    
        masakhane = self._load_masakhane_datasets()
        if masakhane:
            datasets['masakhane'] = masakhane
    
        ontonotes = self.load_ontonotes()
        if ontonotes:
            datasets['ontonotes'] = ontonotes
        else:
            self.logger.warning("OntoNotes data is empty or failed to load")
    
        # Check for NA values in all loaded datasets
        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                if df.isnull().values.any():
                    self.logger.warning(f"{name} dataset contains NA values. Cleaning dataset...")
                    datasets[name] = self.replace_missing_values(df)
            else:
                self.logger.warning(f"{name} is not a DataFrame. Current type: {type(df)}")
    
        return datasets

    def _load_masakhane_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load Masakhane datasets for Swahili, Kinyarwanda, and Luganda from the annotation quality corpus.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of loaded Masakhane datasets.
        """
        target_languages = ['swahili', 'kinyarwanda', 'luganda']
        datasets = {}

        annotation_quality_corpus_dir = os.path.join(self.masakhane_dir, 'annotation_quality_corpus')
        self.logger.info(f"Searching for Masakhane datasets in: {annotation_quality_corpus_dir}")

        if not os.path.exists(annotation_quality_corpus_dir):
            self.logger.warning(f"annotation_quality_corpus directory does not exist at {annotation_quality_corpus_dir}")
            return {}

        for lang in target_languages:
            file_path = os.path.join(annotation_quality_corpus_dir, f'{lang}.txt')
            self.logger.info(f"Attempting to load {lang} dataset from: {file_path}")
            dataset = self._load_text_file(file_path)
            if dataset is not None:
                # Clean and validate the dataset
                dataset = self.replace_missing_values(dataset)
                dataset = self._validate_ner_tags(dataset)
                dataset['language'] = lang  # Add language information
                datasets[lang] = dataset
                self.logger.info(f"Successfully loaded {lang} dataset with {len(dataset)} samples")
            else:
                self.logger.warning(f"Failed to load {lang} dataset")

        if not datasets:
            self.logger.warning("No target language datasets found in Masakhane annotation_quality_corpus.")
        else:
            self.logger.info(f"Loaded {len(datasets)} datasets from Masakhane annotation_quality_corpus: {', '.join(datasets.keys())}")

        return datasets

    def _validate_ner_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean NER tags in the dataset.
    
        Args:
            df (pd.DataFrame): The dataset to validate.
    
        Returns:
            pd.DataFrame: The dataset with validated NER tags.
        """
        valid_ner_tags = {'O', 'B-DATE', 'I-DATE', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG'}
        
        def clean_ner_tags(tags):
            return ' '.join([tag if tag in valid_ner_tags else 'O' for tag in tags.split()])
        
        if 'label' not in df.columns:
            self.logger.warning("'label' column not found in the dataset. Skipping NER tag validation.")
            return df
        
        df['label'] = df['label'].apply(clean_ner_tags)
        return df

    def load_flores_200_benchmark(self) -> Dict[str, pd.DataFrame]:
        """
        Load and prepare the FLORES-200 dataset for translation tasks using the Hugging Face Datasets library.
    
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing DataFrames for 'devtest' splits,
                                     with columns: src_lang, tgt_lang, src_text, tgt_text.
        """
        flores_data = {'devtest': []}
        target_languages = ['swh_Latn', 'kin_Latn', 'lug_Latn']  # ISO codes with script variants for Swahili, Kinyarwanda, and Luganda
        english_code = 'eng_Latn'  # ISO code for English
    
        try:
            for lang in target_languages:
                # Construct the language pair configuration (e.g., swh_Latn-eng_Latn)
                lang_pair = f"{lang}-{english_code}"
    
                # Load the devtest split for each language pair (English ↔ target language)
                dataset = load_dataset("facebook/flores", lang_pair, split='devtest', trust_remote_code=True)
    
                self.logger.info(f"Loaded FLORES-200 devtest for {lang}-English with {len(dataset)} samples")
    
                # Define the correct column names based on the languages
                src_col = f"sentence_{english_code}"
                tgt_col = f"sentence_{lang}"
    
                # Loop through the dataset and extract both translation directions
                for example in dataset:
                    # Ensure the correct columns exist in the dataset
                    if src_col in example and tgt_col in example:
                        # English to Target Language (eng_Latn -> lang)
                        flores_data['devtest'].append({
                            'src_lang': english_code,
                            'tgt_lang': lang,
                            'src_text': example[src_col].strip(),
                            'tgt_text': example[tgt_col].strip(),
                        })
                        # Target Language to English (lang -> eng_Latn)
                        flores_data['devtest'].append({
                            'src_lang': lang,
                            'tgt_lang': english_code,
                            'src_text': example[tgt_col].strip(),
                            'tgt_text': example[src_col].strip(),
                        })
                    else:
                        self.logger.warning(f"Expected columns not found in dataset for {lang_pair}")
    
            # Convert lists to DataFrames
            result = {
                split: pd.DataFrame(data) for split, data in flores_data.items() if data
            }
    
            for split, df in result.items():
                self.logger.info(f"Loaded FLORES-200 {split} dataset with {len(df)} samples")
    
            if not result:
                self.logger.warning("No FLORES-200 data was loaded")
    
            return result
    
        except Exception as e:
            self.logger.error(f"Error loading FLORES-200 dataset: {str(e)}")
            return {}

    def load_ontonotes(self) -> Dict[str, pd.DataFrame]:
        """
        Load OntoNotes 5.0 dataset.
    
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing DataFrames for 'train', 'dev', and 'test' splits.
        """
        if not self.ontonotes_dir:
            self.logger.error("OntoNotes directory not specified in config.")
            return {}
    
        splits = ['train', 'dev', 'test']
        datasets = {}
    
        for split_name in splits:
            split_dir = os.path.join(self.ontonotes_dir, split_name)
            if not os.path.exists(split_dir):
                self.logger.warning(f"Split directory not found: {split_dir}")
                continue
    
            data = self._load_ontonotes_split(split_dir, split_name)
            if data:
                df = pd.DataFrame(data)
                if len(df) > ONTONOTES_SAMPLE_SIZE:
                    df = df.sample(n=ONTONOTES_SAMPLE_SIZE, random_state=self.config['seed'])
                datasets[split_name] = df
                self.logger.info(f"Loaded {len(df)} samples for {split_name} split")
    
        return datasets
    
    def _load_ontonotes_split(self, split_dir: str, split_name: str) -> List[Dict[str, Any]]:
        """
        Recursively load OntoNotes data from a split directory.
    
        Args:
            split_dir (str): Path to the split directory
            split_name (str): Name of the split ('train', 'dev', 'test')
    
        Returns:
            List[Dict[str, Any]]: List of processed data samples
        """
        data = []
        
        def process_file(file_path: str, category: str) -> None:
            """Process individual annotation file"""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    sentences = content.split('\n\n')
                    for sentence in sentences:
                        tokens = []
                        ner_tags = []
                        for line in sentence.split('\n'):
                            parts = line.split()
                            if len(parts) >= 11:
                                tokens.append(parts[3])
                                ner_tags.append(parts[10])
                        if tokens and ner_tags:
                            data.append({
                                'text': ' '.join(tokens),
                                'label': ' '.join(ner_tags),
                                'language': 'eng',
                                'split': split_name,
                                'category': category
                            })
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {str(e)}")
    
        # Walk through all directories and find .gold_conll files
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.endswith('.gold_conll'):
                    file_path = os.path.join(root, file)
                    # Get category from path (first directory after split)
                    rel_path = os.path.relpath(root, split_dir)
                    category = rel_path.split(os.sep)[0]
                    process_file(file_path, category)
    
        if not data:
            self.logger.warning(f"No data found in {split_dir}")
        else:
            self.logger.info(f"Loaded {len(data)} samples from {split_dir}")
            # Log distribution across categories
            category_counts = Counter(item['category'] for item in data)
            self.logger.info(f"Category distribution for {split_name}:")
            for category, count in category_counts.items():
                self.logger.info(f"  {category}: {count} samples")
    
        return data

    def load_experimental_datasets(self, flores_200: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Load the zero-shot and code-switch datasets from the existing FLORES-200 dataset.
    
        Args:
            flores_200 (Dict[str, pd.DataFrame]): The loaded FLORES-200 dataset.
    
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing 'devtest' for both zero-shot and code-switch tasks.
        """
        # Prepare the dictionary to hold datasets
        datasets = {}
    
        try:
            # Reuse FLORES-200 'devtest' split for both zero-shot and code-switch tasks
            devtest_df = flores_200['devtest']
            datasets['zero_shot'] = devtest_df
            datasets['code_switch'] = devtest_df  # Same dataset for both tasks
            self.logger.info(f"Loaded FLORES-200 'devtest' dataset with {len(devtest_df)} samples for both zero-shot and code-switch tasks.")
    
        except KeyError as e:
            self.logger.error(f"Error: 'devtest' split not found in FLORES-200 dataset: {str(e)}")
            datasets['zero_shot'] = pd.DataFrame()  # Return empty DataFrame on failure
            datasets['code_switch'] = pd.DataFrame()  # Return empty DataFrame on failure
    
        return datasets

    
    def generate_code_switched_text(self, swahili_text: str, english_text: str) -> str:
        """
        Simulates code-switching by combining parts of Swahili and English sentences.
        Example of intra-sentence mixing.
        """
        swahili_words = swahili_text.split()
        english_words = english_text.split()
        
        # Simple strategy to alternate between languages
        mixed_sentence = []
        for i in range(max(len(swahili_words), len(english_words))):
            if i % 2 == 0 and i < len(swahili_words):
                mixed_sentence.append(swahili_words[i])
            elif i < len(english_words):
                mixed_sentence.append(english_words[i])
        
        return " ".join(mixed_sentence)

    def _load_text_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a text file into a pandas DataFrame.
    
        Args:
            file_path (str): Path to the text file.
    
        Returns:
            Optional[pd.DataFrame]: Loaded data or None if file not found.
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Warning: File {file_path} does not exist.")
            return None
    
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    
            # Split the content into sentences and remove trailing line breaks
            sentences = [sentence.strip() for sentence in content.strip().split('\n\n')]
    
            # Process each sentence
            processed_data = []
            for sentence in sentences:
                tokens = sentence.strip().split('\n')
                if tokens:  # Ensure there are tokens to process
                    text = []
                    labels = []
                    for token in tokens:
                        parts = token.split()
                        if len(parts) >= 2:  # Ensure we have at least two columns
                            text.append(parts[0])
                            labels.append(parts[1])  # Use the second column for labels
                    processed_data.append({
                        'text': ' '.join(text).strip(),
                        'label': ' '.join(labels).strip()
                    })
    
            df = pd.DataFrame(processed_data)
            self.logger.info(f"Loaded {len(df)} samples from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def replace_missing_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing values in specified columns with default values.

        Args:
            dataset (pd.DataFrame): The dataset to clean.

        Returns:
            pd.DataFrame: The cleaned dataset with missing values replaced.
        """
        # Create a copy of the dataset to avoid modifying the original
        dataset = dataset.copy()
        
        # Define replacements
        replacements = {
            'text': 'unknown',
            'label': 'O',  # Assuming 'O' is a neutral label (outside any entity)
            'language': 'unknown',  # Default for missing language
        }

        # Fill missing values in the DataFrame
        for col, default in replacements.items():
            if col in dataset.columns:
                n_before = dataset[col].isnull().sum()
                # Use direct assignment instead of inplace
                dataset[col] = dataset[col].fillna(default)
                n_after = dataset[col].isnull().sum()
                if n_before > 0:
                    self.logger.info(f"Replaced {n_before} missing values in column '{col}' with '{default}'.")

        return dataset

    def compute_class_weights(self, labels: List[str]) -> Dict[str, float]:
        """
        Compute class weights to handle class imbalance for multi-label classification.

        Args:
            labels (List[str]): List of space-separated label strings.

        Returns:
            Dict[str, float]: Class weights.
        """
        if not labels:
            self.logger.warning("No labels provided for class weight computation.")
            return {}
        
        # Flatten the list of labels
        all_labels = [label for label_set in labels for label in label_set.split()]
        
        # Get unique classes
        classes = list(set(all_labels))
        
        # Count occurrences of each class
        class_counts = Counter(all_labels)
        
        # Compute weights
        total_samples = len(all_labels)
        class_weights = {cls: total_samples / (len(classes) * count) for cls, count in class_counts.items()}
        
        return class_weights


    def prepare_stratified_datasets(self) -> Dict[str, Any]:
        """
        Prepare datasets according to the specified strategy, excluding experimental datasets.

        Returns:
            Dict[str, Any]: A dictionary containing 'train', 'eval', 'benchmark', and 'experimental' datasets.
        """
        datasets = self.load_datasets()
        self.logger.info(f"Loaded datasets: {list(datasets.keys())}")

        # Prepare train, eval, and benchmark datasets
        train_data, eval_data, benchmark_data = [], [], []

        # Process Masakhane datasets
        masakhane_languages = {'swahili': 'swh', 'kinyarwanda': 'kin', 'luganda': 'lug'}
        for lang, iso_code in masakhane_languages.items():
            if 'masakhane' in datasets and lang in datasets['masakhane']:
                data = datasets['masakhane'][lang]
                self.logger.info(f"Masakhane {lang} dataset shape: {data.shape}")
                data['language'] = iso_code
                data['split'] = 'masakhane'
                data = data.sample(n=min(MASAKHANE_SAMPLE_SIZE, len(data)), random_state=self.config['seed'])
                self.logger.info(f"Reduced Masakhane {lang} dataset shape: {data.shape}")
                train, eval_test = train_test_split(data, test_size=0.3, random_state=self.config['seed'])
                eval_, test = train_test_split(eval_test, test_size=0.5, random_state=self.config['seed'])
                train_data.append(train)
                eval_data.append(eval_)
                benchmark_data.append(test)
            else:
                self.logger.warning(f"Masakhane {lang} dataset not found")

        

        # Process OntoNotes data
        if 'ontonotes' in datasets:
            ontonotes_data = datasets['ontonotes']
            for split, df in ontonotes_data.items():
                df['split'] = 'ontonotes_' + split
                if split == 'train':
                    train_data.append(df)
                elif split == 'development':
                    eval_data.append(df)
                else:  # 'test'
                    benchmark_data.append(df)

        # Combine and shuffle the data
        train = pd.concat(train_data, ignore_index=True).sample(frac=1, random_state=self.config['seed'])
        eval = pd.concat(eval_data, ignore_index=True).sample(frac=1, random_state=self.config['seed'])
        benchmark = pd.concat(benchmark_data, ignore_index=True).sample(frac=1, random_state=self.config['seed'])

        # Ensure consistent columns across all datasets
        columns_to_keep = ['text', 'label', 'language', 'split']
        for df in [train, eval, benchmark]:
            for col in columns_to_keep:
                if col not in df.columns:
                    df[col] = 'unknown'  # Add missing columns with a default value

        train = train[columns_to_keep]
        eval = eval[columns_to_keep]
        benchmark = benchmark[columns_to_keep]

        # Further reduce overall dataset sizes
        # train = train.sample(n=min(2000, len(train)), random_state=self.config['seed'])
        # eval = eval.sample(n=min(500, len(eval)), random_state=self.config['seed'])
        # benchmark = benchmark.sample(n=min(500, len(benchmark)), random_state=self.config['seed'])

        self.logger.info(f"Final dataset sizes - Train: {len(train)}, Eval: {len(eval)}, Benchmark: {len(benchmark)}")

        # Log class distributions
        self.log_class_distribution(train, 'train')
        self.log_class_distribution(eval, 'eval')
        self.log_class_distribution(benchmark, 'benchmark')

        # Compute class weights for the training dataset
        class_weights = self.compute_class_weights(train['label'].tolist())
        self.logger.info(f"Class weights computed: {class_weights}")

        # Keep experimental datasets separate
        experimental = datasets.get('experimental', {})

        return {
            'train': train,
            'eval': eval,
            'benchmark': benchmark,
            'class_weights': class_weights,
            'experimental': experimental  # Add experimental datasets separately
        }


    def log_class_distribution(self, dataset: pd.DataFrame, set_name: str):
        """
        Log the distribution of classes in a dataset.

        Args:
            dataset (pd.DataFrame): The dataset to analyze.
            set_name (str): Name of the dataset (e.g., 'train', 'eval', 'benchmark').
        """
        if 'label' in dataset.columns:
            distribution = Counter(dataset['label'])
            self.logger.info(f"Class distribution in {set_name} set:")
            for class_label, count in distribution.items():
                self.logger.info(f"  Class {class_label}: {count}")
        else:
            self.logger.warning(f"No 'label' column found in {set_name} dataset")

    def validate_data(self, dataset: pd.DataFrame, dataset_name: str) -> bool:
        """
        Verify the integrity of the dataset.
    
        Args:
            dataset (pd.DataFrame): The dataset to verify.
            dataset_name (str): Name of the dataset for logging purposes.
    
        Returns:
            bool: True if all checks pass, False otherwise.
        """
        all_checks_passed = True
    
        self.logger.info(f"\nVerifying {dataset_name} dataset:")
        self.logger.info(f"Dataset shape: {dataset.shape}")
        self.logger.info(f"Columns: {dataset.columns.tolist()}")
    
        # Check for missing values
        missing_values = dataset.isnull().sum()
        if missing_values.sum() > 0:
            self.logger.warning(f"Warning: Missing values found in {dataset_name} dataset:")
            for col, count in missing_values[missing_values > 0].items():
                self.logger.warning(f"{col}: {count} missing")
            all_checks_passed = False
    
        # Check for label consistency
        if 'label' in dataset.columns:
            unique_labels = set(dataset['label'].unique())
            self.logger.info(f"Number of unique labels: {len(unique_labels)}")
            self.logger.info(f"Unique labels: {unique_labels}")
            if len(unique_labels) < 2:
                self.logger.warning(f"Warning: {dataset_name} dataset has less than 2 unique labels.")
                all_checks_passed = False
        else:
            self.logger.warning(f"Warning: No 'label' column found in the {dataset_name} dataset.")
            all_checks_passed = False
    
        # Check for language consistency
        if 'language' in dataset.columns:
            unique_languages = dataset['language'].unique()
            self.logger.info(f"Languages in {dataset_name} dataset: {unique_languages}")
            expected_languages = {'kin', 'lug', 'swh'}
            unexpected_languages = set(unique_languages) - expected_languages
            if unexpected_languages:
                self.logger.warning(f"Unexpected languages found in {dataset_name} dataset: {unexpected_languages}")
                all_checks_passed = False
        else:
            self.logger.warning(f"Warning: No 'language' column found in the {dataset_name} dataset.")
            all_checks_passed = False
    
        # Check for split consistency
        if 'split' in dataset.columns:
            unique_splits = dataset['split'].unique()
            self.logger.info(f"Splits in {dataset_name} dataset: {unique_splits}")
            if 'zero_shot' in unique_splits and dataset_name != 'benchmark':
                self.logger.warning(f"'zero_shot' split found in {dataset_name} dataset, expected only in benchmark.")
                all_checks_passed = False
        else:
            self.logger.warning(f"Warning: No 'split' column found in the {dataset_name} dataset.")
            all_checks_passed = False
    
        self.logger.info(f"Sample data:\n{dataset.head()}")
    
        return all_checks_passed


    def print_dataset_info(self, datasets: Dict[str, Any]):
        """
        Print information about the datasets.
        
        Args:
            datasets (Dict[str, Any]): A dictionary of datasets or nested dictionaries to describe.
        """
        for name, data in datasets.items():
            print(f"\n--- {name.upper()} Dataset ---")
            if isinstance(data, pd.DataFrame):
                self._print_dataframe_info(name, data)
            elif isinstance(data, dict):
                for sub_name, sub_data in data.items():
                    if isinstance(sub_data, pd.DataFrame):
                        print(f"\n--- {name.upper()} - {sub_name.upper()} ---")
                        self._print_dataframe_info(f"{name}_{sub_name}", sub_data)
                    else:
                        print(f"{sub_name}: {type(sub_data)}")
            else:
                print(f"Unexpected type for {name}: {type(data)}")
    
    def _print_dataframe_info(self, name: str, dataset: pd.DataFrame):
        """
        Print information about a single DataFrame.
        
        Args:
            name (str): Name of the dataset.
            dataset (pd.DataFrame): DataFrame to describe.
        """
        print(f"Shape: {dataset.shape}")
        print("\nColumn Info:")
        print(dataset.info())
        print("\nSample Data:")
        print(dataset.head())
        print("\nDataset Statistics:")
        stats = self.get_dataset_statistics(dataset)
        for stat, value in stats.items():
            print(f"{stat}: {value}")

    def get_dataset_statistics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic statistics about a dataset.

        Args:
            dataset (pd.DataFrame): The dataset to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing dataset statistics.
        """
        stats = {
            'num_samples': len(dataset),
        }

        if 'text' in dataset.columns:
            stats['avg_text_length'] = dataset['text'].str.len().mean()

        if 'label' in dataset.columns:
            stats.update({
                'num_classes': dataset['label'].nunique(),
                'class_distribution': dataset['label'].value_counts().to_dict(),
            })

        if 'language' in dataset.columns:
            stats['language_distribution'] = dataset['language'].value_counts().to_dict()

        if 'split' in dataset.columns:
            stats['split_distribution'] = dataset['split'].value_counts().to_dict()

        return stats

    def inspect_dataset_structure(self, data: Any, level: Union[int, Dict] = 0) -> str:
        """
        Recursively inspect the structure of the dataset.
    
        Args:
            data: The data to inspect.
            level (Union[int, Dict]): The current nesting level or additional parameters.
    
        Returns:
            str: A string representation of the data structure.
        """
        # Handle the case where level might be passed as a dictionary
        if isinstance(level, dict):
            self.logger.warning(f"Unexpected type for 'level': dict. Using default value 0.")
            level = 0
        elif not isinstance(level, int):
            self.logger.warning(f"Unexpected type for 'level': {type(level)}. Using default value 0.")
            level = 0
    
        indent = "  " * level
        if isinstance(data, dict):
            return "\n".join([f"{indent}{k}: {self.inspect_dataset_structure(v, level+1)}" for k, v in data.items()])
        elif isinstance(data, pd.DataFrame):
            return f"DataFrame(shape={data.shape})"
        elif isinstance(data, np.float64):
            return f"numpy.float64({data})"
        else:
            return f"{type(data).__name__}({data})"
    
    def preprocess_dataset(self, data: Union[pd.DataFrame, Dict[str, Any], float, np.float64]) -> Union[pd.DataFrame, Dict[str, Any], str]:
        """
        Preprocess a dataset, a dictionary of datasets, or individual values.
        
        This method serves as the main entry point for preprocessing, handling different
        types of input and applying the appropriate preprocessing steps.

        Args:
            data (Union[pd.DataFrame, Dict[str, Any], float, np.float64]): The data to preprocess.
                Can be a pandas DataFrame, a dictionary of datasets, or a single numeric value.

        Returns:
            Union[pd.DataFrame, Dict[str, Any], str]: The preprocessed data.
                - If input is a DataFrame, returns a preprocessed DataFrame.
                - If input is a dictionary, returns a dictionary of preprocessed data.
                - If input is a numeric value, returns a formatted string.

        Raises:
            ValueError: If an unsupported data type is provided.
        """
        self.logger.debug(f"Preprocessing data of type: {type(data)}")

        if isinstance(data, pd.DataFrame):
            return self._preprocess_dataframe(data)
        elif isinstance(data, dict):
            return {key: self.preprocess_dataset(value) for key, value in data.items()}
        elif isinstance(data, (float, np.float64)):
            # Convert float to integer if it's a whole number, otherwise format to 5 decimal places
            return str(int(data)) if data.is_integer() else f"{data:.5f}"
        elif isinstance(data, (int, np.integer)):
            return str(data)
        elif isinstance(data, str):
            return data
        else:
            error_msg = f"Unsupported data type for preprocessing: {type(data)}, value: {data}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a single DataFrame.

        This method applies various preprocessing steps to a DataFrame, including
        handling missing values, converting data types, and text processing.

        Args:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        # Remove any rows with missing values
        df = df.dropna()

        # Apply preprocessing to each column
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].dtype.kind in 'fc':  # 'f' for float, 'c' for complex
                df[column] = df[column].apply(self._preprocess_value)

        # Specific processing for 'text' column
        if 'text' in df.columns:
            # Convert text to lowercase
            df['text'] = df['text'].str.lower()

            # Remove special characters (keeping alphanumeric and spaces)
            df['text'] = df['text'].str.replace(r'[^a-z0-9\s]', '', regex=True)

            # Remove extra whitespace
            df['text'] = df['text'].str.strip().str.replace(r'\s+', ' ', regex=True)

        # Ensure labels are strings
        if 'label' in df.columns:
            df['label'] = df['label'].astype(str)

        return df

    def _preprocess_value(self, value: Any) -> str:
        """
        Preprocess a single value.

        This method handles the preprocessing of individual values, converting
        them to appropriate string representations.

        Args:
            value (Any): The value to preprocess.

        Returns:
            str: The preprocessed value as a string.
        """
        if isinstance(value, (float, np.float64)):
            # Convert float to integer if it's a whole number, otherwise format to 5 decimal places
            return str(int(value)) if value.is_integer() else f"{value:.5f}"
        elif isinstance(value, (int, np.integer)):
            return str(value)
        elif isinstance(value, str):
            return value
        else:
            self.logger.warning(f"Unexpected data type in DataFrame: {type(value)}, value: {value}")
            return str(value)
    
    def get_parallel_data(self, flores_data: Dict[str, Dict[str, pd.DataFrame]], lang1: str, lang2: str, split: str = 'dev') -> pd.DataFrame:
        """
        Get parallel data for a given language pair from the FLORES-200 dataset.

        Args:
            flores_data (Dict[str, Dict[str, pd.DataFrame]]): The loaded FLORES-200 dataset.
            lang1 (str): The first language.
            lang2 (str): The second language.
            split (str): The data split to use ('dev' or 'devtest').

        Returns:
            pd.DataFrame: A DataFrame containing parallel text data for the given language pair.
        """
        if lang1 not in flores_data or lang2 not in flores_data:
            self.logger.error(f"One or both languages not found in FLORES-200 dataset: {lang1}, {lang2}")
            return pd.DataFrame()

        data1 = flores_data[lang1][split]
        data2 = flores_data[lang2][split]

        parallel_data = pd.merge(data1, data2, on='id', suffixes=(f'_{lang1}', f'_{lang2}'))
        parallel_data = parallel_data.rename(columns={f'text_{lang1}': 'source_text', f'text_{lang2}': 'target_text'})
        parallel_data['source_lang'] = lang1
        parallel_data['target_lang'] = lang2

        return parallel_data

    def get_translation_samples(self, flores_data: Dict[str, Dict[str, pd.DataFrame]], source_lang: str, target_lang: str, n_samples: int = 5) -> List[Dict[str, str]]:
        """
        Get sample translation pairs for a given language pair.

        Args:
            flores_data (Dict[str, Dict[str, pd.DataFrame]]): The loaded FLORES-200 dataset.
            source_lang (str): The source language code.
            target_lang (str): The target language code.
            n_samples (int): Number of samples to retrieve.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing sample translation pairs.
        """
        parallel_data = self.get_parallel_data(flores_data, source_lang, target_lang)
        if len(parallel_data) < n_samples:
            n_samples = len(parallel_data)
        
        samples = parallel_data.sample(n=n_samples, random_state=self.config['seed'])
        return [
            {
                'source_lang': source_lang,
                'target_lang': target_lang,
                'source_text': row['source_text'],
                'target_text': row['target_text']
            }
            for _, row in samples.iterrows()
        ]

    def print_translation_samples(self, flores_data: Dict[str, Dict[str, pd.DataFrame]], n_samples: int = 5):
        """
        Print sample translation pairs for all language pairs.

        Args:
            flores_data (Dict[str, Dict[str, pd.DataFrame]]): The loaded FLORES-200 dataset.
            n_samples (int): Number of samples to print for each language pair.
        """
        language_pairs = [(lang, 'eng') for lang in ['swh', 'kin', 'lug']] + [('eng', lang) for lang in ['swh', 'kin', 'lug']]
        for source_lang, target_lang in language_pairs:
            print(f"\nSample translations from {source_lang} to {target_lang}:")
            samples = self.get_translation_samples(flores_data, source_lang, target_lang, n_samples)
            for sample in samples:
                print(f"Source ({sample['source_lang']}): {sample['source_text']}")
                print(f"Target ({sample['target_lang']}): {sample['target_text']}")
                print()

    def prepare_translation_evaluation_data(self, flores_data: Dict[str, Dict[str, pd.DataFrame]], samples_per_pair: int = 100) -> pd.DataFrame:
        """
        Prepare a dataset for evaluating translation performance.

        Args:
            flores_data (Dict[str, Dict[str, pd.DataFrame]]): The loaded FLORES-200 dataset.
            samples_per_pair (int): Number of samples to include for each language pair.

        Returns:
            pd.DataFrame: A DataFrame containing data for translation evaluation.
        """
        language_pairs = [(lang, 'eng') for lang in ['swh', 'kin', 'lug']] + [('eng', lang) for lang in ['swh', 'kin', 'lug']]
        evaluation_data = []
        
        for source_lang, target_lang in language_pairs:
            pair_data = self.get_parallel_data(flores_data, source_lang, target_lang, split='devtest')
            
            # Ensure pair_data is not empty before sampling
            if not pair_data.empty:
                if len(pair_data) > samples_per_pair:
                    pair_data = pair_data.sample(n=samples_per_pair, random_state=self.config['seed'])
                evaluation_data.append(pair_data)
            else:
                logging.warning(f"No data found for the language pair: {source_lang} -> {target_lang}")
    
        # Concatenate the evaluation data and reset index
        if evaluation_data:
            return pd.concat(evaluation_data, ignore_index=True)
        else:
            logging.warning("No evaluation data was generated.")
            return pd.DataFrame()  # Return empty DataFrame if no data was collected

    def verify_data_integrity(self, datasets: Dict[str, Any]) -> bool:
        """
        Verify the integrity of the datasets.
        
        Args:
            datasets (Dict[str, Any]): A dictionary of datasets to verify.
        
        Returns:
            bool: True if all checks pass, False otherwise.
        """
        all_checks_passed = True

        expected_keys = ['train', 'eval', 'benchmark', 'class_weights', 'experimental']
        for key in expected_keys:
            if key not in datasets:
                print(f"Warning: Expected key '{key}' not found in datasets.")
                all_checks_passed = False

        for name, dataset in datasets.items():
            if name == 'class_weights':
                if not isinstance(dataset, dict):
                    print(f"Warning: class_weights is not a dictionary. Current type: {type(dataset)}")
                    all_checks_passed = False
                else:
                    print("Class weights validated successfully.")
                continue

            if name == 'experimental':
                if not isinstance(dataset, dict):
                    print(f"Warning: experimental is not a dictionary. Current type: {type(dataset)}")
                    all_checks_passed = False
                else:
                    print("Experimental datasets structure validated.")
                continue

            if isinstance(dataset, pd.DataFrame):
                print(f"\nVerifying {name} dataset:")
                print(f"Dataset type: {type(dataset)}")
                print(f"Dataset shape: {dataset.shape}")
                print(f"Columns: {dataset.columns.tolist()}")
                
                expected_columns = ['text', 'label', 'language', 'split']
                for col in expected_columns:
                    if col not in dataset.columns:
                        print(f"Warning: Expected column '{col}' not found in {name} dataset.")
                        all_checks_passed = False

                # Check for missing values
                missing_values = dataset.isnull().sum()
                missing_percentage = (missing_values / len(dataset)) * 100
                if missing_values.sum() > 0:
                    print(f"Warning: Missing values found in {name} dataset:")
                    for col, pct in missing_percentage[missing_percentage > 0].items():
                        print(f"{col}: {pct:.2f}% missing")
                    all_checks_passed = False

                # Check for label consistency
                if 'label' in dataset.columns:
                    non_null_labels = dataset['label'].dropna()
                    if len(non_null_labels) > 0:
                        unique_labels = set(' '.join(non_null_labels.astype(str)).split())
                        print(f"Number of unique labels: {len(unique_labels)}")
                        print(f"Unique labels: {unique_labels}")
                        if len(unique_labels) < 2:
                            print(f"Warning: {name} dataset has less than 2 unique labels.")
                            all_checks_passed = False
                    else:
                        print("Warning: All labels are null.")
                        all_checks_passed = False

                # Check for text data
                if 'text' in dataset.columns:
                    non_null_texts = dataset['text'].dropna()
                    empty_texts = (non_null_texts.astype(str).str.strip() == '').sum()
                    if empty_texts > 0:
                        print(f"Warning: {empty_texts} empty text entries found in {name} dataset.")
                        all_checks_passed = False
                    else:
                        print("Text data check passed.")

                # Check for language column
                if 'language' in dataset.columns:
                    unique_languages = dataset['language'].unique()
                    print(f"Languages in {name} dataset: {unique_languages}")

                # Check for split column
                if 'split' in dataset.columns:
                    unique_splits = dataset['split'].unique()
                    print(f"Splits in {name} dataset: {unique_splits}")

                print(f"Sample data:\n{dataset.head()}")

            else:
                print(f"Warning: {name} is not a DataFrame. Current type: {type(dataset)}")
                all_checks_passed = False

        return all_checks_passed


