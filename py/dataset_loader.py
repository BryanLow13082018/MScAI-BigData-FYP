import os
import traceback
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache
from sklearn.model_selection import train_test_split
from collections import Counter

class DatasetLoader:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DatasetLoader.
        
        Args:
            config (Dict[str, Any]): Configuration parameters for data loading.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        dirname = os.path.dirname(__file__)

        # Set up directory paths
        self.masakhane_dir = os.path.join(dirname, os.path.abspath(config['data']['masakhane_dir']))
        self.flores_dir = os.path.join(dirname, os.path.abspath(config['data']['flores_dir']))
        self.experiments_dir = os.path.join(dirname, os.path.abspath(config['data']['experiments_dir']))

        # Log directory paths
        self.logger.info(f"Masakhane dir: {self.masakhane_dir}")
        self.logger.info(f"FLORES-200 dir: {self.flores_dir}")
        self.logger.info(f"Experiments dir: {self.experiments_dir}")

        self._check_directories()

    def _check_directories(self):
        """
        Check if all required directories exist.
        """
        for dir_name, dir_path in [
            ("Masakhane", self.masakhane_dir),
            ("FLORES-200", self.flores_dir),
            ("Experiments", self.experiments_dir)
        ]:
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

        flores_200 = self.load_flores_200_benchmark()
        if flores_200:  # Check if the dictionary is not empty
            datasets['flores_200_benchmark'] = flores_200
        else:
            self.logger.warning("FLORES-200 benchmark data is empty or failed to load")

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
                datasets[lang] = dataset
                self.logger.info(f"Successfully loaded {lang} dataset with {len(dataset)} samples")
            else:
                self.logger.warning(f"Failed to load {lang} dataset")

        if not datasets:
            self.logger.warning("No target language datasets found in Masakhane annotation_quality_corpus.")
        else:
            self.logger.info(f"Loaded {len(datasets)} datasets from Masakhane annotation_quality_corpus: {', '.join(datasets.keys())}")

        return datasets

    def load_flores_200_benchmark(self) -> Dict[str, pd.DataFrame]:
        """
        Load and prepare the FLORES-200 dataset for translation tasks.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing DataFrames for 'dev' and 'devtest' splits,
                                    with columns: src_lang, tgt_lang, src_text, tgt_text
        """
        flores_data = {'dev': [], 'devtest': []}
        target_languages = ['swh', 'kin', 'lug']  # ISO codes for Swahili, Kinyarwanda, and Luganda
        english_code = 'eng'
        
        self.logger.info(f"FLORES-200 directory: {self.flores_dir}")
        
        if not os.path.exists(self.flores_dir):
            self.logger.error(f"FLORES-200 directory does not exist: {self.flores_dir}")
            return {}

        try:
            for split in ['dev', 'devtest']:
                split_dir = os.path.join(self.flores_dir, split)
                self.logger.info(f"Checking split directory: {split_dir}")
                if not os.path.exists(split_dir):
                    self.logger.warning(f"Split directory not found: {split_dir}")
                    continue

                files = os.listdir(split_dir)
                self.logger.info(f"Files in {split} directory: {files}")
                
                eng_file = f"{english_code}_Latn.{split}"
                eng_file_path = os.path.join(split_dir, eng_file)
                self.logger.info(f"Looking for English file: {eng_file_path}")
                
                if eng_file in files:
                    self.logger.info(f"Processing English file: {eng_file_path}")
                    try:
                        with open(eng_file_path, 'r', encoding='utf-8') as f:
                            eng_lines = f.readlines()
                        self.logger.info(f"Number of lines in English file: {len(eng_lines)}")
                    except Exception as e:
                        self.logger.error(f"Error reading English file {eng_file_path}: {str(e)}")
                        continue
                    
                    for lang in target_languages:
                        lang_file = f"{lang}_Latn.{split}"
                        lang_file_path = os.path.join(split_dir, lang_file)
                        self.logger.info(f"Checking for language file: {lang_file_path}")
                        
                        if lang_file in files:
                            try:
                                with open(lang_file_path, 'r', encoding='utf-8') as f:
                                    lang_lines = f.readlines()
                                self.logger.info(f"Number of lines in {lang} file: {len(lang_lines)}")
                                
                                if len(eng_lines) != len(lang_lines):
                                    self.logger.warning(f"Mismatch in number of lines between English and {lang} files")
                                    continue
                                
                                for eng_text, lang_text in zip(eng_lines, lang_lines):
                                    flores_data[split].append({
                                        'src_lang': english_code,
                                        'tgt_lang': lang,
                                        'src_text': eng_text.strip(),
                                        'tgt_text': lang_text.strip(),
                                    })
                                    flores_data[split].append({
                                        'src_lang': lang,
                                        'tgt_lang': english_code,
                                        'src_text': lang_text.strip(),
                                        'tgt_text': eng_text.strip(),
                                    })
                            except Exception as e:
                                self.logger.error(f"Error processing {lang} file {lang_file_path}: {str(e)}")
                        else:
                            self.logger.warning(f"Language file not found: {lang_file}")
                else:
                    self.logger.warning(f"English file not found: {eng_file}")

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
            self.logger.error(traceback.format_exc())
            return {}  # Return an empty dictionary on error

    def load_experimental_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load experimental datasets (zero-shot and code-switch) from CSV files.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing 'zero_shot' and 'code_switch' DataFrames.
        """
        zero_shot_path = os.path.join(self.experiments_dir, 'zero_shot_dataset.csv')
        code_switch_path = os.path.join(self.experiments_dir, 'code_switch_dataset.csv')
        
        zero_shot_df = pd.read_csv(zero_shot_path)
        code_switch_df = pd.read_csv(code_switch_path)
        
        # Add necessary columns to match other datasets
        for df in [zero_shot_df, code_switch_df]:
            df['language'] = 'eng'  # Assume English for simplicity
            df['split'] = df.index.map(lambda x: 'zero_shot' if x < len(zero_shot_df) else 'code_switch')
        
        return {
            'zero_shot': zero_shot_df,
            'code_switch': code_switch_df
        }

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

            # Split the content into sentences
            sentences = content.strip().split('\n\n')

            # Process each sentence
            processed_data = []
            for sentence in sentences:
                tokens = sentence.strip().split('\n')
                text = ' '.join([token.split()[0] for token in tokens])
                labels = ' '.join([token.split()[-1] for token in tokens])
                processed_data.append({'text': text, 'label': labels})

            df = pd.DataFrame(processed_data)
            self.logger.info(f"Loaded {len(df)} samples from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def log_class_distribution(self, dataset, set_name):
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

    def prepare_stratified_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare datasets according to the specified strategy.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing 'train', 'eval', and 'benchmark' datasets.
        """
        datasets = self.load_datasets()
        experimental_datasets = self.load_experimental_datasets()
        self.logger.info(f"Loaded datasets: {list(datasets.keys())}")
        self.logger.info(f"Loaded experimental datasets: {list(experimental_datasets.keys())}")
        
        train_data = []
        eval_data = []
        benchmark_data = []
        
        # Process Masakhane datasets
        masakhane_languages = {'swahili': 'swh', 'kinyarwanda': 'kin', 'luganda': 'lug'}
        for lang, iso_code in masakhane_languages.items():
            if 'masakhane' in datasets and lang in datasets['masakhane']:
                data = datasets['masakhane'][lang].dropna()
                self.logger.info(f"Masakhane {lang} dataset shape: {data.shape}")
                data['language'] = iso_code
                data['split'] = 'masakhane'
                # Significantly reduce dataset size
                data = data.sample(n=min(500, len(data)), random_state=self.config['seed'])
                self.logger.info(f"Reduced Masakhane {lang} dataset shape: {data.shape}")
                train, eval_test = train_test_split(data, test_size=0.3, random_state=self.config['seed'])
                eval_, test = train_test_split(eval_test, test_size=0.5, random_state=self.config['seed'])
                train_data.append(train)
                eval_data.append(eval_)
                benchmark_data.append(test)
            else:
                self.logger.warning(f"Masakhane {lang} dataset not found")

        # Process FLORES-200 data
        if 'flores_200_benchmark' in datasets:
            flores_data = datasets['flores_200_benchmark']
            for split, df in flores_data.items():
                self.logger.info(f"FLORES-200 {split} dataset shape: {df.shape}")
                df['split'] = f'flores_200_{split}'
                # Significantly reduce dataset size
                df = df.sample(n=min(200, len(df)), random_state=self.config['seed'])
                self.logger.info(f"Reduced FLORES-200 {split} dataset shape: {df.shape}")
                benchmark_data.append(df)
        else:
            self.logger.warning("FLORES-200 benchmark data not found")

        # Add experimental datasets to benchmark data
        for name, df in experimental_datasets.items():
            self.logger.info(f"Experimental {name} dataset shape: {df.shape}")
            # Reduce dataset size
            df = df.sample(n=min(50, len(df)), random_state=self.config['seed'])
            self.logger.info(f"Reduced Experimental {name} dataset shape: {df.shape}")
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
        train = train.sample(n=min(2000, len(train)), random_state=self.config['seed'])
        eval = eval.sample(n=min(500, len(eval)), random_state=self.config['seed'])
        benchmark = benchmark.sample(n=min(500, len(benchmark)), random_state=self.config['seed'])

        self.logger.info(f"Final dataset sizes - Train: {len(train)}, Eval: {len(eval)}, Benchmark: {len(benchmark)}")

        # Log class distributions
        self.log_class_distribution(train, 'train')
        self.log_class_distribution(eval, 'eval')
        self.log_class_distribution(benchmark, 'benchmark')

        return {
            'train': train,
            'eval': eval,
            'benchmark': benchmark
        }

    def verify_data_integrity(self, datasets: Dict[str, pd.DataFrame]) -> bool:
        """
        Verify the integrity of the datasets.
        
        Args:
            datasets (Dict[str, pd.DataFrame]): A dictionary of datasets to verify.
        
        Returns:
            bool: True if all checks pass, False otherwise.
        """
        all_checks_passed = True

        for name, dataset in datasets.items():
            print(f"\nVerifying {name} dataset:")
            print(f"Dataset type: {type(dataset)}")
            print(f"Dataset shape: {dataset.shape}")
            print(f"Columns: {dataset.columns.tolist()}")
            
            # Check for missing values
            missing_values = dataset.isnull().sum()
            missing_percentage = (missing_values / len(dataset)) * 100
            if missing_values.sum() > 0:
                print(f"Warning: Missing values found in {name} dataset:")
                for col, pct in missing_percentage[missing_percentage > 0].items():
                    print(f"{col}: {pct:.2f}% missing")
                # Instead of failing, we'll just warn and continue
                print("Consider handling these missing values before proceeding.")
            else:
                print("No missing values found.")

            # Check for label consistency
            if 'label' in dataset.columns:
                non_null_labels = dataset['label'].dropna()
                if len(non_null_labels) > 0:
                    unique_labels = set(' '.join(non_null_labels.astype(str)).split())
                    print(f"Number of unique NER tags: {len(unique_labels)}")
                    print(f"Unique NER tags: {unique_labels}")
                    if len(unique_labels) < 2:
                        print(f"Warning: {name} dataset has less than 2 unique NER tags.")
                        all_checks_passed = False
                else:
                    print("Warning: All labels are null.")
                    all_checks_passed = False
            else:
                print("Warning: No 'label' column found in the dataset.")
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
            else:
                print("Warning: No 'text' column found in the dataset.")
                all_checks_passed = False

            # Check for language column
            if 'language' not in dataset.columns:
                print(f"Warning: No 'language' column found in the {name} dataset.")
                all_checks_passed = False
            else:
                unique_languages = dataset['language'].unique()
                print(f"Languages in {name} dataset: {unique_languages}")

            # Check for split column
            if 'split' in dataset.columns:
                unique_splits = dataset['split'].unique()
                print(f"Splits in {name} dataset: {unique_splits}")
            else:
                print("Warning: No 'split' column found in the dataset.")

            print(f"Sample data:\n{dataset.head()}")

        return all_checks_passed

    def print_dataset_info(self, datasets: Dict[str, pd.DataFrame]):
        """
        Print information about the datasets.
        
        Args:
            datasets (Dict[str, pd.DataFrame]): A dictionary of datasets to describe.
        """
        for name, dataset in datasets.items():
            print(f"\n--- {name.upper()} Dataset ---")
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

    def preprocess_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a dataset.
        
        Args:
            dataset (pd.DataFrame): The dataset to preprocess.
        
        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        # Remove any rows with missing values
        dataset = dataset.dropna()
        
        # Convert text to lowercase
        if 'text' in dataset.columns:
            dataset['text'] = dataset['text'].str.lower()
        
            # Remove special characters (keeping alphanumeric and spaces)
            dataset['text'] = dataset['text'].str.replace(r'[^a-z0-9\s]', '', regex=True)
        
            # Remove extra whitespace
            dataset['text'] = dataset['text'].str.strip().str.replace(r'\s+', ' ', regex=True)
        
        # Ensure labels are strings
        if 'label' in dataset.columns:
            dataset['label'] = dataset['label'].astype(str)

        return dataset

    def prepare_translation_dataset(self, flores_data: Dict[str, Dict[str, pd.DataFrame]], samples_per_pair: int = 1000) -> pd.DataFrame:
        """
        Prepare a dataset for translation tasks using FLORES-200 data for Swahili, Kinyarwanda, and Luganda.
        
        Args:
            flores_data (Dict[str, Dict[str, pd.DataFrame]]): The loaded FLORES-200 dataset.
            samples_per_pair (int): Number of samples to include for each language pair.
        
        Returns:
            pd.DataFrame: A DataFrame containing data for translation tasks.
        """
        target_languages = ['swh', 'kin', 'lug']  # ISO codes for Swahili, Kinyarwanda, and Luganda
        english_code = 'eng'
        
        translation_data = []
        for lang in target_languages:
            # English to target language
            pair_data_en_to_lang = self.get_parallel_data(flores_data, english_code, lang)
            if len(pair_data_en_to_lang) > samples_per_pair:
                pair_data_en_to_lang = pair_data_en_to_lang.sample(n=samples_per_pair, random_state=self.config['seed'])
            translation_data.append(pair_data_en_to_lang)
            
            # Target language to English
            pair_data_lang_to_en = self.get_parallel_data(flores_data, lang, english_code)
            if len(pair_data_lang_to_en) > samples_per_pair:
                pair_data_lang_to_en = pair_data_lang_to_en.sample(n=samples_per_pair, random_state=self.config['seed'])
            translation_data.append(pair_data_lang_to_en)

        return pd.concat(translation_data, ignore_index=True)

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

    def get_language_pairs(self, flores_data: Dict[str, Dict[str, pd.DataFrame]]) -> List[tuple]:
        """
        Get all possible language pairs from the FLORES-200 dataset for our target languages.
        
        Args:
            flores_data (Dict[str, Dict[str, pd.DataFrame]]): The loaded FLORES-200 dataset.
        
        Returns:
            List[tuple]: A list of tuples containing all possible language pairs.
        """
        target_languages = ['swh', 'kin', 'lug']  # ISO codes for Swahili, Kinyarwanda, and Luganda
        english_code = 'eng'
        return [(lang, english_code) for lang in target_languages] + [(english_code, lang) for lang in target_languages]

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
        language_pairs = self.get_language_pairs(flores_data)
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
        language_pairs = self.get_language_pairs(flores_data)
        evaluation_data = []
        
        for source_lang, target_lang in language_pairs:
            pair_data = self.get_parallel_data(flores_data, source_lang, target_lang, split='devtest')
            if len(pair_data) > samples_per_pair:
                pair_data = pair_data.sample(n=samples_per_pair, random_state=self.config['seed'])
            evaluation_data.append(pair_data)

        return pd.concat(evaluation_data, ignore_index=True)
    
    def validate_data(self, words, labels):
        if len(words) != len(labels):
            logging.warning(f"Mismatch in words and labels length: words={len(words)}, labels={len(labels)}")
            # Truncate to the shorter length
            min_length = min(len(words), len(labels))
            return words[:min_length], labels[:min_length]
        return words, labels