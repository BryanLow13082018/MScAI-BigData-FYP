# trainers/gemini_trainer.py

import time
import optuna
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from tqdm import tqdm

from models.gemini import GeminiModel

class GeminiTrainer:
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the GeminiTrainer with the model and configuration.

        Args:
            model: An instance of the GeminiModel.
            config: Configuration dictionary with training parameters.
        """
        self.model = model
        self.config = config

        # Validate required configuration keys
        self._validate_config(config)

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        logging.info("GeminiTrainer initialized.")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary for required keys.
        """
        required_keys = ['google_api_token', 'training']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        genai.configure(api_key=config['google_api_token'])

    
    def run_tuning(self, training_data: List[Dict[str, str]]):
        """
        Run tuning on the Gemini model using the provided training data.
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting Gemini model tuning...")
    
            # Create Optuna study for Gemini
            study = optuna.create_study(direction="minimize")
    
            def objective(trial):
                # Define hyperparameters to optimize
                params = {
                    'batch_size': trial.suggest_int('batch_size', 4, 32),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                    'epoch_count': trial.suggest_int('epoch_count', 1, 5)
                }
    
                operation = genai.create_tuned_model(
                    source_model="models/gemini-1.5-flash-001-tuning",
                    training_data=training_data,
                    epoch_count=params['epoch_count'],
                    batch_size=params['batch_size'],
                    learning_rate=params['learning_rate'],
                    display_name=f"tuned_gemini_model_{int(time.time())}"
                )
    
                # Wait for tuning to complete
                for _ in operation.wait_bar():
                    pass
    
                result = operation.result()
                return result.tuning_task.snapshots[-1]  # Return final loss
    
            # Run optimization
            study.optimize(objective, n_trials=10)  # Adjust n_trials as needed
    
            return study  # Return the Optuna study object
    
        except Exception as e:
            logging.error(f"Error during Gemini tuning: {str(e)}")
            logging.error(traceback.format_exc())
            raise


    def evaluate(self, eval_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate the Gemini model using the provided evaluation data.
    
        Args:
            eval_data: List of dictionaries with 'text_input' and 'output'.
    
        Returns:
            dict: Dictionary with evaluation metrics such as evaluation loss.
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting model evaluation...")
            logging.info(f"Number of evaluation samples: {len(eval_data)}")
            
            # Log sample of evaluation data
            logging.info("Sample evaluation data:")
            for i, example in enumerate(eval_data[:3]):  # Show first 3 examples
                logging.info(f"Example {i+1}:")
                logging.info(f"- Input: {example['text_input'][:100]}...")
                logging.info(f"- Expected Output: {example['output'][:100]}...")
            
            total_loss = 0
            num_samples = 0
            successful_generations = 0
            failed_generations = 0
            
            for i, example in enumerate(tqdm(eval_data, desc="Evaluating")):
                input_text = example['text_input']
                expected_output = example['output']
    
                try:
                    # Generate text and measure time
                    start_time = time.time()
                    generated_text = self.model.generate_text(input_text)
                    generation_time = time.time() - start_time
                    
                    if generated_text is not None:
                        # Log occasional samples of generation results
                        if i % 50 == 0:  # Log every 50th example
                            logging.info(f"\nSample generation {i}:")
                            logging.info(f"Input: {input_text[:100]}...")
                            logging.info(f"Expected: {expected_output[:100]}...")
                            logging.info(f"Generated: {generated_text[:100]}...")
                            logging.info(f"Generation time: {generation_time:.2f}s")
    
                        sample_loss = 1 if generated_text != expected_output else 0
                        total_loss += sample_loss
                        successful_generations += 1
                    else:
                        failed_generations += 1
                    
                    num_samples += 1
                    
                except Exception as e:
                    logging.error(f"Error generating text for sample {i}: {str(e)}")
                    failed_generations += 1
                    continue
    
            # Calculate and log metrics
            eval_loss = total_loss / num_samples if num_samples > 0 else float('inf')
            success_rate = successful_generations / len(eval_data) * 100 if len(eval_data) > 0 else 0
            
            logging.info("\nEvaluation Results:")
            logging.info(f"- Total samples processed: {num_samples}")
            logging.info(f"- Successful generations: {successful_generations}")
            logging.info(f"- Failed generations: {failed_generations}")
            logging.info(f"- Success rate: {success_rate:.2f}%")
            logging.info(f"- Evaluation loss: {eval_loss:.4f}")
            logging.info("=" * 50)
    
            return {
                "eval_loss": eval_loss,
                "total_samples": num_samples,
                "successful_generations": successful_generations,
                "failed_generations": failed_generations,
                "success_rate": success_rate
            }
    
        except Exception as e:
            logging.error("=" * 50)
            logging.error(f"Error during evaluation: {str(e)}")
            logging.error("Stack trace:")
            logging.error(traceback.format_exc())
            logging.error("=" * 50)
            return {"error": str(e)}
            

    def generate_text_for_eval(self, prompt: str) -> Optional[str]:
        """
        Generate text using the Gemini model for evaluation purposes.
        
        Args:
            prompt (str): Input text to generate a response.
            
        Returns:
            Optional[str]: Generated text, or None if generation fails.
            
        Raises:
            ValueError: If prompt is empty or invalid.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
            
        try:
            # Get generation parameters from config with defaults
            generation_params = {
                'max_output_tokens': self.config.get('max_output_tokens', 128),
                'temperature': self.config.get('temperature', 0.2),
                'top_p': self.config.get('top_p', 0.8),
                'top_k': self.config.get('top_k', 40)
            }
            
            # Validate parameters
            if generation_params['temperature'] < 0 or generation_params['temperature'] > 1:
                logging.warning(f"Temperature {generation_params['temperature']} out of range [0,1], using default 0.2")
                generation_params['temperature'] = 0.2
                
            if generation_params['top_p'] < 0 or generation_params['top_p'] > 1:
                logging.warning(f"Top-p {generation_params['top_p']} out of range [0,1], using default 0.8")
                generation_params['top_p'] = 0.8
                
            # Log generation attempt
            logging.debug(f"Generating text for prompt of length {len(prompt)} with parameters: {generation_params}")
            
            # Generate text
            response = self.model.generate_text(
                prompt,
                **generation_params
            )
            
            # Log success
            if response:
                logging.debug(f"Successfully generated text of length {len(response)}")
            else:
                logging.warning("Generated text is empty")
                
            return response
            
        except Exception as e:
            logging.error(f"Text generation failed: {str(e)}")
            return None
    
    def evaluate_translation(self, datasets, evaluator, best_params):
        """
        Evaluate translation performance using the Gemini model.

        Args:
            datasets (Dict[str, pd.DataFrame]): Dictionary containing datasets, must include 'devtest' key.
            evaluator (object): An object with an 'evaluate' method to assess translation quality.
            best_params (dict): Dictionary containing hyperparameters, including:
                - 'per_device_train_batch_size' (int): Batch size for processing.

        Returns:
            dict: A dictionary containing evaluation results, including:
                - 'translations': Nested dict with translation scores for each language pair.
                - 'average_score': Overall average translation score across all language pairs.
        """
        try:
            batch_size = best_params.get('per_device_train_batch_size', 16)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            results = {'translations': {}}
            target_languages = ['swh_Latn', 'kin_Latn', 'lug_Latn']
            english_code = 'eng_Latn'

            if 'devtest' not in datasets:
                logging.error("Expected 'devtest' key not found in datasets")
                return results

            df = datasets['devtest']
            results['translations']['devtest'] = {}

            for lang in target_languages:
                eng_to_lang = df[(df['src_lang'] == english_code) & (df['tgt_lang'] == lang)]
                if eng_to_lang.empty:
                    logging.warning(f"No data found for {english_code} to {lang} translation in 'devtest'. Skipping.")
                    continue

                eng_texts = eng_to_lang['src_text'].tolist()
                lang_texts = eng_to_lang['tgt_text'].tolist()
                translations = []

                for i in range(0, len(eng_texts), batch_size):
                    batch_texts = eng_texts[i:i+batch_size]
                    if not batch_texts:
                        break
                    generated_texts = [self.generate_text_for_eval(text) for text in batch_texts]
                    translations.extend(generated_texts)

                scores = evaluator.evaluate(eng_texts[:len(translations)], translations, lang_texts[:len(translations)])
                results['translations']['devtest'][f'{english_code}_to_{lang}'] = scores

            all_scores = [score['average_score'] for score in results['translations']['devtest'].values() if 'average_score' in score]
            results['average_score'] = sum(all_scores) / len(all_scores) if all_scores else float('nan')

            return results

        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            return {'error': str(e)}

    def evaluate_zero_shot(self, zero_shot_df, zero_shot_classifier, best_params):
        """
        Evaluate the model's zero-shot classification performance.

        Args:
            zero_shot_df (pd.DataFrame): Dataset containing zero-shot evaluation data.
            zero_shot_classifier (ZeroShotClassifier): Zero-shot classifier object.
            best_params (dict): Hyperparameters for evaluation (e.g., batch size).

        Returns:
            dict: Performance metrics (accuracy, F1 score).
        """
        logging.info("Starting zero-shot evaluation...")

        src_texts = zero_shot_df['src_text'].tolist()
        tgt_texts = zero_shot_df['tgt_text'].tolist()
        candidate_labels = zero_shot_df['candidate_labels'].tolist() if 'candidate_labels' in zero_shot_df.columns else None
        batch_size = best_params.get('per_device_eval_batch_size', 16)

        zero_shot_classifier.batch_size = batch_size
        results = zero_shot_classifier.evaluate(src_texts, tgt_texts, candidate_labels)

        classification_accuracy = results.get('accuracy', 0.0)
        detailed_results = results.get('results', [])
        predicted_labels = [result['labels'][0] for result in detailed_results]
        classification_f1 = f1_score(tgt_texts, predicted_labels, average='weighted')

        return {
            'classification_accuracy': classification_accuracy,
            'classification_f1': classification_f1,
            'detailed_results': detailed_results
        }

    def evaluate_code_switch(self, code_switch_df, code_switch_classifier, best_params):
        """
        Evaluate the model's code-switch classification performance.

        Args:
            code_switch_df (pd.DataFrame): Dataset containing code-switch evaluation data.
            code_switch_classifier (CodeSwitchClassifier): Code-switch classifier object.
            best_params (dict): Evaluation parameters (e.g., batch size).

        Returns:
            dict: Performance metrics (language accuracy).
        """
        logging.info("Starting code-switch evaluation...")

        src_texts = code_switch_df['src_text'].tolist()
        flores_lang_map = {'eng_Latn': 'eng', 'swh_Latn': 'swh', 'lug_Latn': 'lug', 'kin_Latn': 'kin'}
        expected_languages = [flores_lang_map.get(lang, 'unknown') for lang in code_switch_df['tgt_lang']]
        batch_size = best_params.get('per_device_eval_batch_size', 16)

        code_switch_classifier.batch_size = batch_size
        results = code_switch_classifier.evaluate(src_texts, expected_languages)

        language_accuracy = results.get('accuracy', 0.0)
        detailed_results = results.get('results', [])

        return {
            'language_accuracy': language_accuracy,
            'detailed_results': detailed_results
        }
