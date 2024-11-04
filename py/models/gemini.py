import time
import google.generativeai as genai
import logging
from typing import Optional

class GeminiModel:
    def __init__(self, model_name="models/gemini-1.5-flash-001-tuning"):
        """
        Initialize the Gemini model for tuning and generation.
        
        Args:
            model_name (str): The model name to be used for tuning and inference.
        """
        logging.info(f"Initializing GeminiModel with model_name: {model_name}")
        self.model_name = model_name
        self.model = None
        
        try:
            # Initialize the model during __init__
            logging.info(f"Loading model: {model_name}")
            self.model = genai.GenerativeModel(model_name=self.model_name)
            logging.info(f"Successfully loaded model: {self.model_name}")
            
        except Exception as e:
            logging.error(f"Error initializing Gemini model {self.model_name}: {str(e)}")
            raise

    def generate_text(self, prompt: str) -> Optional[str]:
        """
        Generate text using the Gemini model with safety settings and rate limiting.
        
        Args:
            prompt (str): Input prompt for text generation.
        
        Returns:
            Optional[str]: Generated text or None if generation fails.
        """
        try:
            # Add safety settings
            safety_settings = {
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
            }
            
            # Try to generate with exponential backoff
            max_retries = 3
            base_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(
                        prompt,
                        safety_settings=safety_settings,
                        generation_config={
                            "temperature": 0.1,  # Lower temperature for more focused outputs
                            "candidate_count": 1
                        }
                    )
                    return response.text if response else None
                    
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        time.sleep(delay)
                        continue
                    else:
                        raise
    
        except Exception as e:
            logging.error(f"Error in text generation: {str(e)}")
            return None
