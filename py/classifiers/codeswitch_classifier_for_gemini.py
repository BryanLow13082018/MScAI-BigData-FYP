from models.gemini import GeminiModel  # Ensure that GeminiModel is correctly imported

class CodeSwitchClassifierForGemini:
    """
    CodeSwitchClassifier for identifying languages in code-switched text using Gemini.

    This class uses prompt-based generation to classify the language of text
    containing potential code-switching between multiple languages.
    """

    def __init__(self, model: GeminiModel):
        """
        Initialize the CodeSwitchClassifier for Gemini.

        Args:
            model (GeminiModel): The Gemini model to use for classification.
        """
        self.model = model
        self.languages = ['eng', 'swh', 'kin', 'lug']  # Languages to classify between

    def classify_language(self, text):
        """
        Classify the language used in the text by prompting the Gemini model.

        Args:
            text (str): The text to classify.

        Returns:
            dict: Predicted language and the input text.
        """
        prompt = f"Identify the language used in the following text: '{text}'. Choose from: {', '.join(self.languages)}."
        response = self.model.generate_text(prompt).strip()
        
        return {'text': text, 'predicted_language': response}

    def evaluate(self, texts, true_languages):
        """
        Evaluate code-switched texts.

        Args:
            texts (list): List of texts to classify.
            true_languages (list): List of true languages for each text.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        results = [self.classify_language(text) for text in texts]
        
        # Calculate accuracy
        correct = sum(result['predicted_language'] == true_lang 
                      for result, true_lang in zip(results, true_languages))
        accuracy = correct / len(texts)

        return {
            'accuracy': accuracy,
            'results': results
        }
