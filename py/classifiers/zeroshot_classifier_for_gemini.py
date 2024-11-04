from models.gemini import GeminiModel 
class ZeroShotClassifierForGemini:
    """
    ZeroShotClassifier for classifying texts into categories using Gemini.

    This classifier uses prompt-based generation to classify texts into arbitrary categories.
    """

    def __init__(self, model: GeminiModel):
        """
        Initialize the ZeroShotClassifier for Gemini.

        Args:
            model (GeminiModel): The Gemini model to use for zero-shot classification.
        """
        self.model = model

    def classify(self, text, candidate_labels):
        """
        Perform zero-shot classification by prompting Gemini to choose the best label.

        Args:
            text (str): The text to classify.
            candidate_labels (list): List of candidate labels for classification.

        Returns:
            dict: Classification result with the top label.
        """
        prompt = f"Classify the text: '{text}' into one of these categories: {', '.join(candidate_labels)}."
        response = self.model.generate_text(prompt).strip()
        
        return {'text': text, 'predicted_label': response}

    def evaluate(self, texts, true_labels, candidate_labels):
        """
        Evaluate texts using zero-shot classification.

        Args:
            texts (list): List of texts to classify.
            true_labels (list): List of true labels for each text.
            candidate_labels (list): List of candidate labels for classification.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        results = [self.classify(text, candidate_labels) for text in texts]
        
        # Calculate accuracy
        correct = sum(result['predicted_label'] == true_label 
                      for result, true_label in zip(results, true_labels))
        accuracy = correct / len(texts)

        return {
            'accuracy': accuracy,
            'results': results
        }
