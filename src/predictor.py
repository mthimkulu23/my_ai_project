import os
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

class Predictor:
    def __init__(self, model_path, vectorizer_path):
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model_trainer.load_model(model_path)
            self.data_processor.load_vectorizer(vectorizer_path)
            print("Predictor initialized with loaded model and vectorizer.")
        else:
            print("Warning: Model or vectorizer not found. Please train the model first.")
            self.model_trainer.model = None
            self.data_processor.vectorizer = None


    def predict_sentiment(self, text):
        """Predicts the sentiment of a given text."""
        if self.model_trainer.model is None or self.data_processor.vectorizer is None:
            return "Error: Model or vectorizer not loaded. Cannot make predictions."

        text_processed = self.data_processor.vectorizer.transform([text])

        prediction = self.model_trainer.model.predict(text_processed)
        return prediction[0]