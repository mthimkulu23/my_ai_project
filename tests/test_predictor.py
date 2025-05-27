import unittest
import os
from src.predictor import Predictor
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

TEST_MODEL_PATH = 'models/test_sentiment_model.pkl'
TEST_VECTORIZER_PATH = 'models/test_tfidf_vectorizer.pkl'
TEST_DATA_FILE = 'data/training_data.csv'

class TestPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up for testing: Train a small model for predictable test results.
        This runs once before all tests in this class.
        """
        print("\n--- Setting up test model for Predictor ---")
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()

        df = data_processor.load_data(TEST_DATA_FILE)
        if df is None:
            raise Exception("Test data not found for predictor tests.")

        texts = df['text']
        labels = data_processor.get_labels(df)

        X_features = data_processor.preprocess_data(texts)

        model_trainer.train_model(X_features, labels)

        os.makedirs('models', exist_ok=True)
        model_trainer.save_model(TEST_MODEL_PATH)
        data_processor.save_vectorizer(TEST_VECTORIZER_PATH)
        print("--- Test model setup complete ---")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up after tests: Remove the test model files.
        This runs once after all tests in this class.
        """
        print("\n--- Tearing down test model for Predictor ---")
        if os.path.exists(TEST_MODEL_PATH):
            os.remove(TEST_MODEL_PATH)
        if os.path.exists(TEST_VECTORIZER_PATH):
            os.remove(TEST_VECTORIZER_PATH)
        print("--- Test model teardown complete ---")

    def test_predictor_initialization(self):
        """Test that the predictor initializes correctly when model files exist."""
        predictor = Predictor(TEST_MODEL_PATH, TEST_VECTORIZER_PATH)
        self.assertIsNotNone(predictor.model_trainer.model)
        self.assertIsNotNone(predictor.data_processor.vectorizer)

    def test_predict_positive_sentiment(self):
        """Test prediction for positive sentiment."""
        predictor = Predictor(TEST_MODEL_PATH, TEST_VECTORIZER_PATH)
        sentiment = predictor.predict_sentiment("This is a great product!")
        self.assertEqual(sentiment, "positive")

    def test_predict_negative_sentiment(self):
        """Test prediction for negative sentiment."""
        predictor = Predictor(TEST_MODEL_PATH, TEST_VECTORIZER_PATH)
        sentiment = predictor.predict_sentiment("I hate this, it's terrible.")
        self.assertEqual(sentiment, "negative")

    def test_predictor_without_model(self):
        """Test predictor behavior when model files are missing."""
        temp_model_path = "models/non_existent_model.pkl"
        temp_vectorizer_path = "models/non_existent_vectorizer.pkl"
        predictor = Predictor(temp_model_path, temp_vectorizer_path)
        self.assertIsNone(predictor.model_trainer.model)
        self.assertIsNone(predictor.data_processor.vectorizer)
        result = predictor.predict_sentiment("Some text.")
        self.assertEqual(result, "Error: Model or vectorizer not loaded. Cannot make predictions.")

if __name__ == '__main__':
    unittest.main()