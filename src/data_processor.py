import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class DataProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def load_data(self, file_path):
        """Loads data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            print(f"Error: Data file not found at {file_path}")
            return None

    def preprocess_data(self, texts):
        """Transforms text data into numerical features using TF-IDF."""
        if not hasattr(self.vectorizer, 'idf_'):
            self.vectorizer.fit(texts)
        return self.vectorizer.transform(texts)

    def get_labels(self, df):
        """Extracts sentiment labels from the DataFrame."""
        return df['sentiment']

    def save_vectorizer(self, path):
        """Saves the fitted TF-IDF vectorizer."""
        import joblib
        joblib.dump(self.vectorizer, path)

    def load_vectorizer(self, path):
        """Loads a pre-trained TF-IDF vectorizer."""
        import joblib
        self.vectorizer = joblib.load(path)