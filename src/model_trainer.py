import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self):
        self.model = MultinomialNB()

    def train_model(self, X_train, y_train):
        """Trains the AI model."""
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def evaluate_model(self, X_test, y_test):
        """Evaluates the trained model."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", report)
        return accuracy, report

    def save_model(self, path):
        """Saves the trained model."""
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Loads a pre-trained model."""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")