# My AI Project: Simple Sentiment Analysis

This project demonstrates a basic AI application for sentiment analysis using Python and scikit-learn.

## Project Structure

* `.vscode/`: VS Code specific settings.
* `data/`: Stores training datasets (`training_data.csv`).
* `models/`: Stores trained machine learning models (`sentiment_model.pkl`) and TF-IDF vectorizer (`tfidf_vectorizer.pkl`).
* `src/`: Contains the core Python source code.
    * `data_processor.py`: Handles data loading and text preprocessing (TF-IDF).
    * `model_trainer.py`: Manages model definition, training, and evaluation.
    * `predictor.py`: Provides an interface for making predictions.
* `tests/`: Contains unit tests for the project.
* `venv/`: Python virtual environment for isolated dependencies.
* `main.py`: The main entry point to train the model or run predictions.
* `requirements.txt`: Lists all required Python packages.
* `README.md`: This file.

## Setup and Installation

1.  **Clone the repository (or create the files manually):**
    ```bash
    git clone <your-repo-url>
    cd my_ai_project
    ```
    (If creating manually, first create the `my_ai_project` folder, then all the files and subfolders as outlined above).

2.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure VS Code (Optional but Recommended):**
    * Open the project folder in VS Code (`code .` from the root directory).
    * Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P).
    * Type `Python: Select Interpreter` and choose the `venv` interpreter located in your project folder. This will help VS Code use the correct Python environment.

## How to Run

1.  **Activate your virtual environment** (if not already active).

2.  **Run the main application:**
    ```bash
    python main.py
    ```
    * The first time you run it, it will check for an existing model. If none is found, it will automatically train a new one using the data in `data/training_data.csv`.
    * You will then be prompted to either `(t)rain` a new model or `(p)redict` sentiment.

## How to Run Tests

1.  **Activate your virtual environment.**
2.  **Navigate to the project root.**
3.  **Run tests:**
    ```bash
    python -m unittest discover tests
    ```

## Extending the AI

* **More Complex Models:** Replace `MultinomialNB` with other scikit-learn classifiers (e.g., `LogisticRegression`, `SVC`) or integrate deep learning frameworks like TensorFlow/Keras or PyTorch for more advanced neural network models.
* **Larger Datasets:** Add more diverse and larger datasets to the `data/` folder.
* **Hyperparameter Tuning:** Implement techniques like GridSearchCV or RandomizedSearchCV in `model_trainer.py` to find optimal model parameters.
* **Deployment:** Consider how you would deploy this AI as an API (e.g., using Flask or FastAPI) for wider use.
* **Improved Preprocessing:** Add more sophisticated text preprocessing steps (e.g., stemming, lemmatization, stop word removal, custom tokenizers).