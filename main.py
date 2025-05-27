import speech_recognition as sr
import subprocess
import os
import sys # Import sys to check platform
import pyttsx3
import time
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
from sklearn.model_selection import train_test_split

# --- Configuration Constants ---
DATA_FILE = 'data/training_data.csv'
MODEL_PATH = 'models/sentiment_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
ASSISTANT_NAME = "Jarvis" # Or any name you prefer for your AI

# --- Global variable for the TTS engine ---
engine = None

# --- Voice Engine Initialization ---
def initialize_tts_engine():
    global engine
    if sys.platform == "darwin": # macOS
        print("Detected macOS. Using native 'say' command for TTS.")
        # No pyttsx3 engine needed for macOS 'say'
        engine = None # Explicitly set to None as we won't use pyttsx3 for speaking
    else: # For Linux/Windows, attempt to use pyttsx3 with espeak or auto
        try:
            # Try 'espeak' first if available and desired
            engine = pyttsx3.init('espeak')
            print("Initialized pyttsx3 with 'espeak' driver.")
        except Exception as e:
            print(f"Warning: 'espeak' driver failed to initialize: {e}. Attempting default driver.")
            engine = pyttsx3.init() # Fallback to default driver
            
        if engine: # Only set properties if engine was successfully initialized
            voices = engine.getProperty('voices')
            try:
                # Attempt to set a male voice (often voices[0])
                # You might need to experiment with voices[0], voices[1], etc.
                # based on what pyttsx3 finds on your system.
                engine.setProperty('voice', voices[0].id)
                print(f"Set voice to: {voices[0].name}")
            except IndexError:
                print("Warning: No voices found by pyttsx3. Using default voice.")
            engine.setProperty('rate', 180) # Speed of speech (words per minute)
            print(f"Set speech rate to: {engine.getProperty('rate')}")

def speak(text):
    """Converts text to speech, using 'say' on macOS and pyttsx3 elsewhere."""
    print(f"{ASSISTANT_NAME}: {text}") # Always print what the assistant says

    if sys.platform == "darwin": # macOS
        subprocess.call(['say', text])
    elif engine: # Use pyttsx3 if it was successfully initialized
        engine.say(text)
        engine.runAndWait()
    else:
        print("Warning: TTS engine not initialized. Cannot speak.")

def listen():
    """Listens for audio input and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"{ASSISTANT_NAME}: Listening...")
        r.pause_threshold = 0.8 # Seconds of non-speaking audio before a phrase is considered complete
        r.energy_threshold = 400 # Minimum audio energy to consider for recording
        # r.adjust_for_ambient_noise(source) # Uncomment this line to dynamically adjust to ambient noise
        audio = r.listen(source)
    try:
        print(f"{ASSISTANT_NAME}: Recognizing...")
        query = r.recognize_google(audio, language='en-US') # Using Google Speech Recognition
        print(f"User: {query}")
        return query.lower() # Return lowercase for easier command matching
    except sr.UnknownValueError:
        speak("Sorry, I could not understand your audio.")
        return ""
    except sr.RequestError as e:
        speak(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

# --- Core AI Logic (from previous version, now integrated with voice) ---
def train_new_model():
    """Handles the training process of the AI model."""
    speak("Starting model training process.")
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()

    df = data_processor.load_data(DATA_FILE)
    if df is None:
        speak("Data file not found. Training cannot proceed.")
        return

    texts = df['text']
    labels = data_processor.get_labels(df)

    # Split data for training and testing
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Preprocess training data
    X_train_features = data_processor.preprocess_data(X_train_text)

    # Train the model
    model_trainer.train_model(X_train_features, y_train)

    # Preprocess test data using the *same* fitted vectorizer
    X_test_features = data_processor.preprocess_data(X_test_text)
    model_trainer.evaluate_model(X_test_features, y_test)

    # Save the trained model and vectorizer
    os.makedirs('models', exist_ok=True)
    model_trainer.save_model(MODEL_PATH)
    data_processor.save_vectorizer(VECTORIZER_PATH)
    speak("Model training complete. Model and vectorizer saved.")

def handle_prediction_mode():
    """Handles continuous prediction in voice mode."""
    speak("Starting sentiment prediction mode. Please speak your text for analysis.")
    predictor = Predictor(MODEL_PATH, VECTORIZER_PATH)

    if predictor.model_trainer.model is None or predictor.data_processor.vectorizer is None:
        speak("Prediction aborted. Model or vectorizer not found/loaded. Please ensure the model is trained.")
        return

    while True:
        speak("Waiting for text.")
        user_input_voice = listen() # Listen for voice input

        if "quit" in user_input_voice or "exit" in user_input_voice or "stop prediction" in user_input_voice:
            speak("Exiting sentiment prediction mode. Goodbye.")
            break
        elif user_input_voice: # If something was recognized
            sentiment = predictor.predict_sentiment(user_input_voice)
            speak(f"The predicted sentiment is: {sentiment}")
        time.sleep(0.5) # Short pause to prevent rapid listening loops

def open_application(app_name):
    """Opens a specified application on macOS."""
    if sys.platform == "darwin": # Only works on macOS
        try:
            speak(f"Attempting to open {app_name}.")
            # The 'open -a' command is specific to macOS for opening applications
            subprocess.run(["open", "-a", app_name], check=True)
            speak(f"Opened {app_name} successfully.")
        except subprocess.CalledProcessError:
            speak(f"Sorry, I couldn't find or open {app_name}. Please make sure it's installed and in your Applications folder.")
        except FileNotFoundError:
            speak(f"The 'open' command was not found. This function may only work on macOS.")
    else:
        speak(f"Sorry, opening applications is not yet supported on your operating system.")

# --- Main Interaction Loop (Voice-controlled) ---
def main_voice_assistant():
    """Main loop for the voice assistant."""
    initialize_tts_engine() # Initialize the TTS engine at the start

    speak(f"Hello, I am {ASSISTANT_NAME}. How may I assist you?")

    # Initial check and training if no model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        speak("No trained sentiment model found. I need to train a new one first.")
        train_new_model()
        speak("Training complete. I am now ready.")
    else:
        speak("Trained sentiment model found. I am ready to serve.")

    while True:
        command = listen()

        # --- New and Improved Command Handling ---

        # 1. Personalize Greeting (more flexible)
        if "my name is" in command:
            name_parts = command.split("my name is", 1)
            if len(name_parts) > 1:
                user_name = name_parts[1].strip().title() # Capitalize first letter
                speak(f"It's a pleasure to meet you, {user_name}. Hello {user_name}, how can I assist you today?")
            else:
                speak("I heard you say 'my name is', but I didn't catch your name. Could you please tell me your name?")
        elif "i am Thabang" in command or "this is Thabang" in command: # Specific for your name
            speak("It's a pleasure to meet you, Thabang. Hello Thabang, how can I assist you today?")

        # 2. Maker/Creator
        elif "who built you" in command or "who is your maker" in command or "who created you" in command:
            speak("I am trained and built by Thabang Mthimkulu, who is my maker.")

        # 3. Basic Greetings and Chit-Chat
        elif "hello" in command or "hi" in command:
            speak(f"Hello there. How can I help you today?")
        elif "how are you" in command:
            speak("I am an AI, so I don't have feelings, but I'm functioning perfectly. How can I help you?")
        elif "what is your name" in command:
            speak(f"My name is {ASSISTANT_NAME}.")
        elif "thank you" in command or "thanks" in command:
            speak("You're welcome! Is there anything else I can assist you with?")

        # 4. Desktop Commands
        elif "open safari" in command:
            open_application("Safari")
        elif "open chrome" in command or "open google chrome" in command:
            open_application("Google Chrome")
        elif "open notes" in command:
            open_application("Notes")
        elif "open terminal" in command:
            open_application("Terminal")
        elif "open calculator" in command:
            open_application("Calculator")
        elif "open messages" in command:
            open_application("Messages")

        # 5. Core AI functions
        elif "train model" in command:
            speak("Initiating model training process.")
            train_new_model()
            speak("Training process completed.")
        elif "predict sentiment" in command or "analyze sentiment" in command:
            handle_prediction_mode()
        elif "what don't you understand" in command or "what can you do" in command:
            speak("I am programmed to train a sentiment analysis model, predict sentiment from text, and open certain applications like Safari, Chrome, Notes, or Terminal on your desktop.")
            speak("I can also tell you about my maker. Just try saying 'train model', 'predict sentiment', or 'open Safari'.")


        # 6. Exit Commands
        elif "goodbye" in command or "exit" in command or "shut down" in command or "stop listening" in command:
            speak("Goodbye. I am powering down.")
            break

        # 7. Generic Fallback (modify this message to be more helpful)
        elif command: # If a command was recognized but not specifically handled
            speak("I heard that, but I'm not yet programmed to respond to that specific command.")
            speak("For now, I can train the model, predict sentiment, or help open applications like Safari or Chrome.")
            speak("You can also ask me about my maker or say 'my name is' followed by your name.")

if __name__ == "__main__":
    main_voice_assistant()