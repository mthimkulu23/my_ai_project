import speech_recognition as sr
import subprocess
import os
import sys
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
ASSISTANT_NAME = "Jarvis"

# --- Global variable for the TTS engine ---
engine = None

# --- Voice Engine Initialization ---
def initialize_tts_engine():
    global engine
    if sys.platform == "darwin": # macOS
        print("Detected macOS. Using native 'say' command for TTS.")
        engine = None
    else: # For Linux/Windows, attempt to use pyttsx3 with espeak or auto
        try:
            engine = pyttsx3.init('espeak')
            print("Initialized pyttsx3 with 'espeak' driver.")
        except Exception as e:
            print(f"Warning: 'espeak' driver failed to initialize: {e}. Attempting default driver.")
            engine = pyttsx3.init()
            
        if engine:
            voices = engine.getProperty('voices')
            try:
                engine.setProperty('voice', voices[0].id)
                print(f"Set voice to: {voices[0].name}")
            except IndexError:
                print("Warning: No voices found by pyttsx3. Using default voice.")
            engine.setProperty('rate', 180)
            print(f"Set speech rate to: {engine.getProperty('rate')}")

def speak(text):
    """Converts text to speech, using 'say' on macOS and pyttsx3 elsewhere."""
    print(f"{ASSISTANT_NAME}: {text}")

    if sys.platform == "darwin":
        subprocess.call(['say', text])
    elif engine:
        engine.say(text)
        engine.runAndWait()
    else:
        print("Warning: TTS engine not initialized. Cannot speak.")

def listen():
    """Listens for audio input and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"{ASSISTANT_NAME}: Listening...")
        r.pause_threshold = 0.8
        r.energy_threshold = 400
        # r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        print(f"{ASSISTANT_NAME}: Recognizing...")
        query = r.recognize_google(audio, language='en-US')
        print(f"User: {query}")
        return query.lower()
    except sr.UnknownValueError:
        speak("Sorry, I could not understand your audio.")
        return ""
    except sr.RequestError as e:
        speak(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

# --- Core AI Logic ---
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

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    X_train_features = data_processor.preprocess_data(X_train_text)

    model_trainer.train_model(X_train_features, y_train)

    X_test_features = data_processor.preprocess_data(X_test_text)
    model_trainer.evaluate_model(X_test_features, y_test)

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
        user_input_voice = listen()

        if "quit" in user_input_voice or "exit" in user_input_voice or "stop prediction" in user_input_voice:
            speak("Exiting sentiment prediction mode. Goodbye.")
            break
        elif user_input_voice:
            sentiment = predictor.predict_sentiment(user_input_voice)
            speak(f"The predicted sentiment is: {sentiment}")
        time.sleep(0.5)

def open_application(app_name):
    """Opens a specified application based on the operating system."""
    try:
        speak(f"Attempting to open {app_name}.")
        if sys.platform == "darwin": # macOS
            subprocess.run(["open", "-a", app_name], check=True)
        elif sys.platform == "win32": # Windows
            # For Windows, you might need to specify the full path or use 'start' command
            # For simplicity, we'll try direct execution or 'start'
            try:
                subprocess.run([app_name], check=True, shell=True) # Try direct execution
            except FileNotFoundError:
                subprocess.run(['start', app_name], check=True, shell=True) # Or using 'start'
        elif sys.platform == "linux": # Linux
            subprocess.run(["xdg-open", app_name], check=True) # General way to open files/apps
        else:
            speak(f"Sorry, opening applications is not yet fully supported on your operating system.")
            return # Exit if OS not supported

        speak(f"Opened {app_name} successfully.")
    except subprocess.CalledProcessError:
        speak(f"Sorry, I couldn't find or open {app_name}. Please make sure it's installed and accessible.")
    except FileNotFoundError:
        speak(f"The command to open {app_name} was not found. This function may not work as expected on your OS.")
    except Exception as e:
        speak(f"An unexpected error occurred while trying to open {app_name}: {e}")


def open_website(url):
    """Opens a specified URL in the default web browser."""
    speak(f"Opening {url} in your browser.")
    try:
        if sys.platform == "darwin": # macOS
            subprocess.run(["open", url], check=True)
        elif sys.platform == "win32": # Windows
            os.startfile(url)
        elif sys.platform == "linux": # Linux
            subprocess.run(["xdg-open", url], check=True)
        else:
            speak("Sorry, I can't open web links on this operating system.")
    except Exception as e:
        speak(f"Sorry, I couldn't open the website. There was an error: {e}")

def search_web(query):
    """Performs a Google search for the given query."""
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    open_website(search_url)
    speak(f"Searching for {query} on Google.")

def play_music(song_name):
    """Searches for and plays music on YouTube."""
    speak(f"Searching for {song_name} on YouTube and playing it.")
    # Using the Youtube URL to open directly in a browser
    Youtube_url = f"https://www.youtube.com/results?search_query={song_name.replace(' ', '+')}"
    open_website(Youtube_url) # open_website function will handle opening browser

def get_current_time():
    """Tells the current time."""
    current_time = time.strftime("%I:%M %p")
    speak(f"The current time is {current_time}.")

def get_current_date():
    """Tells the current date."""
    current_date = time.strftime("%A, %B %d, %Y")
    speak(f"Today is {current_date}.")

def list_directory_contents(path="."):
    """Lists contents of a specified directory."""
    if not os.path.isdir(path):
        speak(f"Sorry, {path} is not a valid directory.")
        return

    try:
        contents = os.listdir(path)
        if contents:
            speak(f"Contents of {path}:")
            spoken_items = ", ".join(contents[:5])
            if len(contents) > 5:
                spoken_items += f", and {len(contents) - 5} more."
            speak(spoken_items)
            print(f"Full directory contents of {path}: {contents}")
        else:
            speak(f"The directory {path} is empty.")
    except Exception as e:
        speak(f"Sorry, I couldn't list the directory contents. Error: {e}")

# --- Main Interaction Loop (Voice-controlled) ---
def main_voice_assistant():
    """Main loop for the voice assistant."""
    initialize_tts_engine()

    speak(f"Hello, I am {ASSISTANT_NAME}. How may I assist you?")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        speak("No trained sentiment model found. I need to train a new one first.")
        train_new_model()
        speak("Training complete. I am now ready.")
    else:
        speak("Trained sentiment model found. I am ready to serve.")

    while True:
        command = listen()

        # --- Command Handling ---

        # 1. Personalize Greeting (more flexible)
        if "my name is" in command:
            name_parts = command.split("my name is", 1)
            if len(name_parts) > 1:
                user_name = name_parts[1].strip().title()
                speak(f"It's a pleasure to meet you, {user_name}. Hello {user_name}, how can I assist you today?")
            else:
                speak("I heard you say 'my name is', but I didn't catch your name. Could you please tell me your name?")
        elif "i am Thabang" in command or "this is Thabang" in command:
            speak("It's a pleasure to meet you, Thabang. Hello Thabang, how can I assist you today?")

        # 2. Maker/Creator (Modified)
        elif "who built you" in command or "who is your maker" in command or "who created you" in command:
            speak(f"I am {ASSISTANT_NAME}, a virtual assistant. My capabilities are continuously being developed.")

        # 3. Basic Greetings and Chit-Chat
        elif "hello" in command or "hi" in command:
            speak(f"Hello there. How can I help you today?")
        elif "how are you" in command:
            speak("I am an AI, so I don't have feelings, but I'm functioning perfectly. How can I help you?")
        elif "what is your name" in command:
            speak(f"My name is {ASSISTANT_NAME}.")
        elif "thank you" in command or "thanks" in command:
            speak("You're welcome! Is there anything else I can assist you with?")

        # 4. Desktop Application Commands (More generalized attempt)
        elif "open application" in command:
            app_name_parts = command.split("open application", 1)
            if len(app_name_parts) > 1:
                app_name = app_name_parts[1].strip()
                open_application(app_name)
            else:
                speak("Which application would you like me to open?")
        # Keep specific commands for common apps for better recognition
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
        
        # 5. Web Browse Commands (specific sites + general search)
        elif "open google" in command:
            open_website("https://www.google.com")
        elif "open youtube" in command:
            open_website("https://www.youtube.com") # Corrected YouTube URL
        elif "open wikipedia" in command:
            open_website("https://www.wikipedia.org")
        elif "open ster-kinekor" in command or "go to ster-kinekor" in command:
            open_website("https://www.sterkinekor.com/")
        # General "open website" command (more flexible)
        elif "open website" in command or "go to website" in command or "navigate to website" in command:
            website_query_parts = command.split("website", 1)
            if len(website_query_parts) > 1:
                target_url = website_query_parts[1].strip()
                # Remove common phrases like "dot com", "dot org" from the end if they are just words
                target_url = target_url.replace(" dot com", ".com").replace(" dot org", ".org").replace(" dot net", ".net")
                target_url = target_url.replace(" www ", "www.").replace(" slash ", "/").replace(" colon ", ":")

                # Simple check to prepend https:// if no scheme is provided
                if target_url and not (target_url.startswith("http://") or target_url.startswith("https://")):
                    target_url = "https://" + target_url
                
                if "." in target_url: # Basic validation for a domain
                    open_website(target_url)
                else:
                    speak("Please provide a valid website address, like 'google dot com' or 'example dot org'.")
            else:
                speak("Which website would you like me to open? Please say 'open website' followed by the address.")

        elif "search for" in command:
            query = command.split("search for", 1)[1].strip()
            if query:
                search_web(query)
            else:
                speak("What would you like me to search for?")
        elif "google" in command:
            query = command.split("google", 1)[1].strip()
            if query:
                search_web(query)
            else:
                speak("What would you like me to Google?")

        # 6. Music Playback
        elif "play music" in command or "play a song" in command:
            song_query_parts = command.split("play music", 1)
            if len(song_query_parts) == 1: # Maybe "play a song"
                song_query_parts = command.split("play a song", 1)

            if len(song_query_parts) > 1 and song_query_parts[1].strip():
                song_name = song_query_parts[1].strip()
                play_music(song_name)
            else:
                speak("What song or artist would you like me to play?")

        # 7. Information Commands
        elif "what is the time" in command or "current time" in command:
            get_current_time()
        elif "what is the date" in command or "current date" in command:
            get_current_date()
        elif "list files" in command or "show files" in command or "list directory" in command:
            speak("Which directory would you like me to list? For example, say 'desktop' or 'documents'.")
            dir_command = listen()
            if "desktop" in dir_command:
                list_directory_contents(os.path.expanduser("~/Desktop"))
            elif "documents" in dir_command:
                list_directory_contents(os.path.expanduser("~/Documents"))
            elif "downloads" in dir_command:
                list_directory_contents(os.path.expanduser("~/Downloads"))
            elif "home" in dir_command:
                list_directory_contents(os.path.expanduser("~"))
            else:
                speak("I can only list contents of your Desktop, Documents, Downloads, or Home folder at the moment.")

        # 8. Core AI functions (Sentiment Analysis)
        elif "train model" in command:
            speak("Initiating model training process.")
            train_new_model()
            speak("Training process completed.")
        elif "predict sentiment" in command or "analyze sentiment" in command:
            handle_prediction_mode()
        
        # 9. Help/Capability Inquiry (Updated)
        elif "what can you do" in command:
            speak("I am programmed to train a sentiment analysis model, predict sentiment from text, and search the web.")
            speak("I can open applications, open specific websites like Google, YouTube, Wikipedia, and Ster-Kinekor.")
            speak("I can also try to open any website if you say 'open website' followed by the address, like 'open website google dot com'.")
            speak("I can play music by searching YouTube for songs, tell you the time and date, and list contents of your main folders.")
            speak("You can also ask me about my name or say 'my name is' followed by your name.")

        # 10. Exit Commands
        elif "goodbye" in command or "exit" in command or "shut down" in command or "stop listening" in command:
            speak("Goodbye. I am powering down.")
            break

        # 11. Generic Fallback
        elif command:
            speak("I heard that, but I'm not yet programmed to respond to that specific command.")
            speak("For now, I can train the model, predict sentiment, open applications, open any website, search the web, play music, or tell you the time and date.")
            speak("You can also ask me about my name or say 'my name is' followed by your name.")

if __name__ == "__main__":
    main_voice_assistant()