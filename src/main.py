# src/main.py
import os
from dotenv import load_dotenv

# Import modules from src
from src.config import ASSISTANT_NAME, DATA_FILE, MODEL_PATH, VECTORIZER_PATH, DEFAULT_WEATHER_CITY, GOOGLE_URL, YOUTUBE_URL, WIKIPEDIA_URL, CHATGPT_URL, LINKEDIN_URL, STERKINEKOR_URL
from src.voice_io import speak, listen, initialize_tts_engine
from src.integrations import get_weather, ask_gemini, initialize_gemini
from src.app_launcher import open_email_client, open_application, open_website, search_web, play_music, get_current_time, get_current_date, list_directory_contents
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
from sklearn.model_selection import train_test_split

# Load environment variables from .env file
load_dotenv()

# --- Core AI Logic (Sentiment Analysis) ---
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

    # Preprocess_data should use the same vectorizer for train and test
    # This might require the DataProcessor to handle fitting and transforming
    # or expose the vectorizer.
    # For now, assuming preprocess_data handles vectorization internally
    # and saves/loads it as needed.
    X_train_features = data_processor.preprocess_data(X_train_text, fit_vectorizer=True)

    model_trainer.train_model(X_train_features, y_train)

    X_test_features = data_processor.preprocess_data(X_test_text, fit_vectorizer=False) # Use already fitted vectorizer
    model_trainer.evaluate_model(X_test_features, y_test)

    os.makedirs('models', exist_ok=True)
    model_trainer.save_model(MODEL_PATH)
    data_processor.save_vectorizer(VECTORIZER_PATH)
    speak("Model training complete. Model and vectorizer saved.")

def handle_prediction_mode():
    """Handles continuous prediction in voice mode."""
    speak("Starting sentiment prediction mode. Please speak your text for analysis.")
    predictor = Predictor(MODEL_PATH, VECTORIZER_PATH)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        speak("Sentiment model or vectorizer not found. Please ensure the model is trained before predicting.")
        return

    predictor.load_model_and_vectorizer(MODEL_PATH, VECTORIZER_PATH)

    if predictor.model_trainer.model is None or predictor.data_processor.vectorizer is None:
        speak("Prediction aborted. Model or vectorizer could not be loaded. Please ensure the model is trained.")
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

# --- Main Interaction Loop (Voice-controlled) ---
def main_voice_assistant():
    """Main loop for the voice assistant."""
    initialize_tts_engine()
    initialize_gemini() # Initialize Gemini after TTS

    speak(f"Hello, I am {ASSISTANT_NAME}. How may I assist you?")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        speak("No trained sentiment model found. I need to train a new one first.")
        train_new_model()
        speak("Training complete. I am now ready.")
    else:
        speak("Trained sentiment model found. I am ready to serve.")

    while True:
        command = listen()

        if "goodbye" in command or "exit" in command or "shut down" in command or "stop listening" in command:
            speak("Goodbye. I am powering down.")
            break

        elif "my name is" in command:
            name_parts = command.split("my name is", 1)
            if len(name_parts) > 1:
                user_name = name_parts[1].strip().title()
                speak(f"It's a pleasure to meet you, {user_name}. Hello {user_name}, how can I assist you today?")
            else:
                speak("I heard you say 'my name is', but I didn't catch your name. Could you please tell me your name?")

        elif "who built you" in command or "who is your maker" in command or "who created you" in command:
            speak("I am programmed by Thabang Mthimkulu.")

        elif "hello" in command or "hi" in command:
            speak(f"Hello there. How can I help you today?")
        elif "how are you" in command:
            speak("I am an AI, so I don't have feelings, but I'm functioning perfectly. How can I help you?")
        elif "what is your name" in command:
            speak(f"My name is {ASSISTANT_NAME}.")
        elif "thank you" in command or "thanks" in command:
            speak("You're welcome! Is there anything else I can assist you with?")

        elif "what's the weather" in command or "how's the weather" in command or "weather in" in command:
            city = DEFAULT_WEATHER_CITY
            if "weather in" in command:
                city_parts = command.split("weather in", 1)
                if len(city_parts) > 1:
                    city = city_parts[1].strip()
            get_weather(city)

        elif "open email" in command or "check email" in command or "open my email" in command:
            open_email_client()

        elif "go to desktop" in command or "open desktop folder" in command:
            try:
                desktop_path = os.path.expanduser("~/Desktop")
                open_application(desktop_path) # Reuse open_application if it can handle folders
                speak("Opened your desktop folder.")
            except Exception as e:
                speak(f"Sorry, I couldn't open the desktop folder: {e}")

        elif "list files" in command or "show files" in command or "list directory" in command:
            speak("Which directory would you like me to list? For example, say 'desktop' or 'documents'.")
            dir_command = listen()
            if "desktop" in dir_command:
                list_directory_contents("~/Desktop")
            elif "documents" in dir_command:
                list_directory_contents("~/Documents")
            elif "downloads" in dir_command:
                list_directory_contents("~/Downloads")
            elif "home" in dir_command:
                list_directory_contents("~")
            else:
                speak("I can only list contents of your Desktop, Documents, Downloads, or Home folder at the moment. Please specify one of these.")

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
        elif "open application" in command:
            app_name_parts = command.split("open application", 1)
            if len(app_name_parts) > 1:
                app_name = app_name_parts[1].strip()
                open_application(app_name)
            else:
                speak("Which application would you like me to open? You can say 'open application' followed by the app name, like 'open application Safari'.")

        elif "open google" in command:
            open_website(GOOGLE_URL)
        elif "open youtube" in command:
            open_website(YOUTUBE_URL)
        elif "open wikipedia" in command:
            open_website(WIKIPEDIA_URL)
        elif "open chatgpt" in command: # Added specific ChatGPT command
            open_website(CHATGPT_URL)
        elif "open linkedin" in command:
            open_website(LINKEDIN_URL)
        elif "open ster-kinekor" in command or "go to ster-kinekor" in command:
            open_website(STERKINEKOR_URL)
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
        elif "open website" in command or "go to website" in command or "navigate to website" in command:
            website_query_parts = command.split("website", 1)
            if len(website_query_parts) > 1:
                target_url = website_query_parts[1].strip()
                target_url = target_url.replace(" dot com", ".com").replace(" dot org", ".org").replace(" dot net", ".net")
                target_url = target_url.replace(" www ", "www.").replace(" slash ", "/").replace(" colon ", ":")
                target_url = target_url.replace(" space ", "")

                if target_url and not (target_url.startswith("http://") or target_url.startswith("https://")):
                    target_url = "https://" + target_url

                if "." in target_url:
                    open_website(target_url)
                else:
                    speak("Please provide a valid website address, like 'google dot com' or 'example dot org'.")
            else:
                speak("Which website would you like me to open? Please say 'open website' followed by the address.")

        elif "play" in command and ("music" in command or "song" in command):
            song_query = command.replace("play", "").replace("music", "").replace("song", "").strip()
            if "on youtube" in song_query:
                song_query = song_query.replace("on youtube", "").strip()
            if song_query:
                play_music(song_query)
            else:
                speak("What song or artist would you like me to play?")

        elif "what is the time" in command or "current time" in command:
            get_current_time()
        elif "what is the date" in command or "current date" in command:
            get_current_date()

        elif "train model" in command:
            speak("Initiating model training process.")
            train_new_model()
            speak("Training process completed.")
        elif "predict sentiment" in command or "analyze sentiment" in command:
            handle_prediction_mode()

        elif "answer me" in command or "general question" in command or \
             "tell me about" in command or "who is" in command or \
             "what is" in command or "why is" in command or \
             "how does" in command or "can you explain" in command or \
             "can you answer" in command or "jarvis answer" in command or \
             "ask you a question" in command or "i have a question" in command or \
             "tell me something" in command:

            actual_query = ""
            if "tell me about" in command:
                actual_query = command.split("tell me about", 1)[1].strip()
            elif "who is" in command:
                actual_query = command.split("who is", 1)[1].strip()
            elif "what is" in command:
                actual_query = command.split("what is", 1)[1].strip()
            elif "why is" in command:
                actual_query = command.split("why is", 1)[1].strip()
            elif "how does" in command:
                actual_query = command.split("how does", 1)[1].strip()
            elif "can you explain" in command:
                actual_query = command.split("can you explain", 1)[1].strip()
            elif "can you answer" in command and len(command.split()) > 3:
                actual_query = command.split("can you answer", 1)[1].strip()
            elif "jarvis answer" in command and len(command.split()) > 2:
                actual_query = command.split("jarvis answer", 1)[1].strip()
            elif "tell me something about" in command:
                 actual_query = command.split("tell me something about", 1)[1].strip()

            if not actual_query:
                speak("Please state your question clearly.")
                actual_query = listen()

            if actual_query:
                ask_gemini(actual_query)
            else:
                speak("I didn't hear a question. Please try asking again.")

        elif "answer general questions" in command:
            speak("Yes, I can answer general questions. What would you like to know?")
            actual_query = listen()
            if actual_query:
                ask_gemini(actual_query)
            else:
                speak("I didn't hear your question. Please try again.")

        elif "what can you do" in command:
            speak("I am programmed to train a sentiment analysis model, predict sentiment from text, and search the web.")
            speak("I can open applications like Safari or Calculator, or you can say 'open application' followed by the app name.")
            speak("I can open specific websites like Google, YouTube, Wikipedia, ChatGPT, LinkedIn, and Ster-Kinekor. I can also open any website if you say 'open website' followed by the address, like 'google dot com'.")
            speak("I can play music by searching YouTube for songs, tell you the time and date, and list contents of your main folders, including opening your desktop folder.")
            speak("I can also get you the current weather for a city and open your email client.")
            speak("And I can answer general questions, similar to ChatGPT.")
            speak("You can also ask me about my name or say 'my name is' followed by your name.")

        elif command:
            speak("I heard that, but I'm not yet programmed to respond to that specific command.")
            speak("For now, I can train the model, predict sentiment, open applications, open any website, search the web, play music, tell you the time and date, get weather, open email, or answer general questions.")
            speak("You can also ask me about my name.")
            speak("You can also say 'what can you do' to hear a summary of my current abilities.")


if __name__ == "__main__":
    main_voice_assistant()