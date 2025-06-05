import speech_recognition as sr
import subprocess
import os
import sys
import pyttsx3
import time
import requests # For making HTTP requests (e.g., weather API)
import google.generativeai as genai # For Gemini API
from dotenv import load_dotenv # For loading environment variables
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
from sklearn.model_selection import train_test_split

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants ---
DATA_FILE = 'data/training_data.csv'
MODEL_PATH = 'models/sentiment_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
ASSISTANT_NAME = "Jarvis"

# API Keys from environment variables
# IMPORTANT: Create a .env file in my project root with these lines:
# OPENWEATHER_API_KEY=my_openweather_api_key_here
# GEMINI_API_KEY=my_gemini_api_key_here
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # CORRECTED: retrieves the value for 'GEMINI_API_KEY' 

# Configure Gemini API
gemini_model = None # Initialize to None first
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # Dynamically find a suitable model that supports 'generateContent'
        # We'll prefer models with "gemini-pro" in their name for general tasks
        available_models = [m for m in genai.list_models() if "generateContent" in m.supported_generation_methods]

        # Prioritize 'gemini-pro' if available, as it's the general-purpose model
        found_gemini_pro = next((m for m in available_models if 'gemini-pro' in m.name), None)

        if found_gemini_pro:
            gemini_model = genai.GenerativeModel(found_gemini_pro.name)
            print(f"{ASSISTANT_NAME}: Configured Gemini API with model: {found_gemini_pro.name}")
        elif available_models:
            # Fallback to the first available model if 'gemini-pro' isn't found
            gemini_model = genai.GenerativeModel(available_models[0].name)
            print(f"{ASSISTANT_NAME}: Configured Gemini API with fallback model: {available_models[0].name}")
        else:
            print(f"{ASSISTANT_NAME}: Warning: No suitable Gemini models found that support 'generateContent'.")

    except Exception as e:
        print(f"{ASSISTANT_NAME}: Error configuring Gemini API: {e}")
        print(f"{ASSISTANT_NAME}: ChatGPT-like functionality will be disabled.")
else:
    print(f"{ASSISTANT_NAME}: Warning: GEMINI_API_KEY not found. ChatGPT-like functionality will be disabled.")

# --- Global variable for the TTS engine ---
engine = None

# --- Voice Engine Initialization ---
def initialize_tts_engine():
    global engine
    if sys.platform == "darwin": # macOS
        print("Detected macOS. Using native 'say' command for TTS.")
        engine = None # Use native 'say' command
    else: # For Linux/Windows, attempt to use pyttsx3
        try:
            # Try espeak first as it's often more reliable on Linux
            engine = pyttsx3.init('espeak')
            print("Initialized pyttsx3 with 'espeak' driver.")
        except Exception as e:
            print(f"Warning: 'espeak' driver failed to initialize: {e}. Attempting default driver.")
            try:
                engine = pyttsx3.init() # Fallback to default driver
                print("Initialized pyttsx3 with default driver.")
            except Exception as e_default:
                print(f"Error: Default pyttsx3 driver also failed to initialize: {e_default}. TTS will be disabled.")
                engine = None # Ensure engine is None if both fail

        if engine:
            voices = engine.getProperty('voices')
            if voices:
                try:
                    # Attempt to set a specific voice, e.g., the first one
                    engine.setProperty('voice', voices[0].id)
                    print(f"Set voice to: {voices[0].name}")
                except IndexError:
                    print("Warning: No voices found by pyttsx3. Using system default voice.")
            else:
                print("Warning: No voices found by pyttsx3 at all. Using system default voice if available.")

            engine.setProperty('rate', 180) # Set speech rate
            print(f"Set speech rate to: {engine.getProperty('rate')}")
        else:
            print("TTS engine could not be initialized. Speech output will be through print statements only.")

def speak(text):
    """Converts text to speech, using 'say' on macOS and pyttsx3 elsewhere."""
    print(f"{ASSISTANT_NAME}: {text}") # Always print the text

    if sys.platform == "darwin":
        subprocess.call(['say', text])
    elif engine: # Only use pyttsx3 if it was successfully initialized
        engine.say(text)
        engine.runAndWait()
    else:
        # If no TTS engine, the print statement above is the only output
        pass

def listen():
    """Listens for audio input and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"{ASSISTANT_NAME}: Listening...")
        r.pause_threshold = 0.8
        r.energy_threshold = 400
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

# --- New Functionalities ---

def get_weather(city="Johannesburg"):
    """Fetches and speaks the current weather for a given city."""
    if not OPENWEATHER_API_KEY:
        speak("I cannot get weather information. The OpenWeatherMap API key is not configured.")
        return

    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city}&appid={OPENWEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(complete_url)
        data = response.json()

        if data["cod"] == 200:
            main = data["main"]
            weather = data["weather"][0]
            temperature = main["temp"]
            pressure = main["pressure"]
            humidity = main["humidity"]
            description = weather["description"]

            speak(f"The weather in {city} is {description} with a temperature of {temperature:.1f} degrees Celsius.")
            speak(f"Humidity is {humidity} percent and atmospheric pressure is {pressure} hectopascals.")
        else:
            speak(f"Sorry, I couldn't find weather information for {city}. Please check the city name.")
    except requests.exceptions.ConnectionError:
        speak("I cannot connect to the weather service. Please check your internet connection.")
    except Exception as e:
        speak(f"An error occurred while fetching weather data: {e}")

def open_email_client():
    """Opens the default email client or a webmail service."""
    speak("Opening your email client.")
    if sys.platform == "darwin": # macOS
        subprocess.run(["open", "-a", "Mail"], check=False) # Attempts to open Apple Mail
        # l could also open a webmail like:
        # open_website("https://mail.google.com")
    elif sys.platform == "win32": # Windows
        os.startfile("mailto:") # Opens default email client
        # Or open a webmail:
        # open_website("https://mail.google.com")
    elif sys.platform == "linux": # Linux
        subprocess.run(["xdg-open", "mailto:"], check=False) # Opens default email client
        # Or open a webmail:
        # open_website("https://mail.google.com")
    else:
        speak("Sorry, I don't know how to open the email client on this operating system.")

def ask_gemini(query):
    """Sends a query to the Gemini API and speaks the response."""
    if not gemini_model:
        speak("I cannot answer general questions. The Gemini API is not configured.")
        return

    speak("Thinking...")
    try:
        response = gemini_model.generate_content(query)
        speak(response.text)
    except Exception as e:
        speak(f"I encountered an error while trying to answer your question: {e}")
        speak("Please try asking again.")

# --- Existing Core AI Logic (Sentiment Analysis) ---
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

    # Attempt to load model and vectorizer
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
        time.sleep(0.5)

def open_application(app_name):
    """Opens a specified application based on the operating system."""
    try:
        speak(f"Attempting to open {app_name}.")
        if sys.platform == "darwin": # macOS
            subprocess.run(["open", "-a", app_name], check=True)
        elif sys.platform == "win32": # Windows
            try:
                subprocess.run(['start', app_name], check=True, shell=True)
            except FileNotFoundError:
                speak(f"Could not find {app_name} via 'start' command. Trying direct execution (less reliable).")
                subprocess.run([app_name], check=True, shell=True)
        elif sys.platform == "linux": # Linux
            subprocess.run(["xdg-open", app_name], check=True)
        else:
            speak(f"Sorry, opening applications is not yet fully supported on your operating system.")
            return

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
    speak(f"Searching for {query} on Google.")
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    open_website(search_url)

def play_music(song_name):
    """Searches for and plays music on YouTube."""
    speak(f"Searching for {song_name} on YouTube and playing it.")
    # CORRECTED YouTube URL structure
    Youtube_url = f"https://www.youtube.com/results?search_query={song_name.replace(' ', '+')}"
    open_website(Youtube_url)

def get_current_time():
    """Tells the current time."""
    # Updated to reflect current time in Johannesburg
    current_time = time.strftime("%I:%M %p %Z") # Added %Z for timezone abbreviation
    speak(f"The current time is {current_time}.")

def get_current_date():
    """Tells the current date."""
    current_date = time.strftime("%A, %B %d, %Y")
    speak(f"Today is {current_date}.")

def list_directory_contents(path="."):
    """Lists contents of a specified directory."""
    abs_path = os.path.expanduser(path)

    if not os.path.isdir(abs_path):
        speak(f"Sorry, {path} is not a valid directory or it does not exist.")
        return

    try:
        contents = os.listdir(abs_path)
        if contents:
            speak(f"Contents of {os.path.basename(abs_path)}:")
            spoken_items = ", ".join(contents[:5])
            if len(contents) > 5:
                spoken_items += f", and {len(contents) - 5} more."
            speak(spoken_items)
            print(f"Full directory contents of {abs_path}: {contents}")
        else:
            speak(f"The directory {os.path.basename(abs_path)} is empty.")
    except PermissionError:
        speak(f"Sorry, I don't have permission to access the directory: {os.path.basename(abs_path)}.")
    except Exception as e:
        speak(f"Sorry, I couldn't list the directory contents. Error: {e}")

# --- Main Interaction Loop (Voice-controlled) ---
def main_voice_assistant():
    """Main loop for the voice assistant."""
    initialize_tts_engine()

    speak(f"Hello, I am {ASSISTANT_NAME}. How may I assist you?")

    # Check for model existence and train if needed
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        speak("No trained sentiment model found. I need to train a new one first.")
        train_new_model()
        speak("Training complete. I am now ready.")
    else:
        speak("Trained sentiment model found. I am ready to serve.")

    while True:
        command = listen()

        # --- Command Handling ---
        # Prioritize more specific commands before broader ones if there's overlap

        # 1. Exit Commands (highest priority)
        if "goodbye" in command or "exit" in command or "shut down" in command or "stop listening" in command:
            speak("Goodbye. I am powering down.")
            break

        # 2. Personalize Greeting
        elif "my name is" in command:
            name_parts = command.split("my name is", 1)
            if len(name_parts) > 1:
                user_name = name_parts[1].strip().title()
                speak(f"It's a pleasure to meet you, {user_name}. Hello {user_name}, how can I assist you today?")
            else:
                speak("I heard you say 'my name is', but I didn't catch your name. Could you please tell me your name?")

        # 3. Maker/Creator
        elif "who built you" in command or "who is your maker" in command or "who created you" in command:
            speak("I am programmed by Thabang Mthimkulu.")

        # 4. Basic Greetings and Chit-Chat
        elif "hello" in command or "hi" in command:
            speak(f"Hello there. How can I help you today?")
        elif "how are you" in command:
            speak("I am an AI, so I don't have feelings, but I'm functioning perfectly. How can I help you?")
        elif "what is your name" in command:
            speak(f"My name is {ASSISTANT_NAME}.")
        elif "thank you" in command or "thanks" in command:
            speak("You're welcome! Is there anything else I can assist you with?")

        # 5. Weather Command
        elif "what's the weather" in command or "how's the weather" in command or "weather in" in command:
            city = "Johannesburg" # Default city
            if "weather in" in command:
                city_parts = command.split("weather in", 1)
                if len(city_parts) > 1:
                    city = city_parts[1].strip()
            get_weather(city)

        # 6. Email Command
        elif "open email" in command or "check email" in command or "open my email" in command:
            open_email_client()

        # 7. Desktop and File System Commands
        elif "go to desktop" in command or "open desktop folder" in command:
            try:
                desktop_path = os.path.expanduser("~/Desktop")
                if sys.platform == "darwin": # macOS
                    subprocess.run(["open", desktop_path], check=True)
                elif sys.platform == "win32": # Windows
                    os.startfile(desktop_path)
                elif sys.platform == "linux": # Linux
                    subprocess.run(["xdg-open", desktop_path], check=True)
                speak("Opened your desktop folder.")
            except Exception as e:
                speak(f"Sorry, I couldn't open the desktop folder: {e}")

        elif "list files" in command or "show files" in command or "list directory" in command:
            speak("Which directory would you like me to list? For example, say 'desktop' or 'documents'.")
            dir_command = listen() # Listen for the directory name
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

        # 8. Application Commands
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
        elif "open application" in command: # Generic "open application" after specific ones
            app_name_parts = command.split("open application", 1)
            if len(app_name_parts) > 1:
                app_name = app_name_parts[1].strip()
                open_application(app_name)
            else:
                speak("Which application would you like me to open? You can say 'open application' followed by the app name, like 'open application Safari'.")

        # 9. Web Browse and Search Commands
        elif "open google" in command:
            open_website("https://www.google.com")
        elif "open youtube" in command:
            # Corrected YouTube URL to search directly, not a malformed one
            open_website("https://www.youtube.com")
        elif "open wikipedia" in command:
         
            open_website("https://chatgpt.com")
        elif "open linkedin " in command:
            open_website("https://www.linkedin.com")
            
        elif "open ster-kinekor" in command or "go to ster-kinekor" in command:
            open_website("https://www.sterkinekor.com/")
        elif "search for" in command:
            query = command.split("search for", 1)[1].strip()
            if query:
                search_web(query)
            else:
                speak("What would you like me to search for?")
        elif "google" in command: # If "google" is used as a verb
            query = command.split("google", 1)[1].strip()
            if query:
                search_web(query)
            else:
                speak("What would you like me to Google?")
        elif "open website" in command or "go to website" in command or "navigate to website" in command: # Generic "open website"
            website_query_parts = command.split("website", 1)
            if len(website_query_parts) > 1:
                target_url = website_query_parts[1].strip()
                # Replace common spoken URL parts with actual characters
                target_url = target_url.replace(" dot com", ".com").replace(" dot org", ".org").replace(" dot net", ".net")
                target_url = target_url.replace(" www ", "www.").replace(" slash ", "/").replace(" colon ", ":")
                target_url = target_url.replace(" space ", "") # Remove "space" if spoken in domain

                # Simple check to prepend https:// if no scheme is provided
                if target_url and not (target_url.startswith("http://") or target_url.startswith("https://")):
                    target_url = "https://" + target_url

                if "." in target_url: # Basic validation for a domain
                    open_website(target_url)
                else:
                    speak("Please provide a valid website address, like 'google dot com' or 'example dot org'.")
            else:
                speak("Which website would you like me to open? Please say 'open website' followed by the address.")

        # 10. Music Playback (YouTube) - Improved handling of "play music"
        elif "play" in command and ("music" in command or "song" in command):
            song_query = command.replace("play", "").replace("music", "").replace("song", "").strip()
            if "on youtube" in song_query:
                song_query = song_query.replace("on youtube", "").strip()
            if song_query:
                play_music(song_query)
            else:
                speak("What song or artist would you like me to play?")

        # 11. Information Commands
        elif "what is the time" in command or "current time" in command:
            get_current_time()
        elif "what is the date" in command or "current date" in command:
            get_current_date()

        # 12. Core AI functions (Sentiment Analysis)
        elif "train model" in command:
            speak("Initiating model training process.")
            train_new_model()
            speak("Training process completed.")
        elif "predict sentiment" in command or "analyze sentiment" in command:
            handle_prediction_mode()

        # 13. ChatGPT-like Q&A (General Questions) - UPDATED LOGIC
        # This block should be placed after specific commands (like open app, play music)
        # but before the generic fallback.
        elif "answer me" in command or "general question" in command or \
             "tell me about" in command or "who is" in command or \
             "what is" in command or "why is" in command or \
             "how does" in command or "can you explain" in command or \
             "can you answer" in command or "jarvis answer" in command or \
             "ask you a question" in command or "i have a question" in command or \
             "tell me something" in command: # Added more general triggers

            actual_query = "" # Initialize actual_query

            # Prioritize extracting the question if a common phrase is present
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
            # New specific splits for added triggers
            elif "can you answer" in command and len(command.split()) > 3: # e.g., "can you answer what is the time"
                actual_query = command.split("can you answer", 1)[1].strip()
            elif "jarvis answer" in command and len(command.split()) > 2: # e.g., "jarvis answer how are you"
                actual_query = command.split("jarvis answer", 1)[1].strip()
            elif "tell me something about" in command:
                 actual_query = command.split("tell me something about", 1)[1].strip()

            # If no specific question was extracted, and the command itself is just a trigger,
            # then ask for the full question.
            if not actual_query:
                speak("Please state your question clearly.")
                actual_query = listen() # Listen again specifically for the question

            if actual_query:
                ask_gemini(actual_query)
            else:
                speak("I didn't hear a question. Please try asking again.")

        # If the user just says "answer general questions" without a specific question immediately
        elif "answer general questions" in command:
            speak("Yes, I can answer general questions. What would you like to know?")
            actual_query = listen()
            if actual_query:
                ask_gemini(actual_query)
            else:
                speak("I didn't hear your question. Please try again.")

        # 14. Help/Capability Inquiry (Updated to reflect new features)
        elif "what can you do" in command:
            speak("I am programmed to train a sentiment analysis model, predict sentiment from text, and search the web.")
            speak("I can open applications like Safari or Calculator, or you can say 'open application' followed by the app name.")
            speak("I can open specific websites like Google, YouTube, Wikipedia, and Ster-Kinekor. I can also open any website if you say 'open website' followed by the address, like 'google dot com'.")
            speak("I can play music by searching YouTube for songs, tell you the time and date, and list contents of your main folders, including opening your desktop folder.")
            speak("I can also get you the current weather for a city and open your email client.")
            speak("And I can answer general questions, similar to ChatGPT.")
            speak("You can also ask me about my name or say 'my name is' followed by your name.")

        # 15. Generic Fallback - IMPORTANT: This must be the last `elif`
        elif command: # Only respond if some command was actually recognized (not empty string)
            speak("I heard that, but I'm not yet programmed to respond to that specific command.")
            speak("For now, I can train the model, predict sentiment, open applications, open any website, search the web, play music, tell you the time and date, get weather, open email, or answer general questions.")
            speak("You can also ask me about my name.")
            speak("You can also say 'what can you do' to hear a summary of my current abilities.")


if __name__ == "__main__":
    main_voice_assistant()