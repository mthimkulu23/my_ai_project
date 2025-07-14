import os
from dotenv import load_dotenv

# Import modules from src
from src.config import ASSISTANT_NAME, DATA_FILE, MODEL_PATH, VECTORIZER_PATH, DEFAULT_WEATHER_CITY, GOOGLE_URL, YOUTUBE_URL, WIKIPEDIA_URL, CHATGPT_URL, LINKEDIN_URL, STERKINEKOR_URL
from src.voice_io import speak, listen, initialize_tts_engine
from src.integrations import get_weather, ask_gemini, initialize_gemini
# UPDATED IMPORT: Added open_browser_tab
from src.app_launcher import open_email_client, open_application, open_website, search_web, play_music, get_current_time, get_current_date, list_directory_contents, open_browser_tab
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
    X_train_features = data_processor.preprocess_data(X_train_text) # Removed fit_vectorizer=True, as preprocess_data should handle fit/transform based on internal state
    
    # Ensure vectorizer is fitted only once
    if not hasattr(data_processor.vectorizer, 'idf_'):
        data_processor.vectorizer.fit(texts) # Fit on all available text for consistent vectorization
    
    X_train_features = data_processor.vectorizer.transform(X_train_text)


    model_trainer.train_model(X_train_features, y_train)

    X_test_features = data_processor.vectorizer.transform(X_test_text) # Use already fitted vectorizer
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

        if "quit" in user_input_voice.lower() or "exit" in user_input_voice.lower() or "stop prediction" in user_input_voice.lower():
            speak("Exiting sentiment prediction mode. Goodbye.")
            break
        elif user_input_voice:
            sentiment = predictor.predict_sentiment(user_input_voice)
            speak(f"The predicted sentiment is: {sentiment}")


def main_voice_assistant():
    """Main loop for the voice assistant."""
    initialize_tts_engine()
    initialize_gemini() 

    speak(f"Hello, I am {ASSISTANT_NAME}. How may I assist you?")

   
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        speak("No trained sentiment model found. I need to train a new one first.")
        train_new_model()
        speak("Training complete. I am now ready.")
    else:
        speak("Trained sentiment model found. I am ready to serve.")

    while True:
        command = listen()
        if not command: 
            continue
            
        command_lower = command.lower() 

        if "goodbye" in command_lower or "exit" in command_lower or "shut down" in command_lower or "stop listening" in command_lower:
            speak("Goodbye. I am powering down.")
            break

        elif "my name is" in command_lower:
            name_parts = command_lower.split("my name is", 1)
            if len(name_parts) > 1:
                user_name = name_parts[1].strip().title()
                speak(f"It's a pleasure to meet you, {user_name}. Hello {user_name}, how can I assist you today?")
            else:
                speak("I heard you say 'my name is', but I didn't catch your name. Could you please tell me your name?")

        elif "who built you" in command_lower or "who is your maker" in command_lower or "who created you" in command_lower:
            speak("I am programmed by Thabang Mthimkulu.")

        elif "hello" in command_lower or "hi" in command_lower:
            speak(f"Hello there. How can I help you today?")
        elif "how are you" in command_lower:
            speak("I am an AI, so I don't have feelings, but I'm functioning perfectly. How can I help you?")
        elif "what is your name" in command_lower:
            speak(f"My name is {ASSISTANT_NAME}.")
        elif "thank you" in command_lower or "thanks" in command_lower:
            speak("You're welcome! Is there anything else I can assist you with?")

        elif "what's the weather" in command_lower or "how's the weather" in command_lower or "weather in" in command_lower:
            city = DEFAULT_WEATHER_CITY
            if "weather in" in command_lower:
                city_parts = command_lower.split("weather in", 1)
                if len(city_parts) > 1:
                    city = city_parts[1].strip()
            get_weather(city)

        elif "open email" in command_lower or "check email" in command_lower or "open my email" in command_lower:
            open_email_client()

        elif "go to desktop" in command_lower or "open desktop folder" in command_lower:
            try:
                desktop_path = os.path.expanduser("~/Desktop")
                open_application(desktop_path) 
                speak("Opened your desktop folder.")
            except Exception as e:
                speak(f"Sorry, I couldn't open the desktop folder: {e}")

        elif "list files" in command_lower or "show files" in command_lower or "list directory" in command_lower:
            speak("Which directory would you like me to list? For example, say 'desktop' or 'documents'.")
            dir_command = listen()
            if dir_command: 
                if "desktop" in dir_command.lower():
                    list_directory_contents("~/Desktop")
                elif "documents" in dir_command.lower():
                    list_directory_contents("~/Documents")
                elif "downloads" in dir_command.lower():
                    list_directory_contents("~/Downloads")
                elif "home" in dir_command.lower():
                    list_directory_contents("~")
                else:
                    speak("I can only list contents of your Desktop, Documents, Downloads, or Home folder at the moment. Please specify one of these.")
            else:
                speak("I didn't hear which directory you want me to list.")

        elif "open safari" in command_lower:
            open_application("Safari")
        elif "open chrome" in command_lower or "open google chrome" in command_lower:
            open_application("Google Chrome")
        elif "open notes" in command_lower:
            open_application("Notes")
        elif "open terminal" in command_lower:
            open_application("Terminal")
        elif "open calculator" in command_lower:
            open_application("Calculator")
        elif "open messages" in command_lower:
            open_application("Messages")
        elif "open application" in command_lower:
            app_name_parts = command_lower.split("open application", 1)
            if len(app_name_parts) > 1:
                app_name = app_name_parts[1].strip()
                open_application(app_name)
            else:
                speak("Which application would you like me to open? You can say 'open application' followed by the app name, like 'open application Safari'.")

      
        elif "open a tab on chrome" in command_lower or \
             "open new tab" in command_lower or \
             "open browser tab" in command_lower:
            open_browser_tab() 
            

        elif "open google" in command_lower:
            open_website(GOOGLE_URL)
        elif "open youtube" in command_lower:
            open_website(YOUTUBE_URL)
        elif "open wikipedia" in command_lower:
            open_website(WIKIPEDIA_URL)
        elif "open chatgpt" in command_lower: 
            open_website(CHATGPT_URL)
        elif "open linkedin" in command_lower:
            open_website(LINKEDIN_URL)
        elif "open ster-kinekor" in command_lower or "go to ster-kinekor" in command_lower:
            open_website(STERKINEKOR_URL)
        elif "search for" in command_lower:
            query = command_lower.split("search for", 1)[1].strip()
            if query:
                search_web(query)
            else:
                speak("What would you like me to search for?")
        elif "google" in command_lower:
            query = command_lower.split("google", 1)[1].strip()
            if query:
                search_web(query)
            else:
                speak("What would you like me to Google?")
        elif "open website" in command_lower or "go to website" in command_lower or "navigate to website" in command_lower:
            website_query_parts = command_lower.split("website", 1)
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

        elif "play" in command_lower and ("music" in command_lower or "song" in command_lower):
            song_query = command_lower.replace("play", "").replace("music", "").replace("song", "").strip()
            if "on youtube" in song_query:
                song_query = song_query.replace("on youtube", "").strip()
            if song_query:
                play_music(song_query)
            else:
                speak("What song or artist would you like me to play?")

        elif "what is the time" in command_lower or "current time" in command_lower:
            get_current_time()
        elif "what is the date" in command_lower or "current date" in command_lower:
            get_current_date()

        elif "train model" in command_lower:
            speak("Initiating model training process.")
            train_new_model()
            speak("Training process completed.")
        elif "predict sentiment" in command_lower or "analyze sentiment" in command_lower:
            handle_prediction_mode()

     
        elif any(phrase in command_lower for phrase in [
            "answer me", "general question", "tell me about", "who is",
            "what is", "why is", "how does", "can you explain",
            "can you answer", "jarvis answer", "ask you a question",
            "i have a question", "tell me something",
            "where can we improve", "what should i do", "how can i",
            "advise me", "recommendation", "give me advice", "about",
            "what can you tell me", "can you tell me"
        ]) or (len(command_lower.split()) > 2 and not any(kw in command_lower for kw in ["open", "play", "search", "train", "predict", "time", "date", "weather", "list"])):
        
            
            actual_query = ""
         
            if "tell me about" in command_lower:
                actual_query = command_lower.split("tell me about", 1)[1].strip()
            elif "who is" in command_lower:
                actual_query = command_lower.split("who is", 1)[1].strip()
            elif "what is" in command_lower:
                actual_query = command_lower.split("what is", 1)[1].strip()
            elif "why is" in command_lower:
                actual_query = command_lower.split("why is", 1)[1].strip()
            elif "how does" in command_lower:
                actual_query = command_lower.split("how does", 1)[1].strip()
            elif "can you explain" in command_lower:
                actual_query = command_lower.split("can you explain", 1)[1].strip()
            elif "can you answer" in command_lower and len(command_lower.split()) > 3:
                actual_query = command_lower.split("can you answer", 1)[1].strip()
            elif "jarvis answer" in command_lower and len(command_lower.split()) > 2:
                actual_query = command_lower.split("jarvis answer", 1)[1].strip()
            elif "tell me something about" in command_lower:
                 actual_query = command_lower.split("tell me something about", 1)[1].strip()
            elif "where can we improve on" in command_lower: 
                actual_query = command_lower.split("where can we improve on", 1)[1].strip()
            elif "what should i do about" in command_lower:
                actual_query = command_lower.split("what should i do about", 1)[1].strip()
            elif "how can i" in command_lower:
                actual_query = command_lower.split("how can i", 1)[1].strip()
            elif "advise me on" in command_lower:
                actual_query = command_lower.split("advise me on", 1)[1].strip()
            elif "recommendation for" in command_lower:
                actual_query = command_lower.split("recommendation for", 1)[1].strip()
            elif "give me advice on" in command_lower:
                actual_query = command_lower.split("give me advice on", 1)[1].strip()
            elif "what about" in command_lower:
                actual_query = command_lower.split("what about", 1)[1].strip()
            elif "can you tell me about" in command_lower:
                 actual_query = command_lower.split("can you tell me about", 1)[1].strip()
            elif "tell me" in command_lower and not actual_query: 
                actual_query = command_lower.split("tell me", 1)[1].strip()

           
            if not actual_query:
                actual_query = command_lower 

            if actual_query:
                speak("Let me check that for you.") 
                ask_gemini(actual_query)
            else:
                speak("I didn't hear a question. Please try asking again.") 


        elif "what can you do" in command_lower:
            speak("I am programmed to train a sentiment analysis model, predict sentiment from text, and search the web.")
            speak("I can open applications like Safari or Calculator, or you can say 'open application' followed by the app name.")
            speak("I can open specific websites like Google, YouTube, Wikipedia, ChatGPT, LinkedIn, and Ster-Kinekor. I can also open any website if you say 'open website' followed by the address, like 'google dot com'.")
            speak("I can open a new tab in your browser.") 
            speak("I can play music by searching YouTube for songs, tell you the time and date, and list contents of your main folders, including opening your desktop folder.")
            speak("I can also get you the current weather for a city and open your email client.")
            speak("And I can answer general questions, similar to ChatGPT.")
            speak("You can also ask me about my name or say 'my name is' followed by your name.")

        elif command_lower: # This is the ultimate catch-all for truly unrecognised commands
            speak("I heard that, but I'm not yet programmed to respond to that specific command.")
            speak("For now, I can train the model, predict sentiment, open applications, open any website, search the web, play music, tell you the time and date, get weather, open email, or answer general questions.")
            speak("You can also ask me about my name.")
            speak("You can also say 'what can you do' to hear a summary of my current abilities.")


if __name__ == "__main__":
    main_voice_assistant()