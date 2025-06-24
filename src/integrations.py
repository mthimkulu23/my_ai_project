# src/integrations.py
import requests
import google.generativeai as genai
from src.config import OPENWEATHER_API_KEY, GEMINI_API_KEY, ASSISTANT_NAME, OPENWEATHER_BASE_URL
from src.voice_io import speak # Ensure speak is correctly imported

# Global Gemini model instance
_gemini_model = None

def initialize_gemini():
    """Configures the Gemini API and finds a suitable model."""
    global _gemini_model
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        try:
            preferred_models = ['gemini-1.5-flash', 'gemini-1.5-pro']
            
            selected_model_name = None
            for model_name in preferred_models:
                try:
                    # Attempt to initialize with the preferred model name.
                    # If this line succeeds, it means the model_name is valid
                    # and the model object is created.
                    temp_model = genai.GenerativeModel(model_name)
                    
                    # We remove the 'supported_generation_methods' check here,
                    # as it's not applicable to the GenerativeModel instance.
                    # If it successfully instantiates, it's generally good to go
                    # for generate_content.
                    
                    selected_model_name = model_name
                    break # Found a suitable model, stop searching
                except Exception as e:
                    # Catch specific errors if a model is truly unavailable or deprecated.
                    print(f"{ASSISTANT_NAME}: Failed to initialize model '{model_name}': {e}")
                    continue # Try the next preferred model

            if selected_model_name:
                _gemini_model = genai.GenerativeModel(selected_model_name)
                print(f"{ASSISTANT_NAME}: Configured Gemini API with model: {selected_model_name}")
            else:
                print(f"{ASSISTANT_NAME}: Warning: No suitable Gemini models from preferred list could be initialized.")
                print(f"{ASSISTANT_NAME}: ChatGPT-like functionality will be disabled.")

        except Exception as e:
            print(f"{ASSISTANT_NAME}: Error during Gemini API configuration: {e}")
            print(f"{ASSISTANT_NAME}: ChatGPT-like functionality will be disabled.")
    else:
        print(f"{ASSISTANT_NAME}: Warning: GEMINI_API_KEY not found. ChatGPT-like functionality will be disabled.")

def get_weather(city):
    """Fetches and speaks the current weather for a given city."""
    if not OPENWEATHER_API_KEY:
        speak("I cannot get weather information. The OpenWeatherMap API key is not configured.")
        return

    complete_url = f"{OPENWEATHER_BASE_URL}q={city}&appid={OPENWEATHER_API_KEY}&units=metric"

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

def ask_gemini(query):
    """Sends a query to the Gemini API and speaks the response."""
    if not _gemini_model:
        speak("I cannot answer general questions. The Gemini API is not configured.")
        return

    speak("Thinking...")
    try:
        response = _gemini_model.generate_content(query)
        # Check if the response contains text before speaking
        if hasattr(response, 'text') and response.text:
            speak(response.text)
        else:
            speak("Gemini did not provide a clear text response.")
            print(f"DEBUG: Gemini response details: {response}") # For debugging
    except Exception as e:
        speak(f"I encountered an error while trying to answer your question: {e}")
        speak("Please try asking again.")