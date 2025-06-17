# src/integrations.py
import requests
import google.generativeai as genai
from src.config import OPENWEATHER_API_KEY, GEMINI_API_KEY, ASSISTANT_NAME, OPENWEATHER_BASE_URL
from src.voice_io import speak

# Global Gemini model instance
_gemini_model = None

def initialize_gemini():
    """Configures the Gemini API and finds a suitable model."""
    global _gemini_model
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        try:
            available_models = [m for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
            found_gemini_pro = next((m for m in available_models if 'gemini-pro' in m.name), None)

            if found_gemini_pro:
                _gemini_model = genai.GenerativeModel(found_gemini_pro.name)
                print(f"{ASSISTANT_NAME}: Configured Gemini API with model: {found_gemini_pro.name}")
            elif available_models:
                _gemini_model = genai.GenerativeModel(available_models[0].name)
                print(f"{ASSISTANT_NAME}: Configured Gemini API with fallback model: {available_models[0].name}")
            else:
                print(f"{ASSISTANT_NAME}: Warning: No suitable Gemini models found that support 'generateContent'.")
        except Exception as e:
            print(f"{ASSISTANT_NAME}: Error configuring Gemini API: {e}")
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
        speak(response.text)
    except Exception as e:
        speak(f"I encountered an error while trying to answer your question: {e}")
        speak("Please try asking again.")