# src/config.py

import os

# --- Configuration Constants ---
DATA_FILE = 'data/training_data.csv'
MODEL_PATH = 'models/sentiment_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
ASSISTANT_NAME = "Jarvis"

# Default city for weather
DEFAULT_WEATHER_CITY = "Johannesburg"

# API Keys (loaded from .env in main.py)
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# TTS Engine Properties
TTS_RATE = 180
TTS_VOICE_INDEX = 0 # Example, can be made configurable

# Speech Recognition Properties
SR_PAUSE_THRESHOLD = 0.8
SR_ENERGY_THRESHOLD = 400
SR_LANGUAGE = 'en-US'

# URLs
GOOGLE_URL = "https://www.google.com"
YOUTUBE_URL = "https://www.youtube.com" # Corrected YouTube base URL
WIKIPEDIA_URL = "https://www.wikipedia.org" # Corrected Wikipedia URL
CHATGPT_URL = "https://chatgpt.com"
LINKEDIN_URL = "https://www.linkedin.com"
STERKINEKOR_URL = "https://www.sterkinekor.com/"
OPENWEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"