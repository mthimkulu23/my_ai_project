# src/voice_io.py
import subprocess
import sys
import pyttsx3
import speech_recognition as sr
from src.config import ASSISTANT_NAME, SR_PAUSE_THRESHOLD, SR_ENERGY_THRESHOLD, SR_LANGUAGE, TTS_RATE, TTS_VOICE_INDEX

# Global variable for the TTS engine
_engine = None # Use a leading underscore for internal global variables

def initialize_tts_engine():
    """Initializes the TTS engine based on the operating system."""
    global _engine
    if sys.platform == "darwin": # macOS
        print(f"{ASSISTANT_NAME}: Detected macOS. Using native 'say' command for TTS.")
        _engine = None # Indicate to use native command
    else: # For Linux/Windows, attempt to use pyttsx3
        try:
            _engine = pyttsx3.init('espeak')
            print(f"{ASSISTANT_NAME}: Initialized pyttsx3 with 'espeak' driver.")
        except Exception as e:
            print(f"{ASSISTANT_NAME}: Warning: 'espeak' driver failed to initialize: {e}. Attempting default driver.")
            try:
                _engine = pyttsx3.init() # Fallback to default driver
                print(f"{ASSISTANT_NAME}: Initialized pyttsx3 with default driver.")
            except Exception as e_default:
                print(f"{ASSISTANT_NAME}: Error: Default pyttsx3 driver also failed to initialize: {e_default}. TTS will be disabled.")
                _engine = None

        if _engine:
            voices = _engine.getProperty('voices')
            if voices:
                try:
                    _engine.setProperty('voice', voices[TTS_VOICE_INDEX].id)
                    print(f"{ASSISTANT_NAME}: Set voice to: {voices[TTS_VOICE_INDEX].name}")
                except IndexError:
                    print(f"{ASSISTANT_NAME}: Warning: No voices found by pyttsx3 or index out of range. Using system default voice.")
            else:
                print(f"{ASSISTANT_NAME}: Warning: No voices found by pyttsx3 at all. Using system default voice if available.")

            _engine.setProperty('rate', TTS_RATE)
            print(f"{ASSISTANT_NAME}: Set speech rate to: {_engine.getProperty('rate')}")
        else:
            print(f"{ASSISTANT_NAME}: TTS engine could not be initialized. Speech output will be through print statements only.")

def speak(text):
    """Converts text to speech, using 'say' on macOS and pyttsx3 elsewhere."""
    print(f"{ASSISTANT_NAME}: {text}")

    if sys.platform == "darwin":
        subprocess.call(['say', text])
    elif _engine:
        _engine.say(text)
        _engine.runAndWait()
    # If _engine is None, the print statement is the only output

def listen():
    """Listens for audio input and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"{ASSISTANT_NAME}: Listening...")
        r.pause_threshold = SR_PAUSE_THRESHOLD
        r.energy_threshold = SR_ENERGY_THRESHOLD
        audio = r.listen(source)
    try:
        print(f"{ASSISTANT_NAME}: Recognizing...")
        query = r.recognize_google(audio, language=SR_LANGUAGE)
        print(f"User: {query}")
        return query.lower()
    except sr.UnknownValueError:
        speak("Sorry, I could not understand your audio.")
        return ""
    except sr.RequestError as e:
        speak(f"Could not request results from Google Speech Recognition service; {e}")
        return ""