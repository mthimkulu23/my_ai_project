# src/app_launcher.py
import subprocess
import os
import sys
import time
from src.voice_io import speak

def open_email_client():
    """Opens the default email client or a webmail service."""
    speak("Opening your email client.")
    if sys.platform == "darwin": # macOS
        subprocess.run(["open", "-a", "Mail"], check=False)
    elif sys.platform == "win32": # Windows
        os.startfile("mailto:")
    elif sys.platform == "linux": # Linux
        subprocess.run(["xdg-open", "mailto:"], check=False)
    else:
        speak("Sorry, I don't know how to open the email client on this operating system.")

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
    Youtube_url = f"https://www.youtube.com/results?search_query={song_name.replace(' ', '+')}"
    open_website(Youtube_url)

def get_current_time():
    """Tells the current time."""
    current_time = time.strftime("%I:%M %p %Z")
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