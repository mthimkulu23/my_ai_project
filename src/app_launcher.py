import os
import subprocess
import webbrowser
import platform
from datetime import datetime # Added for time/date functions

# Assuming speak is available or imported within this module
from src.voice_io import speak

def open_email_client():
    """Opens the default email client."""
    speak("Opening your email client.")
    try:
        if platform.system() == "Darwin": # macOS
            subprocess.call(['open', '-a', 'Mail'])
        elif platform.system() == "Windows":
            os.startfile("mailto:")
        else: # Linux (xdg-open is common)
            subprocess.call(['xdg-open', 'mailto:'])
    except Exception as e:
        speak(f"Sorry, I couldn't open the email client: {e}")

def open_application(app_name):
    """Opens a specified application."""
    speak(f"Opening {app_name}.")
    try:
        if platform.system() == "Darwin": # macOS
            # For applications, 'open -a' is usually the best approach
            subprocess.call(['open', '-a', app_name])
        elif platform.system() == "Windows":
            # For Windows, you might need the full path to the .exe or a known command
            # This is a very basic attempt. For robustness, you'd need a mapping or search.
            os.startfile(app_name) # Tries to open by name if it's in PATH or a known association
        else: # Linux
            # For Linux, applications are often launched by their command name
            subprocess.call([app_name.lower()]) # Convert to lowercase, common for Linux commands
            speak(f"Application opening for {platform.system()} is not fully implemented yet for {app_name}. Trying best effort.")
    except FileNotFoundError:
        speak(f"Sorry, I couldn't find the application '{app_name}'. Please ensure it's installed and accessible.")
    except Exception as e:
        speak(f"An error occurred while trying to open {app_name}: {e}")

def open_website(url):
    """Opens a website in the default web browser."""
    speak(f"Opening {url}.")
    try:
        webbrowser.open_new_tab(url)
    except Exception as e:
        speak(f"Sorry, I couldn't open the website: {e}")

def search_web(query):
    """Performs a web search using Google."""
    search_url = f"https://www.google.com/search?q={query}"
    speak(f"Searching the web for {query}.")
    try:
        webbrowser.open_new_tab(search_url)
    except Exception as e:
        speak(f"Sorry, I couldn't perform the web search: {e}")

def play_music(song_query):
    """Searches and plays music on YouTube."""
    # Note: Your original YouTube URL in config was "https://www.youtube.com"
    # This URL for search is likely more effective:
    Youtube_url = f"https://www.youtube.com/results?search_query={song_query}"
    speak(f"Playing {song_query} on YouTube.")
    try:
        webbrowser.open_new_tab(Youtube_url)
    except Exception as e:
        speak(f"Sorry, I couldn't play music: {e}")

def get_current_time():
    """Speaks the current time."""
    now = datetime.now()
    current_time = now.strftime("%I:%M %p") # e.g., 03:05 PM
    speak(f"The current time is {current_time}.")

def get_current_date():
    """Speaks the current date."""
    now = datetime.now()
    current_date = now.strftime("%A, %B %d, %Y") # e.g., Tuesday, June 24, 2025
    speak(f"Today's date is {current_date}.")

def list_directory_contents(path):
    """Lists contents of a specified directory."""
    abs_path = os.path.expanduser(path)
    if not os.path.isdir(abs_path):
        speak(f"Sorry, '{path}' is not a valid directory.")
        return

    # Nicer name for speaking
    display_name = abs_path.split(os.sep)[-1] if abs_path != os.path.expanduser("~") else 'your home directory'
    speak(f"Listing contents of {display_name}.")
    try:
        contents = os.listdir(abs_path)
        if contents:
            speak("Here are some items:")
            # Limit to first few items to avoid very long speeches
            for item in contents[:5]:
                speak(item)
            if len(contents) > 5:
                speak(f"And {len(contents) - 5} more items.")
        else:
            speak("The directory is empty.")
    except Exception as e:
        speak(f"Sorry, I couldn't list the directory contents: {e}")

# --- NEW FUNCTION FOR OPENING BROWSER TABS ---
def open_browser_tab(url="https://www.google.com"):
    """Opens a new tab in the default web browser.
       On macOS, tries to specifically open in Chrome if available.
    """
    speak(f"Opening a new tab.")
    
    system = platform.system()
    
    try:
        if system == "Darwin": # macOS
            # Attempt to open in Google Chrome specifically on macOS
            try:
                # This command is often more reliable for specific browsers on macOS
                subprocess.run(['open', '-a', 'Google Chrome', url], check=True)
            except FileNotFoundError:
                # Fallback to default browser if Chrome isn't found or 'open -a' fails
                print("Google Chrome application not found, trying default browser.")
                webbrowser.open_new_tab(url)
            except subprocess.CalledProcessError as e:
                print(f"Error opening Chrome with 'open -a': {e}, trying default browser.")
                webbrowser.open_new_tab(url)
        else: # Windows/Linux
            webbrowser.open_new_tab(url)
        
        speak("A new browser tab has been opened.")
    except webbrowser.Error as e:
        speak(f"Sorry, I couldn't open a web browser tab: {e}")
    except Exception as e:
        speak(f"An unexpected error occurred while trying to open a tab: {e}")