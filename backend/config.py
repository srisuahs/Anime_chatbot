import os
from dotenv import load_dotenv
from pathlib import Path

# --- .env file loading (Corrected) ---
# This robustly finds the .env file at the project root by going up two levels
# from the current file's location (backend/config.py -> backend -> anime-chatbot)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- JWT Settings ---
SECRET_KEY = os.getenv("SECRET_KEY", "a_super_secret_key_for_dev_only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 # Token will be valid for 1 hour

# --- Google API Settings ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
if not GOOGLE_CLIENT_ID:
    raise RuntimeError("GOOGLE_CLIENT_ID not found. Please check your main project .env file.")

# --- Gemini API Settings ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Please check your main project .env file.")

# --- MyAnimeList API Settings ---
MAL_CLIENT_ID = os.getenv("MAL_CLIENT_ID")
if not MAL_CLIENT_ID:
    raise RuntimeError("MAL_CLIENT_ID not found. Please check your main project .env file.")

MAL_CLIENT_SECRET = os.getenv("MAL_CLIENT_SECRET")
if not MAL_CLIENT_SECRET:
    raise RuntimeError("MAL_CLIENT_SECRET not found. Please check your main project .env file.")

MAL_REDIRECT_URI = os.getenv("MAL_REDIRECT_URI", "http://127.0.0.1:8000/mal/callback") # Backend callback endpoint

# --- Chatbot Settings ---

