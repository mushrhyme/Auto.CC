# ---------- constants.py ----------
import pyaudio
from pathlib import Path

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

DEFAULT_SILENCE_THRESHOLD = 200
DEFAULT_SILENCE_DURATION = 1.5
DEFAULT_REALTIME_UPDATE_INTERVAL = 1.0
MAX_SENTENCE_LENGTH = 50

GPT_MODEL = "gpt-4o-mini-2024-07-18"
TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
TRANSLATION_URL = "https://api.openai.com/v1/chat/completions"

LOG_DIR = Path("logs")
CONFIG_FILE = Path("config.json")