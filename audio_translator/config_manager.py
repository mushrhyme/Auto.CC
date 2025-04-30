# ---------- config_manager.py ----------
import json
from pathlib import Path
from constants import CONFIG_FILE, DEFAULT_SILENCE_THRESHOLD, DEFAULT_SILENCE_DURATION, DEFAULT_REALTIME_UPDATE_INTERVAL

class ConfigManager:
    def __init__(self):
        self.config_file = CONFIG_FILE
        self.config = {
            "silence_threshold": DEFAULT_SILENCE_THRESHOLD,
            "silence_duration": DEFAULT_SILENCE_DURATION,
            "preferred_device": None,
            "update_interval": DEFAULT_REALTIME_UPDATE_INTERVAL,
            "translation_mode": "complete"
        }
        self.load()

    def load(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    self.config.update(saved)
            except IOError as e:
                raise RuntimeError(f"Failed to load config: {e}")

    def save(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            raise RuntimeError(f"Failed to save config: {e}")

    def __getitem__(self, key):
        return self.config.get(key)

    def __setitem__(self, key, value):
        self.config[key] = value