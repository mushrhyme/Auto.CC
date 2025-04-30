# ---------- clients.py ----------
import requests
from backoff import on_exception, expo
from constants import TRANSCRIPTION_URL, TRANSLATION_URL, GPT_MODEL

class WhisperClient:
    def __init__(self, api_key):
        self.api_key = api_key

    @on_exception(expo, (requests.exceptions.RequestException,), max_tries=3)
    def transcribe(self, file_path):
        with open(file_path, 'rb') as f:
            files = {
                'file': (file_path, f, 'audio/wav'),
                'model': (None, 'whisper-1'),
                'language': (None, 'en'),
                'response_format': (None, 'json')
            }
            headers = {'Authorization': f'Bearer {self.api_key}'}
            r = requests.post(TRANSCRIPTION_URL, headers=headers, files=files)
        if r.status_code == 200:
            return r.json().get('text', '')
        raise RuntimeError(f"Transcription error: {r.status_code}\n{r.text}")

class GPTClient:
    def __init__(self, api_key):
        self.api_key = api_key

    @on_exception(expo, (requests.exceptions.RequestException, Exception), max_tries=3)
    def translate(self, text):
        if not text.strip():
            return ''
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': GPT_MODEL,
            'messages': [
                {'role': 'system', 'content': 'You are a professional English to Korean translator.'},
                {'role': 'user', 'content': text}
            ]
        }
        r = requests.post(TRANSLATION_URL, headers=headers, json=data)
        if r.status_code == 200:
            choices = r.json().get('choices', [])
            if choices:
                return choices[0]['message']['content']
        raise RuntimeError(f"Translation error: {r.status_code}\n{r.text}")