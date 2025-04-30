# Directory structure:
# audio_translator/
# ├── constants.py
# ├── config_manager.py
# ├── logger_manager.py
# ├── vad_detector.py
# ├── audio_utils.py
# ├── signal_utils.py
# ├── audio_streamer.py
# ├── clients.py
# └── main.py

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

# ---------- logger_manager.py ----------
import logging
from datetime import datetime
from constants import LOG_DIR

class LoggerManager:
    @staticmethod
    def setup_logger(name="TranslationLogger"):
        LOG_DIR.mkdir(exist_ok=True)
        log_filename = LOG_DIR / f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logger = logging.getLogger(name)
        if logger.handlers:
            for h in logger.handlers:
                logger.removeHandler(h)

        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(log_filename, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        return logger

# ---------- vad_detector.py ----------
import numpy as np
import collections
from constants import DEFAULT_SILENCE_THRESHOLD, DEFAULT_SILENCE_DURATION, RATE, CHUNK

class VADDetector:
    def __init__(self, threshold=None, duration=None, sample_rate=RATE, chunk_size=CHUNK, dynamic_k=1.5):
        self.threshold = threshold or DEFAULT_SILENCE_THRESHOLD
        self.chunk_duration = chunk_size / sample_rate
        self.silence_chunks = int((duration or DEFAULT_SILENCE_DURATION) / self.chunk_duration)
        self.bg_levels = collections.deque(maxlen=int((sample_rate/chunk_size)*5))
        self.bg_mean = self.threshold
        self.bg_std = self.threshold / 2
        self.dynamic_k = dynamic_k
        self.prev_fft = None
        self.silence_count = 0
        self.voice_detected = False

    def update_background(self, rms):
        if rms < self.threshold:
            self.bg_levels.append(rms)
        if self.bg_levels:
            self.bg_mean = np.mean(self.bg_levels)
            self.bg_std = np.std(self.bg_levels)
        self.dynamic_threshold = self.bg_mean + self.dynamic_k * self.bg_std

    def get_audio_features(self, audio_data):
        samples = np.abs(np.frombuffer(audio_data, dtype=np.int16)).astype(np.float32)
        # RMS 계산
        rms = np.sqrt(np.mean(samples**2))
        # Spectral Flux 계산
        fft_mag = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        prev = self.prev_fft if self.prev_fft is not None else fft_mag
        flux = np.sum((fft_mag - prev).clip(min=0))
        self.prev_fft = fft_mag
        return rms, flux

    def should_transcribe(self, audio_data):
        rms, flux = self.get_audio_features(audio_data)
        self.update_background(rms)
        th = self.dynamic_threshold
        speech = (rms > th) or (flux > th * 0.5)
        if speech:
            self.silence_count = 0
            self.voice_detected = True
            return True
        else:
            self.silence_count += 1
            if self.silence_count > self.silence_chunks and self.voice_detected:
                self.voice_detected = False
            return False

# ---------- audio_utils.py ----------
import wave
import tempfile

from constants import RATE

def save_wav(frames, channels, rate=RATE, sampwidth=2, temp=True):
    if not frames:
        return None
    if temp:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        filename = tmp.name
        tmp.close()
    else:
        filename = 'realtime_audio.wav'
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return filename

# ---------- signal_utils.py ----------
import numpy as np

def get_audio_level(audio_data):
    arr = np.abs(np.frombuffer(audio_data, dtype=np.int16))
    sorted_arr = np.sort(arr)
    top = sorted_arr[int(len(sorted_arr)*0.9):] if len(sorted_arr)>0 else arr
    return float(np.mean(top)) if len(top)>0 else float(np.mean(arr))

# ---------- audio_streamer.py ----------
import pyaudio

from constants import FORMAT, CHANNELS, RATE, CHUNK
from signal_utils import get_audio_level

class AudioStreamer:
    def __init__(self, device_index=None):
        self.device_index = device_index
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def list_devices(self):
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                devices.append((i, info.get('name'), int(info.get('defaultSampleRate'))))
        return devices

    def open_stream(self):
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=CHUNK
            )
        except Exception:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

    def read_chunk(self):
        data = self.stream.read(CHUNK, exception_on_overflow=False)
        level = get_audio_level(data)
        return data, level

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

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

# ---------- main.py ----------
import os
import asyncio
import time
from functools import partial

from config_manager import ConfigManager
from logger_manager import LoggerManager
from vad_detector import VADDetector
from audio_streamer import AudioStreamer
from audio_utils import save_wav
from clients import WhisperClient, GPTClient
from constants import RATE, CHUNK, CHANNELS

async def capture_loop(streamer: AudioStreamer, vad: VADDetector, queue: asyncio.Queue, stop_event: asyncio.Event, logger):
    loop = asyncio.get_event_loop()
    recording = False
    frames = []
    silence_count = 0

    while not stop_event.is_set():
        # 블로킹 read_chunk를 스레드 풀에서 실행
        data, level = await loop.run_in_executor(None, streamer.read_chunk)
        logger.debug(f"Audio level: {level:.1f}, Dynamic TH: {vad.dynamic_threshold:.1f}")

        if vad.should_transcribe(data):
            if not recording:
                recording = True
                frames = []
                silence_count = 0
                logger.info("▶️ 발화 시작: 녹음 중")
            frames.append(data)
        else:
            if recording:
                silence_count += 1
                frames.append(data)
                if silence_count >= vad.silence_chunks:
                    logger.info("⏹️ 발화 종료: 버퍼를 큐에 추가합니다.")
                    await queue.put(frames.copy())
                    recording = False
                    frames = []
                    silence_count = 0
        await asyncio.sleep(0)

async def process_loop(queue: asyncio.Queue, whisper: WhisperClient, gpt: GPTClient, stop_event: asyncio.Event, logger):
    loop = asyncio.get_event_loop()
    min_frames = int((RATE * 0.5) / CHUNK)
    while not stop_event.is_set():
        frames = await queue.get()
        if len(frames) < min_frames:
            logger.debug("짧은 발화(0.5초 미만) 무시")
            continue
        # WAV 저장
        path = await loop.run_in_executor(None, partial(save_wav, frames, CHANNELS, RATE))
        # 전사
        text = await loop.run_in_executor(None, partial(whisper.transcribe, path))
        if text:
            logger.info(f"🔤 인식된 텍스트: {text}")
            # 번역
            translation = await loop.run_in_executor(None, partial(gpt.translate, text))
            if translation:
                logger.info(f"📝 번역 결과: {translation}")

async def main_async():
    # 설정 및 로거
    config = ConfigManager()
    logger = LoggerManager.setup_logger()
    logger.info("🔊 비동기 오디오 번역 시스템을 시작합니다.")

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error('❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.')
        return

    # 오디오 스트리머 설정
    streamer = AudioStreamer(device_index=config['preferred_device'])
    devices = streamer.list_devices()
    logger.info(f"🎤 사용 가능한 오디오 입력 장치: {devices}")
    streamer.open_stream()
    logger.info("✅ 스트리밍 시작 완료")

    # VAD 초기화 및 소음 캘리브레이션
    vad = VADDetector(
        threshold=config['silence_threshold'],
        duration=config['silence_duration'],
        sample_rate=RATE,
        chunk_size=CHUNK
    )
    logger.info("🌙 환경 소음 캘리브레이션: 2초간 가만히...")
    for _ in range(int((RATE/CHUNK)*2)):
        data, _ = streamer.read_chunk()
        rms, _ = vad.get_audio_features(data)
        vad.update_background(rms)
        await asyncio.sleep(0.01)
    logger.info(f"🌙 초기 동적 임계치: {vad.dynamic_threshold:.1f}")

    whisper = WhisperClient(api_key)
    gpt = GPTClient(api_key)

    queue = asyncio.Queue()
    stop_event = asyncio.Event()

    # 태스크 생성
    cap_task = asyncio.create_task(capture_loop(streamer, vad, queue, stop_event, logger))
    proc_task = asyncio.create_task(process_loop(queue, whisper, gpt, stop_event, logger))

    # 종료 신호 대기
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        cap_task.cancel()
        proc_task.cancel()
        streamer.close()
        logger.info("🛑 시스템 종료 완료")

if __name__ == '__main__':
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
