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