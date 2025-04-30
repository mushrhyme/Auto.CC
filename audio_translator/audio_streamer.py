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