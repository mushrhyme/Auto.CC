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