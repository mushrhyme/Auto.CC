# ---------- signal_utils.py ----------
import numpy as np

def get_audio_level(audio_data):
    arr = np.abs(np.frombuffer(audio_data, dtype=np.int16))
    sorted_arr = np.sort(arr)
    top = sorted_arr[int(len(sorted_arr)*0.9):] if len(sorted_arr)>0 else arr
    return float(np.mean(top)) if len(top)>0 else float(np.mean(arr))