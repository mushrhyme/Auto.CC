import threading
import time
from audio_translator import AudioTranslator

# 디버깅용 GUI 시그널 스텁: 레벨만 출력
class DebugSignal:
    def __init__(self, prefix=""):
        self.prefix = prefix

    def emit(self, value):
        # 숫자면 소수점 한 자리 포맷, 아니면 문자열 그대로 출력
        if isinstance(value, (int, float)):
            print(f"{self.prefix}{value:.1f}")
        else:
            print(f"{self.prefix}{value}")

class DebugSignals:
    def __init__(self):
        self.audio_level_update = DebugSignal(prefix="Audio level: ")
        self.status_update      = DebugSignal(prefix="Status: ")

def main(run_seconds: float = 30.0):
    translator = AudioTranslator()
    translator.set_gui_signals(DebugSignals())

    # 캡처 스레드 시작
    t = threading.Thread(target=translator.audio_capture, daemon=True)
    t.start()

    print(f"🔴 오디오 캡처 시작 (약 {run_seconds}초)... Ctrl+C로 중단")
    try:
        time.sleep(run_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        # 캡처 종료 플래그
        translator.is_running = False
        t.join()

        # 저장된 프레임을 WAV 로 덤프
        wav_path = translator.save_audio_to_wav(
            frames=translator.audio_frames,
            temp=False,
            channels=translator.audio_frames and translator.audio_frames[0] and 1
        )
        print(f"🔈 저장된 파일: {wav_path}")

        print("🛑 오디오 캡처 종료")

if __name__ == "__main__":
    main()