from translator import Translator
from gui import start_gui
import sys

if __name__ == "__main__":
    # 번역기 초기화 및 백그라운드 시작
    translator = Translator()
    translator.start()

    # GUI 실행 (종료 시 애플리케이션 루프 반환값을 통해 중지)
    exit_code = start_gui(translator)

    # 백그라운드 루프 정리
    translator.stop()
    sys.exit(exit_code)
