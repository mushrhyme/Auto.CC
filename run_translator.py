from audio_translator import AudioTranslator
from gui_translator import start_gui

def main():
    # AudioTranslator 인스턴스 생성
    translator = AudioTranslator()
    
    # 스레드 시작
    translator.start_threads()
    
    # GUI 시작
    start_gui(translator)

if __name__ == "__main__":
    main() 