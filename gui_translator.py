import sys
import time
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject
from PySide6.QtGui import QFont, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QProgressBar, QSlider, QCheckBox, QFrame, QGridLayout
)
import json


class TranslatorSignals(QObject):
    """GUI 업데이트를 위한 신호 클래스"""
    translation_update = Signal(str, str, str)  # 타임스탬프, 번역, 원문
    audio_level_update = Signal(float)  # 현재 오디오 레벨
    status_update = Signal(str)  # 상태 메시지
    voice_detected = Signal(bool)  # 음성 감지 상태

class FloatingSubtitleWindow(QMainWindow):
    """다른 프로그램이 활성화되어도 항상 최상단에 표시되는 자막 창 (화면 하단, 반투명 박스)"""
    def __init__(self, main_window):
        super().__init__(None, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.main_window = main_window
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 중앙 위젯 및 레이아웃 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)  # QGridLayout 사용
        layout.setContentsMargins(10, 5, 10, 5)

        # 각 언어별 레이블 생성
        self.korean_label = self.create_label("한국어")
        self.english_label = self.create_label("영어")
        self.chinese_label = self.create_label("중국어")
        self.japanese_label = self.create_label("일본어")

        # 4분할 배치
        layout.addWidget(self.korean_label, 0, 0)
        layout.addWidget(self.english_label, 0, 1)
        layout.addWidget(self.chinese_label, 1, 0)
        layout.addWidget(self.japanese_label, 1, 1)

        # 창 크기 및 위치 설정
        self.resize(800, 400)
        self.move_to_bottom()

        # 4분할이 균등하게 크기를 조정하도록 설정
        layout.setRowStretch(0, 1)  # 첫 번째 행 (상단)
        layout.setRowStretch(1, 1)  # 두 번째 행 (하단)
        layout.setColumnStretch(0, 1)  # 첫 번째 열 (왼쪽)
        layout.setColumnStretch(1, 1)  # 두 번째 열 (오른쪽)

    def create_label(self, title):
        """언어별 레이블 생성"""
        label = QLabel(title)
        label.setStyleSheet("color: white; font-weight: bold; background-color: rgba(0, 0, 0, 150);")
        label.setFont(QFont("Arial", 14))
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        return label

    def move_to_bottom(self):
        """화면 하단 중앙에 창 위치시키기"""
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = screen_geometry.height() - self.height() - 100  # 하단에서 100픽셀 위
        self.move(x, y)

    def update_subtitles(self, translations):
        """번역 결과를 4분할 화면에 업데이트"""
        self.korean_label.setText(translations.get('korean', ''))
        self.english_label.setText(translations.get('english', ''))
        self.chinese_label.setText(translations.get('chinese', ''))
        self.japanese_label.setText(translations.get('japanese', ''))
    
    def update_font_size(self, size):
        """자막 폰트 크기 업데이트"""
        font = QFont()
        font.setPointSize(size)
        self.korean_label.setFont(font)
        self.english_label.setFont(font)
        self.chinese_label.setFont(font)
        self.japanese_label.setFont(font)
    
    def update_subtitle_height(self, height):
        """자막 높이 업데이트"""
        self.resize(self.width(), height)

    def mousePressEvent(self, event):
        """마우스 클릭 이벤트 - 드래그 시작"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트 - 드래그 처리"""
        if event.buttons() == Qt.LeftButton and self.drag_position:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
    
    def mouseDoubleClickEvent(self, event):
        """더블 클릭으로 메인 창 표시/숨기기"""
        self.main_window.setVisible(not self.main_window.isVisible())
    
    def raise_only(self):
        """창을 최상단으로 올림 (포커스 전환 없음)"""
        self.raise_()
    
    def show(self):
        """창 표시 시 항상 최상단에 있도록 보장 (포커스 전환 없이)"""
        super().show()
        self.raise_()

class AudioTranslatorGUI(QMainWindow):
    def __init__(self, translator):
        # 메인 창은 일반 창으로 표시 (항상 위 플래그 없음)
        super().__init__()
        self.translator = translator
        self.setup_signals()
        # print("AudioTranslator에 GUI 신호 연결 중...")
        translator.set_gui_signals(self.signals)
        
        self.setWindowTitle("오디오 번역기")
        self.setMinimumWidth(400)
        
        self.setup_ui()  # 모든 위젯을 설정하고 나서 스타일을 업데이트
        
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_gui)
        self.update_timer.start(100)
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 상태 표시 영역
        status_layout = QHBoxLayout()
        self.status_label = QLabel("대기 중...")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        # 오디오 레벨 표시
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 1000)
        self.level_bar.setTextVisible(False)
        self.level_bar.setMaximumHeight(15)
        status_layout.addWidget(self.level_bar)
        
        main_layout.addLayout(status_layout)
        
        # 설정 영역
        settings_layout = QVBoxLayout()
        
        # 초기화 버튼 추가
        self.reset_button = QPushButton("초기화")
        self.reset_button.clicked.connect(self.reset_all)
        settings_layout.addWidget(self.reset_button)
        
        main_layout.addLayout(settings_layout)
        
        self.setup_menu()
        
        
        # 음성 감지 임계값
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("음성 감지 임계값:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(100, 1000)
        self.threshold_slider.setValue(self.translator.silence_threshold)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_value_label = QLabel(f"{self.translator.silence_threshold}")
        threshold_layout.addWidget(self.threshold_value_label)
        settings_layout.addLayout(threshold_layout)

        # 침묵 감지 시간
        silence_duration_layout = QHBoxLayout()
        silence_duration_layout.addWidget(QLabel("침묵 감지 시간(초):"))
        self.silence_duration_slider = QSlider(Qt.Horizontal)
        self.silence_duration_slider.setRange(5, 30)  # 0.5초~3초 (10배 적용)
        self.silence_duration_slider.setValue(int(self.translator.silence_duration * 10))
        self.silence_duration_slider.valueChanged.connect(self.update_silence_duration)
        silence_duration_layout.addWidget(self.silence_duration_slider)
        self.silence_duration_value_label = QLabel(f"{self.translator.silence_duration}")
        silence_duration_layout.addWidget(self.silence_duration_value_label)
        settings_layout.addLayout(silence_duration_layout)
        
        # 자막 투명도
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("자막 투명도:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(85)
        self.opacity_slider.valueChanged.connect(self.update_subtitle_opacity)
        opacity_layout.addWidget(self.opacity_slider)
        settings_layout.addLayout(opacity_layout)
        
        # 자막 크기 조절
        font_size_layout = QHBoxLayout()
        font_size_layout.addWidget(QLabel("자막 크기:"))
        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setRange(8, 80)
        self.font_size_slider.setValue(14)  # 기본 크기
        self.font_size_slider.valueChanged.connect(self.update_font_size)
        font_size_layout.addWidget(self.font_size_slider)
        settings_layout.addLayout(font_size_layout)
        
        # 자막 높이 조절
        subtitle_height_layout = QHBoxLayout()
        subtitle_height_layout.addWidget(QLabel("자막 높이:"))
        self.subtitle_height_slider = QSlider(Qt.Horizontal)
        self.subtitle_height_slider.setRange(100, 800)
        self.subtitle_height_slider.setValue(200)  # 기본 높이
        self.subtitle_height_slider.valueChanged.connect(self.update_subtitle_height)
        subtitle_height_layout.addWidget(self.subtitle_height_slider)
        settings_layout.addLayout(subtitle_height_layout)



        main_layout.addLayout(settings_layout)
        
        
        self.setup_menu()
        
        # 자막 창 생성 (FloatingSubtitleWindow는 항상 위로, 포커스 전환 없음)
        self.subtitle_window = FloatingSubtitleWindow(self)
        self.subtitle_window.show()
        
        self.update_subtitle_opacity(self.opacity_slider.value())
        
        # 스타일 적용
        self.update_styles()
        
    def update_styles(self):
        # 자막 레이블 스타일
        self.subtitle_window.korean_label.setStyleSheet("""
            color: white; font-weight: bold;
            background-color: rgba(0, 0, 0, 150);
            border-radius: 10px;
            padding: 5px;
        """)
        self.subtitle_window.english_label.setStyleSheet("""
            color: white; font-weight: bold;
            background-color: rgba(0, 0, 0, 150);
            border-radius: 10px;
            padding: 5px;
        """)
        self.subtitle_window.chinese_label.setStyleSheet("""
            color: white; font-weight: bold;
            background-color: rgba(0, 0, 0, 150);
            border-radius: 10px;
            padding: 5px;
        """)
        self.subtitle_window.japanese_label.setStyleSheet("""
            color: white; font-weight: bold;
            background-color: rgba(0, 0, 0, 150);
            border-radius: 10px;
            padding: 5px;
        """)

        # 상태 바 스타일
        self.status_label.setStyleSheet("""
            color: #fff;
            font-weight: bold;
            background-color: #333;
            padding: 5px;
            border-radius: 5px;
        """)
        
        # 프로그레스 바 스타일
        self.level_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2e2e2e;
                border-radius: 8px;
            }
            QProgressBar::chunk {
                background-color: #3cb371;  /* 초록색으로 채움 */
                border-radius: 8px;
            }
        """)

        # 슬라이더 스타일
        self.threshold_slider.setStyleSheet("""
            QSlider {
                background: #f0f0f0;
                height: 8px;
                border-radius: 5px;
            }
            QSlider::handle {
                background: #3cb371;
                border-radius: 10px;
                width: 15px;
            }
        """)

        # 자막 크기 및 높이 슬라이더 스타일
        self.font_size_slider.setStyleSheet("""
            QSlider {
                background: #f0f0f0;
                height: 8px;
                border-radius: 5px;
            }
            QSlider::handle {
                background: #5bc0de;
                border-radius: 10px;
                width: 15px;
            }
        """)

        self.subtitle_height_slider.setStyleSheet("""
            QSlider {
                background: #f0f0f0;
                height: 8px;
                border-radius: 5px;
            }
            QSlider::handle {
                background: #f0ad4e;
                border-radius: 10px;
                width: 15px;
            }
        """)

    def reset_all(self):
        """초기화 버튼을 눌렀을 때 호출되는 함수로 모든 작업을 중단하고 초기화"""
        # 현재 음성 감지 상태 초기화
        self.translator.voice_detected = False
        self.translator.audio_frames.clear()  # 오디오 버퍼 초기화
        self.translator.silence_threshold = 400  # 기본값으로 초기화
        self.translator.silence_duration = 1.5  # 기본값으로 초기화
        self.status_label.setText("초기화 완료. 음성을 다시 감지할 수 있습니다.")
        self.status_label.setStyleSheet("font-weight: bold; color: #d9534f;")
        
        # 자막 창 업데이트
        self.subtitle_window.update_subtitles({'korean': '', 'english': '', 'chinese': '', 'japanese': ''})

        # 상태 바 리셋
        self.level_bar.setValue(0)

    def update_gui(self):
        if hasattr(self.translator, 'audio_frames') and self.translator.audio_frames:
            with self.translator.buffer_lock:
                if len(self.translator.audio_frames) > 0:
                    audio_level = self.translator.get_audio_level(self.translator.audio_frames[-1])
                    self.signals.audio_level_update.emit(audio_level)
        if hasattr(self.translator, 'voice_detected'):
            self.signals.voice_detected.emit(self.translator.voice_detected)
            
    def setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("파일")
        
        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close_application)
        file_menu.addAction(exit_action)
    
    def setup_signals(self):
        self.signals = TranslatorSignals()
        self.signals.translation_update.connect(self.on_translation_update)
        self.signals.audio_level_update.connect(self.on_audio_level_update)
        self.signals.status_update.connect(self.on_status_update)
        self.signals.voice_detected.connect(self.on_voice_detected)
    
    def update_gui(self):
        if hasattr(self.translator, 'audio_frames') and self.translator.audio_frames:
            with self.translator.buffer_lock:
                if len(self.translator.audio_frames) > 0:
                    audio_level = self.translator.get_audio_level(self.translator.audio_frames[-1])
                    self.signals.audio_level_update.emit(audio_level)
        if hasattr(self.translator, 'voice_detected'):
            self.signals.voice_detected.emit(self.translator.voice_detected)
    
    @Slot(str, str, str)
    def on_translation_update(self, timestamp, translation, original):
        translations = json.loads(translation)  # 전달된 문자열을 딕셔너리로 변환
        if hasattr(self, 'subtitle_window') and self.subtitle_window:
            self.subtitle_window.update_subtitles(translations)
    
    @Slot(float)
    def on_audio_level_update(self, level):
        self.level_bar.setValue(int(level))
        if level > self.translator.silence_threshold:
            self.level_bar.setStyleSheet(
                "QProgressBar { background-color: #f0f0f0; border: 1px solid #c0c0c0; border-radius: 2px; } "
                "QProgressBar::chunk { background-color: #5cb85c; }"
            )
        else:
            self.level_bar.setStyleSheet(
                "QProgressBar { background-color: #f0f0f0; border: 1px solid #c0c0c0; border-radius: 2px; } "
                "QProgressBar::chunk { background-color: #d9534f; }"
            )
    
    @Slot(str)
    def on_status_update(self, status):
        """상태 업데이트 (감지된 언어 표시)"""
        self.status_label.setText(status)
        if "감지 중입니다" in status:
            self.status_label.setStyleSheet("font-weight: bold; color: #5cb85c;")
        else:
            self.status_label.setStyleSheet("font-weight: bold; color: black;")
    
    @Slot(bool)
    def on_voice_detected(self, detected):
        if detected:
            self.status_label.setText("음성 감지 중...")
            self.status_label.setStyleSheet("font-weight: bold; color: #5cb85c;")
        else:
            self.status_label.setText("음성을 인식할 준비가 완료되었습니다")
            self.status_label.setStyleSheet("font-weight: bold; color: black;")
            
    def update_font_size(self):
        size = self.font_size_slider.value()
        self.subtitle_window.update_font_size(size)

    def update_subtitle_height(self):
        height = self.subtitle_height_slider.value()
        self.subtitle_window.update_subtitle_height(height)

    def update_threshold(self):
        value = self.threshold_slider.value()
        self.threshold_value_label.setText(f"{value}")
        self.translator.silence_threshold = value
    
    def update_silence_duration(self):
        value = self.silence_duration_slider.value() / 10.0
        self.silence_duration_value_label.setText(f"{value:.1f}")
        self.translator.silence_duration = value
        self.translator.silence_chunks = int(self.translator.silence_duration / self.translator.chunk_duration)
        self.translator.save_config()
    
    def update_translation_mode(self):
        mode_index = self.mode_combo.currentIndex()
        self.translator.translation_mode = "realtime" if mode_index == 0 else "complete"
        self.translator.save_config()
    
    def update_subtitle_opacity(self, value):
        opacity = value / 100.0
        self.subtitle_window.setWindowOpacity(opacity)
        self.subtitle_window.setStyleSheet(f"background-color: rgba(0, 0, 0, {opacity});")
        
    def show_subtitle_only(self):
        self.hide()
        self.subtitle_window.show()

    
    def close_application(self):
        self.translator.is_running = False
        time.sleep(1)
        QApplication.quit()
    
    def closeEvent(self, event):
        self.close_application()
        event.accept()

def start_gui(translator):
    app = QApplication(sys.argv)
    window = AudioTranslatorGUI(translator)
    window.show()
    sys.exit(app.exec())

# 예시 실행을 위한 DummyTranslator 클래스
class DummyTranslator:
    def __init__(self):
        self.silence_threshold = 400
        self.silence_duration = 1.5
        self.translation_mode = "realtime"
        self.audio_frames = []
        self.voice_detected = False
        self.chunk_duration = 0.064
        from threading import Lock
        self.buffer_lock = Lock()
    
    def set_gui_signals(self, signals):
        self.signals = signals
    
    def save_config(self):
        print("Config saved")

if __name__ == "__main__":
    translator = DummyTranslator()
    start_gui(translator)