import sys
import time
import json
from datetime import datetime
from threading import Lock

from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject
from PySide6.QtGui import QFont, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QProgressBar, QSlider, QGridLayout
)

# ---------- SIGNAL DEFINITION ----------
class TranslatorSignals(QObject):
    """각각의 이벤트 발생을 위한 Signal 정의"""
    translation_update = Signal(str, str, str)  # timestamp, translation (JSON str), original
    audio_level_update = Signal(float)
    status_update = Signal(str)
    voice_detected = Signal(bool)
    translation_started = Signal()
    
# ---------- FLOATING SUBTITLE WINDOW ----------
class FloatingSubtitleWindow(QMainWindow):
    """
    자막 창을 위한 클래스
    자막, 진행 바, 음성 레벨 등을 업데이트하고, 자막 창의 위치와 크기 등을 설정
    """
    def __init__(self, main_window):
        super().__init__(None, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.main_window = main_window
        self.setAttribute(Qt.WA_TranslucentBackground)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 최상위 수직 레이아웃
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(10, 5, 10, 5)
        outer_layout.setSpacing(5)

        # 게이지바 추가
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 1000)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(6)
        self.level_bar.setStyleSheet("""
            QProgressBar { background-color: #2e2e2e; border-radius: 3px; }
            QProgressBar::chunk { background-color: #3cb371; border-radius: 3px; }
        """)
        outer_layout.addWidget(self.level_bar)

        # 자막 그리드
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(5)

        self.korean_label = self.create_label("한국어")
        self.english_label = self.create_label("영어")
        self.chinese_label = self.create_label("중국어")
        self.japanese_label = self.create_label("일본어")

        grid_layout.addWidget(self.korean_label, 0, 0)
        grid_layout.addWidget(self.english_label, 0, 1)
        grid_layout.addWidget(self.chinese_label, 1, 0)
        grid_layout.addWidget(self.japanese_label, 1, 1)

        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        outer_layout.addLayout(grid_layout)

        # 하단 진행 표시 바
        self.translation_bar = QProgressBar()
        self.translation_bar.setRange(0, 100)
        self.translation_bar.setValue(0)
        self.translation_bar.setTextVisible(False)
        self.translation_bar.setFixedHeight(6)
        self.translation_bar.setStyleSheet("""
            QProgressBar { background-color: #2e2e2e; border-radius: 3px; }
            QProgressBar::chunk { background-color: #1e90ff; border-radius: 3px; }
        """)
        outer_layout.addWidget(self.translation_bar)

        # 타이머 설정
        self.translation_timer = QTimer()
        self.translation_timer.timeout.connect(self._update_translation_progress)
        self._progress_value = 0

        self.resize(800, 400)
        self.move_to_bottom()

    # ---------- 자막 및 레이아웃 ----------
    def create_label(self, title):
        """라벨 생성 함수"""
        label = QLabel(title)
        label.setStyleSheet("color: white; font-weight: bold; background-color: rgba(0, 0, 0, 150);")
        label.setFont(QFont("Arial", 14))
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        return label

    def move_to_bottom(self):
        """자막 창을 화면 하단에 위치시키는 함수"""
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = screen_geometry.height() - self.height() - 100
        self.move(x, y)

    def update_subtitles(self, translations):
        """자막을 업데이트하는 함수"""
        self.korean_label.setText(translations.get('korean', ''))
        self.english_label.setText(translations.get('english', ''))
        self.chinese_label.setText(translations.get('chinese', ''))
        self.japanese_label.setText(translations.get('japanese', ''))

    def update_font_size(self, size):
        """자막 폰트 크기 변경 함수"""
        font = QFont()
        font.setPointSize(size)
        for label in [self.korean_label, self.english_label, self.chinese_label, self.japanese_label]:
            label.setFont(font)

    def update_subtitle_height(self, height):
        """자막 높이 변경 함수"""
        self.resize(self.width(), height)

    # ---------- 번역 진행 표시 바 ----------
    def start_translation_progress(self, estimated_duration):
        """번역 시작 시 진행 표시 바 시작"""
        self._progress_value = 0
        self._start_time = time.time()
        self._estimated_duration = estimated_duration
        self.translation_bar.setValue(0)
        self.translation_bar.show()
        self.translation_timer.start(100)

    def _update_translation_progress(self):
        """번역 진행 상황을 갱신하는 함수"""
        if not hasattr(self, '_start_time'):
            return

        elapsed = time.time() - self._start_time
        progress = int((elapsed / self._estimated_duration) * 100)

        if progress < 100:
            self.translation_bar.setValue(progress)
        else:
            self.translation_timer.stop()
            self.translation_bar.setValue(100)

    def complete_translation_progress(self):
        """진행 표시 바를 0으로 리셋하는 함수"""
        QTimer.singleShot(800, lambda: self.translation_bar.setValue(0))  # 0으로 리셋

    # ---------- 마우스 이벤트 ----------
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트 처리"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """마우스 드래그 이동 이벤트 처리"""
        if event.buttons() == Qt.LeftButton and self.drag_position:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """마우스 더블클릭 이벤트 처리"""
        self.main_window.setVisible(not self.main_window.isVisible())

    # ---------- 윈도우 표시 및 최상위 설정 ----------
    def raise_only(self):
        """창을 최상위로 올리는 함수"""
        self.raise_()

    def show(self):
        """창을 화면에 표시하는 함수"""
        super().show()
        self.raise_()

# ---------- MAIN GUI ----------
class AudioTranslatorGUI(QMainWindow):
    """
    메인 GUI 클래스
    자막 창, 음성 레벨, 번역 상태 등의 업데이트를 담당
    """
    def __init__(self, translator):
        super().__init__()
        self.translator = translator
        self.signals = TranslatorSignals()
        self.translator.set_gui_signals(self.signals)

        self.setWindowTitle("오디오 번역기")
        self.setMinimumWidth(400)

        self.setup_ui()
        self.setup_signals()
        self.setup_menu()

        self.subtitle_window = FloatingSubtitleWindow(self)
        self.subtitle_window.show()
        self.update_subtitle_opacity(85)
        self.apply_styles()

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refresh_gui)
        self.update_timer.start(100)

    # ---------- UI 및 메뉴 설정 ----------
    def setup_ui(self):
        """UI 구성"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        self.main_layout.addLayout(self.create_status_layout())
        self.main_layout.addLayout(self.create_settings_layout())

    def create_status_layout(self):
        """상태 표시 레이아웃 생성"""
        layout = QHBoxLayout()
        self.status_label = QLabel("대기 중...")
        layout.addWidget(self.status_label)

        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 1000)
        self.level_bar.setTextVisible(False)
        self.level_bar.setMaximumHeight(15)
        layout.addWidget(self.level_bar)
        return layout

    def create_settings_layout(self):
        """설정 레이아웃 생성"""
        layout = QVBoxLayout()

        self.reset_button = QPushButton("초기화")
        self.reset_button.clicked.connect(self.reset_all)
        layout.addWidget(self.reset_button)

        layout.addLayout(self.create_labeled_slider(
            "음성 감지 임계값:", "threshold_slider", 100, 1000,
            self.translator.silence_threshold, self.update_threshold, "threshold_value_label"
        ))

        layout.addLayout(self.create_labeled_slider(
            "침묵 감지 시간(초):", "silence_duration_slider", 5, 30,
            int(self.translator.silence_duration * 10), self.update_silence_duration, "silence_duration_value_label"
        ))

        layout.addLayout(self.create_slider_only("자막 투명도:", "opacity_slider", 10, 100, 85, self.update_subtitle_opacity))
        layout.addLayout(self.create_slider_only("자막 크기:", "font_size_slider", 8, 80, 14, self.update_font_size))

        # 자막 크기 조절 (세로, 가로 슬라이더 한 줄에 배치)
        subtitle_size_layout = QHBoxLayout()

        # 높이 슬라이더
        subtitle_size_layout.addWidget(QLabel("자막 높이:"))
        self.subtitle_height_slider = QSlider(Qt.Horizontal)
        self.subtitle_height_slider.setRange(100, 800)
        self.subtitle_height_slider.setValue(200)
        self.subtitle_height_slider.valueChanged.connect(self.update_subtitle_height)
        subtitle_size_layout.addWidget(self.subtitle_height_slider)

        # 너비 슬라이더
        subtitle_size_layout.addWidget(QLabel("자막 너비:"))
        self.subtitle_width_slider = QSlider(Qt.Horizontal)
        self.subtitle_width_slider.setRange(400, 1600)
        self.subtitle_width_slider.setValue(800)
        self.subtitle_width_slider.valueChanged.connect(self.update_subtitle_width)
        subtitle_size_layout.addWidget(self.subtitle_width_slider)

        layout.addLayout(subtitle_size_layout)

        return layout

    def create_labeled_slider(self, label_text, slider_name, min_val, max_val, default_val, callback, label_ref):
        """슬라이더와 레이블을 생성하는 함수"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(callback)
        setattr(self, slider_name, slider)
        layout.addWidget(slider)
        value_label = QLabel(str(default_val))
        setattr(self, label_ref, value_label)
        layout.addWidget(value_label)
        return layout

    def create_slider_only(self, label_text, slider_name, min_val, max_val, default_val, callback):
        """슬라이더만 생성하는 함수"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(callback)
        setattr(self, slider_name, slider)
        layout.addWidget(slider)
        return layout

    # ---------- 시그널 연결 및 이벤트 처리 ----------
    def setup_signals(self):
        """시그널 연결 함수"""
        self.signals.translation_update.connect(self.on_translation_update)
        self.signals.audio_level_update.connect(self.on_audio_level_update)
        self.signals.status_update.connect(self.on_status_update)
        self.signals.voice_detected.connect(self.on_voice_detected)
        self.signals.translation_started.connect(self.on_translation_started)

    def setup_menu(self):
        """메뉴 설정 함수"""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("파일")
        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close_application)
        file_menu.addAction(exit_action)

    # ---------- 상태 업데이트 및 번역 ----------
    def apply_styles(self):
        """스타일 적용 함수"""
        self.status_label.setStyleSheet("color: #fff; font-weight: bold; background-color: #333; padding: 5px; border-radius: 5px;")
        self.level_bar.setStyleSheet("""
            QProgressBar { background-color: #2e2e2e; border-radius: 8px; }
            QProgressBar::chunk { background-color: #3cb371; border-radius: 8px; }
        """)

    def reset_all(self):
        """초기화 버튼 클릭 시 호출되는 함수"""
        self.translator.voice_detected = False
        self.translator.audio_frames.clear()
        self.translator.silence_threshold = 400
        self.translator.silence_duration = 1.5
        self.status_label.setText("초기화 완료. 음성을 다시 감지할 수 있습니다.")
        self.status_label.setStyleSheet("font-weight: bold; color: #d9534f;")
        self.subtitle_window.update_subtitles({'korean': '', 'english': '', 'chinese': '', 'japanese': ''})
        self.level_bar.setValue(0)

    def refresh_gui(self):
        """GUI 상태 업데이트 함수"""
        if hasattr(self.translator, 'audio_frames') and self.translator.audio_frames:
            with self.translator.buffer_lock:
                if self.translator.audio_frames:
                    level = self.translator.get_audio_level(self.translator.audio_frames[-1])
                    self.signals.audio_level_update.emit(level)
        if hasattr(self.translator, 'voice_detected'):
            self.signals.voice_detected.emit(self.translator.voice_detected)
           
    # 상태 업데이트 관련 메서드 
    def update_threshold(self):
        """음성 감지 임계값 업데이트 함수"""
        val = self.threshold_slider.value()
        self.threshold_value_label.setText(str(val))
        self.translator.silence_threshold = val

    def update_silence_duration(self):
        """침묵 감지 시간 업데이트 함수"""
        val = self.silence_duration_slider.value() / 10.0
        self.silence_duration_value_label.setText(f"{val:.1f}")
        self.translator.silence_duration = val
        self.translator.silence_chunks = int(val / self.translator.chunk_duration)
        self.translator.save_config()

    def update_font_size(self):
        """자막 폰트 크기 업데이트 함수"""
        self.subtitle_window.update_font_size(self.font_size_slider.value())

    def update_subtitle_height(self):
        """자막 높이 업데이트 함수"""
        self.subtitle_window.update_subtitle_height(self.subtitle_height_slider.value())

    def update_subtitle_width(self):
        """자막 너비 업데이트 함수"""
        width = self.subtitle_width_slider.value()
        self.subtitle_window.resize(width, self.subtitle_window.height())

    def update_subtitle_opacity(self, value):
        """자막 창 투명도 업데이트 함수"""
        self.subtitle_window.setWindowOpacity(value / 100.0)

    # 상태 업데이트 관련 메서드
    @Slot()
    def on_translation_started(self):
        """번역 시작 시 진행 표시 바 시작"""
        self.subtitle_window.start_translation_progress(estimated_duration=1.5)

    @Slot(str, str, str)
    def on_translation_update(self, timestamp, translation, original):
        """번역 업데이트 시 자막과 진행 표시 바 갱신"""
        translations = json.loads(translation)
        self.subtitle_window.update_subtitles(translations)

        # 번역 완료 처리
        QTimer.singleShot(1500, self.subtitle_window.complete_translation_progress)

    # ---------- 오디오 레벨 및 음성 감지 ----------
    @Slot(float)
    def on_audio_level_update(self, level):
        """오디오 레벨 업데이트 시 레벨 바 갱신"""
        self.level_bar.setValue(int(level))
        color = "#5cb85c" if level > self.translator.silence_threshold else "#d9534f"
        self.level_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: #f0f0f0; border-radius: 2px; }}
            QProgressBar::chunk {{ background-color: {color}; }}
        """)

        # 자막창 게이지도 업데이트
        self.subtitle_window.level_bar.setValue(int(level))
        self.subtitle_window.level_bar.setStyleSheet(f"""
            QProgressBar {{ background-color: #2e2e2e; border-radius: 4px; }}
            QProgressBar::chunk {{ background-color: {color}; border-radius: 4px; }}
        """)

    # ---------- 상태 레이블 업데이트 ----------
    @Slot(str)
    def on_status_update(self, status):
        """상태 업데이트 시 상태 레이블 갱신"""
        self.status_label.setText(status)
        color = "#5cb85c" if "감지 중" in status else "black"
        self.status_label.setStyleSheet(f"font-weight: bold; color: {color};")

    @Slot(bool)
    def on_voice_detected(self, detected):
        """음성 감지 시 상태 업데이트"""
        if detected:
            self.status_label.setText("음성 감지 중...")
            self.status_label.setStyleSheet("font-weight: bold; color: #5cb85c;")
        else:
            self.status_label.setText("음성을 인식할 준비가 완료되었습니다")
            self.status_label.setStyleSheet("font-weight: bold; color: black;")
                       
    def close_application(self):
        """애플리케이션 종료 함수"""
        self.translator.is_running = False
        time.sleep(1)
        QApplication.quit()

    def closeEvent(self, event):
        """창 닫기 이벤트 처리"""
        self.close_application()
        event.accept()

# ---------- DUMMY TRANSLATOR FOR TEST ----------
class DummyTranslator:
    def __init__(self):
        self.silence_threshold = 400
        self.silence_duration = 1.5
        self.translation_mode = "realtime"
        self.audio_frames = []
        self.voice_detected = False
        self.chunk_duration = 0.064
        self.buffer_lock = Lock()

    def get_audio_level(self, data):
        return 300.0  # 예시 볼륨

    def set_gui_signals(self, signals):
        self.signals = signals

    def save_config(self):
        print("설정 저장됨")

# ---------- ENTRY ----------
def start_gui(translator):
    app = QApplication(sys.argv)
    window = AudioTranslatorGUI(translator)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    dummy = DummyTranslator()
    start_gui(dummy)