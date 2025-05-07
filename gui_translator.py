import sys
import time
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject
from PySide6.QtGui import QFont, QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QProgressBar, QComboBox, QSlider, QCheckBox, QFrame
)

class TranslatorSignals(QObject):
    """GUI 업데이트를 위한 신호 클래스"""
    translation_update = Signal(str, str, str)  # 타임스탬프, 번역, 원문
    audio_level_update = Signal(float)  # 현재 오디오 레벨
    status_update = Signal(str)  # 상태 메시지
    voice_detected = Signal(bool)  # 음성 감지 상태

class FloatingSubtitleWindow(QMainWindow):
    """다른 프로그램이 활성화되어도 항상 최상단에 표시되는 자막 창 (화면 하단, 반투명 박스)"""
    def __init__(self, main_window):
        # Qt.Tool 플래그 제거하여 독립적으로 표시 (항상 위 플래그 적용)
        super().__init__(None, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.X11BypassWindowManagerHint)
        
        # 창이 포커스를 받지 않도록 설정
        self.setWindowFlag(Qt.WindowDoesNotAcceptFocus, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        self.main_window = main_window
        
        # 투명 배경 설정
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 중앙 위젯 및 레이아웃 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 자막 표시 영역 (반투명 박스)
        self.subtitle_frame = QFrame()
        self.subtitle_frame.setStyleSheet(
            "QFrame { background-color: rgba(0, 0, 0, 150); border-radius: 10px; }"
        )
        subtitle_layout = QVBoxLayout(self.subtitle_frame)
        
        # 번역 텍스트 (한국어)
        self.korean_label = QLabel()
        self.korean_label.setStyleSheet("color: white; font-weight: bold;")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.korean_label.setFont(font)
        self.korean_label.setWordWrap(True)
        self.korean_label.setAlignment(Qt.AlignCenter)
        subtitle_layout.addWidget(self.korean_label)
        
        # 원문 텍스트 (영어) – 토글 가능
        self.english_label = QLabel()
        self.english_label.setStyleSheet("color: lightgray;")
        self.english_label.setFont(QFont("Arial", 12))
        self.english_label.setWordWrap(True)
        self.english_label.setAlignment(Qt.AlignCenter)
        self.english_label.setVisible(False)
        subtitle_layout.addWidget(self.english_label)
        
        layout.addWidget(self.subtitle_frame)
        
        # 초기 크기 및 위치 (화면 하단 중앙)
        self.resize(600, 150)
        self.move_to_bottom()
        
        # 드래그 관련 변수
        self.drag_position = None

        # 단축키 (F2: 모든 창 표시)
        self.show_all_shortcut = QShortcut(QKeySequence(Qt.Key_F2), self)
        self.show_all_shortcut.activated.connect(self.show_all_windows)
        
        # 타이머: 주기적으로 창을 최상단에 올림 (포커스 전환 없이)
        self.top_timer = QTimer(self)
        self.top_timer.timeout.connect(self.raise_only)
        self.top_timer.start(5000)  # 5초마다 실행

    def move_to_bottom(self):
        """화면 하단 중앙에 창 위치시키기"""
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = screen_geometry.height() - self.height() - 100  # 하단에서 100픽셀 위
        self.move(x, y)
    
    def update_subtitle(self, translation, original_text=""):
        """자막 업데이트"""
   
        self.korean_label.setText(translation)
        if original_text:
            self.english_label.setText(original_text)
        print(f"{datetime.now().strftime('%H:%M:%S,%f')[:-3]} - DEBUG - 화면 출력: {self.korean_label.text()}")
    
    def toggle_original_text(self, show):
        """원문 표시 토글"""
        self.english_label.setVisible(show)
        if show:
            self.resize(self.width(), 300)
        else:
            self.resize(self.width(), 220)
    
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
    
    def show_all_windows(self):
        """모든 창 표시"""
        self.main_window.show()
        self.show()
    
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
        print("AudioTranslator에 GUI 신호 연결 중...")
        translator.set_gui_signals(self.signals)
        print("신호 연결 완료!")
        
        self.setWindowTitle("오디오 번역기")
        self.setMinimumWidth(400)
        
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
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("번역 모드:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["실시간 번역", "발화 완료 후 번역"])
        current_mode = 0 if self.translator.translation_mode == "realtime" else 1
        self.mode_combo.setCurrentIndex(current_mode)
        self.mode_combo.currentIndexChanged.connect(self.update_translation_mode)
        mode_layout.addWidget(self.mode_combo)
        settings_layout.addLayout(mode_layout)
        
        show_original_layout = QHBoxLayout()
        self.show_original_checkbox = QCheckBox("원문 함께 표시")
        self.show_original_checkbox.setChecked(False)
        self.show_original_checkbox.stateChanged.connect(self.toggle_original_text)
        show_original_layout.addWidget(self.show_original_checkbox)
        settings_layout.addLayout(show_original_layout)
        
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("자막 투명도:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(85)
        self.opacity_slider.valueChanged.connect(self.update_subtitle_opacity)
        opacity_layout.addWidget(self.opacity_slider)
        settings_layout.addLayout(opacity_layout)
        
        main_layout.addLayout(settings_layout)
        
        shortcut_layout = QHBoxLayout()
        shortcut_label = QLabel("<b>단축키:</b> F1=컨트롤 창 숨기기, F2=컨트롤 창 표시")
        shortcut_layout.addWidget(shortcut_label)
        main_layout.addLayout(shortcut_layout)
        
        self.subtitle_only_shortcut = QShortcut(QKeySequence(Qt.Key_F1), self)
        self.subtitle_only_shortcut.activated.connect(self.show_subtitle_only)
        
        self.show_all_shortcut = QShortcut(QKeySequence(Qt.Key_F2), self)
        self.show_all_shortcut.activated.connect(self.show_all_windows)
        
        self.setup_menu()
        
        # 자막 창 생성 (FloatingSubtitleWindow는 항상 위로, 포커스 전환 없음)
        self.subtitle_window = FloatingSubtitleWindow(self)
        self.subtitle_window.show()
        
        self.update_subtitle_opacity(self.opacity_slider.value())
        
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_gui)
        self.update_timer.start(100)
    
    def setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("파일")
        
        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close_application)
        file_menu.addAction(exit_action)
        
        view_menu = menu_bar.addMenu("보기")
        subtitle_only_action = QAction("자막만 표시 (F1)", self)
        subtitle_only_action.triggered.connect(self.show_subtitle_only)
        view_menu.addAction(subtitle_only_action)
        
        show_all_action = QAction("모든 창 표시 (F2)", self)
        show_all_action.triggered.connect(self.show_all_windows)
        view_menu.addAction(show_all_action)
    
    def setup_signals(self):
        print("GUI 신호 객체 생성...")
        self.signals = TranslatorSignals()
        print("신호 연결 중...")
        self.signals.translation_update.connect(self.on_translation_update)
        self.signals.audio_level_update.connect(self.on_audio_level_update)
        self.signals.status_update.connect(self.on_status_update)
        self.signals.voice_detected.connect(self.on_voice_detected)
        print("신호 연결 완료!")
    
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
        if hasattr(self, 'subtitle_window') and self.subtitle_window:
            # print("자막 창에 업데이트 중...")
            self.subtitle_window.update_subtitle(translation, original if self.show_original_checkbox.isChecked() else "")

    
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
        self.status_label.setText(status)
    
    @Slot(bool)
    def on_voice_detected(self, detected):
        if detected:
            self.status_label.setText("🎤 음성 감지 중...")
            self.status_label.setStyleSheet("font-weight: bold; color: #5cb85c;")
        else:
            self.status_label.setText("대기 중...")
            self.status_label.setStyleSheet("font-weight: bold; color: black;")
    
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
    
    def toggle_original_text(self, state):
        show = (state == Qt.Checked)
        self.subtitle_window.toggle_original_text(show)
    
    def update_subtitle_opacity(self, value):
        opacity = value / 100.0
        self.subtitle_window.setWindowOpacity(opacity)
    
    def show_subtitle_only(self):
        self.hide()
        self.subtitle_window.show()
    
    def show_all_windows(self):
        self.main_window.show()
        self.show()
    
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