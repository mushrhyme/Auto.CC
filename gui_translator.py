import sys
import time
import json
from datetime import datetime
from threading import Thread

from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject
from PySide6.QtGui import QFont, QAction
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QProgressBar, QSlider, QGridLayout, QMessageBox, QDialog, QCheckBox
)
 
# ---------- SIGNAL DEFINITION ----------
class TranslatorSignals(QObject):
    """각각의 이벤트 발생을 위한 Signal 정의"""
    translation_update = Signal(str, str, str)  # timestamp, translation (JSON str), original
    audio_level_update = Signal(float)
    status_update = Signal(str)
    voice_detected = Signal(bool)
    translation_started = Signal()
    stt_original_update = Signal(str)
    
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

        # 번역 결과 저장소 및 인덱스
        self.translation_history = []  # 최대 10개의 번역 결과 저장
        self.current_history_index = -1  # 현재 표시 중인 번역 결과의 인덱스
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 최상위 수직 레이아웃
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(10, 5, 10, 5)
        outer_layout.setSpacing(5)

        
        # 화살표 버튼 레이아웃
        arrow_layout = QHBoxLayout()
        self.left_arrow_button = QPushButton("◀")
        self.right_arrow_button = QPushButton("▶")

        self.left_arrow_button.setFixedSize(30, 30)
        self.right_arrow_button.setFixedSize(30, 30)

        self.left_arrow_button.setStyleSheet("background-color: #444; color: white; border-radius: 15px;")
        self.right_arrow_button.setStyleSheet("background-color: #444; color: white; border-radius: 15px;")

        self.left_arrow_button.clicked.connect(self.on_left_arrow_clicked)
        self.right_arrow_button.clicked.connect(self.on_right_arrow_clicked)

        arrow_layout.addWidget(self.left_arrow_button)
        arrow_layout.addWidget(self.right_arrow_button)
        arrow_layout.addStretch()
        outer_layout.addLayout(arrow_layout)


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
        self.grid_layout = QGridLayout()
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(5)

        self.korean_label = self.create_label("한국어")
        self.english_label = self.create_label("영어")
        self.chinese_label = self.create_label("중국어")
        # self.japanese_label = self.create_label("일본어")

        self.grid_layout.addWidget(self.korean_label, 0, 0)
        self.grid_layout.addWidget(self.english_label, 0, 1)
        self.grid_layout.addWidget(self.chinese_label, 1, 0)
        # self.grid_layout.addWidget(self.japanese_label, 1, 1)

        self.grid_layout.setRowStretch(0, 1)
        self.grid_layout.setRowStretch(1, 1)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)

        outer_layout.addLayout(self.grid_layout)

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

        # 하단 원문 박스 창
        self.stt_original_label = QLabel("")
        self.stt_original_label.setStyleSheet("color: #ffd700; background-color: rgba(0,0,0,180); font-size: 16px; padding: 6px; border-radius: 6px;")
        self.stt_original_label.setAlignment(Qt.AlignCenter)
        
        # ① 긴 문장은 자동 줄바꿈
        self.stt_original_label.setWordWrap(True)
        # ② 창 너비에 맞춰 라벨 폭 고정
        self.stt_original_label.setFixedWidth(self.width() - 20)
        
        outer_layout.addWidget(self.stt_original_label)
        
        # 타이머 설정
        self.translation_timer = QTimer()
        self.translation_timer.timeout.connect(self._update_translation_progress)
        self._progress_value = 0

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

    def on_left_arrow_clicked(self):
        """왼쪽 화살표 버튼 클릭 시 이전 번역 결과 표시"""
        if self.translation_history and self.current_history_index > 0:
            self.current_history_index -= 1
            self.update_subtitles(self.translation_history[self.current_history_index])
        else:
            print("No previous translation available.")

    def on_right_arrow_clicked(self):
        """오른쪽 화살표 버튼 클릭 시 다음 번역 결과 표시"""
        if self.translation_history and self.current_history_index < len(self.translation_history) - 1:
            self.current_history_index += 1
            self.update_subtitles(self.translation_history[self.current_history_index])
        else:
            print("No next translation available.")

    def add_translation_to_history(self, translations):
        """새 번역 결과를 저장소에 추가"""
        if len(self.translation_history) >= 10:
            self.translation_history.pop(0)  # 가장 오래된 번역 결과 제거
        self.translation_history.append(translations)
        self.current_history_index = len(self.translation_history) - 1  # 최신 번역 결과로 인덱스 이동

    def move_to_bottom(self):
        """자막 창을 화면 하단에 위치시키는 함수"""
        screen_geometry = QApplication.primaryScreen().geometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = screen_geometry.height() - self.height() - 100
        self.move(x, y)

    def update_subtitles(self, translations):
        """자막을 업데이트하는 함수"""
        style = "color: white; font-weight: bold; background-color: rgba(0, 0, 0, 150);"
        for lang, label in [
            ('korean', self.korean_label),
            ('english', self.english_label),
            ('chinese', self.chinese_label),
            # ('japanese', self.japanese_label)
        ]:
            text = translations.get(lang, '')
            label.setStyleSheet(style)
            label.setText(text)
        
            # 1) 현재 너비 저장
            current_width = self.width()

            # 2) 레이아웃 재계산 강제
            self.grid_layout.invalidate()           # (Qt 6 부터 activate() 대신 invalidate())
            self.grid_layout.activate()             # 레이아웃 재배치

            # 3) 전체 위젯을 최소 크기로 맞춘 뒤
            self.adjustSize()

            # 4) 너비는 고정하고, 높이만 새로 계산된 값으로 리사이즈
            self.resize(current_width, self.height()+50)

    def update_font_size(self, size):
        """자막 폰트 크기 변경 함수"""
        font = QFont()
        font.setPointSize(size)
        for label in [self.korean_label, self.english_label, self.chinese_label, 
                    #   self.japanese_label
                      ]:
            label.setFont(font)

    def update_subtitle_height(self, height):
        """자막 높이 변경 함수"""
        self.resize(self.width(), height)
        
    def update_visible_languages(self, selected_languages):
        """선택된 언어에 따라 자막 창의 레이아웃을 동적으로 업데이트"""
        # 기존 레이아웃 초기화
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                self.grid_layout.removeWidget(widget)
                widget.setParent(None)

        # 선택된 언어에 따라 동적으로 레이아웃 구성
        language_labels = {
            "korean": self.korean_label,
            "english": self.english_label,
            # "japanese": self.japanese_label,
            "chinese": self.chinese_label
        }

        selected_labels = [language_labels[lang] for lang in selected_languages]

        # 동적으로 그리드 레이아웃 구성
        if len(selected_labels) == 1:
            # 1개 언어: 1행 1열
            self.grid_layout.addWidget(selected_labels[0], 0, 0)
        elif len(selected_labels) == 2:
            # 2개 언어: 1행 2열
            self.grid_layout.addWidget(selected_labels[0], 0, 0)
            self.grid_layout.addWidget(selected_labels[1], 0, 1)
        elif len(selected_labels) == 3:
            # 3개 언어: 1행 3열
            self.grid_layout.addWidget(selected_labels[0], 0, 0)
            self.grid_layout.addWidget(selected_labels[1], 0, 1)
            self.grid_layout.addWidget(selected_labels[2], 0, 2)
        elif len(selected_labels) == 4:
            # 4개 언어: 2행 2열
            self.grid_layout.addWidget(selected_labels[0], 0, 0)
            self.grid_layout.addWidget(selected_labels[1], 0, 1)
            self.grid_layout.addWidget(selected_labels[2], 1, 0)
            self.grid_layout.addWidget(selected_labels[3], 1, 1)

        # 레이아웃 행과 열 설정 초기화
        for i in range(3):  # 최대 3행
            self.grid_layout.setRowStretch(i, 0)
        for i in range(3):  # 최대 3열
            self.grid_layout.setColumnStretch(i, 0)

        # 선택된 언어에 따라 행과 열 설정
        if len(selected_labels) <= 3:
            self.grid_layout.setRowStretch(0, 1)  # 1행만 활성화
            for i in range(len(selected_labels)):
                self.grid_layout.setColumnStretch(i, 1)  # 선택된 언어 개수만큼 열 활성화
        else:
            self.grid_layout.setRowStretch(0, 1)
            self.grid_layout.setRowStretch(1, 1)  # 2행 활성화
            self.grid_layout.setColumnStretch(0, 1)
            self.grid_layout.setColumnStretch(1, 1)  # 2열 활성화
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 창 크기가 변경될 때마다 stt_original_label 폭을 갱신
        self.stt_original_label.setFixedWidth(self.width() - 20)
        
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
        self.subtitle_blue_mode = False
        
        self.setWindowTitle("Nongshim Audio Translator")
        self.setMinimumWidth(400)

        self.setup_ui()
        self.setup_signals()
        self.setup_menu()

        self.subtitle_window = FloatingSubtitleWindow(self)
        self.subtitle_window.show()
        
        self.update_selected_languages()
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

        # 언어 선택 체크박스
        self.language_selection_layout = QHBoxLayout()
        self.language_checkboxes = {}
        self.selected_languages = ["korean", "english", "japanese", "chinese"]  # 디폴트 선택 언어

        language_options = {
            "korean": "한국어",
            "english": "영어",
            # "japanese": "일본어",
            "chinese": "중국어"
        }

        language_label = QLabel("번역 언어 선택:")
        self.language_selection_layout.addWidget(language_label)

        for lang_key, lang_name in language_options.items():
            checkbox = QCheckBox(lang_name)
            checkbox.setChecked(True)  # 디폴트로 모든 언어 선택
            checkbox.stateChanged.connect(self.update_selected_languages)
            self.language_checkboxes[lang_key] = checkbox
            self.language_selection_layout.addWidget(checkbox)
            
        self.selected_languages = list(self.language_checkboxes.keys())
        layout.addLayout(self.language_selection_layout)

        # 작업 초기화 버튼
        self.reset_button = QPushButton("작업 초기화")
        self.reset_button.clicked.connect(self.reset_all)
        self.reset_button.setToolTip("진행 중인 모든 작업을 취소하고 음성 인식 대기 상태로 초기화합니다.")
        layout.addWidget(self.reset_button)

        
        # 음성 감지 임계값 조절
        layout.addLayout(self.create_labeled_slider(
            "음성 감지 임계값:", "threshold_slider", 500, 5000,
            self.translator.silence_threshold, self.update_threshold, "threshold_value_label"
        ))
        self.threshold_slider.setToolTip("음성 레벨이 임계값 이상인 경우에만 발화한 것으로 간주합니다.")
        
        # 음성 감지 임계값 자동 조정 버튼
        self.auto_adjust_button = QPushButton("음성 감지 임계값 자동 조정")
        self.auto_adjust_button.clicked.connect(self.auto_adjust_silence_threshold)
        self.auto_adjust_button.setToolTip("사용자의 실제 음성을 바탕으로 음성 감지 임계값을 자동 조정합니다.")
        layout.addWidget(self.auto_adjust_button)
        
        # 침묵 감지 시간 조절
        layout.addLayout(self.create_labeled_slider(
            "침묵 감지 시간(초):", "silence_duration_slider", 5, 30,
            int(self.translator.silence_duration * 10), self.update_silence_duration, "silence_duration_value_label"
        ))
        self.silence_duration_slider.setToolTip("이 시간 동안 음성이 감지되지 않으면 발화가 종료된 것으로 간주합니다.")

        # 자막 투명도/크기 조절
        layout.addLayout(self.create_slider_only("자막 투명도:", "opacity_slider", 10, 100, 85, self.update_subtitle_opacity))
        layout.addLayout(self.create_slider_only("자막 크기:", "font_size_slider", 8, 80, 14, self.update_font_size))

        subtitle_size_layout = QHBoxLayout()
        
        # 자막 창 높이 조절
        subtitle_size_layout.addWidget(QLabel("자막 높이:"))
        self.subtitle_height_slider = QSlider(Qt.Horizontal)
        self.subtitle_height_slider.setRange(50, 800)
        self.subtitle_height_slider.setValue(100)
        self.subtitle_height_slider.valueChanged.connect(self.update_subtitle_height)
        subtitle_size_layout.addWidget(self.subtitle_height_slider)

        # 자막 창 너비 조절
        subtitle_size_layout.addWidget(QLabel("자막 너비:"))
        self.subtitle_width_slider = QSlider(Qt.Horizontal)
        self.subtitle_width_slider.setRange(500, 2500)
        self.subtitle_width_slider.setValue(1000)
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

    # ---------- 침묵 감지 수준 설정 ----------
    def auto_adjust_silence_threshold(self):
        """침묵 감지 임계값 자동 조정"""
        self.adjustment_dialog = QDialog(self)
        self.adjustment_dialog.setWindowTitle("침묵 감지 임계값 자동 조정")
        self.adjustment_dialog.setModal(True)
        self.adjustment_dialog.setFixedSize(300, 150)

        layout = QVBoxLayout(self.adjustment_dialog)

        self.dialog_label = QLabel("아래 버튼을 누르면 5초 동안 녹음을 시작합니다.\n")
        self.dialog_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.dialog_label)

        self.start_button = QPushButton("오디오 음량 측정")
        self.start_button.clicked.connect(self.start_audio_test)
        layout.addWidget(self.start_button)

        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.countdown_label)

        self.adjustment_dialog.show()

    def start_audio_test(self):
        """음성 테스트 시작"""
        self.start_button.setEnabled(False)  # 버튼 비활성화
        self.dialog_label.setText("테스트가 시작됩니다. 안내에 따라 발화 및 침묵하세요!")
        self.countdown_seconds = 3  # 발화/침묵 각각 3초
        self.is_speaking_phase = True  # 초기 상태: 발화
        self.phase_count = 0  # 발화/침묵 단계를 추적
        self.update_phase_label()
        
        # QProgressBar 초기화
        self.countdown_progress = QProgressBar(self.adjustment_dialog)
        self.countdown_progress.setRange(0, 3)
        self.countdown_progress.setValue(3)
        self.countdown_progress.setTextVisible(False)  # 퍼센트 표시 제거
        self.countdown_progress.setAlignment(Qt.AlignCenter)
        self.countdown_progress.setStyleSheet("""
            QProgressBar { background-color: #2e2e2e; border-radius: 4px; }
            QProgressBar::chunk { background-color: #1e90ff;  /* 파란색 */ border-radius: 4px; }
        """)

        # 부모 레이아웃에 QProgressBar 추가
        parent_layout = self.adjustment_dialog.layout()
        if parent_layout:  # 부모 레이아웃이 존재하는 경우
            parent_layout.addWidget(self.countdown_progress)

        # 카운트다운 타이머 설정
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)  # 1초 간격

        # 음성 레벨 측정을 위한 초기화
        self._speaking_levels = []
        self._silence_levels = []
        self._adjustment_timer = QTimer(self)
        self._adjustment_timer.timeout.connect(self._collect_audio_levels)
        self._adjustment_timer.start(100)  # 100ms 간격으로 오디오 레벨 수집 시작

    def show_adjustment_result(self):
        """결과를 팝업으로 표시"""
        result_popup = QMessageBox(self)
        result_popup.setWindowTitle("결과")
        
        new_threshold = self.translator.silence_threshold       
             
        if new_threshold:
            result_popup.setText(f"임계값이 자동으로 조정되었습니다.\n새로운 임계값: {new_threshold}")
            result_popup.setIcon(QMessageBox.Information)
        else:
            result_popup.setText("오디오 레벨을 감지하지 못했습니다. 다시 시도하세요.")
            result_popup.setIcon(QMessageBox.Warning)
        result_popup.exec()

    def update_phase(self):
        """발화/침묵 상태 전환"""
        self.is_speaking_phase = not self.is_speaking_phase
        self.update_phase_label()

    def update_phase_label(self):
        """발화/침묵 안내 문구 및 색상 업데이트"""
        if self.is_speaking_phase:
            self.dialog_label.setText("3초 동안 발화하세요! \n예) 안녕하세요")
            self.dialog_label.setStyleSheet("""color: black; font-weight: bold; """)
        else:
            self.dialog_label.setText("3초 동안 침묵하세요!")
            self.dialog_label.setStyleSheet("""color: #d9534f; font-weight: bold; """)
    
    def update_countdown(self):
        """카운트다운 업데이트"""
        if self.countdown_seconds > 0:
            self.countdown_progress.setValue(self.countdown_seconds)
            self.countdown_seconds -= 1
        elif self.countdown_seconds == 0:
            self.countdown_progress.setValue(0)
            self.phase_count += 1  # 발화/침묵 단계 증가
            if self.phase_count == 2:  # 발화와 침묵이 모두 끝난 경우
                self.countdown_timer.stop()
                self._adjustment_timer.stop()
                self.dialog_label.setText("테스트가 완료되었습니다!")
                self._finalize_silence_threshold_adjustment()
            else:
                # 다음 단계로 전환
                self.is_speaking_phase = not self.is_speaking_phase
                self.update_phase_label()
                self.countdown_seconds = 3  # 다음 단계의 카운트다운 초기화
                self.countdown_progress.setValue(3)
               
    def _collect_audio_levels(self):
        """오디오 레벨 수집"""
        if hasattr(self.translator, 'audio_frames') and self.translator.audio_frames:
            with self.translator.buffer_lock:
                level = self.translator.get_audio_level(self.translator.audio_frames[-1])
                if self.is_speaking_phase:
                    self._speaking_levels.append(level)
                else:
                    self._silence_levels.append(level)

    def _finalize_silence_threshold_adjustment(self):
        """침묵 감지 임계값 자동 조정 완료"""
        if hasattr(self, '_adjustment_timer') and self._adjustment_timer.isActive():
            self._adjustment_timer.stop()

        if self._speaking_levels and self._silence_levels:
            # 침묵 평균 및 발화 표준편차 계산
            silence_mean = np.mean(self._silence_levels)
            speaking_std = np.std(self._speaking_levels)

            # 새로운 임계값 계산
            new_threshold = int(silence_mean + 1.2 * speaking_std)
            self.translator.silence_threshold = new_threshold
            self.threshold_slider.setValue(new_threshold)
            self.threshold_value_label.setText(str(new_threshold))
            self.status_label.setText("임계값이 자동으로 조정되었습니다.")
        else:
            self.status_label.setText("오디오 레벨을 감지하지 못했습니다. 다시 시도하세요.")
            self.status_label.setStyleSheet("color: #d9534f;")
        # 테스트 완료 후 결과 팝업 표시
        self.show_adjustment_result()
        
        # "오디오 음량 측정" 버튼 다시 활성화
        self.start_button.setEnabled(True)
        self.dialog_label.setText("아래 버튼을 눌러 다시 테스트를 시작하세요.")
        
    # ---------- 시그널 연결 및 이벤트 처리 ----------
    def setup_signals(self):
        """시그널 연결 함수"""
        self.signals.translation_update.connect(self.on_translation_update)
        self.signals.audio_level_update.connect(self.on_audio_level_update)
        self.signals.status_update.connect(self.on_status_update)
        self.signals.voice_detected.connect(self.on_voice_detected)
        self.signals.translation_started.connect(self.on_translation_started)
        self.signals.stt_original_update.connect(self.on_stt_original_update)

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
            QProgressBar::chunk { background-color: #5cb85c; border-radius: 8px; }
        """)
        self.level_bar.setValue(1000) 
        
    def reset_all(self):
        """초기화 버튼 클릭 시 호출되는 함수"""
        # 음성 감지 상태 초기화
        self.translator.voice_detected = False
        self.translator.audio_frames.clear()
        self.translator.silence_count = 0
        self.translator.last_translation = ""
        self.translator.is_running = True # 음성 캡처 루프 활성화
        
        # GUI 상태 초기화
        self.status_label.setText("초기화 완료. 음성을 다시 감지할 수 있습니다.")
        self.status_label.setStyleSheet("font-weight: bold; color: #5cb85c;")
        self.level_bar.setValue(0)
        self.subtitle_window.update_subtitles({'korean': '', 'english': '', 'chinese': '', 'japanese': ''})
        self.subtitle_window.translation_bar.setValue(0)
        self.subtitle_window.level_bar.setValue(0)
        
        # 음성 캡처 루프 재시작
        if hasattr(self.translator, 'audio_capture'):
            Thread(target=self.translator.audio_capture, daemon=True).start()
        else:
            self.status_label.setText("오류: 음성 캡처 루프를 시작할 수 없습니다.")
            self.status_label.setStyleSheet("font-weight: bold; color: #d9534f;")
            
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
    
    def update_selected_languages(self):
        """선택된 언어를 업데이트하고 자막 창과 번역 대상 언어를 동기화"""
        self.selected_languages = [
            lang_key for lang_key, checkbox in self.language_checkboxes.items() if checkbox.isChecked()
        ]

        # 자막 창 업데이트
        self.subtitle_window.update_visible_languages(self.selected_languages)

        # 번역 대상 언어 업데이트
        # self.translator.update_target_languages(self.selected_languages)
        
    # 상태 업데이트 관련 메서드
    @Slot(str)
    def on_stt_original_update(self, original_text):
        self.subtitle_window.stt_original_label.setText(original_text)
    
    @Slot()
    def on_translation_started(self):
        """번역 시작 시 진행 표시 바 시작"""
        self.subtitle_window.start_translation_progress(estimated_duration=1.5)

    @Slot(str, str, str)
    def on_translation_update(self, timestamp, translation, original):
        """번역 업데이트 시 자막과 진행 표시 바 갱신"""        
        if not translation:
            self.status_label.setText("번역이 건너뛰어졌습니다 (발화가 너무 짧음).")
            self.status_label.setStyleSheet("font-weight: bold; color: #d9534f;")
            return
        translations = json.loads(translation)
        self.subtitle_blue_mode = False  # 번역이 오면 파란 자막 모드 해제
        self.subtitle_window.update_subtitles(translations)

        # 번역 결과를 저장소에 추가
        self.subtitle_window.add_translation_to_history(translations)
        
        # 번역 완료 처리
        QTimer.singleShot(1500, self.subtitle_window.complete_translation_progress)

    # ---------- 오디오 레벨 및 음성 감지 ---------- 
    @Slot(float)
    def on_audio_level_update(self, level):
        """오디오 레벨 업데이트 시 게이지바 갱신"""
        if level > self.translator.silence_threshold:
            # 발화 중: 빨간 게이지바가 차오름
            if not self.translator.voice_detected:  # 발화 시작 시 상태 변경
                self.translator.voice_detected = True
            self.level_bar.setValue(int(level))
            self.level_bar.setStyleSheet("""
                QProgressBar { background-color: #2e2e2e; border-radius: 4px; }
                QProgressBar::chunk { background-color: #d9534f; border-radius: 4px; }
            """)
            # 침묵 타이머 초기화
            if hasattr(self, '_silence_timer') and self._silence_timer.isActive():
                self._silence_timer.stop()
        else:
            # 발화 종료를 위한 짧은 침묵 허용
            if self.translator.voice_detected:
                if not hasattr(self, '_silence_timer'):
                    self._silence_timer = QTimer()
                    self._silence_timer.setSingleShot(True)
                    self._silence_timer.timeout.connect(self._handle_silence_end)
                if not self._silence_timer.isActive():
                    self._silence_timer.start(int(self.translator.silence_duration * 1000))  # 침묵 지속 시간(ms)

        # 자막창 게이지도 동일하게 업데이트
        self.subtitle_window.level_bar.setValue(self.level_bar.value())
        self.subtitle_window.level_bar.setStyleSheet(self.level_bar.styleSheet())

    def _handle_silence_end(self):
        """침묵 종료 처리"""
        self.translator.voice_detected = False
        self.level_bar.setValue(1000)
        self.level_bar.setStyleSheet("""
            QProgressBar { background-color: #2e2e2e; border-radius: 4px; }
            QProgressBar::chunk { background-color: #5cb85c; border-radius: 4px; }
        """)
    
    # ---------- 상태 레이블 업데이트 ----------
    @Slot(str)
    def on_status_update(self, status):
        """상태 업데이트 시 상태 레이블 갱신"""
        self.status_label.setText(status)
        color = "#5cb85c" if "감지 중" in status else "black"
        self.status_label.setStyleSheet(f"font-weight: bold; color: {color};")
        
        if "감지 중" in status or "실패" in status or "초기화" in status:
            self.subtitle_blue_mode = False
            default_style = "color: white; font-weight: bold; background-color: rgba(0, 0, 0, 150);"
            for label in [
                self.subtitle_window.korean_label,
                self.subtitle_window.english_label,
                self.subtitle_window.chinese_label,
                # self.subtitle_window.japanese_label
            ]:
                label.setStyleSheet(default_style)

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

# ---------- ENTRY ----------
def start_gui(translator):
    app = QApplication(sys.argv)
    window = AudioTranslatorGUI(translator)
    window.show()
    sys.exit(app.exec())


# # ---------- DUMMY TRANSLATOR FOR TEST ----------
# class DummyTranslator:
    
#     def __init__(self):
#         self.silence_threshold = 400
#         self.silence_duration = 1.5
#         self.translation_mode = "realtime"
#         self.audio_frames = []
#         self.voice_detected = False
#         self.chunk_duration = 0.064
#         self.buffer_lock = Lock()

#     def get_audio_level(self, data):
#         return 300.0  # 예시 볼륨

#     def set_gui_signals(self, signals):
#         self.signals = signals
        
# if __name__ == "__main__":
#     dummy = DummyTranslator()
#     start_gui(dummy)



