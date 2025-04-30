import sys
import time
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QObject
from PySide6.QtGui import QFont, QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QComboBox, QSlider, QCheckBox, QFrame
)

class TranslatorSignals(QObject):
    translation_update = Signal(str, str, str)
    audio_level_update = Signal(float)
    status_update = Signal(str)
    voice_detected = Signal(bool)

class FloatingSubtitleWindow(QMainWindow):
    def __init__(self, main_win):
        super().__init__(None, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.X11BypassWindowManagerHint)
        self.main_win = main_win
        self.setWindowFlag(Qt.WindowDoesNotAcceptFocus, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TranslucentBackground)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10,5,10,5)

        frame = QFrame()
        frame.setStyleSheet("background-color: rgba(0,0,0,150); border-radius:10px;")
        sub_layout = QVBoxLayout(frame)

        self.korean_label = QLabel()
        self.korean_label.setStyleSheet("color:white; font-weight:bold;")
        font = QFont(); font.setPointSize(14); font.setBold(True)
        self.korean_label.setFont(font)
        self.korean_label.setWordWrap(True)
        self.korean_label.setAlignment(Qt.AlignCenter)
        sub_layout.addWidget(self.korean_label)

        self.english_label = QLabel()
        self.english_label.setStyleSheet("color:lightgray;")
        self.english_label.setFont(QFont("Arial",12))
        self.english_label.setWordWrap(True)
        self.english_label.setAlignment(Qt.AlignCenter)
        self.english_label.setVisible(False)
        sub_layout.addWidget(self.english_label)

        layout.addWidget(frame)
        self.resize(600,150)
        self._move_bottom()

        self.drag_pos = None
        self.show_all_shortcut = QShortcut(QKeySequence(Qt.Key_F2), self)
        self.show_all_shortcut.activated.connect(self._show_all)

        self.top_timer = QTimer(self)
        self.top_timer.timeout.connect(self.raise_)
        self.top_timer.start(5000)

    def _move_bottom(self):
        geo = QApplication.primaryScreen().geometry()
        x = (geo.width() - self.width())//2
        y = geo.height() - self.height() - 100
        self.move(x,y)

    def update_subtitle(self, trans, orig=""):
        self.korean_label.setText(trans)
        if orig:
            self.english_label.setText(orig)

    def toggle_original_text(self, show):
        self.english_label.setVisible(show)
        self.resize(self.width(), 200 if show else 120)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self.drag_pos:
            self.move(e.globalPosition().toPoint() - self.drag_pos)
            e.accept()

    def mouseDoubleClickEvent(self, e):
        self.main_win.setVisible(not self.main_win.isVisible())

    def _show_all(self):
        self.main_win.show()
        self.show()

class AudioTranslatorGUI(QMainWindow):
    def __init__(self, translator):
        super().__init__()
        self.translator = translator
        self._setup_signals()
        translator.set_gui_signals(self.signals)

        self.setWindowTitle("오디오 번역기")
        self.setMinimumWidth(400)

        cw = QWidget()
        self.setCentralWidget(cw)
        ml = QVBoxLayout(cw)

        # 상태 + 레벨바
        sl = QHBoxLayout()
        self.status_label = QLabel("대기 중...")
        self.status_label.setStyleSheet("font-weight:bold;")
        sl.addWidget(self.status_label)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0,1000)
        self.level_bar.setTextVisible(False)
        self.level_bar.setMaximumHeight(15)
        sl.addWidget(self.level_bar)
        ml.addLayout(sl)

        # 설정
        cfg = QVBoxLayout()
        # 임계값
        tl = QHBoxLayout()
        tl.addWidget(QLabel("음성 감지 임계값:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(100,1000)
        self.threshold_slider.setValue(self.translator.silence_threshold)
        self.threshold_slider.valueChanged.connect(self._update_threshold)
        tl.addWidget(self.threshold_slider)
        self.threshold_label = QLabel(f"{self.translator.silence_threshold}")
        tl.addWidget(self.threshold_label)
        cfg.addLayout(tl)
        # 침묵 시간
        dl = QHBoxLayout()
        dl.addWidget(QLabel("침묵 감지 시간(초):"))
        self.silence_slider = QSlider(Qt.Horizontal)
        self.silence_slider.setRange(5,30)
        self.silence_slider.setValue(int(self.translator.silence_duration*10))
        self.silence_slider.valueChanged.connect(self._update_silence_duration)
        dl.addWidget(self.silence_slider)
        self.silence_label = QLabel(f"{self.translator.silence_duration:.1f}")
        dl.addWidget(self.silence_label)
        cfg.addLayout(dl)
        # 번역 모드
        mlode = QHBoxLayout()
        mlode.addWidget(QLabel("번역 모드:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["실시간 번역","발화 완료 후 번역"])
        idx = 0 if self.translator.translation_mode=="realtime" else 1
        self.mode_combo.setCurrentIndex(idx)
        self.mode_combo.currentIndexChanged.connect(self._update_mode)
        mlode.addWidget(self.mode_combo)
        cfg.addLayout(mlode)
        # 원문 표시
        ol = QHBoxLayout()
        self.show_orig_cb = QCheckBox("원문 함께 표시")
        self.show_orig_cb.stateChanged.connect(lambda s: self.subtitle_win.toggle_original_text(s==Qt.Checked))
        ol.addWidget(self.show_orig_cb)
        cfg.addLayout(ol)
        # 투명도
        pl = QHBoxLayout()
        pl.addWidget(QLabel("자막 투명도:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10,100)
        self.opacity_slider.setValue(85)
        self.opacity_slider.valueChanged.connect(self._update_opacity)
        pl.addWidget(self.opacity_slider)
        cfg.addLayout(pl)

        ml.addLayout(cfg)

        # 단축키 안내
        sk = QHBoxLayout()
        sk.addWidget(QLabel("<b>단축키:</b> F1=콘트롤 창 숨기기, F2=모두 표시"))
        ml.addLayout(sk)

        self.subtitle_win = FloatingSubtitleWindow(self)
        self.subtitle_win.show()

        self._update_opacity(self.opacity_slider.value())

        self.ctrl_hide = QShortcut(QKeySequence(Qt.Key_F1), self)
        self.ctrl_hide.activated.connect(self._show_subtitle_only)
        self.ctrl_show = QShortcut(QKeySequence(Qt.Key_F2), self)
        self.ctrl_show.activated.connect(self._show_all)

        self._setup_menu()

    def _setup_signals(self):
        self.signals = TranslatorSignals()
        self.signals.translation_update.connect(self._on_translation)
        self.signals.audio_level_update.connect(self._on_level)
        self.signals.status_update.connect(self._on_status)
        self.signals.voice_detected.connect(self._on_voice)

    def _setup_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("파일")
        ex = QAction("종료", self); ex.triggered.connect(self.close)
        fm.addAction(ex)
        vm = mb.addMenu("보기")
        a1 = QAction("자막만 (F1)", self); a1.triggered.connect(self._show_subtitle_only)
        a2 = QAction("모두 표시 (F2)", self); a2.triggered.connect(self._show_all)
        vm.addAction(a1); vm.addAction(a2)

    @Slot(str,str,str)
    def _on_translation(self, ts, tr, ori):
        self.subtitle_win.update_subtitle(tr, ori if self.show_orig_cb.isChecked() else "")

    @Slot(float)
    def _on_level(self, lv):
        self.level_bar.setValue(int(lv))
        style = "QProgressBar::chunk { background-color: #5cb85c; }" if lv>self.translator.silence_threshold else \
                "QProgressBar::chunk { background-color: #d9534f; }"
        self.level_bar.setStyleSheet(style)

    @Slot(str)
    def _on_status(self, st):
        self.status_label.setText(st)

    @Slot(bool)
    def _on_voice(self, det):
        if det:
            self.status_label.setText("🎤 음성 감지 중...")
            self.status_label.setStyleSheet("font-weight:bold;color:#5cb85c;")
        else:
            self.status_label.setText("대기 중...")
            self.status_label.setStyleSheet("font-weight:bold;color:black;")

    def _update_threshold(self):
        v = self.threshold_slider.value()
        self.threshold_label.setText(f"{v}")
        self.translator.silence_threshold = v

    def _update_silence_duration(self):
        v = self.silence_slider.value()/10.0
        self.silence_label.setText(f"{v:.1f}")
        self.translator.silence_duration = v
        self.translator.silence_chunks = int(v / self.translator.chunk_duration)
        self.translator.save_config()

    def _update_mode(self):
        idx = self.mode_combo.currentIndex()
        self.translator.translation_mode = "realtime" if idx==0 else "complete"
        self.translator.save_config()

    def _update_opacity(self, v):
        self.subtitle_win.setWindowOpacity(v/100.0)

    def _show_subtitle_only(self):
        self.hide()
        self.subtitle_win.show()

    def _show_all(self):
        self.show()
        self.subtitle_win.show()
    
    def closeEvent(self, event):
        # 자막 창도 함께 닫고
        if hasattr(self, 'subtitle_win'):
            self.subtitle_win.close()
        # 루프를 완전히 종료
        QApplication.quit()
        event.accept()
        
def start_gui(translator):
    app = QApplication(sys.argv)
    win = AudioTranslatorGUI(translator)
    win.show()
    return app.exec()
