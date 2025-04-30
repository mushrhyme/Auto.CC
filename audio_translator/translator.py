import os
import asyncio
import threading
from datetime import datetime
from functools import partial

from config_manager import ConfigManager
from logger_manager import LoggerManager
from vad_detector import VADDetector
from audio_streamer import AudioStreamer
from audio_utils import save_wav
from clients import WhisperClient, GPTClient
from constants import RATE, CHUNK, CHANNELS
from signal_utils import get_audio_level

class Translator:
    def __init__(self):
        # 설정 불러오기
        self.config = ConfigManager()
        self.silence_threshold = self.config["silence_threshold"]
        self.silence_duration = self.config["silence_duration"]
        self.translation_mode = self.config["translation_mode"]
        self.preferred_device = self.config["preferred_device"]
        self.update_interval = self.config["update_interval"]

        # 로거
        self.logger = LoggerManager.setup_logger()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            raise RuntimeError("OPENAI_API_KEY not set")

        # 클라이언트
        self.whisper = WhisperClient(api_key)
        self.gpt = GPTClient(api_key)

        # 오디오 스트리밍 및 VAD
        self.streamer = AudioStreamer(device_index=self.preferred_device)
        self.vad = VADDetector(
            threshold=self.silence_threshold,
            duration=self.silence_duration,
            sample_rate=RATE,
            chunk_size=CHUNK
        )
        # self._calibrate_ambient()
        # 오디오 스트리밍 및 VAD (캘리브레이션은 start()에서)
        self.streamer = AudioStreamer(device_index=self.preferred_device)
        self.vad = VADDetector(
            threshold=self.silence_threshold,
            duration=self.silence_duration,
            sample_rate=RATE,
            chunk_size=CHUNK
        )
        # 내부 상태
        self.audio_frames = []
        self.buffer_lock = threading.Lock()
        self.voice_detected = False
        self.chunk_duration = CHUNK / RATE
        self.silence_chunks = int(self.silence_duration / self.chunk_duration)

        # asyncio 이벤트 루프
        self.loop = asyncio.new_event_loop()
        self.queue = asyncio.Queue()
        self.stop_event = asyncio.Event()

    def _calibrate_ambient(self):
        self.logger.info("🌙 환경 소음 캘리브레이션: 2초간 측정 중...")
        for _ in range(int((RATE/CHUNK)*2)):
            data, _ = self.streamer.read_chunk()
            rms, _ = self.vad.get_audio_features(data)
            self.vad.update_background(rms)
        self.logger.info(f"🌙 초기 동적 임계치: {self.vad.dynamic_threshold:.1f}")

    def set_gui_signals(self, signals):
        self.signals = signals

    def save_config(self):
        self.config["silence_threshold"] = self.silence_threshold
        self.config["silence_duration"] = self.silence_duration
        self.config["translation_mode"] = self.translation_mode
        self.config.save()
        # VAD 파라미터 즉시 반영
        self.vad.threshold = self.silence_threshold
        self.vad.silence_chunks = int(self.silence_duration / self.vad.chunk_duration)

    def get_audio_level(self, audio_data):
        return get_audio_level(audio_data)

    def start(self):
        self.streamer.open_stream()
        self.logger.info("✅ 오디오 스트리밍 시작")
        # 1) 스트림 열기
        self.streamer.open_stream()
        self.logger.info("✅ 오디오 스트리밍 시작")
        # 2) 환경 소음 캘리브레이션
        self._calibrate_ambient()
        # 3) 캡처/처리 루프 백그라운드 스레드로 시작
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._capture_loop())
        self.loop.create_task(self._process_loop())
        self.loop.run_forever()

    async def _capture_loop(self):
        recording = False
        frames = []
        silence_count = 0

        while not self.stop_event.is_set():
            data, level = await self.loop.run_in_executor(None, self.streamer.read_chunk)
            # 오디오 레벨 GUI에 전달
            self.signals.audio_level_update.emit(level)

            # VAD
            rms, flux = self.vad.get_audio_features(data)
            self.vad.update_background(rms)
            th = self.vad.dynamic_threshold
            speech = (rms > th) or (flux > th * 0.5)

            if speech:
                if not recording:
                    recording = True
                    frames = []
                    silence_count = 0
                    self.signals.status_update.emit("▶️ 발화 시작: 녹음 중")
                frames.append(data)
                self.voice_detected = True
            else:
                if recording:
                    silence_count = 1
                    frames.append(data)
                    if silence_count >= self.vad.silence_chunks:
                        self.signals.status_update.emit("⏹️ 발화 종료: 처리 큐에 추가")
                        await self.queue.put(frames.copy())
                        recording = False
                        frames = []
                        silence_count = 0
                        self.voice_detected = False

            self.signals.voice_detected.emit(self.voice_detected)
            await asyncio.sleep(self.update_interval)

    async def _process_loop(self):
        min_frames = int((RATE * 0.5) / CHUNK)

        while not self.stop_event.is_set():
            frames = await self.queue.get()
            if len(frames) < min_frames:
                continue

            # WAV로 저장
            path = await self.loop.run_in_executor(None, partial(save_wav, frames, CHANNELS, RATE))
            # 전사
            try:
                text = await self.loop.run_in_executor(None, partial(self.whisper.transcribe, path))
            except Exception as e:
                self.logger.error(f"Transcription error: {e}")
                continue

            if text:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.signals.status_update.emit(f"🔤 인식된 텍스트: {text}")
                # 번역
                try:
                    translation = await self.loop.run_in_executor(None, partial(self.gpt.translate, text))
                except Exception as e:
                    self.logger.error(f"Translation error: {e}")
                    continue

                if translation:
                    self.signals.status_update.emit("📝 번역 결과 수신")
                    self.signals.translation_update.emit(timestamp, translation, text)

            await asyncio.sleep(0)

    def stop(self):
        self.stop_event.set()
        self.streamer.close()
        self.loop.call_soon_threadsafe(self.loop.stop)
        if hasattr(self, "thread"):
            self.thread.join()
        self.logger.info("🛑 Translator 중지 완료")
