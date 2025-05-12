import os
import re
import io
import time
import json
import queue
import wave
import asyncio
import logging
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import pyaudio
import requests
import numpy as np
import soundfile as sf
import aiohttp
from langdetect import detect
from PyQt5.QtCore import QTimer
import backoff

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1  
RATE = 8000  # 초당 수집하는 오디오 프레임 수: 높을수록 품질이 좋아지나 더 많은 데이터를 처리해야 함(16000 = Whisper 권장 샘플링 레이트)
CHUNK = 1024 # 한 번에 처리하는 오디오 프레임 수:  작을수록 빠르게 처리되나 품질이 떨어질 수 있음
SILENCE_THRESHOLD = 200 # 평균 진폭이 이 값 미만이면 침묵으로 판단 (기존 400에서 200으로 낮춤)
SILENCE_DURATION = 1.5 # 침묵 지속 시간: 이 시간 동안 침묵이면 발화가 종료된 것으로 간주
REALTIME_UPDATE_INTERVAL = 1.0 # 실시간 번역 업데이트 간격: 음성이 진행 중일 때 현재까지 수집된 버퍼를 주기적으로 처리하여 중간 번역 결과를 보여주는 시간 간격
MAX_SENTENCE_LENGTH = 50 # 최대 문장 길이 (자유롭게 조정 가능)
TARGET_LANGUAGES = {
    'ko': ['Chinese', 'Japanese', 'English'],
    'ja': ['Korean', 'Chinese', 'English'],
    'en': ['Korean', 'Chinese', 'Japanese'],
    'zh': ['Korean', 'Japanese', 'English']
}
# GPT_MODEL = "gpt-3.5-turbo"  
GPT_MODEL = "gpt-4o-mini-2024-07-18"
# API endpoints
TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
TRANSLATION_URL = "https://api.openai.com/v1/chat/completions"

class AudioTranslator:
    def __init__(self):
        self.setup_logging()
        self.load_config()
        self.audio_queue = queue.Queue()
        self.is_running = True
        self.voice_detected = False
        self.silence_count = 0
        self.last_translation = ""
        self.audio_frames = []
        self.buffer_lock = threading.Lock()
       
        # API 키 초기화
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            raise ValueError("OPENAI_API_KEY 환경 변수를 설정해주세요.")

    # ---------- 초기화 및 설정 ----------
    def setup_logging(self):
        """로깅 시스템 구성"""
        # 로그 파일 설정
        log_folder = Path("logs")
        log_folder.mkdir(exist_ok=True)

        # 로그 파일 이름 설정
        self.log_filename = log_folder / f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # 로거 설정
        self.logger = logging.getLogger("TranslationLogger")
       
        # 중요: 기존 핸들러 제거 (로그 중복 방지)
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
       
        self.logger.setLevel(logging.INFO)
       
        # 파일 및 콘솔 핸들러 설정
        file_handler = logging.FileHandler(self.log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
       
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
       
        # 로그 형식 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
       
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
       
    def load_config(self):
        """설정 파일에서 구성 로드 또는 기본값 설정"""
        config_path = Path("config.json")
       
        # 기본 설정
        self.config = {
            "silence_threshold": SILENCE_THRESHOLD,
            "silence_duration": SILENCE_DURATION,
            "preferred_device": 0,  # 기본 장치 인덱스
            "update_interval": REALTIME_UPDATE_INTERVAL,
            "translation_mode": "complete"
        }
       
        # 파일에서 설정 로드
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                # self.logger.debug("Configuration loaded from file")
            except Exception as e:
                self.logger.error(f"Error loading config: {e}", exc_info=True)
       
        # 기존 설정 로드 시도
        self.silence_threshold = self.config["silence_threshold"]
        self.silence_duration = self.config["silence_duration"]
        self.update_interval = self.config["update_interval"]
        self.translation_mode = self.config["translation_mode"]
        self.selected_device = self.config["preferred_device"]  # 추가: selected_device 설정
       
        # 설정에서 변수 설정
        self.chunk_duration = CHUNK / RATE
        self.silence_chunks = int(self.silence_duration / self.chunk_duration)
        self.min_volume_for_display = 200

    def save_config(self):
        """현재 값으로 설정 파일 업데이트"""
        # 현재 값으로 설정 업데이트
        self.config["silence_threshold"] = self.silence_threshold
        self.config["silence_duration"] = self.silence_duration
        self.config["preferred_device"] = self.selected_device
        self.config["update_interval"] = self.update_interval
        self.config["translation_mode"] = self.translation_mode
       
        # 파일 저장
        try:
            with open("config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
            # self.logger.debug("Configuration saved to file")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}", exc_info=True)

    # ---------- 오디오 장치 관리 ----------
    def list_audio_devices(self):
        """사용 가능한 모든 오디오 입력 장치 나열"""
        # PyAudio를 사용하여 사용 가능한 오디오 장치 목록 가져오기
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
       
        self.logger.info("\nAvailable audio input devices:")
        self.logger.info("-" * 50)
       
        # 모든 장치에 대한 정보 표시
        devices = []
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            # 입력 채널이 있는 장치만 표시
            if device_info.get('maxInputChannels') > 0:
                devices.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxInputChannels'),
                    'sample_rate': int(device_info.get('defaultSampleRate'))
                })
               
                self.logger.info(f"Device number {i}: {device_info.get('name')}")
                self.logger.info(f"  Channels: {device_info.get('maxInputChannels')}")
                self.logger.info(f"  Sample rate: {int(device_info.get('defaultSampleRate'))}")
                self.logger.info("-" * 50)
        # PyAudio 정리
        p.terminate()
        return devices

    def check_audio_input(self, device_index):
        """오디오 입력 상태 확인"""
        p = pyaudio.PyAudio()
        try:
            device_info = p.get_device_info_by_index(device_index if device_index is not None else
                                                   p.get_default_input_device_info()['index'])
        except Exception as e:
            self.logger.error(f"오디오 입력 확인 중 오류 발생: {e}", exc_info=True)
        finally:
            p.terminate()
    
    # ---------- 오디어 데이터 처리 ----------
    @staticmethod
    def get_audio_level(audio_data):
        """오디오 데이터의 볼륨 레벨 계산"""
        if len(audio_data) == 0:
            print("Audio data is empty.")  # 디버깅 로그 추가
            return 0
       
        # 더 효율적인 처리를 위해 numpy 배열로 변환
        normalized = np.abs(np.frombuffer(audio_data, dtype=np.int16))
        if len(normalized) == 0:
            print("Normalized audio data is empty.")  # 디버깅 로그 추가
            return 0

        # 더 나은 감지를 위해 상위 10% 샘플 사용
        sorted_samples = np.sort(normalized)
        top_samples = sorted_samples[int(len(sorted_samples) * 0.9):]
       
        # 상위 샘플이 없으면 전체 사용
        if len(top_samples) > 0:
            return np.mean(top_samples)
        return np.mean(normalized)

    def should_transcribe(self, audio_level):
        """오디오 레벨에 따라 음성이 감지되었는지 확인"""
        if audio_level > self.silence_threshold:
            self.silence_count = 0
           
            # 음성 감지 상태 업데이트
            if not self.voice_detected:
                self.voice_start_time = datetime.now()  # 음성 시작 시간 기록
                self.transcribe_start_time = self.voice_start_time
                self.logger.info(f"✅ 발화 감지! Level: {audio_level:.1f}")    
               
                # 감지된 언어를 업데이트
                if hasattr(self, 'gui_signals'):
                    try:
                        # 최근 오디오 프레임을 텍스트로 변환
                        with self.buffer_lock:
                            frames_copy = list(self.audio_frames[-10:])  # 최근 10개의 프레임 복사
                        if frames_copy:
                            audio_file_path = self.save_audio_to_wav(frames_copy)
                            transcription = self.transcribe_audio(audio_file_path)
                            if transcription:
                                detected_language = detect(transcription)  # 텍스트로 언어 감지
                                detected_language = 'zh' if 'zh' in detected_language else detected_language
                                language_map = {
                                    "ko": "한국어",
                                    "en": "영어",
                                    "zh": "중국어",
                                    "ja": "일본어"
                                }
                                language_name = language_map.get(detected_language, "알 수 없음")
                                self.gui_signals.status_update.emit(f"{language_name}를 감지 중입니다...")
                            else:
                                self.gui_signals.status_update.emit("언어 감지 실패")
                        else:
                            self.gui_signals.status_update.emit("오디오 데이터 부족으로 언어 감지 실패")
                    except Exception as e:
                        self.logger.error(f"언어 감지 중 오류 발생: {e}", exc_info=True)
                        self.gui_signals.status_update.emit("언어 감지 실패")

               
            self.voice_detected = True
            return True
        else:
            self.silence_count += 1
           
            # 침묵 후 음성 감지 상태 재설정
            if self.silence_count > self.silence_chunks and self.voice_detected:
                # 발화 종료 로그 제거 (중복 방지)
                self.voice_start_time = None
                self.transcribe_start_time = None
                self.voice_detected = False
           
            return False

    def save_audio_to_wav(self, frames, temp=True, channels=None):
        """오디오 프레임을 WAV 파일로 저장"""
        if not frames:
            return None
       
        # 지정되지 않은 경우 기본 채널 사용
        if channels is None:
            channels = CHANNELS
       
        if temp:
            # 임시 파일에 저장
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_filename = temp_file.name
            temp_file.close()
        else:
            # 고정 파일명에 저장
            temp_filename = "realtime_audio.wav"
       
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 2 bytes for paInt16
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
       
        return temp_filename

    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=1)
    def transcribe_audio(self, audio_file_path):
        """오디오 파일을 텍스트로 변환"""
        try:
            flac_file_path = self._convert_to_flac(audio_file_path)
            return self._call_transcription_api(flac_file_path)
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}", exc_info=True)
            raise
        finally:
            self._cleanup_temp_files(audio_file_path, flac_file_path)

    def _convert_to_flac(self, audio_file_path):
        """WAV 파일을 FLAC 포맷으로 변환"""
        flac_file_path = audio_file_path.replace(".wav", ".flac")
        with sf.SoundFile(audio_file_path) as wav_file:
            data = wav_file.read(dtype='int16')
            sf.write(flac_file_path, data, wav_file.samplerate, format='FLAC')
        return flac_file_path

    def _call_transcription_api(self, flac_file_path):
        """API 호출로 텍스트 변환"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with open(flac_file_path, 'rb') as audio_file:
            files = {
                'file': (os.path.basename(flac_file_path), audio_file, 'audio/flac'),
                'model': (None, 'whisper-1'),
                'response_format': (None, 'json')
            }
            response = requests.post(TRANSCRIPTION_URL, headers=headers, files=files)

        if response.status_code == 200:
            return response.json().get('text', '')
        else:
            self.logger.error(f"Transcription API error: {response.status_code}, {response.text}", exc_info=True)
            return None

    def _cleanup_temp_files(self, *file_paths):
        """임시 파일 삭제"""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    self.logger.error(f"Failed to delete temporary file: {e}", exc_info=True)
                    
    def convert_to_flac(self, audio_data):
        """오디오 데이터를 메모리 내에서 FLAC 포맷으로 변환"""
        try:
            # audio_data가 바이트 형식이 아닌 경우 바이트로 변환
            if isinstance(audio_data, str):
                audio_data = audio_data.encode('utf-8')

            # 음성 데이터를 numpy 배열로 변환
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # FLAC 포맷으로 변환하기 위한 파일-like 객체 생성
            flac_buffer = io.BytesIO()
           
            # soundfile을 사용해 numpy 데이터를 FLAC 형식으로 변환하여 메모리로 저장
            with sf.SoundFile(flac_buffer, 'w', samplerate=RATE, channels=CHANNELS, format='FLAC') as file:
                file.write(audio_np)

            # 메모리에서 FLAC 데이터 반환
            flac_buffer.seek(0)
            return flac_buffer.read()  # 실제 FLAC 데이터 반환
        except Exception as e:
            self.logger.error(f"Error during FLAC conversion: {e}", exc_info=True)
        return None

    def audio_capture(self, device_index=None):
        """오디오 캡처 및 큐에 데이터 추가"""
        self.check_audio_input(device_index)

        p = pyaudio.PyAudio()
        stream = None
        selected_channels = CHANNELS

        def log_audio_level(data):
            """오디오 레벨을 로깅하고 GUI에 업데이트"""
            audio_level = self.get_audio_level(data)
            if hasattr(self, 'gui_signals'):
                self.gui_signals.audio_level_update.emit(audio_level)

        try:
            # 장치 정보 가져오기
            if device_index is not None:
                try:
                    device_info = p.get_device_info_by_index(device_index)
                    max_input_channels = int(device_info.get('maxInputChannels', 1))
                    selected_channels = min(selected_channels, max_input_channels)
                    self.logger.info(f"장치 정보: {device_info.get('name')} (채널: {selected_channels})")
                except Exception as e:
                    self.logger.error(f"장치 정보를 가져오는데 실패했습니다: {e}", exc_info=True)
                    device_index = None

            # 오디오 스트림 열기
            stream = self._open_audio_stream(p, device_index, selected_channels)

            if not stream:
                self.logger.error("오디오 스트림을 열지 못했습니다. 캡처를 종료합니다.")
                return

            self.logger.info(f"현재 침묵 임계값: {self.silence_threshold} (이 값 이상이면 오디오가 감지됨)")

            # 오디오 캡처 루프
            silence_counter = 0
            speech_detected_during_session = False
            volume_monitor_counter = 0

            while self.is_running:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    log_audio_level(data)

                    # 현재 청크의 평균 진폭 계산
                    audio_level = self.get_audio_level(data)

                    # 볼륨 레벨 모니터링 (5초마다)
                    volume_monitor_counter += 1
                    if volume_monitor_counter >= 80:  # 80 * 0.0625 = 5초
                        volume_monitor_counter = 0

                    # 버퍼에 데이터 추가
                    with self.buffer_lock:
                        self.audio_frames.append(data)

                    # 음성 감지 확인
                    if self.should_transcribe(audio_level):
                        silence_counter = 0
                        speech_detected_during_session = True
                    else:
                        silence_counter += 1

                    # 침묵이 지속되면 세션 처리 종료
                    if silence_counter >= self.silence_chunks and len(self.audio_frames) > 0:
                        with self.buffer_lock:
                            frames_copy = list(self.audio_frames)
                            self.audio_frames.clear()

                        # 충분한 데이터가 있고 음성이 감지된 경우에만 처리
                        min_frames = int((RATE * 1.5) / CHUNK)
                        if len(frames_copy) > min_frames and speech_detected_during_session:
                            self.audio_queue.put((frames_copy, selected_channels))

                        silence_counter = 0
                        speech_detected_during_session = False

                except Exception as e:
                    self.logger.error(f"오디오 캡처 중 에러가 발생하였습니다: {e}", exc_info=True)

        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

    def _open_audio_stream(self, p, device_index, channels):
        """오디오 스트림을 열고 실패 시 기본 장치 또는 모노 채널로 시도"""
        try:
            return p.open(
                format=FORMAT,
                channels=channels,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK
            )
        except Exception as e:
            self.logger.error(f"스트림 열기 실패: {e}", exc_info=True)
            self.logger.info("기본 입력 장치를 시도합니다...")

            try:
                default_info = p.get_default_input_device_info()
                default_channels = min(CHANNELS, int(default_info.get('maxInputChannels', 1)))
                return p.open(
                    format=FORMAT,
                    channels=default_channels,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
            except Exception as e2:
                self.logger.error(f"기본 장치도 실패했습니다: {e2}", exc_info=True)
                self.logger.info("모노 채널로 마지막 시도를 합니다...")

                try:
                    return p.open(
                        format=FORMAT,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK
                    )
                except Exception as e3:
                    self.logger.error(f"모든 오디오 스트림 열기 시도가 실패했습니다: {e3}", exc_info=True)
                    return None
    
    # ---------- 번역 처리 ----------
    async def translate_text_async(self, text, size=200):
        """텍스트를 청크로 나누어 비동기 번역"""
        if not text or not text.strip():
            return None

        chunks = [text[i:i + size] for i in range(0, len(text), size)]
        all_translations = await asyncio.gather(*(self.translate_chunk(chunk) for chunk in chunks))
        return self._merge_translations(all_translations)

    def _merge_translations(self, all_translations):
        """번역 결과 병합"""
        final = {}
        for translation_set in all_translations:
            if translation_set:
                for lang, trans in translation_set.items():
                    final.setdefault(lang, []).append(trans)
        return {k: ' '.join(filter(None, v)) for k, v in final.items()}

    async def translate_chunk(self, chunk):
        try:
            lang = detect(chunk)
            targets = TARGET_LANGUAGES.get(lang[:2], [])
            results = await asyncio.gather(*(self.call_translation_api(chunk, t) for t in targets))
            return dict(zip(targets, results))
        except Exception as e:
            self.logger.error(f"Language detection error: {str(e)}")
            return None

    async def call_translation_api(self, chunk, target_lang):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        prompt = f'Translate to {target_lang}: "{chunk}"'
        data = {"model": GPT_MODEL, "messages": [{"role": "user", "content": prompt}]}

        async with aiohttp.ClientSession() as session:
            async with session.post(TRANSLATION_URL, headers=headers, json=data) as resp:
                if resp.status == 200:
                    json_data = await resp.json()
                    return json_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                self.logger.error(f"API error {resp.status}: {await resp.text()}")
                return None
            
    def process_translation_result(self, translation, transcription, prev_translation, accumulated_text, speech_id):
        """번역 결과 처리 및 GUI 업데이트"""
       
        # 새로운 번역 시작 시 초기화
        timestamp = datetime.now().strftime('%H:%M:%S')
       
        prev_translation, accumulated_text = self._update_translation_state(
            translation, prev_translation, accumulated_text, speech_id
        )

        gui_message = self._prepare_gui_message(translation, transcription)
   
        # GUI 신호 발송
        if hasattr(self, 'gui_signals'):
            self.gui_signals.translation_update.emit(
                timestamp,
                json.dumps(gui_message),  # 딕셔너리를 JSON 문자열로 변환
                transcription
            )

            # # 비동기 방식이 아닌 QTimer를 사용하여 GUI 업데이트를 안전하게 처리
            # def emit_signal():
            #     self.gui_signals.translation_update.emit(timestamp, gui_message, translation)

            # # QTimer를 사용해 이벤트 루프에 안전하게 GUI 신호 발송
            # QTimer.singleShot(0, emit_signal)

    def _update_translation_state(self, translation, prev_translation, accumulated_text, speech_id):
        """번역 상태 업데이트"""
        if not hasattr(self, 'translation_results'):
            self.translation_results = {}

        if speech_id not in self.translation_results:
            self.translation_results[speech_id] = {"prev_translation": "", "accumulated_text": ""}

        current_state = self.translation_results[speech_id]
        prev_translation = current_state["prev_translation"]
        accumulated_text = current_state["accumulated_text"]

        if not translation:
            return prev_translation, accumulated_text

        if len(accumulated_text) < MAX_SENTENCE_LENGTH:
            if prev_translation and translation.startswith(prev_translation):
                new_text = translation[len(prev_translation):].strip()
                if new_text:
                    accumulated_text += " " + new_text
            else:
                accumulated_text = translation

        self.translation_results[speech_id]["prev_translation"] = translation
        self.translation_results[speech_id]["accumulated_text"] = accumulated_text
        self.last_translation = accumulated_text
        return prev_translation, accumulated_text
               
    def _prepare_gui_message(self, translation, transcription):
        """GUI 메시지 구성"""
        try:
            detected_language = detect(transcription)
        except Exception as e:
            self.logger.error(f"Language detection error: {e}")
            detected_language = "unknown"

        gui_message = {
            "korean": "",
            "english": "",
            "chinese": "",
            "japanese": ""
        }

        if detected_language == "ko":
            gui_message["korean"] = transcription
        elif detected_language == "en":
            gui_message["english"] = transcription
        elif "zh" in detected_language:
            gui_message["chinese"] = transcription
        elif detected_language == "ja":
            gui_message["japanese"] = transcription
        else:
            self.logger.warning(f"Unsupported language detected: {detected_language}")

        gui_message["korean"] = gui_message["korean"] or translation.get("Korean", "")
        gui_message["english"] = gui_message["english"] or translation.get("English", "")
        gui_message["chinese"] = gui_message["chinese"] or translation.get("Chinese", "")
        gui_message["japanese"] = gui_message["japanese"] or translation.get("Japanese", "")
        return gui_message

 
    async def process_translation_queue(self):
        """번역 큐 처리"""
        current_speech_id = 0

        while self.is_running:
            frames_copy, selected_channels = await self._get_audio_from_queue()
            if frames_copy is None:
                continue

            current_speech_id += 1
            try:
                await self.handle_audio_frames(frames_copy, selected_channels, current_speech_id)
                self.audio_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error in translation processing: {e}", exc_info=True)

    async def _get_audio_from_queue(self):
        """오디오 큐에서 데이터 가져오기"""
        try:
            return self.audio_queue.get(timeout=0.01)
        except queue.Empty:
            await asyncio.sleep(0.01)
        return None, None
    
    async def handle_audio_frames(self, frames, channels, speech_id):
        """오디오 프레임 처리"""
        audio_file_path = self.save_audio_to_wav(frames, channels=channels)

        self.log_timing_on_end_of_speech() # 발화 종료 시간 로깅
        
        # 번역 시작 신호 전송
        self.emit_gui_signal_if_available("translation_started")
        self.logger.info("✅ 번역 시작")

        # 번역 진행 표시 바 시작
        if hasattr(self, 'gui_signals') and hasattr(self.gui_signals, 'translation_started'):
            self.gui_signals.translation_started.emit()
        
        # 음성 추출
        transcription = self.transcribe_audio(audio_file_path)
        if not transcription:
            return

        # 번역 처리
        translation_start = datetime.now()
        translations = await self._perform_translation(transcription)
        translation_end = datetime.now()
        if translations:
            duration = (translation_end - translation_start).total_seconds()
            self.logger.info(f"✅ 번역 종료! 번역 처리 시간: {duration:.2f}초")
            self.process_translation_result(translations, transcription, "", "", speech_id)

    async def _perform_translation(self, transcription):
        """텍스트 번역 수행"""
        translation_start = datetime.now()
        translations = await self.translate_text_async(transcription)
        translation_end = datetime.now()

        self.logger.info(f"✅ 번역 종료! 번역 시간: {(translation_end - translation_start).total_seconds():.2f}초")
        return translations
    
    # ---------- GUI 업데이트 ----------
    def set_gui_signals(self, signals):
        """GUI 신호 객체 설정"""
        # self.logger.debug("GUI 신호 객체 설정됨")
        self.gui_signals = signals

    def emit_gui_signal_if_available(self, signal_name):
        if hasattr(self, 'gui_signals'):
            signal = getattr(self.gui_signals, signal_name, None)
            if signal:
                signal.emit()

    def log_timing_on_end_of_speech(self):
        if hasattr(self, 'transcribe_start_time') and self.transcribe_start_time:
            end_time = datetime.now()
            duration = (end_time - self.transcribe_start_time).total_seconds()
            self.logger.info(f"✅ 발화 종료! 발화 시간: {duration:.2f}초")

    # ---------- 기타 ----------
    def start_threads(self):
        """GUI 모드에서 사용할 스레드만 시작 (사용자 입력 없음)"""
        # 오디오 캡처 및 번역 처리를 위한 스레드 시작
        capture_thread = threading.Thread(
            target=self.audio_capture,
            args=(self.selected_device,),
            daemon=True,
            name="audio_capture_thread"
        )
        capture_thread.start()

        # 비동기 번역 큐 처리
        asyncio_thread = threading.Thread(
            target=lambda: asyncio.run(self.process_translation_queue()),
            daemon=True,
            name="translation_queue_thread"
        )
        asyncio_thread.start()

        self.logger.info("오디오 캡처 및 번역 처리 스레드 시작...")
        self.logger.info("발화 완료 후에만 번역 결과가 표시됩니다.")