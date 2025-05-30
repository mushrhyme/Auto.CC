import os
import time
import json
import queue
import wave
import asyncio
import logging
import random
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
import pyaudio
import requests
import numpy as np
import soundfile as sf
from langdetect import detect
from PyQt5.QtCore import QTimer
import backoff
import socket
import boto3
import websockets
from presigned_url import AWSTranscribePresignedURL
from eventstream import create_audio_event, decode_event


# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1  
RATE = 8000  # 초당 수집하는 오디오 프레임 수: 높을수록 품질이 좋아지나 더 많은 데이터를 처리해야 함(16000 = Whisper 권장 샘플링 레이트)
CHUNK = 1024 # 한 번에 처리하는 오디오 프레임 수:  작을수록 빠르게 처리되나 품질이 떨어질 수 있음
SILENCE_THRESHOLD = 600 # 평균 진폭이 이 값 미만이면 침묵으로 판단
SILENCE_DURATION = 2 # 침묵 지속 시간: 이 시간 동안 침묵이면 발화가 종료된 것으로 간주
REALTIME_UPDATE_INTERVAL = 1.0 # 실시간 자막 업데이트 간격: 음성이 진행 중일 때 현재까지 수집된 버퍼를 주기적으로 처리하여 원문을 보여주는 시간 간격
MAX_SENTENCE_LENGTH = 50 # 최대 문장 길이 (자유롭게 조정 가능)
TARGET_LANGUAGES = {
    'ko': ['zh','ja','en'],
    'ja': ['ko','zh','en'],
    'en': ['ko','zh','ja'],
    'zh': ['ko','ja','en']
}
# GPT_MODEL = "gpt-3.5-turbo"  
GPT_MODEL = "gpt-4o-mini-2024-07-18"

# API endpoints
TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
TRANSLATION_URL = "https://api.openai.com/v1/chat/completions"
# 172.17.17.82:8080

load_dotenv()

class AudioTranslator:
    def __init__(self, translation_mode="aws", language_code="en"):
        self.setup_logging()
        self.load_config()
        self.audio_folder = Path("audio")
        self.audio_folder.mkdir(exist_ok=True)
        self.translation_mode = translation_mode
        self.audio_queue = queue.Queue()
        self.is_running = True
        self.voice_detected = False
        self.silence_count = 0
        self.update_interval = 1.0
        self.detected_language = None
        self.last_translation = ""
        self.audio_frames = []
        self.buffer_lock = threading.Lock()
        self.aws_creds = None    # 캐시할 자격증명 dict
        self.aws_creds_expires = None
        self._partial_timer = None
        self._last_partial = ""             # 마지막으로 보낸 partial 저장
        self.min_partial_length = 4  
        
        # API 키 초기화
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            raise ValueError("OPENAI_API_KEY 환경 변수를 설정해주세요.")
        
        # AWS Transcribe 언어 코드
        language_map = {
            "en": "en-US", 
            "ko": "ko-KR", 
            "ja": "ja-JP", 
            "zh": "zh-CN"
        }
        self.aws_language_code = language_map[language_code] # AWS Transcribe API용
        self.aws_source_lang_code = language_code # AWS Translate API용
        
        self.aws_stream_stop = threading.Event()
        self.use_silence_vad = True
        self._generate_presigned_url()
        
    # ---------- 초기화 및 설정 ----------
    def _ensure_aws_credentials(self):
        """AWS 세션 토큰을 캐싱해서 관리"""
        now = datetime.now(timezone.utc)
        if self.aws_creds is None or now + timedelta(minutes=5) >= self.aws_creds_expires:
            sts = boto3.client("sts", region_name="ap-northeast-2")
            resp = sts.get_session_token(DurationSeconds=3600)
            info = resp["Credentials"]

            self.aws_creds = {
                "aws_access_key_id": info["AccessKeyId"],
                "aws_secret_access_key": info["SecretAccessKey"],
                "aws_session_token": info["SessionToken"],
            }
            self.aws_creds_expires = info["Expiration"] 
       
        session = boto3.session.Session(
            **self.aws_creds,
            region_name="ap-northeast-2"
        )
        self.aws_translate = session.client("translate")
        self.aws_transcribe = AWSTranscribePresignedURL(
            self.aws_creds["aws_access_key_id"],
            self.aws_creds["aws_secret_access_key"],
            self.aws_creds["aws_session_token"],
            region="ap-northeast-2"
        )
        
        
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
        """기본값 설정"""
        # 기본 설정
        self.config = {
            "silence_threshold": SILENCE_THRESHOLD,
            "silence_duration": SILENCE_DURATION,
            "preferred_device": 0,  # 기본 장치 인덱스
        }
        
        # 기존 설정 로드 시도
        self.silence_threshold = self.config["silence_threshold"]
        self.silence_duration = self.config["silence_duration"]
        self.selected_device = self.config["preferred_device"]  # 추가: selected_device 설정
    
        # 설정에서 변수 설정
        self.chunk_duration = CHUNK / RATE
        self.silence_chunks = int(self.silence_duration / self.chunk_duration)
        self.min_volume_for_display = 200

    # ---------- 오디어 데이터 처리 ----------s
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
        top_samples = sorted_samples[int(len(sorted_samples) * 0.8):]
    
        # 상위 샘플이 없으면 전체 사용
        if len(top_samples) > 0:
            return np.mean(top_samples)
        return np.mean(normalized)

    def should_transcribe(self, audio_level):
        """오디오 레벨에 따라 음성이 감지되었는지 확인"""
        if audio_level > self.silence_threshold:        
            # 음성 감지 상태 업데이트
            if not self.voice_detected:
                self.voice_start_time = datetime.now()  # 음성 시작 시간 기록
                self.transcribe_start_time = self.voice_start_time
                self.logger.info(f"✅ 발화 감지! Level: {audio_level:.1f}")    
                self._start_partial_updates()
            self.voice_detected = True
            self.silence_count = 0
            return True
        else:
            self.silence_count += 1
            # 침묵 후 음성 감지 상태 재설정
            if self.silence_count > self.silence_chunks and self.voice_detected:
                # 발화 종료 로그 제거 (중복 방지)
                self.logger.info("✅ [실시간] 발화가 완전히 종료되었습니다!")
                self.voice_start_time = None
                self.transcribe_start_time = None
                self.voice_detected = False
                self._stop_partial_updates()
            return False
        
    # def _start_partial_updates(self):
    #     """부분 자막 송출용 타이머 시작"""
    #     def _send_partial():
    #         try:
    #             with self.buffer_lock:
    #                 frames = list(self.audio_frames)
    #             if not frames:
    #                 return
        
    #             tmp = self.save_audio_to_wav(frames, temp=True)
    #             result = self.transcribe_audio(tmp, ports=8080)
    #             text = (result or {}).get("original_text", "").strip()

    #             # 1) 새 텍스트인가? 2) 충분히 길이가 있는가?
    #             if text and text != self._last_partial and len(text) >= self.min_partial_length:
    #                 self._last_partial = text
    #                 self._emit_stt_original(text)
    #         except Exception as e:
    #             self.logger.error("❌ _send_partial 중 예외 발생: %s", e, exc_info=True)
    #         finally:
    #             # 타이머를 재스케줄
    #             self._partial_timer = threading.Timer(self.update_interval, _send_partial)
    #             self._partial_timer.daemon = True
    #             self._partial_timer.start()

    #     _send_partial()

    def _start_partial_updates(self):
        def _partial_loop():
            while self.is_running:                # ① 항상 돌아감
                if self.voice_detected:           # ② 음성 감지 중일 때만 처리
                    try:
                        with self.buffer_lock:
                            frames = list(self.audio_frames)
                        if frames:
                            tmp = self.save_audio_to_wav(frames, temp=True)
                            result = self.transcribe_audio(tmp, ports=8080)
                            text = (result or {}).get("original_text", "").strip()
                            if text and text != self._last_partial and len(text) >= self.min_partial_length:
                                self._last_partial = text
                                self._emit_stt_original(text)
                    except Exception as e:
                        self.logger.error("❌ 부분 자막 예외: %s", e, exc_info=True)
                time.sleep(self.update_interval)  # ③ 음성 감지 유무 상관없이 주기 대기

        # 중복 스레드 생성 방지
        if getattr(self, "_partial_thread", None) and self._partial_thread.is_alive():
            return

        self._partial_thread = threading.Thread(
            target=_partial_loop, daemon=True, name="partial_update_thread"
        )
        self._partial_thread.start()



    def _stop_partial_updates(self):
        """부분 자막 타이머 취소"""
        if self._partial_timer:
            self._partial_timer.cancel()
            self._partial_timer = None
        self._last_partial = "" 

    def save_audio_to_wav(self, frames, temp=True, channels=None):
        """오디오 프레임을 WAV 파일로 저장"""
        if not frames:
            return None

        if channels is None:
            channels = CHANNELS

        if temp:
            # 기존 임시 파일 로직
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            filename = temp_file.name
            temp_file.close()
        else:
            # 오디오 폴더에 타임스탬프 파일명으로 저장
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = str(self.audio_folder / f"audio_{ts}.wav")

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        return filename
    
    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=1)    
    def transcribe_audio(self, audio_file_path, ports):
        """
        오디오 파일을 텍스트로 변환 (로컬 STT API 사용)
        """
        try:     
            result = self._call_transcription_api(audio_file_path, ports)
            # print(f"call_local_api -> result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"STT 호출 중 오류 발생: {e}", exc_info=True)
            return None
    
    def _call_transcription_api(self, file_path, ports=None):
        """API 호출로 텍스트 변환"""
        # OpenAI API를 서버에서 호출하는 경우
        if self.translation_mode == "server":
            def is_port_open(host, port, timeout=1.0):
                try:
                    with socket.create_connection((host, port), timeout=timeout):
                        return True
                except socket.timeout:
                    self.logger.error(f"포트 {port} 연결 시도 중 타임아웃 발생")
                except ConnectionRefusedError:
                    self.logger.error(f"포트 {port} 연결이 거부되었습니다")
                except Exception as e:
                    self.logger.error(f"포트 {port} 연결 중 알 수 없는 오류 발생: {e}")
                return False
                

            STT_SERVER_IP = "172.17.17.82"
            # 172.26.81.43
            # 172.25.1.95
            available_ports = [ports]
            # [port for port in ports if is_port_open(STT_SERVER_IP, port)]

            if not available_ports:
                self.logger.error("⚠️ 연결 가능한 STT 포트가 없습니다.")
                return None

            port = random.choice(available_ports)
            url = f"http://{STT_SERVER_IP}:{port}/api/transcribe"

            try:
                with open(file_path, 'rb') as f:
                    files = {'audio_file': ('audio.wav', f, 'audio/wav')}
                    response = requests.post(url, files=files, timeout=20)
                    if response.status_code == 200:
                        result = response.json()
                        return result
                    else:
                        self.logger.error(f"STT API 오류 {response.status_code}: {response.text}")
                        return None
            except Exception as e:
                self.logger.error(f"로컬 STT 전송 실패: {e}", exc_info=True)
                return None
        # OpenAI API를 로컬에서 사용하는 경우
        else:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            with open(file_path, 'rb') as audio_file:
                files = {
                    'file': (os.path.basename(file_path), audio_file, 'audio/flac'),
                    'model': (None, 'whisper-1'),
                    'response_format': (None, 'json')
                }
                response = requests.post(TRANSCRIPTION_URL, headers=headers, files=files, timeout=20)

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Transcription API error: {response.status_code}, {response.text}", exc_info=True)
                return None
    
    def log_audio_level(self, data):
        """오디오 레벨을 로깅하고 GUI에 업데이트"""
        audio_level = self.get_audio_level(data)
        if hasattr(self, 'gui_signals'):
            self.gui_signals.audio_level_update.emit(audio_level)
      
    def audio_capture(self, device_index=None):
        """오디오 캡처 및 큐에 데이터 추가"""
        p = pyaudio.PyAudio()
        stream = self._open_audio_stream(p, device_index, CHANNELS)

        try:
            if self.translation_mode=="aws":
                while self.is_running:
                    chunk = stream.read(CHUNK, exception_on_overflow=False)
                    # AWS 스트리밍 루프가 꺼내가도록 큐에 넣는다
                    self.audio_queue.put(chunk)
            else:
                self.logger.info(f"현재 침묵 임계값: {self.silence_threshold} (이 값 이상이면 오디오가 감지됨)")

                # 오디오 캡처 루프
                silence_counter = 0
                speech_detected_during_session = False
                volume_monitor_counter = 0
                while self.is_running:
                    try:
                        chunk = stream.read(CHUNK, exception_on_overflow=False)
                        self.log_audio_level(chunk)

                        # 현재 청크의 평균 진폭 계산
                        audio_level = self.get_audio_level(chunk)

                        # 볼륨 레벨 모니터링 (5초마다)
                        volume_monitor_counter += 1
                        if volume_monitor_counter >= 80:  # 80 * 0.0625 = 5초
                            volume_monitor_counter = 0

                        # 버퍼에 데이터 추가
                        with self.buffer_lock:
                            self.audio_frames.append(chunk)

                        # 음성 감지 확인
                        voice_detect = self.should_transcribe(audio_level)
                        if voice_detect:
                            silence_counter = 0
                            speech_detected_during_session = True
                        else:
                            silence_counter += 1

                        # 침묵이 지속되면 세션 처리 종료
                        if silence_counter >= self.silence_chunks and len(self.audio_frames) > 0:
                            for _ in range(3):  # 약 0.3 ~ 0.4초 분량 더 수집
                                try:
                                    extra_data = stream.read(CHUNK, exception_on_overflow=False)
                                    with self.buffer_lock:
                                        self.audio_frames.append(extra_data)
                                except Exception as e:
                                    self.logger.error(f"추가 오디오 수집 중 오류: {e}", exc_info=True)
                                    
                            with self.buffer_lock:
                                frames_copy = list(self.audio_frames)
                                self.audio_frames.clear()

                            # 충분한 데이터가 있고 음성이 감지된 경우에만 처리
                            min_frames = int((RATE * 1.5) / CHUNK)
                            if len(frames_copy) > min_frames and speech_detected_during_session:
                                self.audio_queue.put((frames_copy, CHANNELS))
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

        # 번역 결과에 언어별 기본값 설정
        for lang in ["English", "Chinese", "Korean"]:
            translation.setdefault(lang, "")
        
        # 키가 있는 언어를 기준으로 업데이트
        for lang, translated_text in translation.items():
            prev_trans = prev_translation.get(lang, "") if isinstance(prev_translation, dict) else prev_translation
            if len(accumulated_text) < MAX_SENTENCE_LENGTH:
                if prev_trans and translated_text.startswith(prev_trans):
                    # print(f"translation: {translation}")
                    new_text = translated_text[len(prev_trans):].strip()
                    if new_text:
                        accumulated_text[lang] += " " + new_text
                else:
                    accumulated_text = translation
            # print(f"accumulated_text: {accumulated_text}")
            self.translation_results[speech_id]["prev_translation"] = translation
            self.translation_results[speech_id]["accumulated_text"] = accumulated_text
            self.last_translation = accumulated_text
            
        return prev_translation, accumulated_text
    
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
        
        try:
            item = self.audio_queue.get(timeout=0.01)
            if isinstance(item, tuple):
                return item
            else:
                # 단일 chunk인 경우 모노 채널로 처리
                return [item], CHANNELS
        except queue.Empty:
            await asyncio.sleep(0.01)
            return None, None

    async def handle_audio_frames(self, frames, channels, speech_id):
        """오디오 프레임 처리"""
        audio_file_path = self.save_audio_to_wav(frames, temp=False, channels=channels)

        self.log_timing_on_end_of_speech() # 발화 종료 시간 로깅
        
        # 번역 시작 신호 전송
        self.emit_gui_signal_if_available("translation_started")
        # self.logger.info("✅ 번역 시작")

        # 번역 진행 표시 바 시작
        if hasattr(self, 'gui_signals') and hasattr(self.gui_signals, 'translation_started'):
            self.gui_signals.translation_started.emit()
        
        # --- OpenAI(server/local) 모드 (기존 로직) ---
        # 음성 추출
        # self.logger.info("8080 연결")
        result = self.transcribe_audio(audio_file_path, 8080)
        transcription = result.get("original_text", "").strip()
        self.aws_source_lang_code = result.get("ori_language", "").strip()
        if transcription is None:
            return
        # 발화 길이 확인
        if transcription and len(transcription.strip()) < 3:
            self.logger.info(f"발화가 너무 짧아 번역을 건너뜁니다: '{transcription}'")
            return None
        self._emit_stt_original(transcription) 
        # if translations is None:
        #     # 번역 처리
        #     lang_map = {'ko': 'Korean', 'zh': 'Chinese', 'en': 'English'}
        #     translations = {'Korean': '', 'Chinese': '', 'English': ''}

        #     # 언어 감지 및 번역
        #     translations = await self._perform_translation(transcription)
        #     if self.detected_language in lang_map:
        #         translations.setdefault(lang_map[self.detected_language], transcription)
            
        # # --- GUI에 원문·번역 내보내기 ---         
        # self._emit_stt_original(transcription) 
        # self.process_translation_result(translations, transcription, "", "", speech_id)
        # self.logger.info(f"✅ 원문: {transcription}")
        
        # for k, v in translations.items():
        #     self.logger.info(f"✅ {k} 번역: {v}")
            
        # AWS Translate 호출
        loop = asyncio.get_running_loop()
                    
        # 번역 대상 언어 목록
        target_codes = TARGET_LANGUAGES.get(self.aws_source_lang_code)
        
        # GUI에 표시할 언어명 매핑
        lang_names = {'ko':'Korean', 'en':'English', 'zh':'Chinese','ja':'Japanese',}
        
        tasks = [
            loop.run_in_executor(
                None,
                lambda src=code: self.aws_translate.translate_text(
                    Text=transcription,
                    SourceLanguageCode=self.aws_source_lang_code,
                    TargetLanguageCode=src
                )
            )
            for code in target_codes
        ]
        # await 모아서 결과 받기
        results = await asyncio.gather(*tasks)
        
        # 결과 딕셔너리로 정리
        translated = {
            lang_names[code]: result.get('TranslatedText','')
            for code, result in zip(target_codes, results)
        }
        
        translated[lang_names[self.aws_source_lang_code]] = transcription
        print(f"translated: {translated}")
        self.process_translation_result(
            translated,
            transcription,
            prev_translation="",
            accumulated_text="",
            speech_id=0
        )
        # self.logger.info(f"✅ 원문: {transcription}")
        # self.logger.info(f"✅ 번역: {translated}")
        
    # async def _perform_translation(self, transcription):
    #     """텍스트 번역 수행"""
    #     translation_start = datetime.now()
    #     translations = await self.translate_text_async(transcription)
    #     translation_end = datetime.now()

    #     self.logger.info(f"✅ 번역 종료! 번역 시간: {(translation_end - translation_start).total_seconds():.2f}초")
    #     return translations

    # async def translate_text_async(self, text, size=200):
    #     """텍스트를 청크로 나누어 비동기 번역"""
    #     if not text or not text.strip():
    #         return None

    #     chunks = [text[i:i+size] for i in range(0, len(text), size)]
    #     all_translations = await asyncio.gather(*(self.translate_chunk(chunk) for chunk in chunks))
    #     return self._merge_translations(all_translations)

    # def _merge_translations(self, all_translations):
    #     """번역 결과 병합"""
    #     final = {}
    #     for translation_set in all_translations:
    #         if translation_set:
    #             for lang, trans in translation_set.items():
    #                 final.setdefault(lang, []).append(trans)
    #     return {k: ' '.join(filter(None, v)) for k, v in final.items()}

    # async def translate_chunk(self, chunk):
    #     try:
    #         self.detected_language = detect(chunk)
    #         targets = TARGET_LANGUAGES.get(self.detected_language[:2], [])
    #         results = await asyncio.gather(*(self.call_translation_api(chunk, t) for t in targets))
    #         return dict(zip(targets, results))
    #     except Exception as e:
    #         self.logger.error(f"Language detection error: {str(e)}")
    #         return None

    # async def call_translation_api(self, chunk, target_lang):
    #     headers = {
    #         "Authorization": f"Bearer {self.api_key}",
    #         "Content-Type": "application/json"
    #     }
    #     prompt = f'Translate to {target_lang}: "{chunk}"'
    #     data = {
    #         "model": GPT_MODEL, 
    #         "messages": [
    #             {"role": "system", 
    #             "content": "You are a translator. Only provide the translation without any explanation."},
    #             {"role": "user", "content": f"Translate to {target_lang} only:\n{chunk}"},
    #             ]
    #         }

    #     async with aiohttp.ClientSession() as session:
    #         async with session.post(TRANSLATION_URL, headers=headers, json=data) as resp:
    #             if resp.status == 200:
    #                 json_data = await resp.json()
    #                 return json_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    #             self.logger.error(f"API error {resp.status}: {await resp.text()}")
    #             return None

    def _prepare_gui_message(self, translation, transcription):
        # print(f"translation: {translation}")
        """GUI 메시지 구성"""
        gui_message = {
            "korean": translation.get("Korean", ""),
            "english": translation.get("English", ""),
            "chinese": translation.get("Chinese", ""),
            "japanese": translation.get("Japanese", "")
        }
        return gui_message
    
    async def _aws_send(self, ws):
        """마이크 버퍼의 청크를 AWS에 전송"""
        loop = asyncio.get_running_loop()
        while not self.aws_stream_stop.is_set():
            try:
                # 큐에서 최대 1초 대기 후 청크를 가져옴
                chunk = await loop.run_in_executor(
                    None, self.audio_queue.get, True, 1.0
                )
                await ws.send(create_audio_event(chunk))
            except (queue.Empty, asyncio.TimeoutError):
                continue
            except websockets.exceptions.ConnectionClosedOK:
                # WS가 정상 종료된 경우, 더 이상 보내지 않음
                break
            except RuntimeError:
                # 이벤트루프가 종료된 경우 (shutdown) 빠져나감
                break

    
    async def _aws_streaming_loop_all(self):
        """하나의 WS 연결에서 send/receive 병행 + 부분/완료 자막 처리"""
        async with websockets.connect(self.aws_url, ping_timeout=None) as ws:
            send_task = asyncio.create_task(self._aws_send(ws))
            last_partial = ""
            try:
                while not self.aws_stream_stop.is_set():
                    try:
                        raw = await ws.recv()
                    except websockets.exceptions.ConnectionClosedOK:
                        self.logger.info("ℹ️ AWS 스트리밍 연결 정상 종료")
                        break

                    # event 파싱
                    header, payload = decode_event(raw)
                    if header.get(':message-type') != 'event':
                        continue
                    results = payload['Transcript']['Results']
                    if not results or not results[0].get('Alternatives'):
                        continue
                    alt = results[0]['Alternatives'][0]
                    text = alt['Transcript']
                    
                    is_partial = results[0].get('IsPartial', True)
                    
                    if is_partial:
                        # 실시간(부분) 자막
                        if text != last_partial:
                            last_partial = text
                            self._emit_gui_realtime_update(text)
        
                    else:
                        # 최종 자막
                        last_partial = ""
                        self._emit_stt_original(text)
                        loop = asyncio.get_running_loop()
                    
                        # 번역 대상 언어 목록
                        target_codes = TARGET_LANGUAGES.get(self.aws_source_lang_code)
                        
                        # GUI에 표시할 언어명 매핑
                        lang_names = {'ko':'Korean', 'en':'English', 'zh':'Chinese','ja':'Japanese',}
                      
                        tasks = [
                            loop.run_in_executor(
                                None,
                                lambda src=code: self.aws_translate.translate_text(
                                    Text=text,
                                    SourceLanguageCode=self.aws_source_lang_code,
                                    TargetLanguageCode=src
                                )
                            )
                            for code in target_codes
                        ]
                        # await 모아서 결과 받기
                        results = await asyncio.gather(*tasks)
                        
                        # 결과 딕셔너리로 정리
                        translated = {
                            lang_names[code]: result.get('TranslatedText','')
                            for code, result in zip(target_codes, results)
                        }
                        
                        translated[lang_names[self.aws_source_lang_code]] = text  # 원문은 영어로 고정
                        # print(f"translated: {translated}")
                        self.process_translation_result(
                            translated,
                            text,
                            prev_translation="",
                            accumulated_text="",
                            speech_id=0
                        )
                        
            finally:
                self.aws_stream_stop.set()
                send_task.cancel()
           
    
    # ---------- GUI 업데이트 ----------
    def log_timing_on_end_of_speech(self):
        if hasattr(self, 'transcribe_start_time') and self.transcribe_start_time:
            end_time = datetime.now()
            duration = (end_time - self.transcribe_start_time).total_seconds()
            self.logger.info(f"✅ 발화 종료! 발화 시간: {duration:.2f}초")

    def set_gui_signals(self, signals):
        """GUI 신호 객체 설정"""
        # self.logger.debug("GUI 신호 객체 설정됨")
        self.gui_signals = signals
    
    def emit_gui_signal_if_available(self, signal_name):
        if hasattr(self, 'gui_signals'):
            signal = getattr(self.gui_signals, signal_name, None)
            if signal:
                signal.emit()

    def _emit_stt_original(self, original_text):
        if hasattr(self, 'gui_signals') and hasattr(self.gui_signals, 'stt_original_update'):
            self.gui_signals.stt_original_update.emit(original_text or "")
        else:
            self.logger.warning("⚠️ stt_original_update 시그널이 존재하지 않음.")
    
    def _emit_gui_realtime_update(self, transcription):
        """GUI에 실시간(부분) 자막 전송 (원문만)"""
        if hasattr(self, 'gui_signals') and getattr(self.gui_signals, 'stt_original_update', None):
            self.gui_signals.stt_original_update.emit(transcription)

    def _generate_presigned_url(self):
        """스트리밍 시 사용할 presigned URL을 미리 생성"""
        self._ensure_aws_credentials()           # 토큰 보장
        self.aws_url = self.aws_transcribe.get_request_url(
            sample_rate=RATE,
            language_code=self.aws_language_code,
            media_encoding="pcm",
            number_of_channels=1,
            # enable_partial_results_stabilization=True
        )
        self.logger.info(f"Presigned URL ready")
    
    def set_translation_mode(self, mode: str):
        """
        mode: "server" 또는 "aws"
        """
        if mode not in ("server", "aws"):
            self.logger.warning(f"지원하지 않는 모드: {mode}")
            return
        self.translation_mode = mode
        self.logger.info(f"⚙️ translation_mode를 '{mode}'로 변경했습니다.")

    def set_translation_mode(self, mode: str):
        """
        mode: "server" 또는 "aws"
        """
        if mode not in ("server", "aws"):
            self.logger.warning(f"지원하지 않는 모드: {mode}")
            return

        # 1) 기존 AWS 스트리밍 중지
        self.aws_stream_stop.set()

        # 2) 서버 모드 쓰레드 종료 플래그가 있다면 세트 (옵션)
        #    예: self.server_stop_event.set()

        # 3) 큐 비우기
        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass

        # 4) 모드 변경
        self.translation_mode = mode
        self.logger.info(f"⚙️ translation_mode를 '{mode}'로 변경했습니다.")

        # 5) 새 모드에 맞게 쓰레드 기동
        if mode == "aws":
            self.aws_stream_stop.clear()
            self._generate_presigned_url()
            threading.Thread(
                target=lambda: asyncio.run(self._aws_streaming_loop_all()),
                daemon=True,
                name="aws_stream"
            ).start()
        else:  # "server"
            # 만약 서버 모드에도 중단 이벤트를 사용했다면 .clear() 해주세요.
            threading.Thread(
                target=lambda: asyncio.run(self.process_translation_queue()),
                daemon=True,
                name="translation_queue_thread"
            ).start()
    # ---------- 기타 ---------- 
    def start_threads(self):
        """오디오 캡처 및 번역 처리를 위한 스레드 시작"""
        # 오디오 캡처 스레드
        capture_thread = threading.Thread(
            target=self.audio_capture,
            args=(self.selected_device,),
            daemon=True,
            name="audio_capture_thread"
        )
        capture_thread.start()

        if self.translation_mode == "aws":
            threading.Thread(
                target=lambda: asyncio.run(self._aws_streaming_loop_all()),
                daemon=True,
                name="aws_stream"
            ).start()
            
        else:
            threading.Thread(
                target=lambda: asyncio.run(self.process_translation_queue()),
                daemon=True,
                name="translation_queue_thread"
            ).start()
