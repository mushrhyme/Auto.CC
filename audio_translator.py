import os
import re
import io
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

import pyaudio
import requests
import numpy as np
import soundfile as sf
import aiohttp
from langdetect import detect
from PyQt5.QtCore import QTimer
import backoff
import socket
import boto3
import websockets
from presigned_url import AWSTranscribePresignedURL
import subprocess
import sys
from eventstream import create_audio_event, decode_event
import string


# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1  
RATE = 8000  # 초당 수집하는 오디오 프레임 수: 높을수록 품질이 좋아지나 더 많은 데이터를 처리해야 함(16000 = Whisper 권장 샘플링 레이트)
CHUNK = 1024 # 한 번에 처리하는 오디오 프레임 수:  작을수록 빠르게 처리되나 품질이 떨어질 수 있음
SILENCE_THRESHOLD = 600 # 평균 진폭이 이 값 미만이면 침묵으로 판단
SILENCE_DURATION = 2 # 침묵 지속 시간: 이 시간 동안 침묵이면 발화가 종료된 것으로 간주
# REALTIME_UPDATE_INTERVAL = 1.0 # 실시간 번역 업데이트 간격: 음성이 진행 중일 때 현재까지 수집된 버퍼를 주기적으로 처리하여 중간 번역 결과를 보여주는 시간 간격
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
# 172.17.17.82:8080

class AudioTranslator:
    def __init__(self):
        self.setup_logging()
        self.load_config()
        self.audio_folder = Path("audio")
        self.translation_mode = "aws" 
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
        
        # API 키 초기화
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
            raise ValueError("OPENAI_API_KEY 환경 변수를 설정해주세요.")
        
        # (A) AWS Transcribe 언어 코드
        self.aws_language_code = "en-US" # en-US  ko-KR

        # (B) AWS Translate API용 소스/타겟 코드
        #    ex) "en", "ko" 등
        self.aws_source_lang_code = "en"
        self.aws_target_lang_code = "ko"

        # (C) GUI에 뿌릴 때 언어명 키
        #    ex) self.aws_target_lang_code == "ko" 이면 "Korean"
        self.translate_target_lang_name = "Korean"
        
        self.aws_stream_stop = threading.Event()
        self.use_silence_vad = True
        self._generate_presigned_url()
        
    # ---------- 초기화 및 설정 ----------
    def _ensure_aws_credentials(self):
        """AWS 세션 토큰을 캐싱해서 관리"""
        now = datetime.now(timezone.utc)
        if self.aws_creds is None or now + timedelta(minutes=5) >= self.aws_creds_expires:
            # 새로 받아오기
            # output = subprocess.run(
            #     ["aws", "sts", "get-session-token", "--duration-seconds", "3600", "--output", "json"],
            #     capture_output=True, text=True
            # )
            # info = json.loads(output.stdout)["Credentials"]
            # 외부 aws-cli 프로세스를 띄우는 대신, 내부 함수 호출로 변경
            sts = boto3.client("sts", region_name="ap-northeast-2")
            resp = sts.get_session_token(DurationSeconds=3600)
            info = resp["Credentials"]

            self.aws_creds = {
                "aws_access_key_id": info["AccessKeyId"],
                "aws_secret_access_key": info["SecretAccessKey"],
                "aws_session_token": info["SessionToken"],
            }
            # ISO8601 문자열 → datetime
            self.aws_creds_expires = info["Expiration"]  # 이미 ISO8601 형식 문자열
       
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

    # ---------- 오디오 장치 관리 ----------
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
        top_samples = sorted_samples[int(len(sorted_samples) * 0.8):]
    
        # 상위 샘플이 없으면 전체 사용
        if len(top_samples) > 0:
            return np.mean(top_samples)
        return np.mean(normalized)

    def audio_capture(self, device_index=None):
        """오디오 캡처 및 큐에 데이터 추가"""
        if self.translation_mode=="aws":
            p = pyaudio.PyAudio()
            stream = self._open_audio_stream(p, device_index, CHANNELS)
            try:
                while self.is_running:
                    chunk = stream.read(CHUNK, exception_on_overflow=False)
                    # AWS 스트리밍 루프가 꺼내가도록 큐에 넣는다
                    self.audio_queue.put(chunk)
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
        else:
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
                if self.translation_mode == "aws":
                    try:
                        while self.is_running and self.translation_mode == "aws":
                            data = stream.read(CHUNK, exception_on_overflow=False)
                            with self.buffer_lock:
                                self.audio_frames.append(data)
                    finally:
                        stream.stop_stream()
                        stream.close()
                        p.terminate()
                else:
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
    
    async def _aws_streaming_transcribe(self, wav_path):
        ws = await websockets.connect(self.aws_url, ping_timeout=None)
        try:
            full_transcript = ""
            last_len = 0

            async def send_audio():
                wf = wave.open(wav_path, 'rb')
                try:
                    chunk_size = int(RATE / 10) 
                    while True:
                        data = wf.readframes(chunk_size)
                        if not data:
                            break
                        await ws.send(create_audio_event(data))
                        await asyncio.sleep(0.1)
                finally:
                    wf.close()

            async def receive_transcript():
                nonlocal full_transcript, last_len
                try:
                    while True:
                        raw = await ws.recv()
                        header, payload = decode_event(raw)
                        if header.get(':message-type') == 'event':
                            results = payload['Transcript']['Results']
                            if results:
                                alt = results[0]['Alternatives'][0]
                                text = alt['Transcript']
                                if not results[0].get('IsPartial', False):
                                    if len(text) > last_len:
                                        full_transcript += text[last_len:]
                                        last_len = len(text)
                except websockets.exceptions.ConnectionClosedOK:
                    pass
                return full_transcript

            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(receive_transcript())

            # 1) 오디오 전송 끝날 때까지 대기
            await send_task
            # 2) 받은 결과가 다 모일 때까지 대기
            transcription = await recv_task
            print("AWS STT 성공:", transcription)
            return transcription

        finally:
            # send/receive가 끝났든 에러났든 반드시 연결 해제
            await ws.close()
    
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
                        target_codes = ['ko','zh','ja']
                        # GUI에 표시할 언어명 매핑
                        lang_names = {'ko':'Korean','zh':'Chinese','ja':'Japanese'}
                      
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
                        
                        translated['English'] = text  # 원문은 영어로 고정
                        
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

        # 비동기 번역 큐 처리 스레드
        if self.translation_mode == "aws":
            threading.Thread(
                target=lambda: asyncio.run(self._aws_streaming_loop_all()),
                daemon=True,
                name="aws_stream"
            ).start()
            
        else:
            # AWS 모드 아니면 기존 큐-배치 스레드
            threading.Thread(
                target=lambda: asyncio.run(self.process_translation_queue()),
                daemon=True,
                name="translation_queue_thread"
            ).start()
        
        # self.start_realtime_transcription_loop()