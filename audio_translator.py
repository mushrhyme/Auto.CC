import pyaudio
import wave
import requests
import threading
import tempfile
import os
import time
import numpy as np
import logging
import json
import queue
from datetime import datetime
from pathlib import Path
import backoff 
import asyncio
from PyQt5.QtCore import QTimer


# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1  
RATE = 16000  # 초당 수집하는 오디오 프레임 수: 높을수록 품질이 좋아지나 더 많은 데이터를 처리해야 함(16000 = Whisper 권장 샘플링 레이트)
CHUNK = 1024 # 한 번에 처리하는 오디오 프레임 수:  작을수록 빠르게 처리되나 품질이 떨어질 수 있음
SILENCE_THRESHOLD = 200 # 평균 진폭이 이 값 미만이면 침묵으로 판단 (기존 400에서 200으로 낮춤)
SILENCE_DURATION = 1.5 # 침묵 지속 시간: 이 시간 동안 침묵이면 발화가 종료된 것으로 간주
REALTIME_UPDATE_INTERVAL = 1.0 # 실시간 번역 업데이트 간격: 음성이 진행 중일 때 현재까지 수집된 버퍼를 주기적으로 처리하여 중간 번역 결과를 보여주는 시간 간격
MAX_SENTENCE_LENGTH = 50 # 최대 문장 길이 (자유롭게 조정 가능)
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

    def find_virtual_audio_device(self):
        """검색하여 Windows용 가상 오디오 장치 인덱스 찾기"""
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        virtual_device_index = None
    
        self.logger.info("\nSearching for virtual audio devices...")
    
        # 모든 장치 검색
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            device_name = device_info.get('name', '').lower()
            # VB-Cable 또는 Virtual Audio Cable 장치 찾기
            if 'vb-cable' in device_name or 'virtual audio cable' in device_name:
                virtual_device_index = i
                self.logger.info(f"✅ Virtual audio device found: {device_info.get('name')} (device number: {i})")
    
        p.terminate()
        return virtual_device_index    
        
    def find_blackhole_device(self):
        """검색하여 가상 오디오 장치 인덱스 찾기"""
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        blackhole_index = None
        
        self.logger.info("\nSearching for virtual audio devices...")
        
        # 모든 장치 검색
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            device_name = device_info.get('name', '').lower()
            # Blackhole 또는 Soundflower 장치 찾기
            if 'blackhole' in device_name or 'soundflower' in device_name:
                blackhole_index = i
                self.logger.info(f"✅ Virtual audio device found: {device_info.get('name')} (device number: {i})")
        
        p.terminate()
        return blackhole_index

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

    @staticmethod
    def get_audio_level(audio_data):
        """오디오 데이터의 볼륨 레벨 계산"""
        if len(audio_data) == 0:
            return 0
        
        # 더 효율적인 처리를 위해 numpy 배열로 변환
        normalized = np.abs(np.frombuffer(audio_data, dtype=np.int16))
        
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
                self.logger.info(f"\n✅  오디오 감지! Level: {audio_level:.1f}")
            self.voice_detected = True
            return True
        else:
            self.silence_count += 1
            
            # 침묵 후 음성 감지 상태 재설정
            if self.silence_count > self.silence_chunks and self.voice_detected:
                self.logger.info(f"\n⏳ 침묵 감지! Level: {audio_level:.1f}")
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

    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=3)
    def transcribe_audio(self, audio_file_path):
        """오디오 파일을 텍스트로 변환 (오류 재시도 포함)"""
        try:
            # API 요청 헤더
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 파일 전송
            with open(audio_file_path, 'rb') as audio_file:
                files = {
                    'file': (os.path.basename(audio_file_path), audio_file, 'audio/wav'),
                    'model': (None, 'whisper-1'),
                    'language': (None, 'en'),
                    'response_format': (None, 'json')
                }
                
                response = requests.post(TRANSCRIPTION_URL, headers=headers, files=files)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('text', '')
            else:
                self.logger.error(f"Transcription API error: {response.status_code}, {response.text}", exc_info=True)
                return None
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}", exc_info=True)
            raise  # Backoff 재시도를 위해 다시 발생
        finally:
            # 임시 파일 삭제
            try:
                if tempfile.gettempdir() in audio_file_path and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
            except Exception as e:
                self.logger.error(f"Failed to delete temporary file: {e}", exc_info=True)

    @backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, Exception), max_tries=3)
    def translate_text(self, text):
        """영어 텍스트를 한국어로 번역 (오류 재시도 포함)"""
        if not text or not text.strip():
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": GPT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": """
You are a professional English to Korean translator.
Translate the following English text into natural and fluent Korean while maintaining the original meaning, tone, and nuance. 
"""
                    },
                    {
                        "role": "user",
                        "content": f"Translate this English text to Korean: \"{text}\""
                    }
                ]
            }
            
            response = requests.post(TRANSLATION_URL, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('choices') and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
            else:
                self.logger.error(f"Translation API error: {response.status_code}, {response.text}", exc_info=True)
            
            return None
        except Exception as e:
            self.logger.error(f"Error during translation: {e}", exc_info=True)
            raise  # Backoff 재시도를 위해 다시 발생

    def is_blackhole_device(self, device_index):
        """현재 선택된 장치가 Blackhole인지 확인"""
        if device_index is None:
            return False
        
        p = pyaudio.PyAudio()
        try:
            device_info = p.get_device_info_by_index(device_index)
            device_name = device_info.get('name', '').lower()
            is_blackhole = 'blackhole' in device_name or 'soundflower' in device_name
            
            if is_blackhole:
                self.logger.warning("\n⚠️ 현재 Blackhole/Soundflower 가상 오디오 장치를 사용 중입니다.")
                self.logger.info("음성이 감지되지 않는다면:")
                self.logger.info("1. 시스템 설정에서 오디오 출력이 Blackhole로 설정되어 있는지 확인하세요.")
                self.logger.info("2. 재생 중인 오디오가 있는지 확인하세요.")
                self.logger.info("3. 오디오 볼륨이 충분히 큰지 확인하세요.")
            
            return is_blackhole
        except Exception as e:
            self.logger.error(f"장치 확인 중 오류 발생: {e}", exc_info=True)
            return False
        finally:
            p.terminate()

    def check_audio_input(self, device_index):
        """오디오 입력 상태 확인"""
        p = pyaudio.PyAudio()
        try:
            device_info = p.get_device_info_by_index(device_index if device_index is not None else 
                                                   p.get_default_input_device_info()['index'])
            
            # 장치 정보 로깅
            self.logger.info("\n🎤 현재 오디오 입력 설정:")
            self.logger.info(f"장치 이름: {device_info.get('name')}")
            self.logger.info(f"입력 채널 수: {device_info.get('maxInputChannels')}")
            self.logger.info(f"기본 샘플링 레이트: {int(device_info.get('defaultSampleRate'))}Hz")
            
            # Blackhole 체크
            if self.is_blackhole_device(device_index):
                return
            
            # 일반 마이크 사용 시 안내
            self.logger.info("\n일반 마이크를 사용 중입니다.")
            self.logger.info("음성이 감지되지 않는다면:")
            self.logger.info("1. 마이크가 정상적으로 연결되어 있는지 확인하세요.")
            self.logger.info("2. 시스템 설정에서 마이크 권한이 허용되어 있는지 확인하세요.")
            self.logger.info("3. 시스템 설정에서 입력 장치가 올바르게 선택되어 있는지 확인하세요.")
            
        except Exception as e:
            self.logger.error(f"오디오 입력 확인 중 오류 발생: {e}", exc_info=True)
        finally:
            p.terminate()

    def audio_capture(self, device_index=None):
        """오디오 캡처 및 큐에 데이터 추가"""
        # 오디오 입력 상태 확인 추가
        self.check_audio_input(device_index)
        
        p = pyaudio.PyAudio()
        
        # 현재 오디오 레벨 주기적으로 출력
        def log_audio_level(data):
            audio_level = self.get_audio_level(data)
            # self.logger.debug(f"Current audio level: {audio_level:.1f} (threshold: {self.silence_threshold})")
            if hasattr(self, 'gui_signals'):
                self.gui_signals.audio_level_update.emit(audio_level)
        
        # 장치 정보 가져오기 및 채널 확인
        selected_channels = CHANNELS
        if device_index is not None:
            try:
                device_info = p.get_device_info_by_index(device_index)
                max_input_channels = int(device_info.get('maxInputChannels', 1))
                if max_input_channels < selected_channels:
                    self.logger.warning(f"경고: 선택한 장치는 {max_input_channels}개 채널만 지원합니다. 자동으로 조정합니다.")
                    selected_channels = max_input_channels
                self.logger.info(f"장치 정보: 최대 입력 채널 = {max_input_channels}, 선택된 채널 = {selected_channels}")
            except Exception as e:
                self.logger.error(f"장치 정보를 가져오는데 실패했습니다: {e}", exc_info=True)
        
        # 오디오 스트림 설정
        stream = None
        try:
            stream = p.open(
                format=FORMAT,
                channels=selected_channels,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK
            )
            self.logger.info(f"오디오 스트림이 성공적으로 열렸습니다. 채널: {selected_channels}")
        except Exception as e:
            self.logger.error(f"선택한 장치로 스트림을 열 수 없습니다: {e}", exc_info=True)
            self.logger.info("기본 입력 장치를 시도합니다...")
            
            try:
                # 기본 장치의 채널 확인
                default_info = p.get_default_input_device_info()
                default_channels = min(CHANNELS, int(default_info.get('maxInputChannels', 1)))
                
                stream = p.open(
                    format=FORMAT,
                    channels=default_channels,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
                self.logger.info(f"기본 장치를 사용합니다. 채널: {default_channels}")
                selected_channels = default_channels
            except Exception as e2:
                self.logger.error(f"기본 장치도 실패했습니다: {e2}", exc_info=True)
                self.logger.info("모노 채널로 마지막 시도를 합니다...")
                
                try:
                    stream = p.open(
                        format=FORMAT,
                        channels=1,  
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK
                    )
                    self.logger.info("모노 채널(1)로 오디오 스트림을 열었습니다.")
                    selected_channels = 1
                except Exception as e3:
                    self.logger.error(f"모든 오디오 스트림 열기 시도가 실패했습니다: {e3}", exc_info=True)
                    return
                
            device_index = None
        
        if not stream:
            self.logger.error("오디오 스트림을 열지 못했습니다. 오디오 캡처를 종료합니다.", exc_info=True)
            return
        
        if device_index is not None:
            device_info = p.get_device_info_by_index(device_index)
            self.logger.info(f"\n'{device_info.get('name')}' 장치를 사용합니다. 시스템 오디오 캡처를 시작합니다...")
        else:
            self.logger.info("\n기본 마이크를 사용합니다.")
        
        self.logger.info(f"현재 침묵 임계값: {self.silence_threshold} (이 값 이상이면 오디오가 감지됨)")
        
        silence_counter = 0
        speech_time_counter = 0
        speech_detected_during_session = False
        volume_monitor_counter = 0
        
        try:
            while self.is_running:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    log_audio_level(data)  # 오디오 레벨 로깅
                except Exception as e:
                    self.logger.error(f"오디오 캡처 중 에러가 발생하였습니다: {e}", exc_info=True)
                    continue
                
                # 현재 청크의 평균 진폭 계산
                audio_level = self.get_audio_level(data)
                
                # 볼륨 레벨 모니터링 (5초마다)
                volume_monitor_counter += 1
                if volume_monitor_counter >= 80:  # 80 * 0.0625 = 5초
                    volume_monitor_counter = 0
                    # 모든 오디오 레벨을 로깅하도록 수정
                    self.logger.info(f"현재 오디오 레벨: {audio_level:.1f} (임계값: {self.silence_threshold})")
                    if audio_level > self.silence_threshold:
                        self.logger.info("✅ 음성이 감지되었습니다!")
                    else:
                        self.logger.info("⏳ 음성이 감지되지 않았습니다.")
                
                # 버퍼에 데이터 추가
                with self.buffer_lock:
                    self.audio_frames.append(data)
                
                # 음성 감지 확인
                if self.should_transcribe(audio_level):
                    silence_counter = 0
                    speech_time_counter += 1
                    speech_detected_during_session = True
                    
                    if speech_time_counter % 16 == 0:  # Show dots about every second
                        print(".", end="", flush=True)
                else:
                    silence_counter += 1
                
                # 침묵이 지속되면 세션 처리 종료
                if silence_counter >= self.silence_chunks and len(self.audio_frames) > 0:
                    with self.buffer_lock:
                        frames_copy = list(self.audio_frames)
                        self.audio_frames.clear()
                    
                    # 충분한 데이터가 있고 음성이 감지된 경우에만 처리 (약 1.5초)
                    min_frames = int((RATE * 1.5) / CHUNK)
                    if len(frames_copy) > min_frames and speech_detected_during_session:
                        # 데이터를 큐에 추가
                        self.audio_queue.put((frames_copy, selected_channels))
                    
                    silence_counter = 0
                    speech_time_counter = 0
                    speech_detected_during_session = False
        
        except Exception as e:
            self.logger.error(f"오디오 캡처 중 에러가 발생하였습니다: {e}", exc_info=True)
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

    def set_gui_signals(self, signals):
        """GUI 신호 객체 설정"""
        # self.logger.debug("GUI 신호 객체 설정됨")
        self.gui_signals = signals


    

    def process_translation_result(self, translation, transcription, prev_translation, accumulated_text):
        """번역 결과 처리 및 GUI 업데이트"""
        
        def log_translation(self, timestamp, translation, transcription, new_text=None):
            """번역 및 원문에 대한 로그 출력"""
            if new_text:
                self.logger.info(f"[{timestamp}] 추가: {new_text}")
                self.logger.debug(f"번역 추가: {new_text}")
            else:
                self.logger.info(f"[{timestamp}] 번역:")
                self.logger.info(f"{translation}")
                self.logger.info(f"원문: {transcription}")
                self.logger.debug(f"새 번역: {translation}")

        # 새로운 번역 시작 시 초기화
        timestamp = datetime.now().strftime('%H:%M:%S')
        # self.logger.info(f"\n[{timestamp}] 새로운 번역 시작")
        # self.logger.info(f"초기화 전 prev_translation: {prev_translation}")
        # self.logger.info(f"초기화 전 accumulated_text: {accumulated_text}")
        
        # prev_translation과 accumulated_text 강제로 빈 문자열로 초기화
        prev_translation = ""
        accumulated_text = ""
        
        # self.logger.info(f"초기화 후 prev_translation: {prev_translation}")
        # self.logger.info(f"초기화 후 accumulated_text: {accumulated_text}")
        
        if not translation or not translation.strip():
            return prev_translation, accumulated_text  # 변경 없음

        new_accumulated = accumulated_text

        # 계속된 발화 확인 (누적 길이가 최대 길이보다 작을 때만)
        if len(accumulated_text) < MAX_SENTENCE_LENGTH:
            if prev_translation and translation.startswith(prev_translation):
                new_text = translation[len(prev_translation):].strip()
                if new_text:
                    new_accumulated = accumulated_text + " " + new_text
                    # log_translation(self, timestamp, translation, transcription, new_text)
            else:
                if len(accumulated_text) >= MAX_SENTENCE_LENGTH:
                    self.logger.info(f"\n[{timestamp}] 최대 길이 도달, 새 문장 시작:")
                # log_translation(self, timestamp, translation, transcription)

                new_accumulated = translation

        # 결과 저장
        self.last_translation = new_accumulated

        # GUI 신호 발송 (GUI 모드인 경우)
        if hasattr(self, 'gui_signals'):
            self.logger.debug(f"GUI 신호 발송: {timestamp}, 번역")
            gui_message = f"(원문) {transcription}\n(번역) {new_accumulated}"
            self.gui_signals.translation_update.emit(timestamp, gui_message, translation)
            
            # # 비동기 방식이 아닌 QTimer를 사용하여 GUI 업데이트를 안전하게 처리
            # def emit_signal():
            #     self.gui_signals.translation_update.emit(timestamp, gui_message, translation)

            # # QTimer를 사용해 이벤트 루프에 안전하게 GUI 신호 발송
            # QTimer.singleShot(0, emit_signal)


        return translation, new_accumulated


    
    def process_translation_queue(self):
        """오디오 큐 처리 및 한국어로 번역"""
        prev_translation = ""
        accumulated_text = ""
        
        while self.is_running:
            try:
                # 큐에서 데이터 가져오기
                try:
                    frames_copy, selected_channels = self.audio_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                
                # 오디오 파일로 저장
                audio_file_path = self.save_audio_to_wav(frames_copy, channels=selected_channels)
                transcription = self.transcribe_audio(audio_file_path)
                
                if transcription and transcription.strip():
                    # 터미널에 영어 원문 출력
                    self.logger.info(f"📝 영어 원문: {transcription}")
                    translation = self.translate_text(transcription)
                    
                    if translation and translation.strip():
                        prev_translation, accumulated_text = self.process_translation_result(
                            translation, transcription, prev_translation, accumulated_text
                        )
                            
                self.audio_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in translation processing: {e}", exc_info=True)

    def process_realtime(self):
        """실시간 번역을 위한 현재 오디오 버퍼 처리"""
        self.logger.info("실시간 번역 스레드 시작")
        prev_translation = ""
        accumulated_text = ""
        last_update_time = time.time()
        
        while self.is_running:
            # 실시간 모드가 아니면 스레드 종료
            if self.translation_mode != "realtime":
                self.logger.info("실시간 번역 모드 비활성화. 스레드 대기 중...")
                time.sleep(1)
                continue

            current_time = time.time()
            # 업데이트 간격에 따라 처리 (기본값: 1초)
            if current_time - last_update_time < self.update_interval:
                time.sleep(0.1)  # CPU 사용량 줄이기 위한 짧은 대기
                continue
            
            last_update_time = current_time
            
            # 음성이 감지된 경우에만 처리
            if self.voice_detected:
                self.logger.info("음성 감지됨, 실시간 처리 중...")
                with self.buffer_lock:
                    frames_copy = list(self.audio_frames) if self.audio_frames else []
                
                try:
                    # 충분한 데이터가 있는 경우에만 처리 (약 1초마다)
                    min_frames = int((RATE * 0.5) / CHUNK)
                    if len(frames_copy) > min_frames:
                        # self.logger.debug(f"오디오 프레임 {len(frames_copy)}개 처리 중...")
                        audio_file_path = self.save_audio_to_wav(frames_copy, channels=1)
                        transcription = self.transcribe_audio(audio_file_path)
                        
                        # 번역
                        if transcription and transcription.strip():
                            translation = self.translate_text(transcription)
                            
                            # 번역 결과가 있으면 출력
                            if translation and translation.strip():
                                prev_translation, accumulated_text = self.process_translation_result(
                                    translation, transcription, prev_translation, accumulated_text
                                )
                                # GUI 신호 발송 및 시간 측정
                            else:
                                self.logger.debug("번역 결과가 없습니다.")
                        else:
                            self.logger.debug("음성 인식 결과가 없습니다.")
                    else:
                        self.logger.debug(f"음성 프레임이 충분하지 않음: {len(frames_copy)}/{min_frames}")
                except Exception as e:
                    self.logger.error(f"실시간 번역 오류: {e}", exc_info=True)
            else:
                # 음성이 감지되지 않을 때는 간단한 로그만 출력
                if time.time() % 5 < 0.1:  # 5초마다 한 번씩만 출력
                    self.logger.info("음성 감지 대기 중...")
    

    def start(self):
        """오디오 번역 시스템 시작"""
        self.logger.info("오디오 캡처 및 번역 시스템 시작 중...")
        self.logger.info(f"로그 파일 위치: {self.log_filename}")
        
        # API key 확인
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.", exc_info=True)
            return
        
        # OS에 따라 가상 오디오 장치 자동 감지
        if os.name == 'nt':  # Windows
            virtual_device_index = self.find_virtual_audio_device()
            device_type = "VB-Cable 또는 Virtual Audio Cable"
        else:  # macOS
            virtual_device_index = self.find_blackhole_device()
            device_type = "Blackhole 또는 Soundflower"
        
        if virtual_device_index is None:
            self.logger.warning(f"\n⚠️ 가상 오디오 장치({device_type})를 찾을 수 없습니다.")
            self.logger.info("시스템 오디오를 캡처하려면 가상 오디오 장치가 필요합니다.")
            self.logger.info("\n가상 장치 없이 계속하면 일반 마이크를 사용합니다.")
        
        # 모든 사용 가능한 오디오 장치 나열
        self.list_audio_devices()
        
        # 디바이스 선택
        self.selected_device = None
        
        self.logger.info("\n참고: 채널 문제로 오류가 발생하면 채널 수가 자동으로 조정됩니다.")
        
        if virtual_device_index is not None:
            use_virtual_device = input(f"\n{device_type} 장치를 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower() or 'y'
            if use_virtual_device == 'y':
                self.selected_device = virtual_device_index
            else:
                try:
                    device_index = int(input("\n사용할 오디오 입력 장치 번호 입력: ").strip())
                    self.selected_device = device_index
                except ValueError:
                    self.logger.warning("잘못된 입력입니다. 기본 마이크를 사용합니다.")
        else:
            try:
                device_index = int(input("\n오디오 입력 장치 번호 입력(기본 마이크는 Enter 키): ").strip() or "-1")
                if device_index >= 0:
                    self.selected_device = device_index
            except ValueError:
                self.logger.warning("잘못된 입력입니다. 기본 마이크를 사용합니다.")
        # 번역 모드 선택
        mode_selection = input(f"\n번역 모드를 선택하세요 (1: 실시간, 2: 발화 완료 후, 기본: {self.translation_mode}): ").strip()
        if mode_selection == "1":
            self.translation_mode = "realtime"
            self.logger.info("실시간 번역 모드가 선택되었습니다.")
        elif mode_selection == "2":
            self.translation_mode = "complete"
            self.logger.info("발화 완료 후 번역 모드가 선택되었습니다.")

        # 업데이트된 설정 저장
        self.save_config()
        self.logger.info("\n종료하려면 Ctrl+C를 누르세요.")
        
        # 스레드 시작
        capture_thread = threading.Thread(target=self.audio_capture, args=(self.selected_device,), daemon=True)
        queue_thread = threading.Thread(target=self.process_translation_queue, daemon=True)
        
        capture_thread.start()
        queue_thread.start()
        
        # 실시간 모드에서만 실시간 번역 스레드 시작
        if self.translation_mode == "realtime":
            realtime_thread = threading.Thread(target=self.process_realtime, daemon=True)
            realtime_thread.start()
            self.logger.info("실시간 번역 업데이트가 활성화되었습니다.")
        else:
            self.logger.info("발화 완료 후에만 번역 결과가 표시됩니다.")
        
        try:
            # 메인 루프
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("\n프로그램을 종료합니다.")
        finally:
            # 정리
            self.is_running = False
            time.sleep(1)
            
            # 임시 파일 삭제
            if os.path.exists("realtime_audio.wav"):
                try:
                    os.remove("realtime_audio.wav")
                except:
                    pass

    def start_threads(self):
        """GUI 모드에서 사용할 스레드만 시작 (사용자 입력 없음)"""
        # Threads 시작
        # 오디오 캡처 및 번역 처리를 위한 스레드 시작
        capture_thread = threading.Thread(
            target=self.audio_capture, 
            args=(self.selected_device,), 
            daemon=True,
            name="audio_capture_thread"
        )
        
        queue_thread = threading.Thread(
            target=self.process_translation_queue, 
            daemon=True,
            name="translation_queue_thread"
        )
        
        self.logger.info("오디오 캡처 및 번역 처리 스레드 시작...")
        capture_thread.start()
        queue_thread.start()
        
        # 실시간 모드에서만 실시간 번역 스레드 시작
        if self.translation_mode == "realtime":
            realtime_thread = realtime_thread = threading.Thread(
            target=self.process_realtime, 
            daemon=True,
            name="realtime_translation_thread"
        )
            realtime_thread.start()
            self.logger.info("실시간 번역 업데이트가 활성화되었습니다.")
        else:
            self.logger.info("발화 완료 후에만 번역 결과가 표시됩니다.")

def main():
    translator = AudioTranslator()
    translator.start()

if __name__ == "__main__":
    main()