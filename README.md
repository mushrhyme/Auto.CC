# Auto.CC

Auto.CC는 실시간 음성 번역 및 자막 생성 애플리케이션입니다. PySide6를 사용한 GUI 인터페이스와 실시간 오디오 처리를 통해 사용자 친화적인 번역 경험을 제공합니다.

## 주요 기능

- 실시간 음성 입력 및 번역
- GUI 기반 사용자 인터페이스
- 다중 스레드를 활용한 효율적인 오디오 처리
- 실시간 자막 생성

## 설치 방법

1. 저장소를 클론합니다:
```bash
git clone [repository-url]
cd Auto.CC
```

2. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
python run_translator.py
```

## 사용된 라이브러리

- PySide6: GUI 인터페이스 구현
- PyAudio: 오디오 입력/출력 처리
- numpy: 수치 연산 및 오디오 데이터 처리
- requests: API 통신
- backoff: API 요청 재시도 처리

## 프로젝트 구조

- `run_translator.py`: 메인 실행 파일
- `audio_translator.py`: 오디오 처리 및 번역 로직
- `gui_translator.py`: GUI 인터페이스 구현
- `requirements.txt`: 프로젝트 의존성 목록

## 라이선스

[라이선스 정보를 여기에 추가하세요]