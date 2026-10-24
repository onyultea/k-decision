# 원격 학습 워크스테이션 가이드 (일반화)

내부 인프라 정보는 이 문서에서 제거되었습니다. 실제 접속 정보는 팀 내부 공유 채널에서 관리합니다.

## 개요

- 로컬(개발 PC): 데이터 생성·스모크 학습·평가·양자화 담당
- 원격(24GB GPU 워크스테이션): 본 학습 전용 — 기존 서비스와 메모리 공존 설정 유지

## SSH 설정 (일반 형태)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
# 원격 ~/.ssh/authorized_keys 에 공개키 추가 후:
ssh <user>@<host> -p <port>
```

## 원격 환경 (일반 형태)

```bash
python3 -m venv ~/venvs/kd
pip install torch transformers peft accelerate
# Qwen3.5 계열 고속 커널(Linux): pip install flash-linear-attention causal-conv1d
```

## 학습 실행 (일반 형태)

```bash
# 프로젝트 동기화 후
cd ~/k-decision/src
python train.py --base Qwen/Qwen3.5-4B --data ../data/mix_train.jsonl --epochs 2
```

## 주의

- 기존 서비스 프로세스와 GPU 메모리 공존: 배치 크기 제한 + nice + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- 세션 끊김 대비: setsid nohup 로 런처 스크립트 실행
