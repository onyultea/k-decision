#!/bin/bash
# serve_systemone 재기동 스크립트 — 세션 끊김에 강한 구조 (스크립트 자체를 setsid 로 실행)
cd /home/kitri/k-decision
pkill -f "src/serve_systemone.py" 2>/dev/null
sleep 3
export PYTHONUNBUFFERED=1
nohup /home/kitri/venvs/kd/bin/python src/serve_systemone.py \
  --merged models/kd-4b-v0.3-merged --port 8111 >> serve.log 2>&1 &
echo "SERVE_PID $!"
