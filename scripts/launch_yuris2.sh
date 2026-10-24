#!/bin/bash
# YURIS-2 학습 런처 — 세션 끊김 무관 실행
cd /home/kitri/k-decision
pkill -f yuris2_train 2>/dev/null
sleep 2
setsid nohup /home/kitri/venvs/kd/bin/python -X utf8 src/yuris2_train.py \
  --backbone BAAI/bge-m3 --epochs 3 --lr 2e-5 --tag yuris2 \
  > /home/kitri/k-decision/train_yuris2.log 2>&1 < /dev/null &
echo "YURIS2_PID $!"
