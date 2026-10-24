#!/bin/bash
# K-Decision v0.2 학습 (도메인+일반 믹스) — 서버 격리 venv, STT 공존 안전 설정
cd /home/kitri/k-decision
pkill -f "venvs/kd/bin/python src/train.py" 2>/dev/null
sleep 2
rm -f train_v02.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup nice -n 10 /home/kitri/venvs/kd/bin/python src/train.py \
  --base Qwen/Qwen3.5-0.8B --data data/mix_v02_train.jsonl \
  --out models/kd-0.8b-v0.2-3090 --epochs 3 --bs 4 --accum 4 --lr 1e-4 \
  > train_v02.log 2>&1 &
echo "V02_PID $!"
