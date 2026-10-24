#!/bin/bash
# K-Decision v0.3 — Qwen3.5-4B 베이스, 도메인+일반 믹스 (3090, STT 공존 안전설정)
cd /home/kitri/k-decision
pkill -f "venvs/kd/bin/python src/train.py" 2>/dev/null
sleep 2
rm -f train_v03.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup nice -n 10 /home/kitri/venvs/kd/bin/python src/train.py \
  --base Qwen/Qwen3.5-4B --data data/mix_v02_train.jsonl \
  --out models/kd-4b-v0.3-3090 --epochs 3 --bs 2 --accum 8 --lr 7e-5 \
  > train_v03.log 2>&1 &
echo "V03_PID $!"
