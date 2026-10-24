#!/bin/bash
# 4B 최종 라운드: 셀프 앙상블 즉시 + v13(r32·3에포크) 학습
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KD5=/home/kitri/venvs/kd/bin/python

echo "[1/2] 4B 셀프 앙상블 (v0.6+v0.8+v0.9 uniform)"
$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-4B \
  --adapters models/kd-v06-3090/adapter,models/kd-v08-3090/adapter,models/kd-v09-3090/adapter \
  --meta models/kd-v09-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag self3_4b > eval_self3.log 2>&1
echo "3way-uniform: $(grep -aoE '"acc": [0-9.]+' reports/self3_4b.json 2>/dev/null | head -1)"

$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-4B \
  --adapters models/kd-v06-3090/adapter,models/kd-v09-3090/adapter \
  --meta models/kd-v09-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag self2_4b > eval_self2.log 2>&1
echo "2way(v6+v9): $(grep -aoE '"acc": [0-9.]+' reports/self2_4b.json 2>/dev/null | head -1)"

echo "[2/2] v13 학습: 4B fp16 r32, v0.9에서 3에포크 continuation"
rm -f train_v13.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-4B --data data/mix_v09_train.jsonl \
  --init_adapter models/kd-v09-3090/adapter \
  --out models/kd-4b-v13-3090 --epochs 3 --bs 2 --accum 8 --lr 2e-5 --lora_r 32 --max_len 1088 \
  > train_v13.log 2>&1
[ -d models/kd-4b-v13-3090/adapter ] || { echo "v13 학습 실패"; exit 1; }

echo "[3/3] v13 판정"
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-4B --adapter models/kd-4b-v13-3090/adapter \
  --meta models/kd-4b-v13-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --bs 4 --out reports --tag kd_v13_typeddec > eval_v13_typeddec.log 2>&1
V13=$(grep -aoE '"acc": [0-9.]+' reports/kd_v13_typeddec.json | head -1)
echo "v13 격전지: $V13 (v0.9=0.8095, 1위=0.826)"
# v13+v0.9 셀프 앙상블
$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-4B \
  --adapters models/kd-4b-v13-3090/adapter,models/kd-v09-3090/adapter \
  --meta models/kd-v09-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag self_v13v09 > eval_self13.log 2>&1
echo "v13+v0.9: $(grep -aoE '"acc": [0-9.]+' reports/self_v13v09.json 2>/dev/null | head -1)"
echo "ROUND_4B_DONE $(date)"
