#!/bin/bash
# KMMLU 재평가 — v11(9B) + v0.9(4B) + 베이스 제로샷
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KD5=/home/kitri/venvs/kd/bin/python

echo "[1/3] v11 KMMLU 재평가"
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v11-3090/adapter --meta models/kd-9b-v11-3090/kdecision_meta.json \
  --data data/kmmlu_eval_test_v2.jsonl --limit 500 --bs 1 --int8 \
  --out reports --tag kmmlu_v2_v11 > eval_kmmlu_v2_v11.log 2>&1
V11=$(grep -aoE '"acc": [0-9.]+' reports/kmmlu_v2_v11.json | head -1)
echo "v11 KMMLU(수정): $V11"

echo "[2/3] v0.9(4B) KMMLU 재평가"
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-4B \
  --adapter models/kd-v09-3090/adapter --meta models/kd-v09-3090/kdecision_meta.json \
  --data data/kmmlu_eval_test_v2.jsonl --limit 500 --bs 4 \
  --out reports --tag kmmlu_v2_v09 > eval_kmmlu_v2_v09.log 2>&1
V09=$(grep -aoE '"acc": [0-9.]+' reports/kmmlu_v2_v09.json | head -1)
echo "v0.9 KMMLU(수정): $V09"

echo "[3/3] 베이스(9B) 제로샷"
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --data data/kmmlu_eval_test_v2.jsonl --limit 500 --bs 1 --int8 \
  --out reports --tag kmmlu_v2_base9b > eval_kmmlu_v2_base.log 2>&1
BASE=$(grep -aoE '"acc": [0-9.]+' reports/kmmlu_v2_base9b.json | head -1)
echo "베이스9B KMMLU(수정): $BASE"

echo "KMMLU_REVAL_DONE $(date)"
