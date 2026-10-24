#!/bin/bash
# v14 수동 평가: 단독 + v11 앙상블 + v13 크로스
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KD5=/home/kitri/venvs/kd/bin/python

echo "[1/3] v14 단독"
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v14-3090/adapter --meta models/kd-9b-v14-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --bs 1 --int8 --out reports --tag kd_v14_typeddec > eval_v14_run.log 2>&1
V14=$(grep -aoE '"acc": [0-9.]+' reports/kd_v14_typeddec.json | head -1)
echo "v14 단독: $V14"

echo "[2/3] v14+v11 앙상블 (9B, adapter swap)"
$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-9B \
  --adapters models/kd-9b-v14-3090/adapter,models/kd-9b-v11-3090/adapter \
  --meta models/kd-9b-v11-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag v14v11 > eval_v14v11.log 2>&1
E1=$(grep -aoE '"acc": [0-9.]+' reports/v14v11.json 2>/dev/null | head -1)
echo "v14+v11: $E1"

echo "[3/3] v14(9B)+v13(4B) 크로스"
$KD5 -X utf8 src/eval_ensemble2.py \
  --pairs "Qwen/Qwen3.5-9B|models/kd-9b-v14-3090/adapter|1|models/kd-9b-v14-3090/kdecision_meta.json,Qwen/Qwen3.5-4B|models/kd-4b-v13-3090/adapter|0|models/kd-4b-v13-3090/kdecision_meta.json" \
  --weights "0.7,0.3" \
  --data data/typeddec_test.jsonl --limit 400 --tag v14v13x > eval_v14v13x.log 2>&1
E2=$(grep -aoE '"acc": [0-9.]+' reports/v14v13x.json 2>/dev/null | head -1)
echo "v14×0.7+v13×0.3: $E2"

echo "V14_EVAL_DONE $(date)"
