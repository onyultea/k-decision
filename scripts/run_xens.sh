#!/bin/bash
# 크로스-베이스 앙상블 평가: v11(9B int8) + v0.9(4B int8)
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/kitri/venvs/kd/bin/python -X utf8 src/eval_ensemble2.py \
  --pairs "Qwen/Qwen3.5-9B|models/kd-9b-v11-3090/adapter|1|models/kd-9b-v11-3090/kdecision_meta.json,Qwen/Qwen3.5-4B|models/kd-v09-3090/adapter|1|models/kd-v09-3090/kdecision_meta.json" \
  --data data/typeddec_test.jsonl --limit 400 --tag xens_9b_4b > eval_xens.log 2>&1
grep -aoE '"acc": [0-9.]+' reports/xens_9b_4b.json 2>/dev/null | head -1 || tail -3 eval_xens.log
