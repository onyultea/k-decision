#!/bin/bash
# v0.9 단독 + 옵션역순 TTA 평가
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/kitri/venvs/kd/bin/python -X utf8 src/eval_ensemble.py \
  --base Qwen/Qwen3.5-4B --adapters models/kd-v09-3090/adapter \
  --meta models/kd-v09-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tta --tag tta_v09 > eval_tta.log 2>&1
grep -aoE '"acc": [0-9.]+' reports/tta_v09.json 2>/dev/null | head -1 || tail -3 eval_tta.log
