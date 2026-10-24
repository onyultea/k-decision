#!/bin/bash
# 스프(v0.6+v0.7 평균 어댑터) 격전지 평가 — 프로젝트 루트 기준 상대경로
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/kitri/venvs/kd/bin/python -X utf8 src/eval.py \
  --base Qwen/Qwen3.5-4B \
  --adapter models/kd-soup-067/adapter \
  --meta models/kd-soup-067/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --bs 4 \
  --out reports --tag kd_soup_typeddec > eval_soup_typeddec.log 2>&1
grep -aoE '"acc": [0-9.]+' reports/kd_soup_typeddec.json 2>/dev/null | head -1 || tail -3 eval_soup_typeddec.log
