#!/bin/bash
# 스프 평가 공용 러너 — 인자: 모델디렉토리, 태그, 데이터
# 사용: bash scripts/run_soup_eval2.sh models/kd-soup-068 kd_soup68_typeddec data/typeddec_test.jsonl 400
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$1; T=$2; D=$3; L=${4:-400}
/home/kitri/venvs/kd/bin/python -X utf8 src/eval.py \
  --base Qwen/Qwen3.5-4B --adapter $M/adapter --meta $M/kdecision_meta.json \
  --data $D --limit $L --bs 4 --out reports --tag $T > eval_$T.log 2>&1
grep -aoE '"acc": [0-9.]+' reports/$T.json 2>/dev/null | head -1 || tail -3 eval_$T.log
