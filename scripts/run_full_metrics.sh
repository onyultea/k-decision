#!/bin/bash
# v11 풀메트릭 산출 + 리더보드 #67 보완 댓글 게시
set -e
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/kitri/venvs/kd/bin/python -X utf8 src/eval_full_metrics.py \
  --adapter models/kd-9b-v11-3090/adapter --meta models/kd-9b-v11-3090/kdecision_meta.json \
  > full_metrics_out.log 2>&1
cat reports/final_metrics_v11.json
echo METRICS_DONE
