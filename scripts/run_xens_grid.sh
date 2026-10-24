#!/bin/bash
# 가중치 앙상블 그리드: 9B 가중치 0.6/0.7/0.8 + v11 TTA — 0.826 돌파 탐색
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P9="Qwen/Qwen3.5-9B|models/kd-9b-v11-3090/adapter|1|models/kd-9b-v11-3090/kdecision_meta.json"
P4="Qwen/Qwen3.5-4B|models/kd-v09-3090/adapter|1|models/kd-v09-3090/kdecision_meta.json"
for W in 0.6 0.7 0.8; do
  /home/kitri/venvs/kd/bin/python -X utf8 src/eval_ensemble2.py \
    --pairs "$P9,$P4" --weights "$W,$(python3 -c "print(1-$W)")" \
    --data data/typeddec_test.jsonl --limit 400 --tag xens_w$W >> eval_xens_grid.log 2>&1
  echo "W=$W: $(grep -aoE '"acc": [0-9.]+' reports/xens_w$W.json 2>/dev/null | head -1)"
done
echo GRID_DONE
