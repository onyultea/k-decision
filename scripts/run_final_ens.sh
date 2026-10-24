#!/bin/bash
# 최종 앙상블 탐색: v11+v12(구조 다양성), 3way(v11+v0.9+v12)
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P9="Qwen/Qwen3.5-9B|models/kd-9b-v11-3090/adapter|1|models/kd-9b-v11-3090/kdecision_meta.json"
P12="Qwen/Qwen3.5-9B|models/kd-9b-v12-3090/adapter|1|models/kd-9b-v12-3090/kdecision_meta.json"
P4="Qwen/Qwen3.5-4B|models/kd-v09-3090/adapter|1|models/kd-v09-3090/kdecision_meta.json"

# 1) v11+v12 2-way
for W in 0.7 0.8; do
  /home/kitri/venvs/kd/bin/python -X utf8 src/eval_ensemble2.py \
    --pairs "$P9,$P12" --weights "$W,$(python3 -c "print(1-$W)")" \
    --data data/typeddec_test.jsonl --limit 400 --tag fens_v11v12_w$W >> eval_final_ens.log 2>&1
  echo "v11v12 W=$W: $(grep -aoE '"acc": [0-9.]+' reports/fens_v11v12_w$W.json 2>/dev/null | head -1)"
done

# 2) 3-way v11+v0.9+v12
/home/kitri/venvs/kd/bin/python -X utf8 src/eval_ensemble2.py \
  --pairs "$P9,$P4,$P12" --weights "0.75,0.1,0.15" \
  --data data/typeddec_test.jsonl --limit 400 --tag fens_3way >> eval_final_ens.log 2>&1
echo "3way: $(grep -aoE '"acc": [0-9.]+' reports/fens_3way.json 2>/dev/null | head -1)"
echo FINAL_ENS_DONE
