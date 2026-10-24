#!/bin/bash
# v11 후속: DI 완료 대기 → 한국인덱스(kjdi) 보완 평가 → 서빙 재기동 → 최종 기록
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[1/3] DI 완료 대기..."
while pgrep -f "decision_index pipeline" > /dev/null; do sleep 300; done
echo "DI 완료: $(date)"

echo "[2/3] kjdi 보완 평가 (서빙 일시 정지 후 bf16)"
pkill -f serve_systemone 2>/dev/null; sleep 5
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v11-3090/adapter --meta models/kd-9b-v11-3090/kdecision_meta.json \
  --data data/kjdi_eval_test.jsonl --limit 3000 --bs 1 --out reports --tag kd_v11_kjdi \
  > eval_v11_kjdi2.log 2>&1
grep -aoE '"acc": [0-9.]+' reports/kd_v11_kjdi.json 2>/dev/null | head -1

echo "[3/3] int8 서빙 재기동"
PYTHONUNBUFFERED=1 nohup $KD5 src/serve_systemone.py \
  --merged models/kd-9b-v11-merged --port 8111 --int8 >> serve.log 2>&1 &
sleep 40
curl -s http://127.0.0.1:8111/health && echo ""
echo "POST_V11_DONE $(date)"
