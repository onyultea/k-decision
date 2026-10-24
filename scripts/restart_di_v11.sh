#!/bin/bash
# v11 DI 재측정: bf16 서빙(OOM) → int8 서빙 교체 후 fresh 재시작
cd /home/kitri/k-decision
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[1/3] 기존 파이프라인·서빙 정리"
pkill -f "decision_index pipeline" 2>/dev/null
pkill -f serve_systemone 2>/dev/null
sleep 5

echo "[2/3] int8 서빙 기동"
PYTHONUNBUFFERED=1 nohup /home/kitri/venvs/kd/bin/python src/serve_systemone.py \
  --merged models/kd-9b-v11-merged --port 8111 --int8 >> serve.log 2>&1 &
for i in $(seq 1 60); do
  sleep 5
  curl -s -m 5 http://127.0.0.1:8111/health && break
done
curl -s -m 5 http://127.0.0.1:8111/health || { echo "서버 기동 실패"; exit 1; }
echo ""

echo "[3/3] DI fresh 재측정"
nohup /home/kitri/venvs/di/bin/python -u -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8111 --option model=k-decision-9b-v11 \
  --suite-dir /home/kitri/di-work --no-verify --fresh --out runs/kdecision-9b-v11 > di_pipeline_v11.log 2>&1 &
echo "DI_V11_RESTARTED $(date)"
