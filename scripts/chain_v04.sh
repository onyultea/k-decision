#!/bin/bash
# v0.4 자동 체인: DI(v0.3) 완료 대기 → v0.4 학습 → 3종 평가 → 병합 → 서버 재기동 → DI(v0.4) 재측정
# 사용: setsid nohup bash scripts/chain_v04.sh > chain_v04.log 2>&1 < /dev/null &
set -u
cd /home/kitri/k-decision
KD=/home/kitri/venvs/kd/bin/python
DI=/home/kitri/venvs/di/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

V3_RUN=runs/kdecision-4b-v0.3
V4_RUN=runs/kdecision-4b-v0.4

echo "[1/7] v0.3 DI 파이프라인 완료 대기..."
while pgrep -f "decision_index pipeline" > /dev/null || [ ! -f $V3_RUN/scores.json ]; do sleep 180; done
echo "v0.3 DI 완료: $(date)"

echo "[2/7] 서버 정리 후 v0.4 학습 시작"
pkill -f serve_systemone; sleep 3
rm -f train_v04.log
nohup nice -n 10 $KD src/train.py \
  --base Qwen/Qwen3.5-4B --data data/mix_v04_train.jsonl \
  --out models/kd-4b-v0.4-3090 --epochs 2 --bs 2 --accum 8 --lr 7e-5 \
  > train_v04.log 2>&1
if [ ! -d models/kd-4b-v0.4-3090/adapter ]; then
  echo "v0.4 학습 실패 - 체인 중단"; exit 1
fi

echo "[3/7] v0.4 3종 평가 (한국생활 홀드아웃 + v0.1 합성 홀드아웃 + KMMLU)"
BASE=Qwen/Qwen3.5-4B
META=models/kd-4b-v0.4-3090/kdecision_meta.json
ADP=models/kd-4b-v0.4-3090/adapter
$KD -X utf8 src/eval.py --base $BASE --adapter $ADP --meta $META --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 4 --out reports --tag kd_v04_klife > eval_v04_klife.log 2>&1
$KD -X utf8 src/eval.py --base $BASE --adapter $ADP --meta $META --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 4 --out reports --tag kd_v04_synth > eval_v04_synth.log 2>&1
$KD -X utf8 src/eval.py --base $BASE --adapter $ADP --meta $META --data data/kmmlu_eval_test.jsonl --limit 500 --bs 4 --out reports --tag kd_v04_kmmlu > eval_v04_kmmlu.log 2>&1

echo "[4/7] v0.4 병합"
$KD src/merge_adapter.py --base $BASE --adapter $ADP --marker_meta $META --out models/kd-4b-v0.4-merged > merge_v04.log 2>&1

echo "[5/7] 서버 재기동 (v0.4)"
nohup $KD src/serve_systemone.py --merged models/kd-4b-v0.4-merged --port 8111 >> serve.log 2>&1 &
sleep 30
curl -s http://127.0.0.1:8111/health || { echo "v0.4 서버 기동 실패"; exit 1; }

echo "[6/7] DI(v0.4) 재측정"
mkdir -p runs
nohup $DI -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8111 --option model=k-decision-4b-v0.4 \
  --suite-dir /home/kitri/di-work --no-verify --fresh --out $V4_RUN > di_pipeline_v04.log 2>&1 &

echo "[7/7] DI(v0.4) 완료 대기"
while pgrep -f "decision_index pipeline" > /dev/null; do sleep 300; done
echo "CHAIN_ALL_DONE $(date)"
