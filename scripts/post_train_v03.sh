#!/bin/bash
# v0.3(4B) 학습 완료 후 후속 자동화: 평가→병합→서버→(리빌드 완료 대기)→임포트→DI 파이프라인
# 임포트 성공 시에만 파이프라인을 돌린다 (v1 오케스트레이터 버그 수정).
set -u
cd /home/kitri/k-decision
KD=/home/kitri/venvs/kd/bin/python
DI=/home/kitri/venvs/di/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1

echo "[1/6] 4B 학습 종료 대기..."
while pgrep -f "venvs/kd/bin/python src/train.py" > /dev/null; do sleep 60; done
if [ ! -d models/kd-4b-v0.3-3090/adapter ]; then
  echo "어댑터 없음 - 학습 실패, 중단"; exit 1
fi
BASE=Qwen/Qwen3.5-4B
META=models/kd-4b-v0.3-3090/kdecision_meta.json
echo "[2/6] v0.3 평가"
$KD -X utf8 src/eval.py --base $BASE --adapter models/kd-4b-v0.3-3090/adapter --meta $META --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 8 --out reports --tag kd_v03_synth > eval_v03_synth.log 2>&1
$KD -X utf8 src/eval.py --base $BASE --adapter models/kd-4b-v0.3-3090/adapter --meta $META --data data/kmmlu_eval_test.jsonl --limit 500 --bs 8 --out reports --tag kd_v03_kmmlu > eval_v03_kmmlu.log 2>&1
echo "[3/6] v0.3 병합"
$KD src/merge_adapter.py --base $BASE --adapter models/kd-4b-v0.3-3090/adapter --marker_meta $META --out models/kd-4b-v0.3-merged > merge_v03.log 2>&1
echo "[4/6] 서버 재기동 (v0.3)"
pkill -f serve_systemone; sleep 3
nohup $KD src/serve_systemone.py --merged models/kd-4b-v0.3-merged --port 8111 > serve.log 2>&1 &
sleep 25
curl -s http://127.0.0.1:8111/health || { echo "서버 기동 실패"; exit 1; }
echo "[5/6] 스위트 리빌드 완료 대기 + 임포트"
ROWS=/home/kitri/di-work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz
while pgrep -f "decision_index suite" > /dev/null || [ ! -f "$ROWS" ]; do sleep 120; done
if [ ! -f /home/kitri/di-work/suite-imported ]; then
  $DI -m decision_index suite import \
    --rows $ROWS \
    --added-rows /home/kitri/di-work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows /home/kitri/di-work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz \
    && touch /home/kitri/di-work/suite-imported \
    || { echo "임포트 실패 - 파이프라인 중단"; exit 1; }
fi
echo "[6/6] DI 파이프라인 (v0.3, 비공개 사전측정)"
mkdir -p runs
$DI -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8111 --option model=k-decision-4b-v0.3 \
  --out runs/kdecision-4b-v0.3 > di_pipeline_v03.log 2>&1
echo "ALL_DONE $(date)"
