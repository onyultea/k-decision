#!/bin/bash
# v0.2 학습 완료 후 전체 후속 작업 자동 실행 (평가→병합→서버→DI 파이프라인)
# 사용: nohup bash scripts/post_train_v02.sh > post_v02.log 2>&1 &
set -u
cd /home/kitri/k-decision
KD=/home/kitri/venvs/kd/bin/python
DI=/home/kitri/venvs/di/bin/python

echo "[1/5] 학습 종료 대기..."
while pgrep -f "venvs/kd/bin/python src/train.py" > /dev/null; do sleep 30; done
if [ ! -d models/kd-0.8b-v0.2-3090/adapter ]; then
  echo "어댑터 없음 - 학습 실패로 추정, 중단"; exit 1
fi
echo "[2/5] v0.2 평가 (합성 홀드아웃 + KMMLU)"
BASE=Qwen/Qwen3.5-0.8B
META=models/kd-0.8b-v0.2-3090/kdecision_meta.json
$KD -X utf8 src/eval.py --base $BASE --adapter models/kd-0.8b-v0.2-3090/adapter --meta $META --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 8 --out reports --tag kd_v02_synth > eval_v02_synth.log 2>&1
$KD -X utf8 src/eval.py --base $BASE --adapter models/kd-0.8b-v0.2-3090/adapter --meta $META --data data/kmmlu_eval_test.jsonl --limit 500 --bs 8 --out reports --tag kd_v02_kmmlu > eval_v02_kmmlu.log 2>&1
echo "[3/5] v0.2 병합"
$KD src/merge_adapter.py --base $BASE --adapter models/kd-0.8b-v0.2-3090/adapter --marker_meta $META --out models/kd-0.8b-v0.2-merged > merge_v02.log 2>&1
echo "[4/5] systemone 서버 기동"
pkill -f "serve_systemone" 2>/dev/null; sleep 2
nohup $KD src/serve_systemone.py --merged models/kd-0.8b-v0.2-merged --port 8111 > serve.log 2>&1 &
sleep 20
curl -s http://127.0.0.1:8111/health || { echo "서버 기동 실패"; exit 1; }
echo "[5/5] DI 스위트 임포트 대기 후 파이프라인"
# 리빌드 완료(선택파일 생성)까지 대기
while pgrep -f "decision_index suite" > /dev/null; do sleep 60; done
if [ ! -f /home/kitri/di-work/suite-imported ]; then
  export HF_TOKEN=$(cat /home/kitri/.hf-token); export HF_HUB_DISABLE_XET=1
  $DI -m decision_index suite import \
    --rows /home/kitri/di-work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows /home/kitri/di-work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows /home/kitri/di-work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz \
    && touch /home/kitri/di-work/suite-imported
fi
mkdir -p runs
nohup $DI -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8111 --option model=k-decision-0.8b-v0.2 \
  --out runs/kdecision-0.8b-v0.2 > di_pipeline.log 2>&1 &
echo "ALL_LAUNCHED $(date)"
