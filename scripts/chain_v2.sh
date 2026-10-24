#!/bin/bash
# K-Decision 마스터 체인 v2
# DI(v0.3) 완료 대기 → v0.4(Qwen3.5-4B) 학습·평가 → v0.5(kanana-2.1B) 학습·평가
# → 한국생활 홀드아웃 정확도로 승자 선정 → 승자 서빙 → DI 최종 재측정
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python    # transformers 5.20.2 (Qwen3.5용)
KD4=/home/kitri/venvs/kd15/bin/python  # transformers 4.57.3 (kanana용)
DI=/home/kitri/venvs/di/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

V3_RUN=runs/kdecision-4b-v0.3
FINAL_RUN=runs/kdecision-final

echo "[1/8] v0.3 DI 완료 대기..."
while pgrep -f "decision_index pipeline" > /dev/null || [ ! -f $V3_RUN/scores.json ]; do sleep 180; done
echo "v0.3 DI 완료: $(date)"

echo "[2/8] v0.4(Qwen3.5-4B) 학습"
pkill -f serve_systemone; sleep 3
rm -f train_v04.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-4B --data data/mix_v04_train.jsonl \
  --out models/kd-4b-v0.4-3090 --epochs 2 --bs 2 --accum 8 --lr 7e-5 \
  > train_v04.log 2>&1
[ -d models/kd-4b-v0.4-3090/adapter ] || { echo "v0.4 학습 실패"; exit 1; }

echo "[3/8] v0.4 4종 평가"
B4=Qwen/Qwen3.5-4B; M4=models/kd-4b-v0.4-3090/kdecision_meta.json; A4=models/kd-4b-v0.4-3090/adapter
$KD5 -X utf8 src/eval.py --base $B4 --adapter $A4 --meta $M4 --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 4 --out reports --tag kd_v04_klife > eval_v04_klife.log 2>&1
$KD5 -X utf8 src/eval.py --base $B4 --adapter $A4 --meta $M4 --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 4 --out reports --tag kd_v04_synth > eval_v04_synth.log 2>&1
$KD5 -X utf8 src/eval.py --base $B4 --adapter $A4 --meta $M4 --data data/kmmlu_eval_test.jsonl --limit 500 --bs 4 --out reports --tag kd_v04_kmmlu > eval_v04_kmmlu.log 2>&1
$KD5 -X utf8 src/eval.py --base $B4 --adapter $A4 --meta $M4 --data data/kobest_eval_test.jsonl --limit 2600 --bs 4 --out reports --tag kd_v04_kobest > eval_v04_kobest.log 2>&1

echo "[4/8] v0.5(kanana-2.1B, bf16) 학습"
rm -f train_v05.log
nice -n 10 $KD4 src/train.py \
  --base kakaocorp/kanana-nano-2.1b-instruct --data data/mix_v04_train.jsonl \
  --out models/kd-knn-2.1b-v0.5-3090 --epochs 2 --bs 4 --accum 4 --lr 7e-5 --bf16 \
  > train_v05.log 2>&1
[ -d models/kd-knn-2.1b-v0.5-3090/adapter ] || { echo "v0.5 학습 실패(스킵하고 v0.4로 진행)"; KNNAcc=0; }

echo "[5/8] v0.5 4종 평가"
if [ -d models/kd-knn-2.1b-v0.5-3090/adapter ]; then
  BK=kakaocorp/kanana-nano-2.1b-instruct; MK=models/kd-knn-2.1b-v0.5-3090/kdecision_meta.json; AK=models/kd-knn-2.1b-v0.5-3090/adapter
  $KD4 -X utf8 src/eval.py --base $BK --adapter $AK --meta $MK --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 4 --out reports --tag kd_v05_klife > eval_v05_klife.log 2>&1
  $KD4 -X utf8 src/eval.py --base $BK --adapter $AK --meta $MK --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 4 --out reports --tag kd_v05_synth > eval_v05_synth.log 2>&1
  $KD4 -X utf8 src/eval.py --base $BK --adapter $AK --meta $MK --data data/kmmlu_eval_test.jsonl --limit 500 --bs 4 --out reports --tag kd_v05_kmmlu > eval_v05_kmmlu.log 2>&1
  $KD4 -X utf8 src/eval.py --base $BK --adapter $AK --meta $MK --data data/kobest_eval_test.jsonl --limit 2600 --bs 4 --out reports --tag kd_v05_kobest > eval_v05_kobest.log 2>&1
fi

echo "[6/8] 승자 선정 (한국생활 홀드아웃 정확도 기준)"
A4k=$(python3 -c "import json;print(json.load(open('reports/kd_v04_klife.json'))['raw']['acc'])" 2>/dev/null || echo 0)
AKk=$(python3 -c "import json;print(json.load(open('reports/kd_v05_klife.json'))['raw']['acc'])" 2>/dev/null || echo 0)
echo "v0.4(Qwen4B) klife=$A4k vs v0.5(kanana2.1B) klife=$AKk"
WIN=$(python3 -c "print('v5' if $AKk > $A4k else 'v4')")

echo "[7/8] 승자 병합 + 서빙 ($WIN)"
if [ "$WIN" = "v5" ]; then
  PY=$KD4; BASE=kakaocorp/kanana-nano-2.1b-instruct
  ADP=models/kd-knn-2.1b-v0.5-3090/adapter; META=models/kd-knn-2.1b-v0.5-3090/kdecision_meta.json
  OUT=models/kd-knn-2.1b-v0.5-merged
else
  PY=$KD5; BASE=Qwen/Qwen3.5-4B
  ADP=models/kd-4b-v0.4-3090/adapter; META=models/kd-4b-v0.4-3090/kdecision_meta.json
  OUT=models/kd-4b-v0.4-merged
fi
$PY src/merge_adapter.py --base $BASE --adapter $ADP --marker_meta $META --out $OUT > merge_final.log 2>&1
nohup $PY src/serve_systemone.py --merged $OUT --port 8111 >> serve.log 2>&1 &
sleep 30
curl -s http://127.0.0.1:8111/health || { echo "서버 기동 실패"; exit 1; }

echo "[8/8] 최종 DI 재측정 (승자 모델)"
mkdir -p runs
nohup $DI -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8111 --option model=k-decision-$WIN \
  --suite-dir /home/kitri/di-work --no-verify --fresh --out $FINAL_RUN > di_pipeline_final.log 2>&1 &
while pgrep -f "decision_index pipeline" > /dev/null; do sleep 300; done
echo "CHAIN2_ALL_DONE winner=$WIN $(date)"
