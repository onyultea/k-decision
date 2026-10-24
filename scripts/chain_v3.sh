#!/bin/bash
# K-Decision 체인 v3 — Darwin-27B 격전 대비 v0.6 (대규모 일반 타이핑드 결정 믹스)
# 체인 v2 완료 대기 → 승자 베이스로 v0.6 학습 → 5종 평가(한국생활/합성/KMMLU/KoBEST/typed-decisions)
# → v0.6이 승자를 이기면 서빙 교체 + DI 최종 재측정
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
KD4=/home/kitri/venvs/kd15/bin/python
DI=/home/kitri/venvs/di/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[1/6] 체인 v2 완료 대기..."
while ! grep -q "CHAIN2_ALL_DONE" chain_v2.log 2>/dev/null; do sleep 300; done
WINNER=$(grep -o "winner=v[45]" chain_v2.log | tail -1 | cut -d= -f2)
echo "체인 v2 승자: $WINNER — $(date)"
# v0.6 베이스는 Qwen(v4 트랙)으로 고정:
#  - 릴리스 라이선스(Kanana는 CC-BY-NC 비상업)
#  - 외부 벤치 우위(KoBEST 81.0 vs 71.9, KMMLU 31.0 vs 28.0)
#  - 한국생활 격차는 미미(99.68 vs 99.78)
PY=$KD5; BASE=Qwen/Qwen3.5-4B; BS=2; ACC=8; BF16=""
V2KLIFE=$(python3 -c "import json;print(json.load(open('reports/kd_v04_klife.json'))['raw']['acc'])" 2>/dev/null || echo 0)

echo "[2/6] v0.6b 믹스 재구성(증류+도구 추가) 및 학습 (베이스=$BASE)"
pkill -f serve_systemone 2>/dev/null  # 체인v2 잔여 서빙 정리(GPU 메모리 확보)
sleep 5
bash scripts/compose_v06b.sh > compose_v06b.log 2>&1 || echo "재구성 실패 — 기존 믹스로 진행"
rm -f train_v06.log
nice -n 10 $PY src/train.py \
  --base $BASE --data data/mix_v06_train.jsonl \
  --out models/kd-v06-3090 --epochs 1.5 --bs $BS --accum $ACC --lr 7e-5 $BF16 \
  > train_v06.log 2>&1
[ -d models/kd-v06-3090/adapter ] || { echo "v0.6 학습 실패"; exit 1; }

echo "[3/6] v0.6 6종 평가"
MB=$BASE; MM=models/kd-v06-3090/kdecision_meta.json; MA=models/kd-v06-3090/adapter
$PY -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 4 --out reports --tag kd_v06_klife > eval_v06_klife.log 2>&1
$PY -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 4 --out reports --tag kd_v06_synth > eval_v06_synth.log 2>&1
$PY -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kmmlu_eval_test.jsonl --limit 500 --bs 4 --out reports --tag kd_v06_kmmlu > eval_v06_kmmlu.log 2>&1
$PY -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kobest_eval_test.jsonl --limit 2600 --bs 4 --out reports --tag kd_v06_kobest > eval_v06_kobest.log 2>&1
$PY -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/typeddec_test.jsonl --limit 400 --bs 4 --out reports --tag kd_v06_typeddec > eval_v06_typeddec.log 2>&1
$PY -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kjdi_eval_test.jsonl --limit 3000 --bs 4 --out reports --tag kd_v06_kjdi > eval_v06_kjdi.log 2>&1

echo "[4/6] v0.6 vs 체인v2승자 비교"
V6KLIFE=$(python3 -c "import json;print(json.load(open('reports/kd_v06_klife.json'))['raw']['acc'])" 2>/dev/null || echo 0)
echo "v0.6 klife=$V6KLIFE vs v2승자 klife=$V2KLIFE"

echo "[5/6] 승자 서빙"
if python3 -c "exit(0 if $V6KLIFE > $V2KLIFE else 1)"; then
  echo "v0.6 승리 — 서빙 교체 + DI 재측정"
  $PY src/merge_adapter.py --base $MB --adapter $MA --marker_meta $MM --out models/kd-v06-merged > merge_v06.log 2>&1
  pkill -f serve_systemone; sleep 3
  nohup $PY src/serve_systemone.py --merged models/kd-v06-merged --port 8111 >> serve.log 2>&1 &
  sleep 30
  curl -s http://127.0.0.1:8111/health || { echo "서버 기동 실패"; exit 1; }
  nohup $DI -m decision_index pipeline --engine http \
    --option base_url=http://127.0.0.1:8111 --option model=k-decision-v0.6 \
    --suite-dir /home/kitri/di-work --no-verify --fresh --out runs/kdecision-v0.6 > di_pipeline_v06.log 2>&1 &
  while pgrep -f "decision_index pipeline" > /dev/null; do sleep 300; done
  echo "CHAIN3_ALL_DONE winner=v6 $(date)"
else
  echo "v0.6 미승 — 체인 v2 승자 유지 (DI 이미 완료)"
  echo "CHAIN3_ALL_DONE winner=$WINNER $(date)"
fi
