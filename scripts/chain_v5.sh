#!/bin/bash
# K-Decision 체인 v5 — v0.8 (격전지 5배 + 규정 준거) Darwin 역전 최종 시도
# 즉시 실행(대기 없음): 믹스 구성 → 스모크 → 학습 → 7종 평가(규정 포함)
# → typeddec>0.812 달성 시 병합·서빙·DI 재측정
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
DI=/home/kitri/venvs/di/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f serve_systemone 2>/dev/null; sleep 3

echo "[1/6] v0.8 믹스 구성"
bash scripts/compose_v08.sh > compose_v08.log 2>&1 || { echo "믹스 실패"; exit 1; }

echo "[2/6] 스모크(0.8B 3스텝)"
head -n 300 data/mix_v08_train.jsonl > data/smoke_mix.jsonl
$KD5 src/train.py --base Qwen/Qwen3.5-0.8B --data data/smoke_mix.jsonl \
  --out /tmp/kd-smoke --max_steps 3 --bs 1 --accum 1 --lr 7e-5 > smoke_v08.log 2>&1
[ -d /tmp/kd-smoke/adapter ] || { echo "스모크 실패"; exit 1; }
rm -rf /tmp/kd-smoke data/smoke_mix.jsonl

echo "[3/6] v0.8 학습 (Qwen3.5-4B, 믹스 $(grep -c '' data/mix_v08_train.jsonl))"
rm -f train_v08.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-4B --data data/mix_v08_train.jsonl \
  --out models/kd-v08-3090 --epochs 1.5 --bs 2 --accum 8 --lr 7e-5 \
  > train_v08.log 2>&1
[ -d models/kd-v08-3090/adapter ] || { echo "v0.8 학습 실패"; exit 1; }

echo "[4/6] v0.8 7종 평가"
MB=Qwen/Qwen3.5-4B; MM=models/kd-v08-3090/kdecision_meta.json; MA=models/kd-v08-3090/adapter
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/typeddec_test.jsonl --limit 400 --bs 4 --out reports --tag kd_v08_typeddec > eval_v08_typeddec.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 4 --out reports --tag kd_v08_klife > eval_v08_klife.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 4 --out reports --tag kd_v08_synth > eval_v08_synth.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kmmlu_eval_test.jsonl --limit 500 --bs 4 --out reports --tag kd_v08_kmmlu > eval_v08_kmmlu.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kobest_eval_test.jsonl --limit 2600 --bs 4 --out reports --tag kd_v08_kobest > eval_v08_kobest.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kjdi_eval_test.jsonl --limit 3000 --bs 1 --out reports --tag kd_v08_kjdi > eval_v08_kjdi.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/regul_v08_test.jsonl --limit 100 --bs 1 --out reports --tag kd_v08_regul > eval_v08_regul.log 2>&1

V6TDC=$(python3 -c "import json;print(json.load(open('reports/kd_v06_typeddec.json'))['raw']['acc'])" 2>/dev/null || echo 0)
V8TDC=$(python3 -c "import json;print(json.load(open('reports/kd_v08_typeddec.json'))['raw']['acc'])" 2>/dev/null || echo 0)
V8KLIFE=$(python3 -c "import json;print(json.load(open('reports/kd_v08_klife.json'))['raw']['acc'])" 2>/dev/null || echo 0)
echo "격전지: v0.6=$V6TDC vs v0.8=$V8TDC / klife=$V8KLIFE / Darwin=0.812"

echo "[5/6] 판정 — v0.6 갱신 조건: (typeddec>V6) and (klife>=0.995)"
if python3 -c "exit(0 if $V8TDC > $V6TDC and $V8KLIFE >= 0.995 else 1)"; then
  echo "v0.8 승리 — 병합·서빙·DI 재측정"
  $KD5 src/merge_adapter.py --base $MB --adapter $MA --marker_meta $MM --out models/kd-v08-merged > merge_v08.log 2>&1
  nohup $KD5 src/serve_systemone.py --merged models/kd-v08-merged --port 8111 >> serve.log 2>&1 &
  sleep 30
  curl -s http://127.0.0.1:8111/health || { echo "서버 기동 실패"; exit 1; }
  nohup $DI -m decision_index pipeline --engine http \
    --option base_url=http://127.0.0.1:8111 --option model=k-decision-v0.8 \
    --suite-dir /home/kitri/di-work --no-verify --fresh --out runs/kdecision-v0.8 > di_pipeline_v08.log 2>&1 &
  while pgrep -f "decision_index pipeline" > /dev/null; do sleep 300; done
else
  echo "v0.8 미갱신 — v0.6 유지"
fi
echo "CHAIN5_ALL_DONE $(date)"
