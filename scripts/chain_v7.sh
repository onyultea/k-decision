#!/bin/bash
# K-Decision 체인 v7 — v10 (Qwen3.5-9B QLoRA) Darwin 격파 최후 카드
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
DI=/home/kitri/venvs/di/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f serve_systemone 2>/dev/null; sleep 3

echo "[1/4] 스모크(0.8B 3스텝, qlora 경로)"
head -n 300 data/mix_v09_train.jsonl > data/smoke_mix.jsonl
$KD5 src/train.py --base Qwen/Qwen3.5-0.8B --data data/smoke_mix.jsonl \
  --out /tmp/kd-smoke10 --max_steps 3 --bs 1 --accum 1 --lr 7e-5 --max_len 1088 --qlora --bf16 > smoke_v10.log 2>&1
[ -d /tmp/kd-smoke10/adapter ] || { echo "스모크 실패"; exit 1; }
rm -rf /tmp/kd-smoke10 data/smoke_mix.jsonl

echo "[2/4] v10 학습 (Qwen3.5-9B QLoRA, v0.9 믹스 재사용)"
rm -f train_v10.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-9B --data data/mix_v09_train.jsonl \
  --out models/kd-9b-v10-3090 --epochs 1 --bs 1 --accum 16 --lr 5e-5 --max_len 1088 --qlora --bf16 \
  > train_v10.log 2>&1
[ -d models/kd-9b-v10-3090/adapter ] || { echo "v10 학습 실패"; exit 1; }

echo "[3/4] 판정 평가"
MB=Qwen/Qwen3.5-9B; MM=models/kd-9b-v10-3090/kdecision_meta.json; MA=models/kd-9b-v10-3090/adapter
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/typeddec_test.jsonl --limit 400 --bs 1 --out reports --tag kd_v10_typeddec > eval_v10_typeddec.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 1 --out reports --tag kd_v10_klife > eval_v10_klife.log 2>&1
V9TDC=$(python3 -c "import json;print(json.load(open('reports/kd_v09_typeddec.json'))['raw']['acc'])" 2>/dev/null || echo 0)
V10TDC=$(python3 -c "import json;print(json.load(open('reports/kd_v10_typeddec.json'))['raw']['acc'])" 2>/dev/null || echo 0)
V10KLIFE=$(python3 -c "import json;print(json.load(open('reports/kd_v10_klife.json'))['raw']['acc'])" 2>/dev/null || echo 0)
echo "격전지: v0.9=$V9TDC vs v10=$V10TDC (Darwin=0.812) / klife=$V10KLIFE"

echo "[4/4] 판정"
if python3 -c "exit(0 if $V10TDC > 0.812 else 1)"; then
  echo "★ DARWIN 격파 성공 (v10=$V10TDC) — 전체 평가 + 병합·서빙·DI"
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 1 --out reports --tag kd_v10_synth > eval_v10_synth.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kmmlu_eval_test.jsonl --limit 500 --bs 1 --out reports --tag kd_v10_kmmlu > eval_v10_kmmlu.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kobest_eval_test.jsonl --limit 2600 --bs 1 --out reports --tag kd_v10_kobest > eval_v10_kobest.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kjdi_eval_test.jsonl --limit 3000 --bs 1 --out reports --tag kd_v10_kjdi > eval_v10_kjdi.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/regul_v08_test.jsonl --limit 100 --bs 1 --out reports --tag kd_v10_regul > eval_v10_regul.log 2>&1
  $KD5 src/merge_adapter.py --base $MB --adapter $MA --marker_meta $MM --out models/kd-9b-v10-merged > merge_v10.log 2>&1
  nohup $KD5 src/serve_systemone.py --merged models/kd-9b-v10-merged --port 8111 >> serve.log 2>&1 &
  sleep 30
  curl -s http://127.0.0.1:8111/health || { echo "서버 기동 실패"; exit 1; }
  nohup $DI -m decision_index pipeline --engine http \
    --option base_url=http://127.0.0.1:8111 --option model=k-decision-9b-v10 \
    --suite-dir /home/kitri/di-work --no-verify --fresh --out runs/kdecision-9b-v10 > di_pipeline_v10.log 2>&1 &
  while pgrep -f "decision_index pipeline" > /dev/null; do sleep 300; done
  echo "CHAIN7_ALL_DONE DARWIN_DEFEATED_9B $(date)"
else
  echo "v10 미달 — 4B v0.9(80.95%)가 최종. 격차 0.25pp."
  echo "CHAIN7_ALL_DONE 4b_final $(date)"
fi
