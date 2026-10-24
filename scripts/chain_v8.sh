#!/bin/bash
# K-Decision 체인 v8 — v11 (Qwen3.5-9B int8 LoRA) 진짜 최후 카드
# v10 4비트 실패 원인(양자 정밀도)에 정확히 대응: int8 = 4비트의 16배 정밀도
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
DI=/home/kitri/venvs/di/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f serve_systemone 2>/dev/null; sleep 3

echo "[1/4] 스모크(0.8B 3스텝, int8 경로)"
head -n 300 data/mix_v09_train.jsonl > data/smoke_mix.jsonl
$KD5 src/train.py --base Qwen/Qwen3.5-0.8B --data data/smoke_mix.jsonl \
  --out /tmp/kd-smoke11 --max_steps 3 --bs 1 --accum 1 --lr 7e-5 --max_len 1088 --int8 --bf16 > smoke_v11.log 2>&1
[ -d /tmp/kd-smoke11/adapter ] || { echo "스모크 실패"; exit 1; }
rm -rf /tmp/kd-smoke11 data/smoke_mix.jsonl

echo "[2/4] v11 학습 (Qwen3.5-9B int8, r32, 2에포크)"
rm -f train_v11.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-9B --data data/mix_v09_train.jsonl \
  --out models/kd-9b-v11-3090 --epochs 2 --bs 1 --accum 16 --lr 1e-4 --lora_r 32 --max_len 1088 --int8 --bf16 \
  > train_v11.log 2>&1
[ -d models/kd-9b-v11-3090/adapter ] || { echo "v11 학습 실패"; exit 1; }

echo "[3/4] 판정 평가"
MB=Qwen/Qwen3.5-9B; MM=models/kd-9b-v11-3090/kdecision_meta.json; MA=models/kd-9b-v11-3090/adapter
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/typeddec_test.jsonl --limit 400 --bs 1 --out reports --tag kd_v11_typeddec > eval_v11_typeddec.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 1 --out reports --tag kd_v11_klife > eval_v11_klife.log 2>&1
V9TDC=$(python3 -c "import json;print(json.load(open('reports/kd_v09_typeddec.json'))['raw']['acc'])" 2>/dev/null || echo 0)
V11TDC=$(python3 -c "import json;print(json.load(open('reports/kd_v11_typeddec.json'))['raw']['acc'])" 2>/dev/null || echo 0)
V11KLIFE=$(python3 -c "import json;print(json.load(open('reports/kd_v11_klife.json'))['raw']['acc'])" 2>/dev/null || echo 0)
echo "격전지: v0.9=$V9TDC vs v11=$V11TDC (Darwin=0.812) / klife=$V11KLIFE"

echo "[4/4] 판정"
if python3 -c "exit(0 if $V11TDC > 0.812 else 1)"; then
  echo "★ DARWIN 격파 성공 (v11=$V11TDC) — 전체 평가 + 병합·서빙·DI"
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 1 --out reports --tag kd_v11_synth > eval_v11_synth.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kmmlu_eval_test.jsonl --limit 500 --bs 1 --out reports --tag kd_v11_kmmlu > eval_v11_kmmlu.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kobest_eval_test.jsonl --limit 2600 --bs 1 --out reports --tag kd_v11_kobest > eval_v11_kobest.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kjdi_eval_test.jsonl --limit 3000 --bs 1 --out reports --tag kd_v11_kjdi > eval_v11_kjdi.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/regul_v08_test.jsonl --limit 100 --bs 1 --out reports --tag kd_v11_regul > eval_v11_regul.log 2>&1
  $KD5 src/merge_adapter.py --base $MB --adapter $MA --marker_meta $MM --out models/kd-9b-v11-merged > merge_v11.log 2>&1
  nohup $KD5 src/serve_systemone.py --merged models/kd-9b-v11-merged --port 8111 >> serve.log 2>&1 &
  sleep 30
  curl -s http://127.0.0.1:8111/health || { echo "서버 기동 실패"; exit 1; }
  nohup $DI -m decision_index pipeline --engine http \
    --option base_url=http://127.0.0.1:8111 --option model=k-decision-9b-v11 \
    --suite-dir /home/kitri/di-work --no-verify --fresh --out runs/kdecision-9b-v11 > di_pipeline_v11.log 2>&1 &
  while pgrep -f "decision_index pipeline" > /dev/null; do sleep 300; done
  echo "CHAIN8_ALL_DONE DARWIN_DEFEATED_INT8 $(date)"
else
  echo "v11 미달 — v0.9(80.95%) 최종 확정. 하드웨어 내 4B 한계 도달."
  echo "CHAIN8_ALL_DONE final_v09 $(date)"
fi
