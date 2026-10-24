#!/bin/bash
# K-Decision 체인 v6 — v0.9 Darwin 격파 최종 라운드
# v0.6 어댑터 이어학습: max_len 1088 + 격전지 소프트라벨 6배
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
DI=/home/kitri/venvs/di/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f serve_systemone 2>/dev/null; sleep 3

echo "[1/5] v0.9 믹스 구성 (소프트라벨 재변환 포함)"
bash scripts/compose_v09.sh > compose_v09.log 2>&1 || { echo "믹스 실패"; exit 1; }
grep -a "" compose_v09.log | tail -3

echo "[2/5] 스모크(0.8B 3스텝, init_adapter+max_len 경로)"
head -n 300 data/mix_v09_train.jsonl > data/smoke_mix.jsonl
$KD5 src/train.py --base Qwen/Qwen3.5-0.8B --data data/smoke_mix.jsonl \
  --out /tmp/kd-smoke9 --max_steps 3 --bs 1 --accum 1 --lr 7e-5 --max_len 1088 > smoke_v09.log 2>&1
[ -d /tmp/kd-smoke9/adapter ] || { echo "스모크 실패"; exit 1; }
rm -rf /tmp/kd-smoke9 data/smoke_mix.jsonl

echo "[3/5] v0.9 학습 (v0.6 어댑터 이어학습, lr 3e-5, 1에포크)"
rm -f train_v09.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-4B --data data/mix_v09_train.jsonl \
  --init_adapter models/kd-v06-3090/adapter \
  --out models/kd-v09-3090 --epochs 1 --bs 2 --accum 8 --lr 3e-5 --max_len 1088 \
  > train_v09.log 2>&1
[ -d models/kd-v09-3090/adapter ] || { echo "v0.9 학습 실패"; exit 1; }

echo "[4/5] 판정 평가 (격전지 → 한국생활 순)"
MB=Qwen/Qwen3.5-4B; MM=models/kd-v09-3090/kdecision_meta.json; MA=models/kd-v09-3090/adapter
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/typeddec_test.jsonl --limit 400 --bs 4 --out reports --tag kd_v09_typeddec > eval_v09_typeddec.log 2>&1
$KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 4 --out reports --tag kd_v09_klife > eval_v09_klife.log 2>&1
V6TDC=$(python3 -c "import json;print(json.load(open('reports/kd_v06_typeddec.json'))['raw']['acc'])" 2>/dev/null || echo 0)
V9TDC=$(python3 -c "import json;print(json.load(open('reports/kd_v09_typeddec.json'))['raw']['acc'])" 2>/dev/null || echo 0)
V9KLIFE=$(python3 -c "import json;print(json.load(open('reports/kd_v09_klife.json'))['raw']['acc'])" 2>/dev/null || echo 0)
echo "격전지: v0.6=$V6TDC vs v0.9=$V9TDC (Darwin=0.812) / klife=$V9KLIFE"

echo "[5/5] 판정"
if python3 -c "exit(0 if $V9TDC > 0.812 else 1)"; then
  echo "★ DARWIN 격파 성공 ($V9TDC > 0.812) — 전체 평가 + 병합·서빙·DI"
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/synth_v01_test.jsonl --val data/synth_v01_val.jsonl --limit 800 --bs 4 --out reports --tag kd_v09_synth > eval_v09_synth.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kmmlu_eval_test.jsonl --limit 500 --bs 4 --out reports --tag kd_v09_kmmlu > eval_v09_kmmlu.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kobest_eval_test.jsonl --limit 2600 --bs 4 --out reports --tag kd_v09_kobest > eval_v09_kobest.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/kjdi_eval_test.jsonl --limit 3000 --bs 1 --out reports --tag kd_v09_kjdi > eval_v09_kjdi.log 2>&1
  $KD5 -X utf8 src/eval.py --base $MB --adapter $MA --meta $MM --data data/regul_v08_test.jsonl --limit 100 --bs 1 --out reports --tag kd_v09_regul > eval_v09_regul.log 2>&1
  $KD5 src/merge_adapter.py --base $MB --adapter $MA --marker_meta $MM --out models/kd-v09-merged > merge_v09.log 2>&1
  nohup $KD5 src/serve_systemone.py --merged models/kd-v09-merged --port 8111 >> serve.log 2>&1 &
  sleep 30
  curl -s http://127.0.0.1:8111/health || { echo "서버 기동 실패"; exit 1; }
  nohup $DI -m decision_index pipeline --engine http \
    --option base_url=http://127.0.0.1:8111 --option model=k-decision-v0.9 \
    --suite-dir /home/kitri/di-work --no-verify --fresh --out runs/kdecision-v0.9 > di_pipeline_v09.log 2>&1 &
  while pgrep -f "decision_index pipeline" > /dev/null; do sleep 300; done
  echo "CHAIN6_ALL_DONE DARWIN_DEFEATED $(date)"
elif python3 -c "exit(0 if $V9TDC > $V6TDC else 1)"; then
  echo "v0.9가 v0.6는 돌파했으나 Darwin 미달 — 9B 카드로 진행 필요"
  echo "CHAIN6_ALL_DONE partial $(date)"
else
  echo "v0.9 미돌파 — 9B QLoRA 카드 필요"
  echo "CHAIN6_ALL_DONE failed $(date)"
fi
