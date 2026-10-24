#!/bin/bash
# K-Decision v16 — Decider-4B 스타일 데이터 다양화 (1위 레시피 기반)
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f serve_systemone 2>/dev/null; sleep 3

echo "[1/4] 데이터 다양화 생성"
$KD5 -X utf8 src/gen_decider_style.py > gen_decider.log 2>&1
tail -8 gen_decider.log

echo "[2/4] 스모크"
head -n 300 data/mix_v16_train.jsonl > data/smoke_v16.jsonl
$KD5 src/train.py --base Qwen/Qwen3.5-0.8B --data data/smoke_v16.jsonl \
  --out /tmp/kd-smoke16 --max_steps 3 --bs 1 --accum 1 --lr 7e-5 --max_len 1088 --int8 --bf16 > smoke_v16.log 2>&1
[ -d /tmp/kd-smoke16/adapter ] || { echo "스모크 실패"; exit 1; }
rm -rf /tmp/kd-smoke16 data/smoke_v16.jsonl

echo "[3/4] v16 학습 (9B int8 r32, 신규 학습 1에포크)"
rm -f train_v16.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-9B --data data/mix_v16_train.jsonl \
  --out models/kd-9b-v16-3090 --epochs 1 --bs 1 --accum 16 --lr 1e-4 --lora_r 32 --max_len 1088 --int8 --bf16 \
  > train_v16.log 2>&1
[ -d models/kd-9b-v16-3090/adapter ] || { echo "v16 학습 실패"; exit 1; }

echo "[4/4] 평가 (격전지 + 한국생활 + 앙상블)"
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v16-3090/adapter --meta models/kd-9b-v16-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --bs 1 --int8 --out reports --tag kd_v16_typeddec > eval_v16.log 2>&1
V16=$(grep -aoE '"acc": [0-9.]+' reports/kd_v16_typeddec.json | head -1 | grep -oE '[0-9.]+')
echo "v16 격전지: $V16"

$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v16-3090/adapter --meta models/kd-9b-v16-3090/kdecision_meta.json \
  --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 1 --int8 --out reports --tag kd_v16_klife > eval_v16_klife.log 2>&1
V16K=$(grep -aoE '"acc": [0-9.]+' reports/kd_v16_klife.json | head -1 | grep -oE '[0-9.]+')
echo "v16 한국생활: $V16K"

$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-9B \
  --adapters models/kd-9b-v16-3090/adapter,models/kd-9b-v11-3090/adapter \
  --meta models/kd-9b-v11-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag v16v11 > eval_v16v11.log 2>&1
E1=$(grep -aoE '"acc": [0-9.]+' reports/v16v11.json 2>/dev/null | head -1 | grep -oE '[0-9.]+')
echo "v16+v11: $E1"

BEST=$(python3 -c "print(max(float('$V16'), float('$E1' or 0)))")
echo "최고: $BEST (v11=0.818, 1위=0.826)"
if python3 -c "exit(0 if $BEST > 0.826 else 1)"; then
  echo "★ 1위 돌파! ($BEST)"
  echo "CHAIN_V16_VICTORY $BEST $(date)"
else
  echo "미달 — v11 유지"
  echo "CHAIN_V16_DONE $BEST $(date)"
fi
