#!/bin/bash
# K-Decision v15 — 데이터 정밀 공략 (오류분석 기반)
# 전략: ①격전지 4배 증강(옵션셔플×3+소프트라벨) ②agent_trace 집중보강
#        ③v11 이어학식(저LR) ④멀티체크포인트 앙상블
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f serve_systemone 2>/dev/null; sleep 3

echo "[1/5] 증강 데이터 생성"
$KD5 -X utf8 src/gen_typeddec_augmented.py > gen_aug.log 2>&1
tail -4 gen_aug.log

echo "[2/5] v15 학습 (9B int8 r32, v11 이어학습, 1에포크)"
rm -f train_v15.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-9B --data data/mix_v15_train.jsonl \
  --init_adapter models/kd-9b-v11-3090/adapter \
  --out models/kd-9b-v15-3090 --epochs 1 --bs 1 --accum 16 --lr 3e-5 --lora_r 32 --max_len 1088 --int8 --bf16 \
  > train_v15.log 2>&1
[ -d models/kd-9b-v15-3090/adapter ] || { echo "v15 학습 실패"; exit 1; }

echo "[3/5] v15 평가 (격전지 + 한국생활)"
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v15-3090/adapter --meta models/kd-9b-v15-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --bs 1 --int8 --out reports --tag kd_v15_typeddec > eval_v15.log 2>&1
V15=$(grep -aoE '"acc": [0-9.]+' reports/kd_v15_typeddec.json | head -1 | grep -oE '[0-9.]+')
echo "v15 격전지: $V15"

$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v15-3090/adapter --meta models/kd-9b-v15-3090/kdecision_meta.json \
  --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 1 --int8 --out reports --tag kd_v15_klife > eval_v15_klife.log 2>&1
V15K=$(grep -aoE '"acc": [0-9.]+' reports/kd_v15_klife.json | head -1 | grep -oE '[0-9.]+')
echo "v15 한국생활: $V15K"

echo "[4/5] v15+v11 앙상블"
$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-9B \
  --adapters models/kd-9b-v15-3090/adapter,models/kd-9b-v11-3090/adapter \
  --meta models/kd-9b-v11-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag v15v11 > eval_v15v11.log 2>&1
E1=$(grep -aoE '"acc": [0-9.]+' reports/v15v11.json 2>/dev/null | head -1 | grep -oE '[0-9.]+')
echo "v15+v11: $E1"

echo "[5/5] 최종 판정"
BEST=$(python3 -c "print(max(float('$V15'), float('$E1' or 0)))")
echo "최고: $BEST (v11=0.818, 1위=0.826)"
if python3 -c "exit(0 if $BEST > 0.826 else 1)"; then
  echo "★ 1위 돌파!"
  echo "CHAIN_V15_VICTORY $BEST $(date)"
else
  echo "미달 — v11 유지"
  echo "CHAIN_V15_DONE $BEST $(date)"
fi
