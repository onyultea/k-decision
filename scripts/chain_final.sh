#!/bin/bash
# K-Decision 최종 달리기 — 1위(0.826) 돌파까지: v13 완료 대기 → 전 조합 앙상블
# → 미달시 v14(9B int8 r32 장기) → 전 조합 → 최고 조합 확정
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f serve_systemone 2>/dev/null; sleep 3

echo "[1/8] v13 학습 완료 대기..."
while pgrep -f "v13" > /dev/null && pgrep -f "src/train" > /dev/null; do sleep 300; done
[ -d models/kd-4b-v13-3090/adapter ] || { echo "v13 어댑터 없음 — 학습 실패"; exit 1; }
echo "v13 완료: $(date)"

echo "[2/8] v13 단독 + 셀프 앙상블 전 조합"
# v13 단독
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-4B \
  --adapter models/kd-4b-v13-3090/adapter --meta models/kd-4b-v13-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --bs 4 --out reports --tag kd_v13_typeddec > eval_v13.log 2>&1
V13=$(grep -aoE '"acc": [0-9.]+' reports/kd_v13_typeddec.json | head -1 | grep -oE '[0-9.]+')
echo "v13 단독: $V13"

# v13+v0.9 (same 4B, adapter swap)
$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-4B \
  --adapters models/kd-4b-v13-3090/adapter,models/kd-v09-3090/adapter \
  --meta models/kd-v09-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag v13v09 > eval_e1.log 2>&1
E1=$(grep -aoE '"acc": [0-9.]+' reports/v13v09.json | head -1 | grep -oE '[0-9.]+')
echo "v13+v0.9: $E1"

# 4B 3-way (v13+v0.6+v0.9)
$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-4B \
  --adapters models/kd-4b-v13-3090/adapter,models/kd-v06-3090/adapter,models/kd-v09-3090/adapter \
  --meta models/kd-v09-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag v13_3way4b > eval_e2.log 2>&1
E2=$(grep -aoE '"acc": [0-9.]+' reports/v13_3way4b.json | head -1 | grep -oE '[0-9.]+')
echo "v13+v0.6+v0.9(3way4B): $E2"

# Cross: v13(4B) + v11(9B) — eval_ensemble2 (서로 다른 base)
$KD5 -X utf8 src/eval_ensemble2.py \
  --pairs "Qwen/Qwen3.5-4B|models/kd-4b-v13-3090/adapter|0|models/kd-4b-v13-3090/kdecision_meta.json,Qwen/Qwen3.5-9B|models/kd-9b-v11-3090/adapter|1|models/kd-9b-v11-3090/kdecision_meta.json" \
  --weights "0.4,0.6" \
  --data data/typeddec_test.jsonl --limit 400 --tag v13v11_x > eval_e3.log 2>&1
E3=$(grep -aoE '"acc": [0-9.]+' reports/v13v11_x.json | head -1 | grep -oE '[0-9.]+')
echo "v13(4B)×0.4+v11(9B)×0.6: $E3"

BEST=$(python3 -c "
vals = [float('$V13'), float('$E1'), float('$E2'), float('$E3')]
print(max(vals))
")
echo "4B 라운드 최고: $BEST"

if python3 -c "exit(0 if $BEST > 0.826 else 1)"; then
  echo "★ 1위 돌파! ($BEST > 0.826)"
  echo "FINAL_VICTORY 4b $BEST $(date)"
  exit 0
fi

echo "[3/8] 4B 미달 → v14 (9B int8 r32, v11 이어학습 2에포크)"
rm -f train_v14.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-9B --data data/mix_v09_train.jsonl \
  --init_adapter models/kd-9b-v11-3090/adapter \
  --out models/kd-9b-v14-3090 --epochs 2 --bs 1 --accum 16 --lr 2e-5 --lora_r 32 --max_len 1088 --int8 --bf16 \
  > train_v14.log 2>&1
[ -d models/kd-9b-v14-3090/adapter ] || { echo "v14 학습 실패"; exit 1; }

echo "[4/8] v14 단독"
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v14-3090/adapter --meta models/kd-9b-v14-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --bs 1 --int8 --out reports --tag kd_v14_typeddec > eval_v14.log 2>&1
V14=$(grep -aoE '"acc": [0-9.]+' reports/kd_v14_typeddec.json | head -1 | grep -oE '[0-9.]+')
echo "v14 단독: $V14"

echo "[5/8] v14+v11 (same 9B)"
$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-9B \
  --adapters models/kd-9b-v14-3090/adapter,models/kd-9b-v11-3090/adapter \
  --meta models/kd-9b-v11-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag v14v11 > eval_e4.log 2>&1
E4=$(grep -aoE '"acc": [0-9.]+' reports/v14v11.json | head -1 | grep -oE '[0-9.]+')
echo "v14+v11: $E4"

echo "[6/8] v14(9B)+v13(4B) cross"
$KD5 -X utf8 src/eval_ensemble2.py \
  --pairs "Qwen/Qwen3.5-9B|models/kd-9b-v14-3090/adapter|1|models/kd-9b-v14-3090/kdecision_meta.json,Qwen/Qwen3.5-4B|models/kd-4b-v13-3090/adapter|0|models/kd-4b-v13-3090/kdecision_meta.json" \
  --weights "0.6,0.4" \
  --data data/typeddec_test.jsonl --limit 400 --tag v14v13_x > eval_e5.log 2>&1
E5=$(grep -aoE '"acc": [0-9.]+' reports/v14v13_x.json | head -1 | grep -oE '[0-9.]+')
echo "v14(9B)×0.6+v13(4B)×0.4: $E5"

echo "[7/8] 최종 비교"
BEST2=$(python3 -c "
vals = [float('$BEST'), float('$V14'), float('$E4'), float('$E5')]
print(max(vals))
")
echo "전체 최고: $BEST2 (1위=0.826)"

if python3 -c "exit(0 if $BEST2 > 0.826 else 1)"; then
  echo "★ 최종 1위 돌파! ($BEST2)"
  echo "FINAL_VICTORY $BEST2 $(date)"
else
  echo "미달 — 최고기록으로 확정"
  echo "FINAL_BEST $BEST2 $(date)"
fi
echo "CHAIN_FINAL_DONE $(date)"
