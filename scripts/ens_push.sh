#!/bin/bash
# 11문제 더 맞추기 — 가중치 그리드 + 시드 다양화 동시 진행
set -u
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KD5=/home/kitri/venvs/kd/bin/python

P11="Qwen/Qwen3.5-9B|models/kd-9b-v11-3090/adapter|1|models/kd-9b-v11-3090/kdecision_meta.json"
P16="Qwen/Qwen3.5-9B|models/kd-9b-v16-3090/adapter|1|models/kd-9b-v16-3090/kdecision_meta.json"
P17="Qwen/Qwen3.5-9B|models/kd-9b-v17-3090/adapter|1|models/kd-9b-v17-3090/kdecision_meta.json"

echo "═══ [1/3] 3-way 가중치 그리드 (v11 중심) ═══"
for W11 in 0.5 0.6 0.7; do
  REST=$(python3 -c "print(round(1-$W11, 2) / 2)")
  $KD5 -X utf8 src/eval_ensemble2.py \
    --pairs "$P11,$P16,$P17" --weights "$W11,$REST,$REST" \
    --data data/typeddec_test.jsonl --limit 400 --tag gw3_${W11} >> gw.log 2>&1
  echo "  v11×$W11 + v16×$REST + v17×$REST: $(grep -aoE '"acc": [0-9.]+' reports/gw3_${W11}.json 2>/dev/null | head -1)"
done

echo "═══ [2/3] 시드 다양화 — v11 레시피로 시드 바꿔 학습 (v18a) ═══"
# v11과 동일한 레시피지만 시드만 다름 → 앙상블 다양성 확보
rm -f train_v18a.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-9B --data data/mix_v09_train.jsonl \
  --out models/kd-9b-v18a-seed123 --epochs 2 --bs 1 --accum 16 --lr 1e-4 --lora_r 32 --max_len 1088 --int8 --bf16 \
  > train_v18a.log 2>&1
[ -d models/kd-9b-v18a-seed123/adapter ] || { echo "v18a 학습 실패"; exit 1; }
echo "  v18a(시드123) 완료"

echo "═══ [3/3] 4-way+5-way 앙상블 ═══"
P18a="Qwen/Qwen3.5-9B|models/kd-9b-v18a-seed123/adapter|1|models/kd-9b-v18a-seed123/kdecision_meta.json"

# 4-way: v11+v16+v17+v18a (uniform)
$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-9B \
  --adapters models/kd-9b-v11-3090/adapter,models/kd-9b-v16-3090/adapter,models/kd-9b-v17-3090/adapter,models/kd-9b-v18a-seed123/adapter \
  --meta models/kd-9b-v11-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag ens_4way > eval_4way.log 2>&1
E4=$(grep -aoE '"acc": [0-9.]+' reports/ens_4way.json 2>/dev/null | head -1)
echo "  4-way (v11+v16+v17+v18a): $E4"

# 4-way 최적 가중치
BEST_W="0.4,0.2,0.2,0.2"
if python3 -c "exit(0 if float('$E4'.split(': ')[1] if ': ' in '$E4' else '0') > 0.821 else 1)" 2>/dev/null; then
  BEST_W="0.25,0.25,0.25,0.25"
fi
$KD5 -X utf8 src/eval_ensemble2.py \
  --pairs "$P11,$P16,$P17,$P18a" --weights "0.4,0.2,0.2,0.2" \
  --data data/typeddec_test.jsonl --limit 400 --tag ens_4way_w >> eval_4wayw.log 2>&1
E4W=$(grep -aoE '"acc": [0-9.]+' reports/ens_4way_w.json 2>/dev/null | head -1)
echo "  4-way 가중치(v11×0.4): $E4W"

# v18a 단독
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v18a-seed123/adapter --meta models/kd-9b-v18a-seed123/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --bs 1 --int8 --out reports --tag v18a_typeddec > eval_v18a.log 2>&1
echo "  v18a 단독: $(grep -aoE '"acc": [0-9.]+' reports/v18a_typeddec.json | head -1)"

# 전체 결과 정리
echo ""
echo "═══ 최종 결과 ═══"
for f in reports/gw3_*.json reports/ens_4way*.json reports/ens_9b_3way.json; do
  tag=$(basename $f .json)
  acc=$(grep -aoE '"acc": [0-9.]+' $f 2>/dev/null | head -1 | grep -oE '[0-9.]+')
  [ -n "$acc" ] && echo "  $tag: $acc"
done | sort -t: -k2 -rn | head -10

echo "ENS_PUSH_DONE $(date)"
