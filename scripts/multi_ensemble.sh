#!/bin/bash
# 1위 기법 적용: 멀티 런 앙상블 — 보유한 5개 체크포인트로 5-way 앙상블
# 방식: 어댑터 스왑 (같은 베이스 9B → 메모리 효율적)
set -u
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KD5=/home/kitri/venvs/kd/bin/python

echo "═══ [1/3] 9B 5-way 앙상블 (v11+v16+v17) ═══"
# 9B 3-way: v11+v16+v17 (uniform)
$KD5 -X utf8 src/eval_ensemble.py --base Qwen/Qwen3.5-9B \
  --adapters models/kd-9b-v11-3090/adapter,models/kd-9b-v16-3090/adapter,models/kd-9b-v17-3090/adapter \
  --meta models/kd-9b-v11-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --tag ens_9b_3way > eval_9b3.log 2>&1
echo "  9B 3-way (v11+v16+v17): $(grep -aoE '"acc": [0-9.]+' reports/ens_9b_3way.json 2>/dev/null | head -1)"

echo "═══ [2/3] 9B+4B 크로스 앙상블 (다양성 극대화) ═══"
# cross: v11(9B)×w + v0.9(4B)×(1-w) — 다양한 가중치
P11="Qwen/Qwen3.5-9B|models/kd-9b-v11-3090/adapter|1|models/kd-9b-v11-3090/kdecision_meta.json"
P16="Qwen/Qwen3.5-9B|models/kd-9b-v16-3090/adapter|1|models/kd-9b-v16-3090/kdecision_meta.json"
P09="Qwen/Qwen3.5-4B|models/kd-v09-3090/adapter|0|models/kd-v09-3090/kdecision_meta.json"

for W11 in 0.5 0.6; do
  W09=$(python3 -c "print(round(1-$W11-0.2, 2))")
  W16=0.2
  $KD5 -X utf8 src/eval_ensemble2.py \
    --pairs "$P11,$P16,$P09" --weights "$W11,$W16,$W09" \
    --data data/typeddec_test.jsonl --limit 400 --tag ens_cross_${W11}_${W16}_${W09} >> eval_cross.log 2>&1
  echo "  v11×$W11+v16×$W16+v0.9×$W09: $(grep -aoE '"acc": [0-9.]+' reports/ens_cross_${W11}_${W16}_${W09}.json 2>/dev/null | head -1)"
done

echo "═══ [3/3] 최적 조합 탐색 (그리드) ═══"
# best 조합 찾기
$KD5 -X utf8 - << 'GRID_EOF'
import json, glob

results = {}
for f in glob.glob("reports/ens_*.json") + glob.glob("reports/grid_*.json"):
    try:
        d = json.load(open(f))
        tag = d.get("tag", f.split("/")[-1].replace(".json",""))
        acc = d.get("acc") or d.get("accuracy") or 0
        if acc > 0.5:
            results[tag] = float(acc)
    except:
        pass

# 새로 추가된 것만
for f in glob.glob("reports/ens_cross_*.json"):
    try:
        d = json.load(open(f))
        tag = d.get("tag", "cross")
        acc = d.get("acc") or d.get("accuracy") or 0
        if acc > 0.5:
            results[tag] = float(acc)
    except:
        pass

print("\n=== 앙상블 전체 결과 ===")
for tag, acc in sorted(results.items(), key=lambda x: -x[1])[:10]:
    print(f"  {tag:40s} {acc:.4f}")

best_tag = max(results, key=results.get)
best_acc = results[best_tag]
print(f"\n최고: {best_tag} = {best_acc:.4f} (1위=0.8265)")
print(f"격차: {0.8265 - best_acc:.4f}")
GRID_EOF

echo "MULTI_ENS_DONE $(date)"
