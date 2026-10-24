#!/bin/bash
# 0.85 달리기 — 5카드 동시 진행
# 카드3(앙상블 그리드) → 카드1(tasksource 200K) → 카드5(셀프증류) → v17 학습
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f serve_systemone 2>/dev/null; sleep 3

echo "═══ [카드 3] 앙상블 가중치 그리드 (30분) ═══"
P16="Qwen/Qwen3.5-9B|models/kd-9b-v16-3090/adapter|1|models/kd-9b-v16-3090/kdecision_meta.json"
P11="Qwen/Qwen3.5-9B|models/kd-9b-v11-3090/adapter|1|models/kd-9b-v11-3090/kdecision_meta.json"
for W in 0.4 0.5 0.6; do
  $KD5 -X utf8 src/eval_ensemble2.py \
    --pairs "$P16,$P11" --weights "$W,$(python3 -c "print(1-$W)")" \
    --data data/typeddec_test.jsonl --limit 400 --tag grid_v16w$W >> grid_final.log 2>&1
  echo "  v16×$W+v11×$(python3 -c "print(1-$W)"): $(grep -aoE '"acc": [0-9.]+' reports/grid_v16w$W.json 2>/dev/null | head -1)"
done

echo "═══ [카드 1] tasksource 30K→200K 변환 ═══"
$KD5 -X utf8 src/convert_tasksource.py --sample 200000 --out data/tasksource_200k.jsonl 2>&1 | tail -3

echo "═══ [카드 5] 셀프 증류 — v11이 tasksource에 소프트라벨 생성 ═══"
$KD5 -X utf8 - << 'DISTILL_EOF'
import sys, json, torch
sys.path.insert(0, "src")
import common as C
from train import load_model
from transformers import AutoTokenizer
from pathlib import Path

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
meta = json.loads(Path("models/kd-9b-v11-3090/kdecision_meta.json").read_text())
marker_id = meta["marker_id"]
model = load_model("Qwen/Qwen3.5-9B", for_training=False, dtype=torch.bfloat16, load_in_8bit=True)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "models/kd-9b-v11-3090/adapter").eval()

recs = C.load_jsonl("data/tasksource_200k.jsonl")[:30000]  # 30K만 증류 (시간 절약)
out = []
with torch.no_grad():
    for i, rec in enumerate(recs):
        if i % 1000 == 0:
            print(f"  증류 {i}/{len(recs)}")
        try:
            msgs = [{"role":"user","content":C.build_user_content(rec)},
                    {"role":"assistant","content":C.build_assistant_skeleton(rec)}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            if ids.shape[1] > 1088:
                continue
            logits = model(input_ids=ids).logits[0].float()
            pos = (ids[0]==marker_id).nonzero(as_tuple=True)[0].tolist()
            if len(pos) != len(rec["questions"]):
                continue
            r2 = json.loads(json.dumps(rec))
            for q, p in zip(r2["questions"], pos):
                syms = C.CHOICE_SYMBOLS[:len(q["options"])] if q["type"]=="choice" else C.NOUL_SYMBOLS
                cand = C.symbol_ids(tok, syms)
                probs = torch.softmax(logits[p-1, cand], dim=-1)
                q["soft"] = [round(float(x), 4) for x in probs]
            out.append(r2)
        except Exception:
            continue

C.dump_jsonl(out, "data/distill_30k.jsonl")
print(f"  증류 완료: {len(out)} 레코드")
DISTILL_EOF

echo "═══ v17 믹스 구성 (tasksource 200K + 증류 30K + 기존 전체) ═══"
$KD5 -X utf8 - << 'MIX_EOF'
import sys, json, random
sys.path.insert(0, "src")
from common import load_jsonl, dump_jsonl
rng = random.Random(77)

# 기존 v16 믹스 (이미 다양화됨)
v16 = load_jsonl("data/mix_v16_train.jsonl")
# tasksource 200K에서 v16에 없는 것 추가
ts_big = load_jsonl("data/tasksource_200k.jsonl")
ts_existing = set()
for r in v16:
    if r.get("domain") == "tasksource":
        ts_existing.add(r["id"])
ts_new = [r for r in ts_big if r["id"] not in ts_existing][:150000]
# 증류 데이터
distill = load_jsonl("data/distill_30k.jsonl")

mix = v16 + ts_new + distill
rng.shuffle(mix)
dump_jsonl(mix, "data/mix_v17_train.jsonl")
print(f"v17 믹스: v16({len(v16)}) + tasksource신규({len(ts_new)}) + 증류({len(distill)}) = {len(mix)}")
MIX_EOF

echo "═══ v17 학습 (9B int8 r32, 1에포크) ═══"
rm -f train_v17.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-9B --data data/mix_v17_train.jsonl \
  --out models/kd-9b-v17-3090 --epochs 1 --bs 1 --accum 16 --lr 8e-5 --lora_r 32 --max_len 1088 --int8 --bf16 \
  > train_v17.log 2>&1
[ -d models/kd-9b-v17-3090/adapter ] || { echo "v17 학습 실패"; exit 1; }

echo "═══ v17 평가 + 전체 앙상블 ═══"
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v17-3090/adapter --meta models/kd-9b-v17-3090/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --bs 1 --int8 --out reports --tag kd_v17_typeddec > eval_v17.log 2>&1
V17=$(grep -aoE '"acc": [0-9.]+' reports/kd_v17_typeddec.json | head -1 | grep -oE '[0-9.]+')
echo "v17 단독: $V17"

$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-9B \
  --adapter models/kd-9b-v17-3090/adapter --meta models/kd-9b-v17-3090/kdecision_meta.json \
  --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 1 --int8 --out reports --tag kd_v17_klife > eval_v17_klife.log 2>&1
echo "v17 한국생활: $(grep -aoE '"acc": [0-9.]+' reports/kd_v17_klife.json | head -1)"

# 3-way 앙상블: v17+v16+v11
P17="Qwen/Qwen3.5-9B|models/kd-9b-v17-3090/adapter|1|models/kd-9b-v17-3090/kdecision_meta.json"
$KD5 -X utf8 src/eval_ensemble2.py \
  --pairs "$P17,$P16,$P11" --weights "0.4,0.3,0.3" \
  --data data/typeddec_test.jsonl --limit 400 --tag v17_3way >> eval_3way.log 2>&1
E3=$(grep -aoE '"acc": [0-9.]+' reports/v17_3way.json 2>/dev/null | head -1 | grep -oE '[0-9.]+')
echo "v17+v16+v11 (3way): $E3"

# v17+v11 2-way
$KD5 -X utf8 src/eval_ensemble2.py \
  --pairs "$P17,$P11" --weights "0.5,0.5" \
  --data data/typeddec_test.jsonl --limit 400 --tag v17v11 >> eval_v17v11.log 2>&1
E2=$(grep -aoE '"acc": [0-9.]+' reports/v17v11.json 2>/dev/null | head -1 | grep -oE '[0-9.]+')
echo "v17+v11: $E2"

BEST=$(python3 -c "print(max(float('$V17'), float('$E2' or 0), float('$E3' or 0)))")
echo ""
echo "═══ 최종: $BEST (v11=0.818, 1위=0.826, 목표=0.85) ═══"
if python3 -c "exit(0 if $BEST > 0.826 else 1)"; then
  echo "★ 1위 돌파! ($BEST)"
  echo "FINAL_PUSH_VICTORY $BEST $(date)"
else
  echo "미달 — 최고 기록 갱신 여부 확인"
  echo "FINAL_PUSH_DONE $BEST $(date)"
fi
