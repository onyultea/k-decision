#!/bin/bash
# YURIS-Decision — v2(K-Decision-9B v2)를 Qwen3.5-0.8B로 증류
# 1단계: 교사(v18a)가 학습 데이터에 소프트라벨 생성
# 2단계: 학생(0.8B)이 소프트라벨로 학습
# 3단계: 평가 (격전지 + 한국생활)
set -u
cd /home/kitri/k-decision
KD5=/home/kitri/venvs/kd/bin/python
export HF_TOKEN=$(cat /home/kitri/.hf-token)
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pkill -f serve_systemone 2>/dev/null; sleep 3

echo "═══ [1/4] 증류 데이터 생성 — 교사(v18a 9B)가 소프트라벨 부여 ═══"
$KD5 -X utf8 - << 'DISTILL_EOF'
import sys, json, torch, random
sys.path.insert(0, "src")
import common as C
from train import load_model
from transformers import AutoTokenizer
from pathlib import Path

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
meta = json.loads(Path("models/kd-9b-v18a-seed123/kdecision_meta.json").read_text())
marker_id = meta["marker_id"]
model = load_model("Qwen/Qwen3.5-9B", for_training=False, dtype=torch.bfloat16, load_in_8bit=True)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "models/kd-9b-v18a-seed123/adapter").eval()

# 증류 대상: 기존 믹스에서 샘플링 + tasksource
from common import load_jsonl
mix = load_jsonl("data/mix_v09_train.jsonl")
rng = random.Random(99)
rng.shuffle(mix)
# 다양성 확보: 도메인별 균형 샘플링
by_dom = {}
for r in mix:
    d = r.get("domain", "unknown")
    by_dom.setdefault(d, []).append(r)

distill_set = []
per_dom = 3000
for dom, recs in by_dom.items():
    distill_set.extend(recs[:per_dom])
rng.shuffle(distill_set)
print(f"증류 대상: {len(distill_set)} 레코드 (도메인 {len(by_dom)}개)")

out = []
with torch.no_grad():
    for i, rec in enumerate(distill_set):
        if i % 2000 == 0:
            print(f"  증류 {i}/{len(distill_set)}")
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

C.dump_jsonl(out, "data/yuris_distill_train.jsonl")
print(f"증류 완료: {len(out)} 레코드 (soft labels 포함)")
n_soft = sum(1 for r in out for q in r["questions"] if "soft" in q)
print(f"소프트라벨 질문: {n_soft}")
DISTILL_EOF

echo "═══ [2/4] 스모크 (0.8B) ═══"
head -n 200 data/yuris_distill_train.jsonl > data/yuris_smoke.jsonl
$KD5 src/train.py --base Qwen/Qwen3.5-0.8B --data data/yuris_smoke.jsonl \
  --out /tmp/yuris-smoke --max_steps 3 --bs 4 --accum 2 --lr 2e-4 --max_len 1088 --bf16 \
  > /dev/null 2>&1
[ -d /tmp/yuris-smoke/adapter ] || { echo "스모크 실패"; exit 1; }
rm -rf /tmp/yuris-smoke data/yuris_smoke.jsonl
echo "  스모크 통과"

echo "═══ [3/4] YURIS-Decision 학습 (Qwen3.5-0.8B, fp16, r32, 3에포크) ═══"
rm -f train_yuris.log
nice -n 10 $KD5 src/train.py \
  --base Qwen/Qwen3.5-0.8B --data data/yuris_distill_train.jsonl \
  --out models/yuris-decision-0.8b --epochs 3 --bs 8 --accum 2 --lr 2e-4 --lora_r 32 --max_len 1088 --bf16 \
  > train_yuris.log 2>&1
[ -d models/yuris-decision-0.8b/adapter ] || { echo "YURIS 학습 실패"; exit 1; }
echo "  YURIS 학습 완료"

echo "═══ [4/4] 평가 (격전지 + 한국생활) ═══"
$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-0.8B \
  --adapter models/yuris-decision-0.8b/adapter --meta models/yuris-decision-0.8b/kdecision_meta.json \
  --data data/typeddec_test.jsonl --limit 400 --bs 4 --out reports --tag yuris_typeddec > eval_yuris.log 2>&1
YTD=$(grep -aoE '"acc": [0-9.]+' reports/yuris_typeddec.json | head -1)
echo "  YURIS 격전지: $YTD"

$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-0.8B \
  --adapter models/yuris-decision-0.8b/adapter --meta models/yuris-decision-0.8b/kdecision_meta.json \
  --data data/klife_v02_test.jsonl --val data/klife_v02_val.jsonl --limit 1000 --bs 4 --out reports --tag yuris_klife > eval_yuris_klife.log 2>&1
YKL=$(grep -aoE '"acc": [0-9.]+' reports/yuris_klife.json | head -1)
echo "  YURIS 한국생활: $YKL"

$KD5 -X utf8 src/eval.py --base Qwen/Qwen3.5-0.8B \
  --adapter models/yuris-decision-0.8b/adapter --meta models/yuris-decision-0.8b/kdecision_meta.json \
  --data data/kobest_eval_test.jsonl --limit 2600 --bs 4 --out reports --tag yuris_kobest > eval_yuris_kobest.log 2>&1
YKB=$(grep -aoE '"acc": [0-9.]+' reports/yuris_kobest.json | head -1)
echo "  YURIS KoBEST: $YKB"

echo ""
echo "═══ YURIS-Decision 결과 ═══"
echo "  격전지: $YTD (교사 v2=0.8295, v11=0.818)"
echo "  한국생활: $YKL"
echo "  KoBEST: $YKB"
echo "YURIS_DONE $(date)"
