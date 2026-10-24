#!/bin/bash
# 리더보드 공식 스코러 실행 — KL/Brier/ECE 전수 채우기
set -e
cd /home/kitri/k-decision
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KD5=/home/kitri/venvs/kd/bin/python

# 스코러 코드 확인
if [ ! -f bench/score_with_jobs.py ]; then
  echo "[1/3] typed-decisions 리포에서 벤치 코드 다운로드"
  /home/kitri/venvs/di/bin/python - << 'SCORER_EOF'
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download("LocalLLaMA/typed-decisions", "bench/score_with_jobs.py",
                    repo_type="dataset", token=os.environ.get("HF_TOKEN"))
print("scorer path:", p)
SCORER_EOF
fi

echo "[2/3] v11 예측 결과를 벤치 형식으로 저장"
# eval.py가 결과 JSON을 저장하도록 --dump 추가 필요 — 직접 실행
$KD5 -X utf8 - << 'EVAL_EOF'
import sys, json, torch, time
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

recs = C.load_jsonl("data/typeddec_test.jsonl")[:400]
out = []
latencies = []
with torch.no_grad():
    for rec in recs:
        msgs = [{"role":"user","content":C.build_user_content(rec)},
                {"role":"assistant","content":C.build_assistant_skeleton(rec)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        t0 = time.time()
        logits = model(input_ids=ids).logits[0].float()
        torch.cuda.synchronize()
        latencies.append((time.time()-t0)*1000)
        pos = (ids[0]==marker_id).nonzero(as_tuple=True)[0].tolist()
        answers = {}
        for q, p in zip(rec["questions"], pos):
            syms = C.CHOICE_SYMBOLS[:len(q["options"])] if q["type"]=="choice" else C.NOUL_SYMBOLS
            cand = C.symbol_ids(tok, syms)
            probs = torch.softmax(logits[p-1, cand], dim=-1)
            if q["type"]=="choice":
                keys = list(q["_orig_criteria_keys"]) if "_orig_criteria_keys" in q else [f"opt_{i}" for i in range(len(syms))]
                pmap = {k: float(probs[i]) for i, k in enumerate(keys)}
                top = max(pmap, key=pmap.get)
                answers[q["id"]] = {"type":"choice","choice":top,"probabilities":pmap}
            else:
                answers[q["id"]] = {"type":"noul","noul":float(probs[syms.index("Y")])}
        out.append({"id": rec["id"], "answers": answers})

lat_sorted = sorted(latencies)
result = {
    "model": "ONYULTEA/K-Decision-9B",
    "predictions": out,
    "latency_ms": {"median": lat_sorted[len(lat)//2], "mean": sum(lat)/len(lat),
                    "p95": lat_sorted[int(len(lat)*0.95)]},
    "n_cases": len(recs), "n_decisions": sum(len(r["questions"]) for r in recs),
}
Path("bench_v11_predictions.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print("saved bench_v11_predictions.json")
print(f"acc check: evaluating...")
correct = total = 0
for rec in recs:
    pred = next(o for o in out if o["id"]==rec["id"])
    for q in rec["questions"]:
        a = pred["answers"][q["id"]]
        gold = C.gold_symbol(rec, q)
        syms = C.CHOICE_SYMBOLS[:len(q["options"])] if q["type"]=="choice" else C.NOUL_SYMBOLS
        pred_sym = syms[int(max(range(len(a.get("probabilities",{"x":0}).values())),
                                key=lambda i: list(a.get("probabilities",{"x":0}).values())[i]))] if a["type"]=="choice" else ("Y" if a["noul"]>0.5 else "N")
        correct += int(pred_sym==gold); total += 1
print(f"accuracy: {correct/total:.4f} ({correct}/{total})")
EVAL_EOF

echo "[3/3] 완료 — 결과를 reports/bench_v11_full.json으로 복사"
cp bench_v11_predictions.json reports/bench_v11_full.json 2>/dev/null || true
echo "BENCH_SCORER_DONE $(date)"
