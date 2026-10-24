"""v11 격전지 오류 분석 — 어디서 잃는지 정확히 진단."""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

import common as C
from train import load_model


def load_gold_full():
    f = glob.glob("/home/kitri/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/"
                  "snapshots/*/all/test-00000-of-00001.parquet")[0]
    rows = pq.read_table(f).to_pylist()
    out = {}
    for row in rows:
        gold = row["gold"]
        if isinstance(gold, str):
            gold = json.loads(gold)
        qs = row["questions"]
        if isinstance(qs, str):
            qs = json.loads(qs)
        if isinstance(qs, list) and qs and isinstance(qs[0], str):
            qs = [json.loads(q) for q in qs]
        if isinstance(qs, dict):
            qs = [dict(v, id=k) for k, v in qs.items()]
        la = row.get("label_agreement", "{}")
        if isinstance(la, str):
            la = json.loads(la)
        out[row["id"]] = {"gold": gold, "questions": qs, "la": la,
                          "workflow": row.get("workflow", "?")}
    return out


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    meta = json.loads(Path("models/kd-9b-v11-3090/kdecision_meta.json").read_text())
    marker_id = meta["marker_id"]
    model = load_model("Qwen/Qwen3.5-9B", for_training=False,
                       dtype=torch.bfloat16, load_in_8bit=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, "models/kd-9b-v11-3090/adapter").eval()

    gold_data = load_gold_full()
    recs = C.load_jsonl("data/typeddec_test.jsonl")[:400]
    orig_ids = list(gold_data.keys())

    # 오류 분석 축적기
    by_workflow = defaultdict(lambda: [0, 0])  # [correct, total]
    by_qtype = defaultdict(lambda: [0, 0])
    by_agree = defaultdict(lambda: [0, 0])  # argmax_agree true/false
    by_confidence = defaultdict(lambda: [0, 0])  # bucket
    errors = []

    for i, rec in enumerate(recs):
        gd = gold_data[orig_ids[i]]
        la = gd["la"]
        wf = gd["workflow"]
        gold = gd["gold"]

        msgs = [{"role": "user", "content": C.build_user_content(rec)},
                {"role": "assistant", "content": C.build_assistant_skeleton(rec)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        pos = (ids[0] == marker_id).nonzero(as_tuple=True)[0].tolist()

        for qi, (q, p) in enumerate(zip(rec["questions"], pos)):
            syms = C.CHOICE_SYMBOLS[:len(q["options"])] if q["type"] == "choice" else C.NOUL_SYMBOLS
            cand = C.symbol_ids(tok, syms)
            probs = torch.softmax(logits[p - 1, cand], dim=-1)
            pred_sym = syms[int(probs.argmax())]
            gold_sym = C.gold_symbol(rec, q)
            conf = float(probs.max())
            ok = pred_sym == gold_sym

            by_workflow[wf][0] += ok; by_workflow[wf][1] += 1
            by_qtype[q["type"]][0] += ok; by_qtype[q["type"]][1] += 1

            # label agreement에서 이 질문의 agree 여부
            q_la = la.get(q["id"], {})
            agree = q_la.get("argmax_agree")
            if agree is None:
                agree_str = "unknown"
            else:
                agree_str = "agree" if agree else "disagree"
            by_agree[agree_str][0] += ok; by_agree[agree_str][1] += 1

            # confidence bucket
            bucket = f"{int(conf * 10) / 10:.1f}"
            by_confidence[bucket][0] += ok; by_confidence[bucket][1] += 1

            if not ok:
                g = gold.get(q["id"], {})
                errors.append({
                    "case": i, "qid": q["id"], "type": q["type"],
                    "workflow": wf, "agree": agree_str,
                    "pred": pred_sym, "gold": gold_sym,
                    "confidence": round(conf, 3),
                    "gold_label": g.get("label"),
                    "gold_probs": g.get("probabilities"),
                })

    print("=== 도메인별 ===")
    for wf, (c, t) in sorted(by_workflow.items(), key=lambda x: -x[1][1]):
        print(f"  {wf:40s} {c:4d}/{t:4d} = {c/t:.3f}")

    print("\n=== 질문 타입별 ===")
    for qt, (c, t) in by_qtype.items():
        print(f"  {qt:10s} {c:4d}/{t:4d} = {c/t:.3f}")

    print("\n=== 주석자 일치도별 ===")
    for ag, (c, t) in by_agree.items():
        print(f"  {ag:12s} {c:4d}/{t:4d} = {c/t:.3f}")

    print("\n=== 자신감별 ===")
    for cf, (c, t) in sorted(by_confidence.items()):
        print(f"  conf={cf}  {c:4d}/{t:4d} = {c/t:.3f}")

    print(f"\n=== 총 오류: {len(errors)} ===")
    print(f"  disagree에서 틀림: {sum(1 for e in errors if e['agree']=='disagree')}")
    print(f"  agree에서 틀림: {sum(1 for e in errors if e['agree']=='agree')}")
    print(f"  unknown에서 틀림: {sum(1 for e in errors if e['agree']=='unknown')}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/error_analysis_v11.json").write_text(
        json.dumps({"errors": errors,
                    "by_workflow": {k: v for k, v in by_workflow.items()},
                    "by_agree": {k: v for k, v in by_agree.items()},
                    "total_errors": len(errors)}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
