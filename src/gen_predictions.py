"""v18a 전체 예측 파일 생성 — 벤치마크 스코러(score_with_jobs.py) 대응.

모든 400 케이스 × 모든 질문 × 모든 옵션에 대한 확률을
원본 criteria 키로 매핑하여 predictions.jsonl로 저장.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

import common as C
from train import load_model


def load_test_orig():
    f = glob.glob("/home/kitri/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/"
                  "snapshots/*/all/test-00000-of-00001.parquet")[0]
    rows = pq.read_table(f).to_pylist()
    out = []
    for row in rows:
        gold = row["gold"]
        if isinstance(gold, str):
            gold = json.loads(gold)
        qs = row["questions"]
        if isinstance(qs, str):
            qs = json.loads(qs)
        if isinstance(qs, dict):
            qs = [dict(v, id=k) for k, v in qs.items()]
        elif isinstance(qs, list) and qs and isinstance(qs[0], str):
            qs = [json.loads(q) for q in qs]
        out.append({"id": row["id"], "gold": gold, "orig_qs": qs})
    return out


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    meta = json.loads(Path("models/kd-9b-v18a-seed123/kdecision_meta.json").read_text())
    marker_id = meta["marker_id"]
    model = load_model("Qwen/Qwen3.5-9B", for_training=False,
                       dtype=torch.bfloat16, load_in_8bit=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, "models/kd-9b-v18a-seed123/adapter").eval()

    orig_rows = load_test_orig()
    recs = C.load_jsonl("data/typeddec_test.jsonl")[:400]

    predictions = []
    for i, rec in enumerate(recs):
        gr = orig_rows[i]
        orig_qs = gr["orig_qs"]

        msgs = [{"role": "user", "content": C.build_user_content(rec)},
                {"role": "assistant", "content": C.build_assistant_skeleton(rec)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        pos = (ids[0] == marker_id).nonzero(as_tuple=True)[0].tolist()

        answers = {}
        for qi, (q, p) in enumerate(zip(rec["questions"], pos)):
            qid = q["id"]
            oq = next((o for o in orig_qs if o.get("id") == qid), None)
            gtype = oq.get("type") if oq else q["type"]
            crit = oq.get("criteria") if oq else None

            if gtype == "noul":
                syms = C.NOUL_SYMBOLS
                cand = C.symbol_ids(tok, syms)
                probs = torch.softmax(logits[p - 1, cand], dim=-1)
                answers[qid] = {
                    "type": "noul",
                    "noul": float(probs[0]),  # P(yes)
                }
            elif gtype in ("choice", "score"):
                syms = C.CHOICE_SYMBOLS[:len(q["options"])]
                cand = C.symbol_ids(tok, syms)
                probs = torch.softmax(logits[p - 1, cand], dim=-1)

                # 원본 criteria 키에 확률 매핑
                if isinstance(crit, dict):
                    keys = list(crit.keys())
                    prob_map = {k: float(probs[j]) for j, k in enumerate(keys) if j < len(probs)}
                    pred_key = max(prob_map, key=prob_map.get)
                    answers[qid] = {
                        "type": "choice",
                        "choice": pred_key,
                        "probabilities": prob_map,
                    }
                elif isinstance(crit, list):
                    # score: 인덱스가 곧 키
                    prob_map = {str(j): float(probs[j]) for j in range(len(probs))}
                    pred_key = max(prob_map, key=prob_map.get)
                    answers[qid] = {
                        "type": "choice",
                        "choice": pred_key,
                        "probabilities": prob_map,
                    }

        predictions.append({"id": gr["id"], "answers": answers})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/400")

    Path("bench_v2_predictions.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in predictions) + "\n",
        encoding="utf-8")
    print(f"저장 완료: bench_v2_predictions.jsonl ({len(predictions)} 케이스)")

    # 검증: 몇 개 샘플 확인
    for p in predictions[:2]:
        for qid, a in p["answers"].items():
            print(f"  {p['id']}/{qid}: type={a['type']} pred={a.get('choice', a.get('noul'))}")


if __name__ == "__main__":
    main()
