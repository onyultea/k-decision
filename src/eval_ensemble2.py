"""크로스-베이스 앙상블 평가 — (base, adapter) 쌍 여러 개의 필드 확률 산술평균.

같은 베이스 다중 어댑터는 실패(80.70)했지만, 다른 베이스(9B+4B)는 오류
패턴이 달라 앙상블 이득이 남. GPU 메모리: 9B int8(10GB)+4B int8(~6GB).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

import common as C
from train import load_model


def load_pair(base, adapter, int8):
    from peft import PeftModel
    m = load_model(base, for_training=False,
                   dtype=torch.bfloat16 if int8 else None,
                   load_in_8bit=int8)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    return m


@torch.no_grad()
def field_probs(model, tok, rec, marker_id):
    msgs = [
        {"role": "user", "content": C.build_user_content(rec)},
        {"role": "assistant", "content": C.build_assistant_skeleton(rec)},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    logits = model(input_ids=ids).logits[0].float()
    pos = (ids[0] == marker_id).nonzero(as_tuple=True)[0].tolist()
    out = []
    for q, p in zip(rec["questions"], pos):
        syms = C.CHOICE_SYMBOLS[: len(q["options"])] if q["type"] == "choice" else C.NOUL_SYMBOLS
        cand = C.symbol_ids(tok, syms)
        out.append(torch.softmax(logits[p - 1, cand], dim=-1).cpu())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="쉼표 구분 'base|adapter|int8(0/1)|meta' 목록")
    ap.add_argument("--weights", default=None,
                    help="쉼표 구분 가중치 (기본 균등). 예: 0.7,0.3")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--tag", default="xens")
    args = ap.parse_args()

    models = []
    for spec in args.pairs.split(","):
        base, adapter, i8, meta = spec.split("|")
        tok = AutoTokenizer.from_pretrained(base)
        marker_id = json.loads(Path(meta).read_text(encoding="utf-8"))["marker_id"]
        m = load_pair(base, adapter or None, i8 == "1")
        models.append((m, tok, marker_id))
        print(f"[loaded] {base} + {adapter or 'base'} (int8={i8})")
    weights = [float(x) for x in args.weights.split(",")] if args.weights \
        else [1.0 / len(models)] * len(models)

    records = C.load_jsonl(args.data)[: args.limit]
    correct = total = 0
    for rec in records:
        probs_sum = None
        for (m, tok, mid), w in zip(models, weights):
            probs = field_probs(m, tok, rec, mid)
            scaled = [p * w for p in probs]
            probs_sum = scaled if probs_sum is None else \
                [(a + b) for a, b in zip(probs_sum, scaled)]
        for q, p in zip(rec["questions"], probs_sum):
            syms = C.CHOICE_SYMBOLS[: len(q["options"])] if q["type"] == "choice" else C.NOUL_SYMBOLS
            pred = syms[int(p.argmax())]
            correct += int(pred == C.gold_symbol(rec, q))
            total += 1
    acc = correct / total
    print(f"[xens {args.tag}] acc={acc:.4f} ({correct}/{total})")
    out = Path(args.out) / f"{args.tag}.json"
    out.write_text(json.dumps({"tag": args.tag, "acc": acc, "pairs": args.pairs,
                               "n": total}, ensure_ascii=False, indent=2),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
