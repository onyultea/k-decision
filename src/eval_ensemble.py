"""앙상블 평가 — 한 베이스에 어댑터 여러 개를 올리고 필드별 확률을 산술평균.

모델 스프(가중치 평균)와 달리 출력 확률 앙상블은 서로 다른 실수 패턴을
합산한다. 옵션 순서 TTA(원순서+역순서 평균)도 선택 가능.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

import common as C
from train import load_model


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
    assert len(pos) == len(rec["questions"])
    out = []
    for q, p in zip(rec["questions"], pos):
        syms = C.CHOICE_SYMBOLS[: len(q["options"])] if q["type"] == "choice" else C.NOUL_SYMBOLS
        cand = C.symbol_ids(tok, syms)
        out.append(torch.softmax(logits[p - 1, cand], dim=-1).cpu())
    return out


def reversed_rec(rec):
    """choice 옵션을 역순으로 뒤집은 사본 (TTA용)."""
    import copy
    r2 = copy.deepcopy(rec)
    for q, q2 in zip(rec["questions"], r2["questions"]):
        if q["type"] == "choice":
            opts = list(reversed(q["options"]))
            q2["options"] = opts
            gi = C.gold_symbol(rec, q)
            idx = "ABCDEFGH".index(gi)
            r2["answers"][q["id"]] = "ABCDEFGH"[len(opts) - 1 - idx]
    return r2


def probs_for_order(model, tok, rec, marker_id, reverse):
    r = reversed_rec(rec) if reverse else rec
    return field_probs(model, tok, r, marker_id), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", required=True, help="쉼표 구분 어댑터 경로들")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--tta", action="store_true", help="옵션 역순 TTA 추가")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--tag", default="ensemble")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    marker_id = json.loads(Path(args.meta).read_text(encoding="utf-8"))["marker_id"]
    model = load_model(args.base, for_training=False).cuda().eval()

    from peft import PeftModel
    pm = PeftModel.from_pretrained(model, args.adapters.split(",")[0],
                                   adapter_name="a")
    for i, ad in enumerate(args.adapters.split(",")[1:], start=1):
        pm.load_adapter(ad, adapter_name=f"m{i}")
    names = ["a"] + [f"m{i}" for i in range(1, len(args.adapters.split(",")))]

    records = C.load_jsonl(args.data)[: args.limit]
    correct = total = 0
    for rec in records:
        orders = [False] + ([True] if args.tta else [])
        acc_probs = None
        golds = {}
        for reverse in orders:
            r = reversed_rec(rec) if reverse else rec
            order_probs = []
            for name in names:
                pm.set_adapter(name)
                probs, rr = probs_for_order(pm, tok, r, marker_id, reverse)
                order_probs.append(probs)
            stacked = [[torch.stack([op[k] for op in order_probs]).mean(0)
                        for k in range(len(rr["questions"]))]]
            if acc_probs is None:
                acc_probs = stacked[0]
                golds = {q["id"]: C.gold_symbol(rr, q) for q in rr["questions"]}
            else:
                acc_probs = [(a + b) / 2 for a, b in zip(acc_probs, stacked[0])]
        for q, p in zip(rr["questions"], acc_probs):
            syms = C.CHOICE_SYMBOLS[: len(q["options"])] if q["type"] == "choice" else C.NOUL_SYMBOLS
            pred = syms[int(p.argmax())]
            correct += int(pred == golds[q["id"]])
            total += 1
    acc = correct / total
    print(f"[ensemble {args.tag}] acc={acc:.4f} ({correct}/{total})")
    out = Path(args.out) / f"{args.tag}.json"
    out.write_text(json.dumps({"tag": args.tag, "acc": acc,
                               "adapters": args.adapters, "tta": args.tta,
                               "n": total}, ensure_ascii=False, indent=2),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
