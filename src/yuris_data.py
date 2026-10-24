"""YURIS 데이터셋 — (state, question, option) 쌍 단위 인코딩 + 교사 소프트라벨.

각 질문의 옵션 수만큼 시퀀스를 생성하고, 배치는 (질문, 옵션) 쌍 단위로 구성.
손실은 질문 단위로 groupby하여 옵션 내 softmax를 계산.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from common import load_jsonl, dump_jsonl


class YurisPairDataset(Dataset):
    """질문×옵션 쌍 단위 데이터셋.

    레코드: {state, question, qtype(choice/noul), options[], gold_idx, teacher_probs[]|None}
    """

    def __init__(self, records, tokenizer, max_len=1088):
        self.tok = tokenizer
        self.max_len = max_len
        self.items = []
        skipped = 0
        for r in records:
            state = r["state"]
            q = r["question"]
            qtype = r["qtype"]
            options = r["options"]
            gold_idx = r["gold_idx"]
            teacher = r.get("teacher_probs")
            for oi, opt in enumerate(options):
                text = f"{state} </s> {q} </s> {opt}"
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if len(ids) > max_len:
                    skipped += 1
                    continue
                self.items.append({
                    "input_ids": ids,
                    "qtype": qtype,
                    "n_options": len(options),
                    "opt_index": oi,
                    "gold_idx": gold_idx,
                    "teacher_probs": teacher,
                })
        if skipped:
            print(f"[yuris-data] 길이 초과 스킵: {skipped} 쌍")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class YurisCollator:
    def __init__(self, tokenizer, max_len=1088):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        maxlen = min(max(len(b["input_ids"]) for b in batch), self.max_len)
        input_ids, attn = [], []
        for b in batch:
            ids = b["input_ids"][:maxlen]
            input_ids.append(ids + [self.tok.pad_token_id or 0] * (maxlen - len(ids)))
            attn.append([1] * len(ids) + [0] * (maxlen - len(ids)))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "meta": batch,
        }


def records_from_typed(typed_jsonl: str, with_soft: bool = False, seed=42):
    """타이핑드 결정 JSONL → YURIS 질문×옵션 레코드.

    gold probabilities가 있으면 teacher_probs로 사용.
    """
    rng = random.Random(seed)
    recs = load_jsonl(typed_jsonl)
    out = []
    for rec in recs:
        state = rec["state"]
        for q in rec["questions"]:
            options = q["options"]
            gold_sym = rec["answers"][q["id"]]
            gold_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(gold_sym) \
                if q["type"] == "choice" else ("YN".index(gold_sym))
            teacher = q.get("soft")
            if not with_soft or not teacher or len(teacher) != len(options):
                teacher = None
            out.append({
                "state": state, "question": q.get("text", "결정하라"),
                "qtype": q["type"], "options": options,
                "gold_idx": gold_idx, "teacher_probs": teacher,
            })
    return out


def records_from_raw(raw_jsonl: str, with_soft: bool = False, seed=42):
    """원본 parquet 변환 형태(row with state/questions/gold) → YURIS 레코드."""
    import json as _json
    out = []
    with open(raw_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            gold = row.get("gold")
            if isinstance(gold, str):
                gold = _json.loads(gold)
            qs = row.get("questions")
            if isinstance(qs, str):
                qs = _json.loads(qs)
            if isinstance(qs, dict):
                qs = [dict(v, id=k) for k, v in qs.items()]
            elif isinstance(qs, list) and qs and isinstance(qs[0], str):
                qs = [_json.loads(q) for q in qs]
            state = row.get("state", "")
            if not isinstance(state, str) or len(state) > 6000:
                continue
            for q in qs:
                qid = q.get("id")
                g = gold.get(qid, {})
                kind = q.get("type") or g.get("type")
                crit = q.get("criteria")
                instr = q.get("instructions", "")
                gp = g.get("probabilities", {})
                if kind == "noul":
                    label = g.get("label")
                    if label is not None:
                        gold_idx = 0 if str(label).lower() in ("true", "yes", "1") else 1
                    else:
                        gold_idx = 0 if g.get("noul", 0.5) > 0.5 else 1
                    teacher = [g.get("noul", 0.5), 1.0 - g.get("noul", 0.5)] \
                        if isinstance(g.get("noul"), (int, float)) else None
                    out.append({
                        "state": state, "question": instr, "qtype": "noul",
                        "options": ["Yes", "No"], "gold_idx": gold_idx,
                        "teacher_probs": teacher if with_soft else None,
                    })
                elif kind == "choice" and isinstance(crit, dict):
                    keys = list(crit.keys())
                    options = [f"{k}: {crit[k]}" if len(str(crit[k])) > 15 else str(k)
                               for k in keys]
                    label = g.get("label")
                    if label not in keys:
                        continue
                    teacher = None
                    if with_soft and gp and len(gp) == len(keys):
                        teacher = [float(gp.get(k, 0)) for k in keys]
                    out.append({
                        "state": state, "question": instr, "qtype": "choice",
                        "options": options, "gold_idx": keys.index(label),
                        "teacher_probs": teacher if with_soft else None,
                    })
                elif kind == "score" and isinstance(crit, list):
                    options = [str(c) for c in crit]
                    label = g.get("label")
                    if label is None or not str(label).isdigit() or int(label) >= len(options):
                        continue
                    teacher = None
                    if with_soft and gp and len(gp) == len(options):
                        teacher = [float(gp.get(str(j), 0)) for j in range(len(options))]
                    out.append({
                        "state": state, "question": instr + " (level)", "qtype": "choice",
                        "options": options, "gold_idx": int(label),
                        "teacher_probs": teacher if with_soft else None,
                    })
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def convert_dataset_to_yuris(typed_jsonl: str, out_jsonl: str, with_soft: bool = False):
    """타이핑드 결정 JSONL → YURIS 학습 형식으로 저장 (서버 배포용)."""
    import json as _json
    recs = load_jsonl(typed_jsonl)
    out = []
    for rec in recs:
        state = rec["state"]
        for q in rec["questions"]:
            options = q["options"]
            gold_sym = rec["answers"][q["id"]]
            gold_idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(gold_sym) \
                if q["type"] == "choice" else "YN".index(gold_sym)
            out.append({
                "state": state, "question": q.get("text", "결정하라"),
                "qtype": q["type"], "options": options,
                "gold_idx": gold_idx, "teacher_probs": q.get("soft"),
            })
    Path(out_jsonl).write_text(
        "\n".join(_json.dumps(r, ensure_ascii=False) for r in out) + "\n",
        encoding="utf-8")
    print(f"[yuris-convert] {typed_jsonl} → {out_jsonl}: {len(out)} 질문")
