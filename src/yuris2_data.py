"""YURIS-2 데이터셋 — 상태 그룹 기반 (state 1회 인코딩 + 옵션 배치).

typeddec train parquet에서 gold probabilities + label_agreement를 그대로 사용:
  - teacher_probs: gold.probabilities (주석자 분포)
  - agree: argmax_agree (메타인지 헤드 타깃)
"""
from __future__ import annotations

import glob
import json

import pyarrow.parquet as pq


def load_typeddec_train():
    f = glob.glob("/home/kitri/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/"
                  "snapshots/*/all/train-*.parquet")[0]
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
        la = row.get("label_agreement", "{}")
        if isinstance(la, str):
            la = json.loads(la)
        state = row.get("state", "")
        if not isinstance(state, str) or not state:
            continue
        out.append({"state": state, "questions": qs, "gold": gold, "la": la,
                    "workflow": row.get("workflow", "?")})
    return out


def build_groups(rows, tokenizer, state_max=1024, opt_max=384):
    """state 그룹 단위 데이터 구성.

    Returns: list of groups
      group = {"state_ids", "state_mask",
               "pairs": [{"opt_ids","opt_mask","qtype","gold_idx","n",
                          "teacher_probs","agree"}], "workflow"}
    """
    groups = []
    skipped = 0
    for row in rows:
        state = row["state"]
        s_ids = tokenizer(state, add_special_tokens=False)["input_ids"][:state_max]
        if not s_ids:
            skipped += 1
            continue
        pairs = []
        qmeta = []  # (start, n, qtype, gold_idx, teacher_probs|None, agree)
        start = 0
        ok = True
        for q in row["questions"]:
            qid = q.get("id")
            g = row["gold"].get(qid, {})
            kind = q.get("type") or g.get("type")
            crit = q.get("criteria")
            instr = q.get("instructions", "")
            gp = g.get("probabilities", {})
            la_q = row.get("la", {}).get(qid, {})
            agree = la_q.get("argmax_agree")

            if kind == "choice" and isinstance(crit, dict):
                keys = list(crit.keys())
                label = g.get("label")
                if label not in keys:
                    ok = False
                    break
                gold_idx = keys.index(label)
                for k in keys:
                    opt_text = f"{instr} Option: {k} = {crit[k]}"
                    o_ids = tokenizer(opt_text, add_special_tokens=False)["input_ids"][:opt_max]
                    soft = None
                    if gp and len(gp) == len(keys):
                        soft = [float(gp.get(kk, 0)) for kk in keys]
                    pairs.append({"opt_ids": o_ids, "opt_mask": [1] * len(o_ids),
                                  "qtype": "choice", "gold_idx": gold_idx,
                                  "n": len(keys), "teacher_probs": soft,
                                  "agree": agree})
                qmeta.append({"start": start, "n": len(keys), "qtype": "choice",
                              "gold_idx": gold_idx, "teacher_probs": soft,
                              "agree": agree})
                start += len(keys)
            elif kind == "score" and isinstance(crit, list):
                label = g.get("label")
                if label is None or not str(label).isdigit() or int(label) >= len(crit):
                    ok = False
                    break
                gold_idx = int(label)
                for li, lvl in enumerate(crit):
                    opt_text = f"{instr} Level {li}: {lvl}"
                    o_ids = tokenizer(opt_text, add_special_tokens=False)["input_ids"][:opt_max]
                    soft = None
                    if gp and len(gp) == len(crit):
                        soft = [float(gp.get(str(j), 0)) for j in range(len(crit))]
                    pairs.append({"opt_ids": o_ids, "opt_mask": [1] * len(o_ids),
                                  "qtype": "choice", "gold_idx": gold_idx,
                                  "n": len(crit), "teacher_probs": soft,
                                  "agree": agree})
                qmeta.append({"start": start, "n": len(crit), "qtype": "choice",
                              "gold_idx": gold_idx,
                              "teacher_probs": ([float(gp.get(str(j), 0))
                                                 for j in range(len(crit))]
                                                if gp and len(gp) == len(crit) else None),
                              "agree": agree})
                start += len(crit)
            elif kind == "noul":
                opt_text = f"{instr} Answer: Yes"
                o_ids = tokenizer(opt_text, add_special_tokens=False)["input_ids"][:opt_max]
                label = g.get("label")
                if label is None:
                    pt = g.get("noul", 0.5)
                    gold_idx = 0 if pt > 0.5 else 1
                else:
                    gold_idx = 0 if str(label).lower() in ("true", "yes", "1") else 1
                soft = None
                if isinstance(g.get("noul"), (int, float)):
                    pt = float(g["noul"])
                    soft = [pt, 1 - pt]
                pairs.append({"opt_ids": o_ids, "opt_mask": [1] * len(o_ids),
                              "qtype": "noul", "gold_idx": gold_idx,
                              "n": 2, "teacher_probs": soft,
                              "agree": agree})
                qmeta.append({"start": start, "n": 2, "qtype": "noul",
                              "gold_idx": gold_idx,
                              "teacher_probs": soft, "agree": agree})
            else:
                ok = False
                break
        if not ok or not pairs:
            skipped += 1
            continue
        groups.append({
            "state_ids": s_ids, "state_mask": [1] * len(s_ids),
            "pairs": pairs, "workflow": row.get("workflow", "?"),
            "questions_meta": qmeta,
        })
    if skipped:
        print(f"[yuris2-data] 스킵: {skipped} 케이스")
    return groups
