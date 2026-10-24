"""bzantium/korean-jev-decision-index (한국어 디시전 인덱스, 82.5k) → 평가셋 변환 (서버 실행).

sample 열이 JSON 문자열: state/question(instructions, criteria?)/target.
- noul: target {true/false 가중치} → Y/N
- choice: criteria 옵션 + target 가중치 → argmax 심볼
test 스플릿 전용 평가 데이터 — K-Decision 한국 인덱스 점수용.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def to_record(row, idx):
    sample = row.get("sample")
    if isinstance(sample, str):
        try:
            sample = json.loads(sample)
        except Exception:
            return None
    if not isinstance(sample, dict):
        return None
    state = sample.get("state")
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    if len(state) > 6000:
        return None
    q = sample.get("question") or {}
    instr = q.get("instructions") or q.get("instruction") or "결정하라"
    crit = q.get("criteria")
    target = sample.get("target")
    kind = row.get("type") or sample.get("type")
    if kind == "noul" or (isinstance(target, dict) and set(map(str.lower, map(str, target))) <= {"true", "false"}):
        qs = [{"id": "q0", "type": "noul", "text": instr}]
        t_true = 0.5
        if isinstance(target, dict):
            for k, v in target.items():
                if str(k).lower() == "true":
                    t_true = float(v)
        ans = "Y" if t_true >= 0.5 else "N"
    elif isinstance(crit, dict) and len(crit) >= 2 and isinstance(target, dict):
        keys = list(crit)
        weights0 = [float(target.get(k, 0) or 0) for k in keys]
        if sum(weights0) <= 0:
            return None
        gold = max(range(len(keys)), key=lambda i: weights0[i])
        # 옵션 8개 초과 시: 정답 포함 8개로 재구성
        if len(keys) > 8:
            others = [i for i in range(len(keys)) if i != gold][:7]
            rng = random.Random(idx)
            rng.shuffle(others)
            keep = sorted(others[:7] + [gold])
            keys = [keys[i] for i in keep]
            weights0 = [weights0[i] for i in keep]
            gold = keep.index(gold)
        opts = [str(crit[k]) or str(k) for k in keys]
        qs = [{"id": "q0", "type": "choice", "text": instr, "options": opts}]
        ans = "ABCDEFGH"[gold]
    else:
        return None
    return {"id": f"kjdi-{idx:06d}",
            "domain": f"kjdi:{row.get('area', '?')}",
            "state": state, "questions": qs, "answers": {"q0": ans}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kjdi_eval_test.jsonl")
    ap.add_argument("--sample", type=int, default=3000)
    args = ap.parse_args()

    files = sorted(glob.glob(
        "/home/kitri/.cache/huggingface/hub/datasets--bzantium--korean-jev-decision-index/"
        "snapshots/*/**/*.parquet", recursive=True))
    # default 서브셋 우선 (complete_cases/paired_* 제외)
    files = [f for f in files if "/default/" in f.replace("\\", "/")] or files
    print(f"parquet {len(files)}개: {[f.split('/')[-3] for f in files[:5]]}")
    recs, dropped = [], 0
    for fp in files:
        for row in pq.read_table(fp).to_pylist():
            r = to_record(row, len(recs) + dropped)
            if r:
                recs.append(r)
            else:
                dropped += 1
    files = [f for f in files if "paired_" not in f]  # default 서브셋만
    random.Random(21).shuffle(recs)
    recs = recs[: args.sample]
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"변환 {len(recs)} / 드롭 {dropped} → 저장 {out}")


if __name__ == "__main__":
    main()
