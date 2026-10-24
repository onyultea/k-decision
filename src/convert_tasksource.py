"""tasksource/tasksource-jev-typed-decisions(full/) → K-Decision 포맷 변환 (서버 실행).

열: state, kind(choice|score|noul), options(리스트), target(원-핫), question, ...
- choice/score → choice (options 를 선택지로)
- noul → noul (options 2개 가정, 첫번째가 yes 계열인지 확인)
- 긴 state(>6000자) 스킵, 단일 필드 레코드로 출력
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

MAX_STATE = 6000


def convert():
    files = sorted(glob.glob(
        "/home/kitri/.cache/huggingface/hub/datasets--tasksource--tasksource-jev-typed-decisions/"
        "snapshots/*/full/train-*.parquet"))
    appends = sorted(glob.glob(
        "/home/kitri/.cache/huggingface/hub/datasets--tasksource--tasksource-jev-typed-decisions/"
        "snapshots/*/full/train-append-*.parquet"))
    files += appends
    print(f"parquet {len(files)}개")
    recs = []
    skipped = long_skipped = 0
    for fp in files:
        for row in pq.read_table(fp).to_pylist():
            state = row.get("state")
            kind = row.get("kind")
            options = row.get("options") or []
            target = row.get("target") or []
            question = row.get("question") or "결정하라"
            if not isinstance(state, str):
                state = json.dumps(state, ensure_ascii=False)
            if len(state) > MAX_STATE:
                long_skipped += 1
                continue
            if not options or len(options) < 2 or len(target) != len(options):
                skipped += 1
                continue
            gold = max(range(len(target)), key=lambda i: target[i] or 0)
            if kind == "noul" and len(options) == 2:
                qs = [{"id": "q0", "type": "noul", "text": str(question)}]
                # target 원-핫: [1,0] 이면 첫옵션=정답 — 통상 [yes,no] 순서
                ans = "Y" if gold == 0 else "N"
            elif kind in ("choice", "score"):
                opts = [str(o)[:120] for o in options]
                if len(opts) > 8:
                    # 8개 초과는 상위 정답포함 8개로 재구성(정답 유지)
                    others = [o for i, o in enumerate(opts) if i != gold][:7]
                    pos = random.randrange(8)
                    opts = others[:pos] + [opts[gold]] + others[pos:]
                    gold = pos
                qs = [{"id": "q0", "type": "choice", "text": str(question), "options": opts}]
                ans = "ABCDEFGH"[gold]
            else:
                skipped += 1
                continue
            recs.append({"id": f"ts-{len(recs):06d}", "domain": "tasksource",
                         "state": state, "questions": qs, "answers": {"q0": ans}})
    print(f"변환 {len(recs)} / 스킵 {skipped} / 장문스킵 {long_skipped}")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tasksource_train.jsonl")
    ap.add_argument("--sample", type=int, default=35000)
    args = ap.parse_args()
    recs = convert()
    random.Random(11).shuffle(recs)
    recs = recs[: args.sample]
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {out} ({len(recs)}레코드)")


if __name__ == "__main__":
    main()
