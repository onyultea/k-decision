"""격전지 훈련 데이터 대폭 확장 — 오류 분석 기반 정밀 공략.

핵심 전략:
1. 애매한 문제(disagree)의 gold 확률분포를 소프트라벨로 직접 학습
2. agent_trace_observability 도메인 집중 보강
3. 옵션 셔플 증강 (각 케이스 × 3 permuations)
4. tasksource에서 벤치 4도메인 유사 데이터 추가
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

import common as C


def load_parquet(split):
    f = glob.glob(f"/home/kitri/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/"
                  f"snapshots/*/all/{split}-00000-of-00001.parquet")[0]
    return pq.read_table(f).to_pylist()


def to_record(row, idx, rng, shuffle_options=False):
    gold = row["gold"]
    if isinstance(gold, str):
        gold = json.loads(gold)
    qs_raw = row["questions"]
    if isinstance(qs_raw, str):
        qs_raw = json.loads(qs_raw)
    if isinstance(qs_raw, dict):
        qs_raw = [dict(v, id=k) for k, v in qs_raw.items()]
    elif isinstance(qs_raw, list) and qs_raw and isinstance(qs_raw[0], str):
        qs_raw = [json.loads(q) for q in qs_raw]

    state = row["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)

    qs, answers = [], {}
    for q in qs_raw:
        qid = q["id"]
        kind = q.get("type") or gold.get(qid, {}).get("type")
        instr = q.get("instructions", "")
        crit = q.get("criteria")
        g = gold.get(qid, {})
        gp = g.get("probabilities", {})

        if kind == "choice" and isinstance(crit, dict):
            keys = list(crit.keys())
            descs = {k: str(crit[k]) for k in keys}

            # 옵션 셔플 증강
            if shuffle_options and len(keys) > 2:
                perm = list(range(len(keys)))
                rng.shuffle(perm)
                keys = [keys[i] for i in perm]

            opts = [f"{k}: {descs[k]}" if len(descs.get(k, "")) > 20 else str(k) for k in keys]
            label = g.get("label")
            if label not in keys and label not in descs:
                continue

            # gold 심볼
            orig_idx = list(crit.keys()).index(label) if label in crit else 0
            if shuffle_options:
                # 원본 인덱스 → 셔플된 인덱스
                new_idx = perm.index(orig_idx) if orig_idx < len(perm) else orig_idx
            else:
                new_idx = orig_idx

            qd = {"id": qid, "type": "choice", "text": instr, "options": opts}
            # 소프트라벨: gold 확률분포를 옵션 순서에 맞게 재배열
            if gp and abs(sum(gp.values()) - 1) < 0.1:
                if shuffle_options:
                    soft = [float(gp.get(list(crit.keys())[p], 0)) for p in perm]
                else:
                    soft = [float(gp.get(k, 0)) for k in crit.keys()]
                if min(soft) >= 0 and max(soft) <= 1:
                    qd["soft"] = soft
            qs.append(qd)
            answers[qid] = "ABCDEFGH"[new_idx]

        elif kind == "noul":
            qd = {"id": qid, "type": "noul", "text": instr}
            qs.append(qd)
            label = g.get("label")
            if label is not None:
                answers[qid] = "Y" if str(label).lower() in ("true", "yes", "1") else "N"
            else:
                pt = g.get("noul", 0.5)
                answers[qid] = "Y" if pt > 0.5 else "N"

        elif kind == "score" and isinstance(crit, list):
            opts = [str(c) for c in crit]
            label = g.get("label")
            matched = int(label) if label and str(label).isdigit() and int(label) < len(opts) else None
            if matched is None:
                continue
            qd = {"id": qid, "type": "choice", "text": instr + " (수준 선택)", "options": opts}
            if gp and abs(sum(gp.values()) - 1) < 0.1:
                qd["soft"] = [float(gp.get(str(i), 0)) for i in range(len(opts))]
            qs.append(qd)
            answers[qid] = "ABCDEFGH"[matched]

    if len(qs) < 2:
        return None
    return {"id": f"aug-{idx:05d}", "domain": "typeddec_aug",
            "state": state, "questions": qs, "answers": answers}


def main():
    rng = random.Random(42)
    train_rows = load_parquet("train")

    # 1) 원본 + 옵션 셔플 × 3 (총 4배)
    all_records = []
    for i, row in enumerate(train_rows):
        # 원본
        r = to_record(row, i, rng, shuffle_options=False)
        if r:
            all_records.append(r)
        # 셔플 3번
        for s in range(3):
            r = to_record(row, len(all_records) + s * 10000 + i, rng, shuffle_options=True)
            if r:
                all_records.append(r)

    print(f"격전지 증강: 원본 {len(train_rows)} → {len(all_records)} 레코드")

    n_soft = sum(1 for r in all_records for q in r["questions"] if "soft" in q)
    print(f"소프트라벨 질문: {n_soft}")

    # 2) 기존 믹스와 합성
    from common import load_jsonl
    base_mix = load_jsonl("data/mix_v09_train.jsonl")

    # typeddec 원본을 제거하고 증강버전으로 교체
    base_no_tdc = [r for r in base_mix if r.get("domain") != "typeddec"]
    print(f"기존 믹스에서 typeddec 제거: {len(base_mix)} → {len(base_no_tdc)}")

    mix = base_no_tdc + all_records
    rng.shuffle(mix)

    from common import dump_jsonl
    dump_jsonl(mix, "data/mix_v15_train.jsonl")
    print(f"v15 믹스: {len(mix)} 레코드")


if __name__ == "__main__":
    main()
