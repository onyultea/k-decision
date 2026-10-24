"""추가 학습 풀 변환 (서버 실행):
1) jaeyong2/jev-decision-claude-distill (Apache-2.0): 한국어 행 우선 샘플 → 학습 레코드
   (questions 구조: instruction/options/descriptions/label/teacher_probs)
2) Emulated-Inc/tool-decision-training-pool (CC-BY-4.0): 선호쌍 → '응답 유형' choice 레코드
   (chosen_response 가 TOOLCALL 이면 도구호출, 질문형이면 정보요청, 거부면 기각, 아니면 직접답변)
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

MAX_STATE = 6000


# ── claude-distill ────────────────────────────────────────────────────────────
def _parse_distill_questions(raw):
    """questions 열: JSON 문자열(또는 dict)의 리스트 → 파싱된 dict 리스트."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for q in raw or []:
        if isinstance(q, str):
            try:
                q = json.loads(q)
            except Exception:
                continue
        if isinstance(q, dict):
            out.append(q)
    return out


def conv_distill_row(row, idx):
    lang = row.get("lang") or ""
    state = row.get("state")
    if not isinstance(state, str) or len(state) > MAX_STATE:
        return None
    qs, answers = [], {}
    parsed = _parse_distill_questions(row.get("questions"))
    for qi, q in enumerate(parsed):
        instr = str(q.get("instruction") or "결정하라")
        options = q.get("options") or []
        descs = q.get("descriptions") or []
        label = q.get("label")
        if not options or len(options) < 2 or label is None or not (0 <= int(label) < len(options)):
            continue
        gold = int(label)
        # 소프트 타깃: teacher_probs {선생: [옵션정렬 확률]} → 선생 평균
        tp = q.get("teacher_probs") or {}
        prob_lists = [v for v in tp.values()
                      if isinstance(v, list) and len(v) == len(options)]
        if prob_lists:
            soft = [sum(p[i] for p in prob_lists) / len(prob_lists)
                    for i in range(len(options))]
            if min(soft) < 0 or max(soft) > 1:
                soft = None
        else:
            soft = None
        # 옵션 표기: 설명이 있으면 '라벨: 설명' 결합, 아니면 라벨만
        opts = []
        for i, o in enumerate(options):
            o = str(o)[:60]
            d = str(descs[i])[:110] if i < len(descs) else ""
            opts.append(f"{o}: {d}" if d else o)
        soft_kept = None
        if len(opts) > 8:  # 정답 포함 8개로 축소
            others = [i for i in range(len(opts)) if i != gold][:7]
            import random as _r
            _r.Random(idx * 7 + qi).shuffle(others)
            keep = sorted(others[:7] + [gold])
            opts = [opts[i] for i in keep]
            if soft is not None:
                soft_kept = [soft[i] for i in keep]
            gold = keep.index(gold)
        elif soft is not None:
            soft_kept = soft
        qid = f"q{qi}"
        qd = {"id": qid, "type": "choice", "text": instr, "options": opts}
        if soft_kept is not None and abs(sum(soft_kept) - 1) < 0.05:
            qd["soft"] = soft_kept
        qs.append(qd)
        answers[qid] = "ABCDEFGH"[gold]
    if not qs:
        return None
    return {"id": f"cd-{idx:06d}", "domain": f"distill:{lang}",
            "state": state, "questions": qs, "answers": answers}


def convert_distill(max_ko, max_other):
    files = sorted(glob.glob(
        "/home/kitri/.cache/huggingface/hub/datasets--jaeyong2--jev-decision-claude-distill/"
        "snapshots/*/**/*.parquet", recursive=True))
    ko, other = [], []
    for fp in files:
        for i, row in enumerate(pq.read_table(fp).to_pylist()):
            r = conv_distill_row(row, i)
            if not r:
                continue
            if "ko" in r["domain"]:
                ko.append(r)
            else:
                other.append(r)
    random.Random(31).shuffle(ko)
    random.Random(32).shuffle(other)
    out = ko[:max_ko] + other[:max_other]
    print(f"distill: 한국어 {len(ko)}중 {min(max_ko,len(ko))}, 기타 {len(other)}중 {min(max_other,len(other))}")
    return out


# ── tool-decision pool ────────────────────────────────────────────────────────
def classify(chosen: str) -> str:
    if "<TOOLCALL>" in chosen:
        return "도구호출"
    s = chosen.strip().lower()
    if s.startswith(("clarify", "could you", "can you provide", "which", "what is the")) or "?" in chosen[:120]:
        return "정보요청"
    if any(w in s[:60] for w in ("cannot", "can't", "unable", "i don't", "no suitable")):
        return "기각(도구없음)"
    return "직접답변"


def _parse_tools(tools):
    out = []
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            return []
    if isinstance(tools, dict):
        tools = list(tools.values())
    for t in tools or []:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except Exception:
                out.append({"name": t[:40], "description": ""})
                continue
        if isinstance(t, dict):
            out.append(t)
    return out


def conv_tool_row(row, idx):
    tools = _parse_tools(row.get("tools"))
    msgs = row.get("messages") or []
    if not msgs:
        return None
    user = ""
    for m in msgs:
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except Exception:
                continue
        if m.get("role") == "user":
            user = m.get("content", "")
            if not isinstance(user, str):
                user = json.dumps(user, ensure_ascii=False)
            break
    if not user:
        return None
    tool_desc = "; ".join(
        f"{t.get('name','?')}: {str(t.get('description',''))[:80]}" for t in tools[:8])
    state = f"사용가능 도구: {tool_desc or '없음'}\n사용자 요청: {str(user)[:800]}"
    if len(state) > MAX_STATE:
        return None
    opts = ["도구호출", "정보요청", "직접답변", "기각(도구없음)"]
    gold = classify(str(row.get("chosen_response") or ""))
    return {"id": f"tool-{idx:06d}", "domain": "tooldec",
            "state": state,
            "questions": [{"id": "q0", "type": "choice",
                           "text": "에이전트의 올바른 응답 유형은?", "options": opts}],
            "answers": {"q0": "ABCDEFGH"[opts.index(gold)]}}


def convert_tools(max_n):
    files = sorted(glob.glob(
        "/home/kitri/.cache/huggingface/hub/datasets--Emulated-Inc--tool-decision-training-pool/"
        "snapshots/*/**/*.parquet", recursive=True))
    # json 원본이면 parquet 자동변환 파일이 없을 수 있음 → json 대응
    if not files:
        files = sorted(glob.glob(
            "/home/kitri/.cache/huggingface/hub/datasets--Emulated-Inc--tool-decision-training-pool/"
            "snapshots/*/**/*.json", recursive=True))
        recs = []
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    r = conv_tool_row(row, i)
                    if r:
                        recs.append(r)
        random.Random(33).shuffle(recs)
        print(f"tool(json): {len(recs)}")
        return recs[:max_n]
    recs = []
    for fp in files:
        for i, row in enumerate(pq.read_table(fp).to_pylist()):
            r = conv_tool_row(row, i)
            if r:
                recs.append(r)
    random.Random(33).shuffle(recs)
    print(f"tool(parquet): {len(recs)}")
    return recs[:max_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-distill", default="data/distill_train.jsonl")
    ap.add_argument("--out-tools", default="data/tooldec_train.jsonl")
    ap.add_argument("--ko", type=int, default=8000)
    ap.add_argument("--other", type=int, default=4000)
    ap.add_argument("--tools", type=int, default=6000)
    args = ap.parse_args()
    for out, recs in ((args.out_distill, convert_distill(args.ko, args.other)),
                      (args.out_tools, convert_tools(args.tools))):
        p = Path(out)
        with p.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"저장 {p}: {len(recs)}")


if __name__ == "__main__":
    main()
