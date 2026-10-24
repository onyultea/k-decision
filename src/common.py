"""K-Decision 공통 모듈: 레코드 스키마, 프롬프트 빌더, 마커/심볼 처리.

레코드 형식 (JSONL 1행):
{
  "id": "...",
  "domain": "public|medical|cs|agent",
  "state": "상태를 서술하는 한국어 텍스트",
  "questions": [
    {"id": "q0", "type": "choice", "text": "...", "options": ["...","..."]},
    {"id": "q1", "type": "noul",  "text": "..."}
  ],
  "answers": {"q0": "B", "q1": "Y"}
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# ── 마커 ────────────────────────────────────────────────────────────────────
# 학습 시 어시스턴트 응답 스킬레톤의 결정 슬롯에 들어가는 토큰.
# 단일 토큰인 것을 런타임에 검증하며, 후보 중 컨텍스트 안에서 단일 토큰으로
# 잘리는 것을 우선 사용한다.
MARKER_CANDIDATES = ["□", "▪", "◆", "◇", "■", "@", "|", "~", "^", "$", "%"]  # 폴백 ASCII 포함(Llama계 vocab 대비)
MARKER = "□"  # resolve_marker() 가 확정

CHOICE_SYMBOLS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # 최대 26옵션 (학습데이터는 ≤8 사용)
NOUL_SYMBOLS = ["Y", "N"]
NOUL_LABELS = {"Y": "예", "N": "아니오"}

INSTR_KO = (
    "당신은 한국어 의사결정 엔진이다. [상태]와 [질문]들을 읽고, 각 질문의 "
    "정답 심볼을 예측하라. 질문 유형: choice(단일선택), noul(예/아니오). "
    "설명 없이 각 필드의 심볼을 결정하며, 모든 필드는 한 번에 판단한다."
)


def resolve_marker(tokenizer) -> str:
    """컨텍스트(실제 스키마 문자열) 안에서 단일 토큰으로 잘리는 마커를 반환.

    주의: 콜론 뒤 공백이 있으면 '공백+마커'가 병합 토큰이 되므로 스키마은
    반드시 무공백 형식('{"q0":□,"q1":□}') 이어야 한다.
    """
    global MARKER
    for cand in MARKER_CANDIDATES:
        skeleton = '{"q0":%s,"q1":%s}' % (cand, cand)
        ids = tokenizer.encode(skeleton, add_special_tokens=False)
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(cand_ids) == 1 and ids.count(cand_ids[0]) == 2:
            MARKER = cand
            return cand
    raise RuntimeError("단일 토큰 마커를 찾지 못함. MARKER_CANDIDATES 확장 필요.")


def symbol_ids(tokenizer, symbols):
    """심볼들이 모두 단일 토큰인지 검증하고 id 리스트를 반환."""
    out = []
    for s in symbols:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"심볼 {s!r}이(가) 단일 토큰이 아님: {ids}")
        out.append(ids[0])
    return out


def field_symbols(q: dict):
    if q["type"] == "choice":
        n = len(q["options"])
        assert 2 <= n <= len(CHOICE_SYMBOLS), f"choice 옵션 수 오류: {n}"
        return CHOICE_SYMBOLS[:n]
    if q["type"] == "noul":
        return NOUL_SYMBOLS
    raise ValueError(f"지원하지 않는 질문 타입: {q['type']}")


def gold_symbol(rec: dict, q: dict) -> str:
    ans = rec["answers"][q["id"]]
    assert ans in field_symbols(q), f"금드 심볼 오류: {q['id']}={ans}"
    return ans


def build_user_content(rec: dict) -> str:
    lines = [INSTR_KO, "", "[상태]", rec["state"], "", "[질문]"]
    for q in rec["questions"]:
        if q["type"] == "choice":
            pairs = ", ".join(
                f"{s}={o}" for s, o in zip(field_symbols(q), q["options"])
            )
            lines.append(f"- {q['id']} (choice) {q['text']} 옵션: {pairs}")
        else:  # noul
            pairs = ", ".join(f"{s}={l}" for s, l in NOUL_LABELS.items())
            lines.append(f"- {q['id']} (noul) {q['text']} 옵션: {pairs}")
    lines.append("")
    lines.append("[결정] 각 필드 심볼을 JSON으로 답하라.")
    return "\n".join(lines)


def build_assistant_skeleton(rec: dict) -> str:
    # 무공백 JSON — 마커가 주변 토큰과 병합되지 않는 형식(resolve_marker 참조)
    slots = ",".join(f'"{q["id"]}":{MARKER}' for q in rec["questions"])
    return "{" + slots + "}"


def build_eval_candidate_ids(tokenizer, rec: dict):
    """필드별 제약 디코딩에 쓸 후보 토큰 id. field_symbols 와 순서 일치."""
    return [symbol_ids(tokenizer, field_symbols(q)) for q in rec["questions"]]


# JSONL 입출력은 io_jsonl 모듈로 이관 (경로 검증 내장)
from io_jsonl import dump_jsonl, load_jsonl  # noqa: F401,E402
