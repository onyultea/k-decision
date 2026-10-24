"""Decider-4B 스타일 데이터 다양화 — 1위 레시피에서 추출한 핵심 5종.

1. 대조쌍 (contrastive): 기존 데이터에서 팩트 하나 뒤집기 → 답 변경
2. 룰 기반 (rules): 검증 가능한 규칙 조건부 판단
3. JSON state 다양화: 다양한 구조/크기의 JSON 입력
4. 단일질문 분리: 멀티질문을 개별 질문으로
5. 옵션 패딩: 소규모 라벨을 무관 라벨로 확장 (up to 20)
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from common import load_jsonl, dump_jsonl

rng = random.Random(2026)

# ── 1. 대조쌍 (contrastive pairs) ──────────────────────────────────────────

CONTRAST_FLIPS = {
    "긴급": [("긴급", "일반"), ("당일", "내일"), ("즉시", "여유")],
    "환불": [("환불", "교환"), ("가능", "불가능")],
    "위반": [("위반", "적합"), ("초과", "이내"), ("초과", "미달")],
    "예": [("예", "아니오"), ("Y", "N")],
}

def make_contrastive(recs, n_target):
    """기존 레코드에서 noul 질문의 정답을 뒤집고 state의 핵심 단어도 뒤집기."""
    out = []
    noul_recs = [r for r in recs if any(q["type"] == "noul" for q in r["questions"])]
    rng.shuffle(noul_recs)
    for rec in noul_recs:
        if len(out) >= n_target:
            break
        r2 = json.loads(json.dumps(rec))
        state = r2["state"]
        flipped = False
        for word, antonyms in CONTRAST_FLIPS.items():
            for orig, antonym in antonyms:
                if orig in state:
                    state = state.replace(orig, antonym, 1)
                    for q in r2["questions"]:
                        if q["type"] == "noul":
                            old = r2["answers"][q["id"]]
                            r2["answers"][q["id"]] = "N" if old == "Y" else "Y"
                    flipped = True
                    break
            if flipped:
                break
        if not flipped:
            # 수치 뒤집기: "21학점" → "15학점"
            nums = re.findall(r"(\d{1,3})\s*(학점|년|개월|%|명)", state)
            if nums:
                old_num, unit = nums[0]
                new_num = str(max(1, int(old_num) - 5))
                state = state.replace(f"{old_num}{unit}", f"{new_num}{unit}", 1)
                for q in r2["questions"]:
                    if q["type"] == "noul":
                        old = r2["answers"][q["id"]]
                        r2["answers"][q["id"]] = "N" if old == "Y" else "Y"
                flipped = True
        if flipped:
            r2["state"] = state
            r2["id"] = f"ctr-{len(out):05d}"
            r2["domain"] = f"contrast:{rec.get('domain','?')}"
            out.append(r2)
    return out

# ── 2. 룰 기반 (programmatic rules) ────────────────────────────────────────

RULE_TEMPLATES = [
    ("予算", "예산", "한도", ["초과", "이내"], lambda x, limit: x > limit),
    ("기한", "기한", "일", ["지남", "이내"], lambda x, limit: x > limit),
    ("횟수", "횟수", "회", ["초과", "미달"], lambda x, limit: x > limit),
    ("인원", "정원", "명", ["초과", "이내"], lambda x, limit: x > limit),
    ("점수", "기준점", "점", ["미달", "통과"], lambda x, limit: x < limit),
]

def make_rules(n_target):
    """검증 가능한 규칙 조건부 판단 — 숫자와 조건으로 정답이 결정됨."""
    out = []
    subjects = ["프로젝트", "구매 신청", "휴가 신청", "시스템 접근", "보고서 제출",
                "교육 수강", "연장 근로", "출장", "예산 집행", "계약 연장"]
    for i in range(n_target):
        rule_name, rule_ko, unit, outcomes, check = rng.choice(RULE_TEMPLATES)
        limit = rng.choice([3, 5, 7, 10, 14, 20, 30, 50, 70, 100])
        actual = limit + rng.choice([-2, -1, 0, 0, 1, 2, rng.randint(3, 10)])
        subj = rng.choice(subjects)
        state = (f"[규칙] {rule_ko} {rule_name}은(는) {limit}{unit}까지 허용됩니다.\n"
                 f"[상황] {subj} 요청값은 {actual}{unit}입니다.")
        violated = check(actual, limit)
        qs = [
            {"id": "q_rule", "type": "noul", "text": f"이 요청은 규칙을 {outcomes[0]}합니까?"},
            {"id": "q_margin", "type": "noul", "text": f"여유가 있습니까? (요청 < 한도)"},
            {"id": "q_action", "type": "choice", "text": "적절한 조치는?",
             "options": ["승인", "조건부 승인", "반려", "상급 확인"]},
        ]
        answers = {
            "q_rule": "Y" if violated else "N",
            "q_margin": "N" if actual >= limit else "Y",
            "q_action": "C" if violated else ("B" if actual > limit * 0.8 else "A"),
        }
        # 상급 확인: 초과분이 크면
        if violated and actual > limit * 1.5:
            answers["q_action"] = "D"
        out.append({"id": f"rule-{i:05d}", "domain": "rules",
                    "state": state, "questions": qs, "answers": answers})
    return out

# ── 3. JSON state 다양화 ────────────────────────────────────────────────────

def make_json_states(n_target):
    """다양한 구조의 JSON state에서 판단 — 에이전트 트레이스 도메인 보강."""
    tools = ["web_search", "sql_query", "calculator", "calendar", "mail_send",
             "file_read", "api_call", "human_review"]
    statuses = ["success", "failed", "timeout", "partial", "pending"]
    out = []
    for i in range(n_target):
        steps = rng.randint(2, 8)
        tool_calls = []
        for s in range(steps):
            tool_calls.append({
                "step": s + 1,
                "tool": rng.choice(tools),
                "status": rng.choice(statuses),
                "duration_ms": rng.randint(50, 3000),
            })
        errors = sum(1 for c in tool_calls if c["status"] in ("failed", "timeout"))
        state = json.dumps({
            "task": rng.choice(["보고서 생성", "데이터 조회", "메일 발송",
                                "일정 관리", "파일 처리", "API 통합"]),
            "agent_version": "v1.2",
            "trace": tool_calls,
            "constraints": [f"max_{rng.choice(['time','cost','steps'])}_{rng.randint(1,10)}"],
        }, ensure_ascii=False)
        qs = [
            {"id": "q_health", "type": "noul", "text": "전체 워크플로우가 성공했습니까?"},
            {"id": "q_errors", "type": "choice", "text": "오류 수준은?",
             "options": ["없음", "1~2개", "3개 이상"]},
            {"id": "q_retry", "type": "noul", "text": "재시도가 권장됩니까?"},
        ]
        error_level = "A" if errors == 0 else ("B" if errors <= 2 else "C")
        answers = {
            "q_health": "Y" if errors == 0 else "N",
            "q_errors": error_level,
            "q_retry": "Y" if errors > 0 and rng.random() < 0.7 else "N",
        }
        out.append({"id": f"json-{i:05d}", "domain": "json_state",
                    "state": state, "questions": qs, "answers": answers})
    return out

# ── 4. 단일질문 분리 ────────────────────────────────────────────────────────

def make_single_question(recs, n_target):
    """멀티질문 레코드를 개별 질문으로 분리 — 각 질문의 독립적 판단력 강화."""
    out = []
    multi = [r for r in recs if len(r["questions"]) >= 2]
    rng.shuffle(multi)
    for rec in multi:
        if len(out) >= n_target:
            break
        for q in rec["questions"]:
            if len(out) >= n_target:
                break
            out.append({
                "id": f"single-{len(out):05d}",
                "domain": f"single:{rec.get('domain','?')}",
                "state": rec["state"],
                "questions": [q],
                "answers": {q["id"]: rec["answers"][q["id"]]},
            })
    return out

# ── 5. 옵션 패딩 (소규모 → 대규모) ─────────────────────────────────────────

PAD_LABELS = ["관련없음", "해당사항 없음", "기타", "미분류", "불명확",
              "추가 확인 필요", "외부 문의", "자동 처리", "수동 검토", "보류",
              "긴급 아니함", "일반 처리", "최저 우선순위", "취소", "연기",
              "부분 승인", "조건부", "임시", "테스트", "드라이런"]

def make_padded(recs, n_target):
    """choice 질문의 옵션에 무관 라벨을 추가해 선택지 폭을 넓힘."""
    out = []
    choice_recs = [r for r in recs if any(q["type"] == "choice" and len(q["options"]) <= 6
                                           for q in r["questions"])]
    rng.shuffle(choice_recs)
    for rec in choice_recs:
        if len(out) >= n_target:
            break
        r2 = json.loads(json.dumps(rec))
        for q in r2["questions"]:
            if q["type"] == "choice" and len(q["options"]) <= 6:
                orig_len = len(q["options"])
                gold_idx = "ABCDEFGH".index(r2["answers"][q["id"]])
                # 무관 라벨 추가
                n_pad = min(rng.randint(2, 14), 20 - orig_len)
                pads = rng.sample(PAD_LABELS, n_pad)
                # 정답 위치 무작위 배치
                new_opts = q["options"][:] + pads
                rng.shuffle(new_opts)
                new_gold = new_opts.index(q["options"][gold_idx])
                q["options"] = new_opts
                r2["answers"][q["id"]] = "ABCDEFGHIJKLMNOPQRST"[new_gold]
        r2["id"] = f"pad-{len(out):05d}"
        r2["domain"] = f"padded:{rec.get('domain','?')}"
        out.append(r2)
    return out

# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    all_train = load_jsonl("data/mix_v09_train.jsonl")
    typeddec_train = load_jsonl("data/typeddec_train.jsonl")
    base = all_train + typeddec_train

    parts = {
        "contrastive": make_contrastive(base, 8000),
        "rules": make_rules(15000),
        "json_state": make_json_states(10000),
        "single_question": make_single_question(base, 15000),
        "padded": make_padded(base, 8000),
    }

    print("=== 생성 결과 ===")
    total_new = 0
    for name, recs in parts.items():
        n_dec = sum(len(r["questions"]) for r in recs)
        print(f"  {name:20s} {len(recs):6,} 레코드 {n_dec:6,} 결정")
        total_new += len(recs)

    # 기존 v09 믹스에 추가
    mix = all_train + typeddec_train
    for recs in parts.values():
        mix.extend(recs)
    rng.shuffle(mix)
    print(f"\nv16 믹스: 기존 {len(all_train)+len(typeddec_train):,} + 신규 {total_new:,} = {len(mix):,}")
    dump_jsonl(mix, "data/mix_v16_train.jsonl")

if __name__ == "__main__":
    main()
