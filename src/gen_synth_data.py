"""K-Decision-Data v0.1 — 온율티 사업영역(공공·의료·CS·에이전트) 합성 타이핑드 결정 데이터 생성.

도메인별 규칙 엔진에서 상태 문장을 생성하고, 규칙에서 금드 심볼을 결정한다.
학습/검증/평가 분할은 서로 다른 시드로 생성해 템플릿 반복을 줄인다.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import load_jsonl  # noqa: F401  (호환용)


# ── 공공(민원) ───────────────────────────────────────────────────────────────
CIVIL = {
    "부서": {
        "단전": ["전기과", "한국전력"], "단수": ["상하수도과", "한국수자원공사"],
        "가스": ["가스안전과", "도시가스사"], "도로파손": ["도로과", "도로관리사업소"],
        "쓰레기": ["환경위생과", "폐기물관리과"], "소음": ["환경과", "환경관리단"],
        "상수도누수": ["상하수도과", "수도사업소"], "가로등": ["도로과", "조명관리반"],
        "불법주차": ["교통지도과", "경찰서"], "국민연금": ["연금정책과", "국민연금공단"],
        "건축물철거": ["건축과", "허가민원실"], "재해지원": ["안전총괄과", "재난안전본부"],
        "홍수예경보": ["하천과", "재난상황실"],
    },
    "증상문장": {
        "단전": ["집에 전기가 완전히 나갔어요", "아파트 복도 전등이 다 꺼졌습니다",
                 "분전반에서 타는 냄새가 나요", "계량기가 멈췄어요"],
        "단수": ["수도꼭지에서 물이 안 나옵니다", "옥상 물탱크 물이 안 내려와요",
                 "동 전체가 물이 끊겼어요", "급수관 공사 후 물이 안 나옵니다"],
        "가스": ["가스 냄새가 심하게 나요", "보일러에 가스가 샙니다",
                 "도시가스 배관에서 치직거리는 소리가 나요"],
        "도로파손": ["아스팔트에 커다란 구멍이 생겼어요", "보도블럭이 파여 넘어질 뻔했어요",
                    "차도에 포트홀이 여러 개 있습니다"],
        "쓰레기": ["대형 쓰레기가 2주째 방치돼요", "무단 투기 쓰레기가 산더미예요",
                   "재활용 분리수거가 안 되고 있어요"],
        "소음": ["밤 12시에 공사 소음이 시끄러워요", "이웃집 노래방 소음이 심합니다",
                 "새벽에 배달 오토바이 소음이 커요"],
        "상수도누수": ["지하 주차장 천장에서 물이 뚝뚝 떨어져요", "도로에서 물이 솟고 있어요"],
        "가로등": ["골목 가로등이 일주일째 꺼져 있어요", "보안등이 고장나 밤에 어둡습니다"],
        "불법주차": ["소방차 진입로에 차가 주차돼 있어요", "교차로 모퉁이 불법주차가 심합니다"],
        "국민연금": ["연금 납부 내역이 이상해요", "노령 연금 지급일이 지났는데 안 왔어요"],
        "건축물철거": ["무허가 철거 공사가 진행되고 있어요", "철거 예정 통보 없이 공사를 시작했습니다"],
        "재해지원": ["태풍으로 지붕이 날아갔어요", "침수로 가전제품이 전부 망가졌습니다"],
        "홍수예경보": ["하천 수위가 급격히 올라가고 있어요", "하천 범람 위험 경보가 왔어요"],
    },
    "긴급키": {"가스", "단전", "단수", "홍수예경보", "재해지원", "불법주차"},
    "즉시처리키": {"가스", "단수", "홍수예경보", "도로파손", "불법주차"},
}
DEPTS_FLAT = sorted({d for v in CIVIL["부서"].values() for d in v})

# ── 의료 ────────────────────────────────────────────────────────────────────
MED = {
    "진료과": {
        "복통": ["소화기내과", "일반외과"], "흉통": ["순환기내과", "응급실"],
        "발열": ["감염내과", "소아과"], "골절": ["정형외과"], "두통": ["신경과"],
        "피부발진": ["피부과"], "치통": ["치과"], "안구통": ["안과"],
        "정신불안": ["정신건강의학과"], "요통": ["재활의학과", "정형외과"],
        "호흡곤란": ["호흡기내과", "응급실"], "식중독": ["소화기내과", "응급실"],
    },
    "증상문장": {
        "복통": ["배가 몹시 아파요", "오른쪽 아랫배가 찌르듯 아픕니다", "속이 뒤틀리듯 아파요"],
        "흉통": ["가슴이 조이듯 아파요", "왼쪽 가슴에 통증이 있어요", "가슴 통증과 식은땀이 나요"],
        "발열": ["38.9도 열이 나요", "사흘째 열이 안 떨어져요", "아이가 고열입니다"],
        "골절": ["계단에서 넘어져 팔이 부러진 것 같아요", "손목이 심하게 부어 움직일 수 없어요"],
        "두통": ["머리가 깨질 것 같아요", "편두통이 일주일째예요", "갑자기 극심한 두통이 왔어요"],
        "피부발진": ["온몸에 두드러기가 돋았어요", "발진이 가렵고 부어올라요"],
        "치통": ["이가 욱신거려 잠을 못 자요", "잇몸이 많이 부었어요"],
        "안구통": ["눈이 빠질 것처럼 아파요", "시야가 흐려지고 눈이 아파요"],
        "정신불안": ["공황 발작이 반복돼요", "일주일째 잠을 못 자고 불안해요"],
        "요통": ["허리가 펴지지 않아요", "방사통이 다리까지 내려가요"],
        "호흡곤란": ["숨을 쉬기 힘들어요", "호흡이 가쁘고 어지러워요"],
        "식중독": ["집단 식사 후 설사와 구토가 시작됐어요", "회 먹고 복통과 구토가 있어요"],
    },
    "응급키": {"흉통", "호흡곤란", "골절", "식중독"},
    "소아키": {"발열"},
}
DEPTS_MED_FLAT = sorted({d for v in MED["진료과"].values() for d in v})

# ── CS / 커머스 ─────────────────────────────────────────────────────────────
CS = {
    "인텐트": {
        "환불": ["CS-환불팀", "결제관리팀"], "교환": ["CS-교환팀", "품질관리팀"],
        "배송": ["CS-배송팀", "물류파트너"], "AS": ["CS-기술팀", "AS센터"],
        "회원": ["CS-회원팀", "계정관리팀"], "취소": ["CS-환불팀", "주문관리팀"],
    },
    "문장": {
        "환불": ["결제했는데 돈을 돌려받고 싶어요", "환불 처리가 언제 되나요",
                 "상품이 마음에 안 들어서 환불 요청합니다"],
        "교환": ["사이즈가 안 맞아 교환하고 싶어요", "다른 색상으로 바꾸고 싶어요",
                 "상품에 하자가 있어 교환 요청합니다"],
        "배송": ["배송이 5일째 소식이 없어요", "운송장이 조회되지 않아요",
                 "배송지를 변경하고 싶어요"],
        "AS": ["제품이 고장났어요", "보증 기간 안에 수리받고 싶어요",
               "기기가 켜지지 않아요"],
        "회원": ["비밀번호를 잊어버렸어요", "회원 탈퇴하고 싶어요",
                 "아이디 찾기를 하고 싶어요"],
        "취소": ["방금 주문 취소하고 싶어요", "주문을 취소해 주세요"],
    },
    "악성키": set(),  # 문구에서 판정
    "악성단서": ["욕설", "고소하겠다", "매스컴", "폭로", "즉시 답 안 주면"],
}
TEAMS_FLAT = sorted({t for v in CS["인텐트"].values() for t in v})

# ── 에이전트 게이팅 ─────────────────────────────────────────────────────────
AGENT = {
    "도구": {
        "검색": ["web_search", "portal_lookup"],
        "DB조회": ["sql_query", "crm_lookup"],
        "계산": ["calculator", "unit_convert"],
        "대형LLM": ["reasoning_llm", "deep_model"],
        "일정": ["calendar", "scheduler"],
    },
    "문장": {
        "검색": ["최신 뉴스 찾아줘", "오늘 춘천 날씨 알려줘", "이번 주 경기 일정 검색해줘"],
        "DB조회": ["우리 매장 매출 보여줘", "지난달 재고 현황 조회해줘", "고객 목록 좀 찾아줘"],
        "계산": ["부가세 계산해줘", "12% 할인하면 얼마야", "환율 곱해서 계산해줘"],
        "대형LLM": ["이 계약서 조항의 법적 리스크 분석해줘", "전략 수립 보고서 초안을 깊이 있게 작성해줘",
                    "복잡한 추론이 필요한 문제야, 단계별로 분석해줘"],
        "일정": ["내일 오전 회의 잡아줘", "다음 주 월요일 알림 설정해줘", "일정 비어있는 시간 보여줘"],
    },
    "즉답불가키": {"대형LLM", "DB조회"},  # 외부 리소스 필요 → 즉답 불가
}
TOOLS_FLAT = sorted({t for v in AGENT["도구"].values() for t in v})

HONOR = ["{s}", "{s}요", "{s}입니다", "{s}어요", "{s}네요", "{s}요~", "{s}!"]


def style(rng, s):
    t = rng.choice(HONOR)
    return t.format(s=s) if "{s}" in t else t


# ── 시나리오 생성기 ──────────────────────────────────────────────────────────
def gen_public(rng):
    key = rng.choice(list(CIVIL["부서"].keys()))
    sent = rng.choice(CIVIL["증상문장"][key])
    state = f"민원 접수: {style(rng, sent)} (접수경로: {rng.choice(['인터넷민원','콜센터','방문','이장실'])})"
    dept = rng.choice(CIVIL["부서"][key])
    others = rng.sample([d for d in DEPTS_FLAT if d != dept], 3)
    opts = [dept] + others
    rng.shuffle(opts)
    urgent = key in CIVIL["긴급키"]
    immediate = key in CIVIL["즉시처리키"]
    rec = {
        "domain": "public", "state": state,
        "questions": [
            {"id": "q_route", "type": "choice", "text": "담당 부서는?", "options": opts},
            {"id": "q_urgent", "type": "noul", "text": "긴급 처리 대상인가?"},
            {"id": "q_imm", "type": "noul", "text": "즉시 처리(당일) 대상인가?"},
        ],
        "answers": {"q_route": "ABCDEFG"[opts.index(dept)], "q_urgent": "Y" if urgent else "N",
                    "q_imm": "Y" if immediate else "N"},
    }
    return rec


def gen_medical(rng):
    key = rng.choice(list(MED["진료과"].keys()))
    sent = rng.choice(MED["증상문장"][key])
    state = f"환자 응대 채팅: {style(rng, sent)} (환자: {rng.choice(['30대 여성','50대 남성','8세 아동','70대 남성','20대 남성'])})"
    dept = rng.choice(MED["진료과"][key])
    others = rng.sample([d for d in DEPTS_MED_FLAT if d != dept], 3)
    opts = [dept] + others
    rng.shuffle(opts)
    emerg = key in MED["응급키"] or ("아동" in state and key in MED["소아키"])
    rec = {
        "domain": "medical", "state": state,
        "questions": [
            {"id": "q_dept", "type": "choice", "text": "1차 진료과/응대는?", "options": opts},
            {"id": "q_emerg", "type": "noul", "text": "응급실 우선 안내 대상인가?"},
            {"id": "q_child", "type": "noul", "text": "소아(만 12세 이하) 대상인가?"},
        ],
        "answers": {"q_dept": "ABCDEFG"[opts.index(dept)], "q_emerg": "Y" if emerg else "N",
                    "q_child": "Y" if "아동" in state else "N"},
    }
    return rec


def gen_cs(rng):
    key = rng.choice(list(CS["인텐트"].keys()))
    sent = rng.choice(CS["문장"][key])
    toxic = (rng.random() < 0.25) and rng.random() < 0.7
    if toxic:
        suffix = rng.choice([" (고객이 화를 내며 고소하겠다고 함)", " (욕설 포함, 매스컴 폭로 언급)"])
        state = f"고객 문의: {style(rng, sent)}{suffix}"
    else:
        state = f"고객 문의: {style(rng, sent)}"
    team = rng.choice(CS["인텐트"][key])
    others = rng.sample([t for t in TEAMS_FLAT if t != team], 3)
    opts = [team] + others
    rng.shuffle(opts)
    reasons = ["단순변심", "상품불량", "배송지연", "오주문", "가격변동"]
    rec = {
        "domain": "cs", "state": state,
        "questions": [
            {"id": "q_intent", "type": "choice", "text": "라우팅 팀은?", "options": opts},
            {"id": "q_toxic", "type": "noul", "text": "에스컬레이션(악성/법적 대응) 대상인가?"},
            {"id": "q_reason", "type": "choice", "text": "가장 가능성 높은 사유는?",
             "options": rng.sample(reasons, 4) if key in ("환불", "교환", "취소")
             else ["해당없음", "단순변심", "상품불량", "배송지연"]},
        ],
        "answers": {},  # 아래에서 채움
    }
    if key in ("환불", "교환", "취소"):
        gold_reason = "상품불량" if "하자" in state or "고장" in state else (
            "배송지연" if "늦" in state or "5일" in state else "단순변심")
        if gold_reason not in rec["questions"][2]["options"]:
            rec["questions"][2]["options"][0] = gold_reason
        rec["answers"] = {
            "q_intent": "ABCDEFG"[opts.index(team)],
            "q_toxic": "Y" if toxic else "N",
            "q_reason": "ABCDEFG"[rec["questions"][2]["options"].index(gold_reason)],
        }
    else:
        rec["answers"] = {"q_intent": "ABCDEFG"[opts.index(team)],
                          "q_toxic": "Y" if toxic else "N",
                          "q_reason": "A"}  # '해당없음' 인덱스는 옵션 순서에 따름
        if rec["questions"][2]["options"][0] != "해당없음":
            rec["questions"][2]["options"] = ["해당없음", "단순변심", "상품불량", "배송지연"]
    return rec


def gen_agent(rng):
    key = rng.choice(list(AGENT["도구"].keys()))
    sent = rng.choice(AGENT["문장"][key])
    state = f"에이전트 지시: {style(rng, sent)}"
    tool = rng.choice(AGENT["도구"][key])
    others = rng.sample([t for t in TOOLS_FLAT if t != tool], 3)
    opts = [tool] + others
    rng.shuffle(opts)
    rec = {
        "domain": "agent", "state": state,
        "questions": [
            {"id": "q_tool", "type": "choice", "text": "호출할 도구는?", "options": opts},
            {"id": "q_defer", "type": "noul", "text": "대형 추론 모델 지연(defer) 대상인가?"},
            {"id": "q_local", "type": "noul", "text": "온디바이스(로컬) 처리 가능한가?"},
        ],
        "answers": {"q_tool": "ABCDEFG"[opts.index(tool)],
                    "q_defer": "Y" if key == "대형LLM" else "N",
                    "q_local": "N" if key in AGENT["즉답불가키"] else "Y"},
    }
    return rec


GENS = [gen_public, gen_medical, gen_cs, gen_agent]


def make_split(n, seed):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        rec = GENS[i % len(GENS)](rng)
        rec["id"] = f"{rec['domain']}-{seed}-{i:05d}"
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--test", type=int, default=800)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": make_split(args.train, 1001),
        "val": make_split(args.val, 2002),
        "test": make_split(args.test, 3003),
    }
    from common import dump_jsonl
    for name, recs in splits.items():
        p = out / f"synth_v01_{name}.jsonl"
        dump_jsonl(recs, p.as_posix())
        n_fields = sum(len(r["questions"]) for r in recs)
        print(f"{p}: {len(recs)}레코드 / {n_fields}결정")


if __name__ == "__main__":
    main()
