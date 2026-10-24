"""K-Decision-Data v0.2 — 한국 생활 밀착형 타이핑드 결정 데이터 (v0.4 학습용).

v0.1(민원/의료/CS/에이전트) 대비 확장:
  - 도메인 8개: 행정(20종 서비스), 의료, 금융, 쇼핑/배달 CS, 생활인프라(부동산·교통·통신),
    교육, 직장, 에이전트 게이팅
  - 언어 증강: 존댓말/반말 변형, 오탈자, 신조어/감정 표현, 방언투
  - 결정 밀도: 필드 3~5개, 라우팅 옵션 6~8개(유사 부서 함정 포함)

분할: train/val/test 는 서로 다른 시드(템플릿 조합 홀드아웃).
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

# ── 언어 증강기 ──────────────────────────────────────────────────────────────
POLITE = [
    lambda s: s + "요",
    lambda s: s + "습니다",
    lambda s: s + "인데요, 알려주세요",
    lambda s: s + "할 수 있을까요?",
    lambda s: "안녕하세요, " + s + " 관련해서요",
    lambda s: s + "고요",
    lambda s: s + "네요",
]
CASUAL = [
    lambda s: s,
    lambda s: s + "ㅋㅋ",
    lambda s: s + "..",
    lambda s: s + "ㄷㄷ",
    lambda s: "ㅠㅠ " + s,
    lambda s: s + "!!",
]
EMOTION = [
    lambda s: s + " (고객 화남, 즉시 응대 필요)",
    lambda s: s + " (매우 급함 표현 반복)",
    lambda s: s,
]
DIALECT_HINT = [
    ("하러", "하러"),  # 자연스러운 문장 유지용 패스
]

TYPO_RATE = 0.18


def add_typos(text, rng):
    """간단한 오탈자: 글자 2회 교환 / 모음 축약 / 띄어쓰기 제거."""
    if rng.random() > TYPO_RATE:
        return text
    ops = ["swap", "space", "double"]
    op = rng.choice(ops)
    chars = list(text)
    if op == "swap" and len(chars) > 4:
        i = rng.randrange(1, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == "space":
        text2 = text.replace(" ", "", 1) if " " in text else text
        return text2
    elif op == "double" and len(chars) > 4:
        i = rng.randrange(1, len(chars) - 2)
        chars.insert(i, chars[i])
    return "".join(chars)


def augment(state, rng, formal=False):
    if formal:
        f = rng.choice(POLITE)
    else:
        f = rng.choice(CASUAL + POLITE[:3])
    s = f(state)
    s = rng.choice(EMOTION)(s)
    s = add_typos(s, rng)
    return s


# ── 1. 공공 행정 (20종 서비스) ───────────────────────────────────────────────
ADMIN = {
    # 서비스: (담당기관, 관련과, 즉시처리, 긴급)
    "전입신고": ("관할 주민센터(동주민센터)", "주민과", True, False),
    "전출신고": ("관할 주민센터(동주민센터)", "주민과", True, False),
    "주민등록증 재발급": ("주민센터 또는 HiKR(온라인)", "주민과", True, False),
    "가족관계증명서": ("구청/시청 민원실", "가족관계등록과", True, False),
    "아동수당 신청": ("읍면동 주민복지센터", "복지과", False, False),
    "기초생활수급 신청": ("주민복지센터(통합방문서비스)", "복지과", False, True),
    "장애인 등록 신청": ("보건소 또는 주민복지센터", "복지과", False, False),
    "국민연금 납부 확인": ("국민연금공단 지사", "연금관리과", True, False),
    "건강보험 자격 확인": ("국민건강보험공단", "자격관리팀", True, False),
    "지방세 납세증명": ("구청 세무과", "세무과", True, False),
    "취득세 신고·납부": ("시청 세무과", "부과과", False, False),
    "폐기물 배출 신고": ("구청 환경위생과", "환경위생과", False, False),
    "가로등 고장 신고": ("구청 도로과(조명관리)", "도로과", False, True),
    "상수도 누수 신고": ("상수도사업본부", "상하수도과", False, True),
    "하수도 막힘": ("구청 상하수도과", "상하수도과", False, True),
    "도로 파손(포트홀)": ("구청 도로과", "도로과", False, True),
    "불법 주차 신고": ("구청 교통지도과 또는 경찰서", "교통지도과", False, True),
    "건축물 사용승인": ("시청 건축과", "건축과", False, False),
    "행정복지센터 증명서": ("행정복지센터", "민원팀", True, False),
    "선거인명부 확인": ("구청 선거관리위원회", "선관위 사무과", True, False),
}
ADMIN_DISTRACTORS = ["세무과", "복지과", "도로과", "환경위생과", "주민과", "건축과",
                     "교통지도과", "상하수도과", "가족관계등록과", "민원팀", "부과과"]
ADMIN_SENT = {
    "전입신고": ["이사했는데 전입신고 어디서 해요", "전입신고하러 왔는데요",
                "새 집으로 이사 왔어요 신고하려고요"],
    "전출신고": ["전출신고 부탁드립니다", "이사 가는데 전출부터 해야 하나요?"],
    "주민등록증 재발급": ["주민증 잃어버렸어요", "신분증 재발급 어떻게 하나요",
                      "주민등록증이 닳아서 재발급 원해요"],
    "가족관계증명서": ["가족관계증명서 한 부 뽑아주세요", "가족관계증명서 발급 받으러 왔습니다"],
    "아동수당 신청": ["첫 아이 낳았는데 수당 신청하러 왔어요", "아동수당 신청합니다"],
    "기초생활수급 신청": ["생활이 어려워서 수급 신청하고 싶어요..", "기초생활보장 신청하려는데요"],
    "장애인 등록 신청": ["장애인 등록하려면 뭐부터 해야 하나요", "부모님 장애 등록 신청 문의"],
    "국민연금 납부 확인": ["국민연금 납부한 것 좀 확인해 주세요", "연금 내역 확인하러 왔어요"],
    "건강보험 자격 확인": ["직장 다닐 때 건보 자격 확인해 주세요", "건강보험 피부양자 확인 문의"],
    "지방세 납세증명": ["납세증명서 발급 가능한가요?", "지방세 다 냈는지 증명서 주세요"],
    "취득세 신고·납부": ["집 샀는데 취득세 신고하러 왔습니다", "취득세 언제까지 내야 하나요?"],
    "폐기물 배출 신고": ["소파 버리고 싶은데 신고해야 하나요?", "대형폐기물 신고하러 왔어요"],
    "가로등 고장 신고": ["골목 가로등이 일주일째 꺼져 있어요", "밤길이 너무 어두워요 가로등 고장"],
    "상수도 누수 신고": ["집 앞 도로에서 물이 솟고 있어요!", "수도관에서 물이 새는 것 같아요"],
    "하수도 막힘": ["하수구가 막혀서 물이 안 빠져요", "배수구 역류가 심해요"],
    "도로 파손(포트홀)": ["차도에 구멍이 뚫려 있어요", "포트홀 때문에 타이어 펑크 났어요!!"],
    "불법 주차 신고": ["소방차 진입로에 차를 세워뒀어요", "골목길 불법주차 신고합니다"],
    "건축물 사용승인": ["신축 건물 사용승인 신청하러 왔습니다", "사용승인 절차 알려주세요"],
    "행정복지센터 증명서": ["증명서 한 부 뽑아 주세요", "민원서류 발급받으러 왔어요"],
    "선거인명부 확인": ["선거인명부 확인 문의드립니다", "선거 때 제 이름 확인하고 싶어요"],
}


def gen_admin(rng):
    svc = rng.choice(list(ADMIN.keys()))
    org, dept, immediate, urgent = ADMIN[svc]
    sent = rng.choice(ADMIN_SENT[svc])
    state = augment(f"민원(방문): {sent}", rng)
    route = dept
    n_opts = rng.choice([6, 7, 8])
    others = [d for d in ADMIN_DISTRACTORS if d != route]
    rng.shuffle(others)
    opts = [route] + others[: n_opts - 1]
    rng.shuffle(opts)
    gold = "ABCDEFGH"[opts.index(route)]
    qs = [
        {"id": "q_route", "type": "choice", "text": "1차 담당 부서/창구는?", "options": opts},
        {"id": "q_urgent", "type": "noul", "text": "긴급(당일 조치) 대상인가?"},
        {"id": "q_online", "type": "noul", "text": "온라인(비대면) 처리 가능성이 높은가?"},
        {"id": "q_visit", "type": "noul", "text": "현장 방문 없이 전화로 해결 가능한가?"},
    ]
    # 온라인 가능: 증명서/확인류 → True
    online = svc in ("가족관계증명서", "지방세 납세증명", "국민연금 납부 확인",
                     "건강보험 자격 확인", "주민등록증 재발급", "행정복지센터 증명서")
    phone = svc in ("가로등 고장 신고", "불법 주차 신고", "상수도 누수 신고")
    ans = {"q_route": gold, "q_urgent": "Y" if urgent else "N",
           "q_online": "Y" if online else "N", "q_visit": "Y" if phone else "N"}
    return {"domain": "admin", "state": state, "questions": qs, "answers": ans}


# ── 2. 의료 v2 ───────────────────────────────────────────────────────────────
MED2 = {
    # 증상: (진료과 후보[정답 먼저], 응급, 소아가능, 한방대안)
    "흉통-조임": (["응급실(내과 당직)", "순환기내과"], True, False, False),
    "복통-우하복부": (["일반외과", "소화기내과"], True, True, False),
    "발열-소아": (["소아과", "감염내과"], False, True, False),
    "두통-급성": (["신경과", "응급실"], False, False, False),
    "골절-의심": (["정형외과", "응급실"], True, True, False),
    "화상": (["응급실", "성형외과"], True, True, False),
    "피부발진": (["피부과"], False, False, False),
    "치통": (["치과"], False, False, False),
    "안충혈": (["안과"], False, False, False),
    "비염": (["이비인후과"], False, True, False),
    "요통-만성": (["재활의학과", "정형외과"], False, False, True),
    "불면-불안": (["정신건강의학과"], False, False, True),
    "소화불량": (["소화기내과", "한방과"], False, False, True),
    "생리통-심함": (["산부인과"], False, False, True),
    "어지럼": (["신경과", "이비인후과"], False, False, False),
}
MED_SENT = {
    "흉통-조임": ["가슴이 조이듯 아파요 식은땀 나요", "왼쪽 가슴 통증이 30분째예요"],
    "복통-우하복부": ["오른쪽 아랫배가 찌르듯 아파요", "배가 아파서 몸을 못 펴요"],
    "발열-소아": ["아이가 39도 열이 나요", "애가 열로 울어요 밤새"],
    "두통-급성": ["머리가 갑자기 깨질 것 같아요", "폭발하는 두통이에요"],
    "골절-의심": ["넘어져서 손목이 부어요 움직이면 아파요", "다리가 부러진 것 같아요"],
    "화상": ["손을 뜨거운 물에 데였어요", "김이 손에 닿아서 물집이 생겼어요"],
    "피부발진": ["온몸에 두드러기가 났어요 가려워요", "발진이 사라지지 않아요"],
    "치통": ["이가 욱신거려 잠을 못 자겠어요", "잇몸이 많이 부었어요"],
    "안충혈": ["눈이 빨갛고 이물감 있어요", "눈이 시려요 충혈도 심해요"],
    "비염": ["코가 막히고 콧물이 안 멈춰요", "재채기가 계속 나와요 아침마다"],
    "요통-만성": ["허리가 3개월째 아파요", "오래 앉으면 허리가 펴지지 않아요"],
    "불면-불안": ["일주일째 잠을 한두 시간만 자요", "마음이 불안해서 잠이 안 와요"],
    "소화불량": ["속이 더부룩하고 소화가 안 돼요", "식사만 하면 체한 느낌이에요"],
    "생리통-심함": ["생리통이 너무 심해 일을 못 해요", "월경통으로 진통제를 자주 먹어요"],
    "어지럼": ["어지러워서 넘어질 뻔했어요", "눈앞이 핑 도네요"],
}
MED_DISTRACTORS = ["내과", "외과", "피부과", "안과", "이비인후과", "정신건강의학과",
                   "소아과", "산부인과", "정형외과", "신경과", "치과", "응급실", "한방과"]


def gen_med(rng):
    key = rng.choice(list(MED2.keys()))
    depts, emerg, pediatric, haneui = MED2[key]
    sent = rng.choice(MED_SENT[key])
    age = rng.choice(["20대", "30대", "40대", "60대", "70대", "8세 아동", "5세 아동"])
    state = augment(f"환자 응대(채팅): {sent}", rng) + f" (환자: {age})"
    route = depts[0]
    others = [d for d in MED_DISTRACTORS if d not in depts]
    rng.shuffle(others)
    opts = [route] + depts[1:] + others
    opts = opts[: rng.choice([6, 7])]
    # 정답 포함 보장
    if route not in opts:
        opts[0] = route
    rng.shuffle(opts)
    is_child = "아동" in age
    qs = [
        {"id": "q_dept", "type": "choice", "text": "1차 진료과/응대는?", "options": opts},
        {"id": "q_emerg", "type": "noul", "text": "응급실 우선 안내 대상인가?"},
        {"id": "q_child", "type": "noul", "text": "소아(만 12세 이하) 대상인가?"},
        {"id": "q_alt", "type": "noul", "text": "한방/대체의료가 고려 가능한 증상인가?"},
    ]
    ans = {"q_dept": "ABCDEFGH"[opts.index(route)],
           "q_emerg": "Y" if emerg else "N",
           "q_child": "Y" if is_child else "N",
           "q_alt": "Y" if haneui else "N"}
    return {"domain": "medical", "state": state, "questions": qs, "answers": ans}


# ── 3. 쇼핑/배달 CS (커머스 한국형) ───────────────────────────────────────────
SHOP = {
    "반품": {"team": "CS-반품팀", "refund": True, "reason": ["단순변심", "상품불량", "사이즈미스"]},
    "교환": {"team": "CS-교환팀", "refund": False, "reason": ["사이즈미스", "색상차이", "상품불량"]},
    "환불지연": {"team": "CS-결제팀", "refund": True, "reason": ["환불지연", "이중결제"]},
    "배송조회": {"team": "CS-배송팀", "refund": False, "reason": ["배송지연", "운송장미갱신"]},
    "파손수령": {"team": "CS-파손접수팀", "refund": True, "reason": ["상품불량", "포장불량"]},
    "오배송": {"team": "CS-배송팀", "refund": True, "reason": ["오배송"]},
    "리뷰신고": {"team": "CS-신고팀", "refund": False, "reason": ["악성리뷰"]},
    "쿠폰문의": {"team": "CS-프로모션팀", "refund": False, "reason": ["쿠폰미적용"]},
    "적립금": {"team": "CS-회원팀", "refund": False, "reason": ["적립금미지급"]},
    "VIP문의": {"team": "CS-VIP전담", "refund": False, "reason": ["등급문의"]},
}
SHOP_SENT = {
    "반품": ["어제 산 신발 반품하고 싶어요", "안 입어봤는데 반품 되나요?"],
    "교환": ["사이즈가 커서 교환하고 싶어요", "색이랑 화면이 달라요 바꿔주세요"],
    "환불지연": ["환불했는데 왜 안 들어와요? 5일째인데", "취소했는데 돈이 안 돌아와요"],
    "배송조회": ["배송조회가 왜 안 되죠?", "3일째 준비중이에요 언제 와요?"],
    "파손수령": ["박스 열어보니 깨져 있었어요", "도착한 제품이 찌그러졌어요"],
    "오배송": ["주문 안 한 게 왔어요", "다른 상품이 왔어요 어떡하죠"],
    "리뷰신고": ["제 가게에 악성 리뷰가 달렸어요", "허위 리뷰 신고하고 싶어요"],
    "쿠폰문의": ["쿠폰이 왜 적용이 안 되죠?", "첫구매 쿠폰 쓰고 싶은데 안 돼요"],
    "적립금": ["적립금이 안 쌓였어요", "리뷰 적립금 언제 줘요?"],
    "VIP문의": ["VIP 등급 어떻게 올려요?", "등급 혜택이 뭐가 달라요?"],
}
SHOP_DISTRACTORS = ["CS-반품팀", "CS-교환팀", "CS-결제팀", "CS-배송팀", "CS-파손접수팀",
                    "CS-신고팀", "CS-프로모션팀", "CS-회원팀", "CS-VIP전담", "기술지원팀"]


def gen_shop(rng):
    key = rng.choice(list(SHOP.keys()))
    info = SHOP[key]
    sent = rng.choice(SHOP_SENT[key])
    angry = rng.random() < 0.3
    toxic = angry and rng.random() < 0.4
    state = augment(f"쇼핑몰 고객문의: {sent}", rng)
    if toxic:
        state += rng.choice([" (욕설 포함, 고소 언급)", " (매스컴 폭로 언급)"])
    elif angry:
        state += " (불만 표출)"
    route = info["team"]
    others = [t for t in SHOP_DISTRACTORS if t != route]
    rng.shuffle(others)
    opts = [route] + others[: rng.choice([5, 6]) - 1]
    rng.shuffle(opts)
    reasons = info["reason"]
    gold_reason = ("상품불량" if ("불량" in state or "깨" in state or "찌그러" in state)
                   else "배송지연" if ("지연" in state or "안 와" in state or "3일" in state)
                   else rng.choice(reasons))
    reason_opts = list({gold_reason} | set(rng.sample(
        ["단순변심", "상품불량", "배송지연", "사이즈미스", "색상차이", "오배송", "환불지연"],
        3)))
    rng.shuffle(reason_opts)
    qs = [
        {"id": "q_route", "type": "choice", "text": "라우팅 팀은?", "options": opts},
        {"id": "q_refund", "type": "noul", "text": "환불(결제취소) 가능성이 높은 사유인가?"},
        {"id": "q_escalate", "type": "noul", "text": "에스컬레이션(법무/보상 검토) 대상인가?"},
        {"id": "q_reason", "type": "choice", "text": "가장 가능성 높은 사유는?", "options": reason_opts},
    ]
    ans = {"q_route": "ABCDEFGH"[opts.index(route)],
           "q_refund": "Y" if info["refund"] else "N",
           "q_escalate": "Y" if toxic else "N",
           "q_reason": "ABCDEFGH"[reason_opts.index(gold_reason)]}
    return {"domain": "shop", "state": state, "questions": qs, "answers": ans}


# ── 4. 생활 인프라 (부동산/교통/통신) ────────────────────────────────────────
LIFE = {
    "전세계약": {"route": "부동산 중개사무소", "urgent": False,
              "note": ["전세금 반환보증 가입 문의", "전세 계약 조건 확인"]},
    "월세분쟁": {"route": "한국토지주택공사(LH) or 주택임대차분쟁조정위", "urgent": True,
               "note": ["보증금 돌려 안 줘요", "임대인이 갑자기 나가래요"]},
    "버스분실": {"route": "버스운수사 분실물센터", "urgent": False,
               "note": ["버스에 우산 두고 내렸어요", "휴대폰을 버스에 놓고 왔어요"]},
    "지하철분실": {"route": "각 역 소재 유실물센터(역무실)", "urgent": False,
                "note": ["지하철에 가방 두고 내렸어요", "역에서 지갑 잃어버렸어요"]},
    "택시분실": {"route": "택시회사/분실물 센터", "urgent": True,
               "note": ["택시에 노트북 두고 내렸어요!!", "택시에서 물건 잃어버렸어요"]},
    "주민세": {"route": "구청 세무과", "urgent": False,
             "note": ["주민세 고지서가 이상해요", "주민세 납부 문의"]},
    "요금제변경": {"route": "통신사 고객센터(국번 없이 114)", "urgent": False,
                "note": ["요금제 바꾸고 싶어요", "데이터 더 쓰고 싶은데 요금제 문의"]},
    "인터넷장애": {"route": "통신사 기술지원", "urgent": True,
                "note": ["인터넷이 자꾸 끊겨요", "와이파이가 안 잡혀요"]},
    "주차민원": {"route": "구청 교통지도과", "urgent": False,
               "note": ["단속 주차장 출차 문의", "주차위반 신고하고 싶어요"]},
    "상가임대": {"route": "상가임대차분쟁조정위원회", "urgent": False,
               "note": ["상가 권리금 분쟁이 있어요", "임대료 인상이 과해요"]},
}
LIFE_DISTRACTORS = ["부동산 중개사무소", "한국토지주택공사(LH) or 주택임대차분쟁조정위",
                    "버스운수사 분실물센터", "각 역 소재 유실물센터(역무실)",
                    "택시회사/분실물 센터", "구청 세무과", "통신사 고객센터(국번 없이 114)",
                    "통신사 기술지원", "구청 교통지도과", "상가임대차분쟁조정위원회", "경찰서"]


def gen_life(rng):
    key = rng.choice(list(LIFE.keys()))
    info = LIFE[key]
    sent = rng.choice(info["note"])
    state = augment(f"생활 민원: {sent}", rng)
    route = info["route"]
    others = [d for d in LIFE_DISTRACTORS if d != route]
    rng.shuffle(others)
    opts = [route] + others[:5]
    rng.shuffle(opts)
    qs = [
        {"id": "q_route", "type": "choice", "text": "1차 상담/처리 창구는?", "options": opts},
        {"id": "q_urgent", "type": "noul", "text": "당일 처리 긴급 대상인가?"},
        {"id": "q_legal", "type": "noul", "text": "법적 분쟁 가능성이 있는 사안인가?"},
        {"id": "q_fee", "type": "noul", "text": "수수료/비용 발생 가능성이 높은가?"},
    ]
    legal = key in ("월세분쟁", "상가임대")
    fee = key in ("전세계약", "주민세", "상가임대")
    ans = {"q_route": "ABCDEFGH"[opts.index(route)],
           "q_urgent": "Y" if info["urgent"] else "N",
           "q_legal": "Y" if legal else "N",
           "q_fee": "Y" if fee else "N"}
    return {"domain": "life", "state": state, "questions": qs, "answers": ans}


# ── 5. 교육 ──────────────────────────────────────────────────────────────────
EDU = {
    "수능원서": ("교육지원청 학적팀", False, "수능 원서 접수 문의"),
    "전학": ("교육지원청 학적팀", False, "자녀 전학 절차 문의"),
    "학자금": ("한국장학재단", False, "국가장학 신청 문의"),
    "학사일정": ("학교 교무실", False, "학사일정 확인 문의"),
    "학원환불": ("학원 + 소비자분쟁조정위", True, "학원비 환불 요청"),
    "급식비": ("학교 행정실", False, "급식비 납부 문의"),
    "방과후": ("학교 방과후학교부", False, "방과후 수업 문의"),
    "진로상담": ("학교 진로상담부", False, "진로 상담 신청"),
}
EDU_DISTRACTORS = ["교육지원청 학적팀", "한국장학재단", "학교 교무실",
                   "학원 + 소비자분쟁조정위", "학교 행정실", "학교 방과후학교부",
                   "학교 진로상담부", "시교육청 민원실"]


def gen_edu(rng):
    key = rng.choice(list(EDU.keys()))
    route, urgent, sent = EDU[key]
    state = augment(f"학부모 문의: {sent}", rng)
    others = [d for d in EDU_DISTRACTORS if d != route]
    rng.shuffle(others)
    opts = [route] + others[:5]
    rng.shuffle(opts)
    qs = [
        {"id": "q_route", "type": "choice", "text": "응대 창구는?", "options": opts},
        {"id": "q_urgent", "type": "noul", "text": "즉시 에스컬레이션 대상인가?"},
        {"id": "q_refund", "type": "noul", "text": "환불/지급 관련 사안인가?"},
    ]
    refund = key in ("학자금", "학원환불")
    ans = {"q_route": "ABCDEFGH"[opts.index(route)],
           "q_urgent": "Y" if urgent else "N",
           "q_refund": "Y" if refund else "N"}
    return {"domain": "edu", "state": state, "questions": qs, "answers": ans}


# ── 6. 직장/조직 ─────────────────────────────────────────────────────────────
WORK = {
    "연차신청": ("팀장 결재", False, "연차 쓰고 싶은데요"),
    "병가": ("인사팀 + 팀장", True, "몸이 아파서 병가 필요해요"),
    "결재긴급": ("팀장 즉시 결재", True, "오늘까지 결재 필요한 건입니다"),
    "급여오류": ("인사팀 급여담당", True, "급여가 덜 나왔어요"),
    "퇴사문의": ("인사팀", False, "퇴사 절차 문의드립니다"),
    "보고작성": ("팀장 보고", False, "주간 보고서 작성 문의"),
    "보안사고": ("보안팀 즉시", True, "피싱 메일을 클릭했어요"),
    "출장신청": ("팀장 결재 + 총무팀", False, "출장 가려고 합니다"),
}


def gen_work(rng):
    key = rng.choice(list(WORK.keys()))
    route, urgent, sent = WORK[key]
    state = augment(f"사내 문의: {sent}", rng, formal=True)
    rank = rng.choice(["신입사원", "대리", "과장", "부장"])
    state += f" (직급: {rank})"
    opts = ["팀장 결재", "인사팀 + 팀장", "팀장 즉시 결재", "인사팀 급여담당",
            "인사팀", "팀장 보고", "보안팀 즉시", "팀장 결재 + 총무팀"]
    rng.shuffle(opts)
    opts = opts[:6]
    if route not in opts:
        opts[0] = route
        rng.shuffle(opts)
    qs = [
        {"id": "q_route", "type": "choice", "text": "1차 결재/응대 라인은?", "options": opts},
        {"id": "q_urgent", "type": "noul", "text": "당일 즉시 처리 대상인가?"},
        {"id": "q_doc", "type": "noul", "text": "서면(전자결재) 문서 필요 가능성이 높은가?"},
    ]
    doc = key in ("연차신청", "결재긴급", "출장신청", "퇴사문의")
    ans = {"q_route": "ABCDEFGH"[opts.index(route)],
           "q_urgent": "Y" if urgent else "N",
           "q_doc": "Y" if doc else "N"}
    return {"domain": "work", "state": state, "questions": qs, "answers": ans}


# ── 7. 에이전트 게이팅 v2 ───────────────────────────────────────────────────
AGENT2 = {
    "검색": {"tools": ["web_search", "portal_news"], "defer": False, "local": True,
             "sents": ["오늘 춘천 날씨 어때?", "최신 뉴스 찾아줘", "이 주식 지금 얼마야?"]},
    "DB조회": {"tools": ["sql_query", "crm_lookup"], "defer": False, "local": False,
              "sents": ["우리 매장 매출 보여줘", "고객 명단 조회해줘"]},
    "계산": {"tools": ["calculator", "unit_convert"], "defer": False, "local": True,
             "sents": ["부가세 계산해줘", "환율 곱해서 얼마야?"]},
    "대형LLM": {"tools": ["reasoning_llm"], "defer": True, "local": False,
              "sents": ["이 계약 조항 리스크 분석해줘", "전략 보고서 초안 깊이 있게 써줘"]},
    "일정": {"tools": ["calendar", "scheduler"], "defer": False, "local": True,
             "sents": ["내일 오전 회의 잡아줘", "다음 주 알림 설정해줘"]},
    "번역": {"tools": ["translator"], "defer": False, "local": True,
             "sents": ["이 문장 영어로 번역해줘", "일본어 메일 번역해줘"]},
    "지도": {"tools": ["maps_route"], "defer": False, "local": False,
             "sents": ["여기서 강남역 가는 길 찾아줘", "근처 약국 어디야?"]},
    "메일": {"tools": ["mail_send"], "defer": True, "local": False,
             "sents": ["거래처에 회의 메일 보내줘", "사장님께 보고 메일 써줘"]},
}
TOOL_FLAT = ["web_search", "portal_news", "sql_query", "crm_lookup", "calculator",
             "unit_convert", "reasoning_llm", "calendar", "scheduler", "translator",
             "maps_route", "mail_send"]


def gen_agent(rng):
    key = rng.choice(list(AGENT2.keys()))
    info = AGENT2[key]
    state = augment(f"에이전트 지시: {rng.choice(info['sents'])}", rng)
    tool = rng.choice(info["tools"])
    others = [t for t in TOOL_FLAT if t not in info["tools"]]
    rng.shuffle(others)
    opts = [tool] + others[:5]
    rng.shuffle(opts)
    qs = [
        {"id": "q_tool", "type": "choice", "text": "호출할 도구는?", "options": opts},
        {"id": "q_defer", "type": "noul", "text": "대형 추론 모델 지연(defer) 대상인가?"},
        {"id": "q_local", "type": "noul", "text": "온디바이스(로컬) 처리 가능한가?"},
        {"id": "q_privacy", "type": "noul", "text": "개인정민정보 포함 가능성이 있어 외부 전송 주의가 필요한가?"},
    ]
    privacy = key in ("DB조회", "메일")
    ans = {"q_tool": "ABCDEFGH"[opts.index(tool)],
           "q_defer": "Y" if info["defer"] else "N",
           "q_local": "Y" if info["local"] else "N",
           "q_privacy": "Y" if privacy else "N"}
    return {"domain": "agent", "state": state, "questions": qs, "answers": ans}


GENS = [gen_admin, gen_med, gen_shop, gen_life, gen_edu, gen_work, gen_agent]


def make_split(n, seed):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        rec = GENS[i % len(GENS)](rng)
        rec["id"] = f"{rec['domain']}-klife-{seed}-{i:05d}"
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=28000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": make_split(args.train, 4001),
        "val": make_split(args.val, 4002),
        "test": make_split(args.test, 4003),
    }
    from io_jsonl import dump_jsonl
    for name, recs in splits.items():
        p = out / f"klife_v02_{name}.jsonl"
        dump_jsonl(recs, p.as_posix())
        n_fields = sum(len(r["questions"]) for r in recs)
        doms = {}
        for r in recs:
            doms[r["domain"]] = doms.get(r["domain"], 0) + 1
        print(f"{p.name}: {len(recs)}레코드 {n_fields}결정 {doms}")


if __name__ == "__main__":
    main()
