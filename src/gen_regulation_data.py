"""v0.8 규정 준거 판단 데이터 — 한림대 규정 3종(학칙/교원인사/교원업적평가) 기반.

설계(클로즈드 월드 — 디시전 모델의 최적 영역):
  A. 조항 주제 라우팅 (choice): 규정 발췌 → 다루는 사안은?
  B. 수치 요건 판단 (noul): '이내/초과/이상/미만 + 수치' 조항 → 시나리오 위반 여부
  C. 적용 대상 판단 (noul): 조항 대상(학생/교원/대학원생 등) vs 시나리오 주체

입력: data/reg_2_1_1.txt(학칙), reg_4_1_1.txt(교원인사), reg_4_1_2.txt(업적평가)
출력: data/regul_v08_{train,val,test}.jsonl
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from common import dump_jsonl

TOPICS = {
    "입학·전형": ["입학", "전형", "모집", "지원자격", "편입학", "재입학", "시간제"],
    "등록·수납": ["등록금", "수납", "납부", "반환", "분납"],
    "수업·이수학점": ["이수학점", "수강신청", "학점", "수업연한", "재학연한", "수료", "시간제등록", "계절학기"],
    "휴학·복학·제적": ["휴학", "복학", "제적", "자퇴", "복수학적", "유급"],
    "졸업·학위": ["졸업", "학위", "논문", "심사", "졸업유예"],
    "장학·지원": ["장학", "수당", "지원금", "감면"],
    "교원 임용·인사": ["임용", "채용", "신규임용", "겸임", "파견", "휴직", "복직", "면직", "정년", "인사"],
    "승진·재임용": ["승진", "재임용", "승진임용", "업적평가", "성과평가", "근무평정"],
    "연구·업적": ["연구업적", "논문실적", "연구비", "학술", "저술", "연구년"],
    "징계·징계사항": ["징계", "징계사항", "파면", "해임", "정직", "감봉", "명예훼손"],
    "학사 행정·조직": ["학장", "학부", "학과", "대학원", "위원회", "규정", "세칙", "조직", "정원"],
    "시설·부속기관": ["부속기관", "부설", "도서관", "박물관", "시설", "기숙사"],
    "성적 평가": ["성적", "평점", "학사경고", "우수", "급제"],
    "교육과정": ["교육과정", "전공", "복수전공", "부전공", "연계전공", "교양"],
}
TOPIC_LIST = list(TOPICS.keys())
# 우선순위(구체→일반): 매칭된 것 중 가장 구체적인 주제 채택
PRIORITY = ["징계·징계사항", "휴학·복학·제적", "승진·재임용", "연구·업적", "입학·전형",
            "졸업·학위", "등록·수납", "수업·이수학점", "성적 평가", "장학·지원",
            "교원 임용·인사", "교육과정", "시설·부속기관", "학사 행정·조직"]

SUBJECTS = {
    "학부학생": ["학부 학생", "학생이", "재학생", "신입생", "편입생"],
    "대학원생": ["대학원생", "석사과정", "박사과정", "대학원 학생"],
    "교원": ["교원", "교수", "조교수", "부교수", "전임강사", "시간강사"],
    "직원": ["직원", "행정직", "사무직"],
}

BASE_TITLES = {
    "reg_2_1_1": "한림대학교 학칙",
    "reg_4_1_1": "한림대학교 교원 인사 규정",
    "reg_4_1_2": "한림대학교 교원 업적 평가 규정",
}


def load_sources():
    """data/reg_*.txt 전체를 규정 소스로 로드(제목 미지정 파일은 파일명 사용)."""
    files = {}
    for p in sorted(Path("data").glob("reg_*.txt")):
        if p.stem in ("reg_train", "reg_val", "reg_test"):
            continue
        files[p.stem] = BASE_TITLES.get(p.stem, p.stem)
    return files

NUM_UNITS = {"학점": "학점", "년": "년", "년 이내": "년", "학기": "학기", "개월": "개월",
             "인": "명", "명": "명", "%": "%", "회": "회", "점": "점", "배": "배"}


def chunk_text(t, size=900):
    t = re.sub(r"개정 \d{4}\.\s?\d{1,2}\.\s?\d{1,2}\.?", "", t)
    t = re.sub(r"제정 \d{4}\.\s?\d{1,2}\.\s?\d{1,2}\.?", "", t)
    # 짧은 라인 병합 → 문단 블록화
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    merged, cur = [], ""
    for l in lines:
        cur = (cur + " " + l).strip()
        if len(cur) >= size:
            merged.append(cur)
            cur = ""
    if len(cur) > 80:
        merged.append(cur)
    # 너무 긴 블록 분할
    out = []
    for blk in merged:
        for i in range(0, len(blk), size):
            piece = blk[i:i + size]
            if len(piece) > 80:
                out.append(piece)
    return out


def classify(chunk):
    for topic in PRIORITY:
        for kw in TOPICS[topic]:
            if kw in chunk:
                return topic
    return None


def find_subject(chunk):
    for subj, kws in SUBJECTS.items():
        for kw in kws:
            if kw in chunk:
                return subj
    return None


def find_thresholds(chunk):
    """'N단위 이내/초과/이상/이하/미만' 패턴 → (수치, 단위, 한정어) 목록."""
    out = []
    for m in re.finditer(r"(\d{1,3})\s*(학점|년|개월|학기|명|인|회|점|%|배)\s*(이내|초과|이상|이하|미만|까지)?", chunk):
        n, unit, bound = int(m.group(1)), m.group(2), m.group(3) or ""
        out.append((n, unit, bound))
    return out


def scenario_for(n, unit, bound, subj, rng):
    """수치 요건 시나리오 — 규정 한도 vs 시나리오 값의 관계로 위반/적합 생성."""
    if unit == "학점":
        base = f"{subj or '학부학생'}이(가) 이번 학기 {n + rng.choice([0, 0, 3, 6])}학점을 수강신청하려 한다"
    elif unit in ("년", "개월"):
        span = n + rng.choice([0, 0, 1, 2])
        base = f"{subj or '학생'}이(가) {span}{unit} 동안 해당 제도를 이용하려 한다"
    elif unit == "학기":
        base = f"{subj or '학생'}이(가) {n + rng.choice([0, 0, 1])}학기를 초과하지 않고 이수하려 한다"
    elif unit in ("명", "인"):
        base = f"해당 위원회를 {n + rng.choice([0, 0, 2])}명으로 구성하려 한다"
    elif unit == "%":
        base = f"비율 기준 {n + rng.choice([0, 0, 5, 10])}%인 경우를 검토하고 있다"
    elif unit == "회":
        base = f"{n + rng.choice([0, 0, 1])}회 재신청하려 한다"
    else:
        base = f"기준치 {n + rng.choice([0, 0, 2])}{unit}인 경우"
    # 위반 판정: 한도+초과값 → 초과했으면 '초과' 시나리오
    # (bound 이내/이하/까지 → 값이 한도 넘으면 위반; 이상 → 미달이 위반)
    return base


def gen_records(seed, n_target):
    rng = random.Random(seed)
    recs = []
    for fkey, fname in load_sources().items():
        p = Path("data") / f"{fkey}.txt"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        for ch in chunk_text(text):
            topic = classify(ch)
            subj = find_subject(ch)
            snippet = ch[:600]
            # A. 주제 라우팅
            if topic:
                others = rng.sample([t for t in TOPIC_LIST if t != topic], min(5, len(TOPIC_LIST) - 1))
                opts = [topic] + others
                rng.shuffle(opts)
                recs.append({
                    "domain": "regul",
                    "state": f"[{fname} 발췌] {snippet}",
                    "questions": [{"id": "q_topic", "type": "choice",
                                   "text": "이 조항이 다루는 사안은?", "options": opts}],
                    "answers": {"q_topic": "ABCDEFGH"[opts.index(topic)]},
                })
            # B. 수치 요건 (위반/적합 noul) — 상한형만
            if topic:
                for n, unit, bound in find_thresholds(ch)[:2]:
                    val = n + rng.choice([0, 0, 3, 6]) if unit == "학점" else n + rng.choice([0, 0, 1, 2])
                    if bound in ("이내", "이하", "까지", "초과") or bound == "":
                        scen = scenario_for(n, unit, bound, subj, rng)
                        if unit == "학점":
                            violate = rng.random() < 0.5
                            val = n + rng.choice([0, -2]) if not violate else n + rng.choice([3, 6])
                            scen = f"{subj or '학부학생'}이(가) 학기당 {val}학점을 신청했다"
                        else:
                            violate = val > n
                        recs.append({
                            "domain": "regul",
                            "state": f"[{fname}] {snippet}\n\n[상황] {scen}",
                            "questions": [
                                {"id": "q_violate", "type": "noul",
                                 "text": "조항에 비추어 규정 위반(또는 초과) 소지가 있는가?"},
                                {"id": "q_topic2", "type": "noul",
                                 "text": "이 조항은 수치 기준을 명시하고 있는가?"},
                            ],
                            "answers": {"q_violate": "Y" if violate else "N",
                                        "q_topic2": "Y"},
                        })
            # C. 적용 대상
            if subj and topic and rng.random() < 0.5:
                other_subj = rng.choice([s for s in SUBJECTS if s != subj])
                chosen = rng.choice([subj, subj, other_subj])  # 2:1 해당
                role = {"학부학생": "3학년 학부 학생", "대학원생": "박사과정 대학원생",
                        "교원": "전임교수", "직원": "행정직 직원"}[chosen]
                recs.append({
                    "domain": "regul",
                    "state": f"[{fname} 발췌] {snippet}\n\n[상황] {role}이(가) 문의했다",
                    "questions": [{"id": "q_apply", "type": "noul",
                                   "text": "이 조항의 직접 적용 대상에 해당하는가?"}],
                    "answers": {"q_apply": "Y" if chosen == subj else "N"},
                })
    rng.shuffle(recs)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=1500)
    args = ap.parse_args()
    allr = gen_records(91, args.train * 2)
    # 분할: 시드 기반 재생성으로 홀드아웃 분리 (파일·조항 오염 방지엔 한계 — 개략 분리)
    rng = random.Random(7)
    rng.shuffle(allr)
    n = len(allr)
    tr, va, te = allr[: int(n * 0.8)], allr[int(n * 0.8): int(n * 0.9)], allr[int(n * 0.9):]
    for name, recs in (("train", tr), ("val", va), ("test", te)):
        for i, r in enumerate(recs):
            r["id"] = f"regul-{name}-{i:05d}"
        dump_jsonl(recs, (Path(args.out) / f"regul_v08_{name}.jsonl").as_posix())
        print(name, len(recs))


if __name__ == "__main__":
    main()
