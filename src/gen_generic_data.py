"""v0.2 일반 지식 믹스 데이터 — KMMLU 붕괴(도메인 과적합) 해결용.

라이선스 검증된 소스만 사용:
  - OpenBookQA main   (CC BY 4.0)  — datasets-server 로 수신
  - AI2 ARC Challenge/Easy (CC BY-SA 4.0) — datasets-server 로 수신
  - 자체 한국 상식 팩트뱅크(규칙 생성, 저작권 무결)

보안: 모든 외부 요청은 ALLOWED_HOSTS(https 고정) 대상으로만, 사설/루프백 IP 차단,
리다이렉트 미추적. RNG는 데이터 셔플용(암호화 용도 아님)이나 시드는 secrets 로 생성.
출력: 타이핑드 결정 레코드 (choice 단일 필드) — gen_synth_data 와 동일 스키마.
"""
from __future__ import annotations

import argparse
import ipaddress
import random
import secrets
import socket
import urllib.parse
import urllib.request
from pathlib import Path

from common import dump_jsonl

ALLOWED_HOSTS = {"datasets-server.huggingface.co"}


def _assert_safe_url(url: str):
    """SSRF 방지: https 고정, 호스트 allowlist, 사설/루프백/링크로컬 IP 차단."""
    p = urllib.parse.urlsplit(url)
    if p.scheme != "https" or p.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"허용되지 않은 요청 대상: {url}")
    for info in socket.getaddrinfo(p.hostname, 443, proto=socket.IPPROTO_TCP):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"금지된 주소로 해석됨: {ip}")


def get(url):
    _assert_safe_url(url)
    # 리다이렉트 미추적: allowlist 밖으로 새어나가는 것을 원천 차단
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def fetch(dataset, config, split, total):
    rows, offset = [], 0
    q = urllib.parse.quote
    while len(rows) < total:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={q(dataset, safe='')}"
               f"&config={q(config, safe='')}&split={split}&offset={offset}&length=100")
        try:
            data = get(url)
        except Exception as e:
            print(f"[warn] {dataset} {offset}: {e}")
            break
        batch = [r["row"] for r in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows[:total]


def mcq_record(state, options, gold_idx, rng, dom="generic"):
    opts = list(options)
    rng.shuffle(opts)
    return {
        "domain": dom,
        "state": state,
        "questions": [{"id": "q0", "type": "choice", "text": "정답은?", "options": opts}],
        "answers": {"q0": "ABCDEFGH"[opts.index(options[gold_idx])]},
    }


def from_obqa(n, rng):
    rows = fetch("allenai/openbookqa", "main", "train", n)
    out = []
    for r in rows:
        ch, key = r["choices"], r["answerKey"]
        if key not in ch["label"]:
            continue
        gold = ch["label"].index(key)
        out.append(mcq_record(f"문제: {r['question_stem']}", ch["text"], gold, rng, "obqa"))
    return out


def from_arc(n, rng):
    rows = fetch("allenai/ai2_arc", "ARC-Challenge", "train", n // 2)
    rows += fetch("allenai/ai2_arc", "ARC-Easy", "train", n - len(rows))
    out = []
    for r in rows:
        ch = r["choices"]
        if r["answerKey"] not in ch["label"]:
            continue
        gold = ch["label"].index(r["answerKey"])
        out.append(mcq_record(f"문제: {r['question']}", ch["text"], gold, rng, "arc"))
    return out


# ── 한국 상식 팩트뱅크 (규칙 생성) ────────────────────────────────────────────
FACTS = {
    "나라수도": [("대한민국", "서울"), ("일본", "도쿄"), ("중국", "베이징"), ("프랑스", "파리"),
                ("이탈리아", "로마"), ("스페인", "마드리드"), ("영국", "런던"), ("독일", "베를린"),
                ("러시아", "모스크바"), ("미국", "워싱턴 D.C."), ("캐나다", "오타와"), ("호주", "캔버라"),
                ("브라질", "브라질리아"), ("이집트", "카이로"), ("인도", "뉴델리"), ("태국", "방콕"),
                ("베트남", "하노이"), ("인도네시아", "자카르타"), ("필리핀", "마닐라"), ("몽골", "울란바토르"),
                ("그리스", "아테네"), ("튀르키예", "앙카라"), ("포르투갈", "리스본"), ("네덜란드", "암스테르담")],
    "한국사": [("세종대왕이 한글을 착제한 시기", "1443년(반포 1446년)"), ("임진왜란 발발 연도", "1592년"),
              ("을사늑약 체결 연도", "1905년"), ("8·15 광복 연도", "1945년"), ("삼국통일을 완성한 나라", "신라"),
              ("고려를 건국한 왕", "태조 왕건"), ("조선을 건국한 왕", "태조 이성계"),
              ("훈민정음 창제 취지를 밝힌 문서", "훈민정음 해례본"), ("조선 최후의 황제", "순종"),
              ("동학농민운동 발생 연도", "1894년"), ("을미사변으로 시해된 왕비", "명성황후(민비)"),
              ("경주 불국사 석굴암 본존불", "석가여래좌상")],
    "과학": [("물의 화학식", "H2O"), ("이산화탄소의 화학식", "CO2"), ("태양계에서 가장 큰 행성", "목성"),
            ("인체에서 가장 큰 기관", "피부"), ("빛의 속도(진공, 근사)", "초속 약 30만 km"),
            ("광합성을 하는 세포 내 소기관", "엽록체"), ("가장 가벼운 원소", "수소"),
            ("혈액형을 결정하는 주요 항원 체계", "ABO식"), ("소리가 전파되지 않는 상태", "진공"),
            ("지구 대기에서 비율이 가장 높은 기체", "질소"), ("인체의 정상 체온(근사)", "36.5°C"),
            ("태양계에서 지구는 몇 번째 행성", "세 번째")],
    "속담": [("가는 말이 고와야 오는 말이 곱다'의 의미", "남에게 좋게 대하면 좋게 대답받는다"),
            ("열 길 물속은 알아도 한 길 사람 속은 모른다'의 의미", "사람의 마음은 헤아리기 어렵다"),
            ("호랑이도 제 말 하면 온다'의 의미", "이야기 중인 당사자가 마침 나타난다"),
            ("가랑비에 옷 젖는다'의 의미", "작은 일이 모여 큰 손해가 된다"),
            ("우물 안 개구리'의 의미", "견문이 좁고 세상 모르고 잘난 체한다"),
            ("콩 한 쪽도 나눠 먹는다'의 의미", "아주 친하고 정이 깊다")],
}

QTEMPLATES = {
    "나라수도": ["{q}의 수도는?", "{q}의 수도 도시로 올은 것은?"],
    "한국사": ["{q}은(는)?", "{q}에 대한 설명으로 올은 것은?"],
    "과학": ["{q}은(는)?", "{q}로 올은 것은?"],
    "속담": ["{q}"],
}


def gen_facts(n, rng):
    out, i = [], 0
    keys = list(FACTS.keys())
    while len(out) < n:
        k = keys[i % len(keys)]
        i += 1
        items = FACTS[k]
        q, gold = items[rng.randrange(len(items))]
        distractors = rng.sample([a for _, a in items if a != gold], 3)
        opts = [gold] + distractors
        state = "문제: " + rng.choice(QTEMPLATES[k]).format(q=q)
        out.append(mcq_record(state, opts, 0, rng, f"fact:{k}"))
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--obqa", type=int, default=5000)
    ap.add_argument("--arc", type=int, default=3500)
    ap.add_argument("--facts", type=int, default=3000)
    args = ap.parse_args()

    # 데이터 셔플용 RNG — 예측 불가 시드(secrets) 사용
    rng = random.Random(secrets.randbits(64) ^ 77)
    recs = []
    for fn, n in ((from_obqa, args.obqa), (from_arc, args.arc)):
        got = []
        try:
            got = fn(n, rng)
        except Exception as e:
            print(f"[warn] {fn.__name__} 실패: {e}")
        print(f"{fn.__name__}: {len(got)}")
        recs += got
    facts = gen_facts(args.facts, rng)
    print(f"facts: {len(facts)}")
    recs += facts
    rng.shuffle(recs)
    for j, r in enumerate(recs):
        r["id"] = f"gen-{j:05d}"
    out = Path(args.out) / "generic_v02_train.jsonl"
    dump_jsonl(recs, out.as_posix())
    print(f"저장: {out} ({len(recs)}레코드)")


if __name__ == "__main__":
    import json  # noqa: F401  (get() 에서 사용)
    main()
