"""KoBEST(skt/kobest_v1, CC-BY-SA-4.0) → K-Decision 평가 포맷 변환 (평가 전용).

5개 태스크를 타이핑드 결정으로 매핑:
  boolq    → noul   (문단+질문 → 예/아니오)
  copa     → choice(2) (전제에 이어지는 결과)
  hellaswag→ choice(4) (문맥에 이어지는 결말)
  sentineg → noul   (긍정/부정)
  wic      → noul   (두 문맥에서 같은 의미인지)

출력은 K-Decision Index 0.1 한국 공개 태스크의 재료가 된다.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import urllib.parse
import urllib.request
from pathlib import Path

ALLOWED_HOSTS = {"datasets-server.huggingface.co"}


def _assert_safe_url(url: str):
    p = urllib.parse.urlsplit(url)
    if p.scheme != "https" or p.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"허용되지 않은 요청 대상: {url}")
    for info in socket.getaddrinfo(p.hostname, 443, proto=socket.IPPROTO_TCP):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"금지된 주소로 해석됨: {ip}")


def get(url):
    _assert_safe_url(url)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def fetch(subset, split, total):
    rows, offset = [], 0
    q = urllib.parse.quote
    while len(rows) < total:
        url = (f"https://datasets-server.huggingface.co/rows?dataset=skt%2Fkobest_v1"
               f"&config={q(subset)}&split={q(split)}&offset={offset}&length=100")
        data = get(url)
        batch = [r["row"] for r in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows[:total]


def conv_boolq(row, i):
    label = "Y" if row["label"] == 1 else "N"
    return {"id": f"kobest-boolq-{i:05d}", "domain": "kobest:boolq",
            "state": f"문단: {row['paragraph']}\n질문: {row['question']}",
            "questions": [{"id": "q0", "type": "noul", "text": "질문의 답이 '예'인가?"}],
            "answers": {"q0": label}}


def conv_copa(row, i):
    opts = [row["alternative_1"], row["alternative_2"]]
    gold = row["label"]
    q = row.get("question", "다음에 일어난 일은?")
    return {"id": f"kobest-copa-{i:05d}", "domain": "kobest:copa",
            "state": f"전제: {row['premise']}",
            "questions": [{"id": "q0", "type": "choice", "text": q, "options": opts}],
            "answers": {"q0": "AB"[gold]}}


def conv_hellaswag(row, i):
    opts = [row[f"ending_{k}"] for k in range(1, 5)]
    gold = row["label"]
    return {"id": f"kobest-hswag-{i:05d}", "domain": "kobest:hellaswag",
            "state": f"문맥: {row['context']}",
            "questions": [{"id": "q0", "type": "choice",
                           "text": "이어지는 결말은?", "options": opts}],
            "answers": {"q0": "ABCD"[gold]}}


def conv_sentineg(row, i):
    label = "Y" if row["label"] == 1 else "N"  # 1=긍정
    return {"id": f"kobest-sentineg-{i:05d}", "domain": "kobest:sentineg",
            "state": f"문장: {row['sentence']}",
            "questions": [{"id": "q0", "type": "noul", "text": "긍정적 문장인가?"}],
            "answers": {"q0": label}}


def conv_wic(row, i):
    label = "Y" if row["label"] == 1 else "N"
    return {"id": f"kobest-wic-{i:05d}", "domain": "kobest:wic",
            "state": f"대상 단어: {row['word']}\n문맥1: {row['context_1']}\n문맥2: {row['context_2']}",
            "questions": [{"id": "q0", "type": "noul",
                           "text": "두 문맥에서 단어의 의미가 같은가?"}],
            "answers": {"q0": label}}


TASKS = {
    "boolq": (conv_boolq, 700),
    "copa": (conv_copa, 500),
    "hellaswag": (conv_hellaswag, 500),
    "sentineg": (conv_sentineg, 500),
    "wic": (conv_wic, 500),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kobest_eval_test.jsonl")
    ap.add_argument("--split", default="validation")  # test는 리더보드용 라벨 미공개 가능성 대비
    args = ap.parse_args()

    recs = []
    for subset, (conv, n) in TASKS.items():
        try:
            rows = fetch(subset, args.split, n)
        except Exception as e:
            print(f"[warn] {subset}: {e}")
            continue
        made = [conv(r, i) for i, r in enumerate(rows)]
        recs.extend(made)
        print(f"{subset}: {len(made)}")

    out = Path(args.out)
    if out.parent == Path("data"):
        out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {out} ({len(recs)}레코드)")


if __name__ == "__main__":
    main()
