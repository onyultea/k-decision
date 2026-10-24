"""KMMLU(HAERAE-HUB/KMMLU, CC-BY-ND-4.0) → K-Decision 평가 포맷 변환.

평가(evaluation) 전용 변환기다. ND 조항 때문에 학습 데이터로는 사용하지 않는다.
datasets-server API 로 전체 다운로드 없이 test 일부를 가져온다.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co"
ALLOWED_HOSTS = {"datasets-server.huggingface.co"}


def _assert_safe_url(url: str):
    """SSRF 방지: https 고정, 호스트 allowlist, 사설/루프백 IP 차단."""
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


def fetch_rows(dataset, config, split, total):
    rows, offset = [], 0
    while len(rows) < total:
        url = f"{API}/rows?dataset={urllib.parse.quote(dataset, safe='')}" \
              f"&config={config}&split={split}&offset={offset}&length=100"
        data = get(url)
        batch = [r["row"] for r in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows[:total]


def map_row(row, idx):
    """컬럼명을 탐지해 (question, options, gold_index) 추출."""
    keys = list(row.keys())
    qkey = next((k for k in keys if k.lower() in ("question", "문제")), keys[0])
    question = str(row[qkey])
    # 선택지: 리스트이거나 Choice1..5 / A..E 컬럼
    options, gold = None, None
    if any(isinstance(row.get(k), list) for k in keys):
        ckey = next(k for k in keys if isinstance(row.get(k), list))
        options = [str(o) for o in row[ckey]]
    else:
        opt_cols = [k for k in keys if k.lower().startswith(("choice", "option"))]
        if opt_cols:
            options = [str(row[k]) for k in sorted(opt_cols)]
        else:
            abc = [k for k in keys if len(k) == 1 and k.upper() in "ABCDE"]
            options = [str(row[k]) for k in sorted(abc, key=lambda x: x.upper())]
    gkey = next((k for k in keys if k.lower() in ("answer", "gold", "label", "정답")), None)
    if gkey is not None:
        g = row[gkey]
        if isinstance(g, (int, float)):
            gold = int(g) if int(g) < len(options) else int(g) - 1
        else:
            s = str(g).strip().upper()
            if s in "ABCDE"[: len(options)]:
                gold = "ABCDE".index(s)
            else:
                for i, o in enumerate(options):
                    if o.strip() == str(g).strip():
                        gold = i
                        break
    if not options or gold is None or gold >= len(options):
        return None
    return {
        "id": f"kmmlu-{idx:05d}", "domain": "kmmlu",
        "state": f"문제: {question}",
        "questions": [{"id": "q0", "type": "choice",
                       "text": "정답은?", "options": options}],
        "answers": {"q0": "ABCDEFGH"[gold]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kmmlu_eval_test.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--split", default="test")
    ap.add_argument("--subjects", type=int, default=10,
                    help="과목 수(과목당 n/subjects 행 샘플링)")
    args = ap.parse_args()

    splits = get(f"{API}/splits?dataset=HAERAE-HUB%2FKMMLU")
    cfgs = sorted({c["config"] for c in splits.get("splits", [])
                   if c["split"] == args.split})
    print(f"가용 {args.split} 과목 {len(cfgs)}개")
    rng = __import__("random").Random(7)
    subjects = rng.sample(cfgs, min(args.subjects, len(cfgs)))
    per = max(args.n // len(subjects), 10)

    rows, recs, dropped = [], [], 0
    for subj in subjects:
        try:
            subj_rows = fetch_rows("HAERAE-HUB/KMMLU", subj, args.split, per)
        except Exception as e:
            print(f"[skip] {subj}: {e}")
            continue
        for row in subj_rows:
            rec = map_row(row, len(recs))
            if rec:
                rec["domain"] = f"kmmlu:{subj}"
                rows.append(row)
                recs.append(rec)
            else:
                dropped += 1

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"과목 {len(subjects)}개 / 저장: {out} ({len(recs)}레코드, {dropped}매핑실패)")
    if rows:
        print("샘플 검증:", json.dumps(recs[0]["questions"][0]["options"][:2], ensure_ascii=False),
              "gold:", recs[0]["answers"]["q0"])


if __name__ == "__main__":
    main()
