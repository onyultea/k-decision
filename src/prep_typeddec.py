"""LocalLLaMA/typed-decisions (Apache-2.0) → K-Decision 포맷 변환.

한 케이스에 5개 타이핑드 질문(multi-field 단일패스에 최적).
score 타입은 레벨 옵션의 choice로 변환해 학습·평가 호환성 유지.
train(1.2k 케이스)은 v0.6 학습 믹스로, test(400케이스/2,000결정)는
Darwin-27B(0.812) 대결용 평가셋으로 사용한다.
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


def fetch(config, split):
    """parquet 직접 다운로드 후 행 반환 (rows API 레이트리밋 회피)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "LocalLLaMA/typed-decisions",
        f"{config}/{split}-00000-of-00001.parquet",
        repo_type="dataset",
    )
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def to_record(row, idx, with_soft=False):
    def parse(v):
        return v if isinstance(v, dict) else json.loads(v)

    questions = parse(row["questions"])
    gold = parse(row["gold"])
    state = row["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    qs, answers = [], {}
    for qid, q in questions.items():
        kind = q.get("type") or gold.get(qid, {}).get("type")
        instr = q.get("instructions", "")
        crit = q.get("criteria")
        g = gold.get(qid, {})
        probs_gold = g.get("probabilities") if isinstance(g.get("probabilities"), dict) else None
        if kind == "choice":
            if not isinstance(crit, dict) or len(crit) < 2:
                return None
            keys = list(crit)
            opts = [str(crit[k]) or str(k) for k in keys]
            label = g.get("label")
            if label not in keys:
                return None
            qd = {"id": qid, "type": "choice", "text": instr, "options": opts}
            if with_soft and probs_gold and abs(sum(probs_gold.values()) - 1) < 0.05:
                qd["soft"] = [float(probs_gold.get(k, 0.0)) for k in keys]
            qs.append(qd)
            answers[qid] = "ABCDEFGH"[keys.index(label)]
        elif kind == "noul":
            qd = {"id": qid, "type": "noul", "text": instr}
            qs.append(qd)
            label = g.get("label")
            if label is not None:  # "true"/"false" 문자열 우선
                answers[qid] = "Y" if str(label).lower() in ("true", "yes", "y", "1") else "N"
            else:
                p_true = g.get("noul", 0.5)
                answers[qid] = "Y" if float(p_true) > 0.5 else "N"
        elif kind == "score":
            # criteria = 레벨 설명 리스트, gold.label = "0".."N" 인덱스 문자열
            if not isinstance(crit, list) or len(crit) < 2:
                return None
            opts = [str(c) for c in crit]
            qd = {"id": qid, "type": "choice", "text": instr + " (수준 선택)",
                  "options": opts}
            label = g.get("label")
            matched = None
            if label is not None and str(label).isdigit() and int(label) < len(opts):
                matched = int(label)
            if matched is None:
                return None
            if with_soft and probs_gold and abs(sum(probs_gold.values()) - 1) < 0.05:
                qd["soft"] = [float(probs_gold.get(str(i), 0.0)) for i in range(len(opts))]
            qs.append(qd)
            answers[qid] = "ABCDEFGH"[matched]
        else:
            return None
    if len(qs) < 2:
        return None
    return {"id": f"tdc-{idx:05d}", "domain": "typeddec",
            "state": state, "questions": qs, "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", default="data/typeddec_train.jsonl")
    ap.add_argument("--out-test", default="data/typeddec_test.jsonl")
    ap.add_argument("--soft-train", action="store_true",
                    help="train에 gold 주석자 확률 소프트라벨 부착")
    args = ap.parse_args()

    for split, out in (("train", args.out_train), ("test", args.out_test)):
        recs, dropped = [], 0
        rows = fetch("all", split)
        for i, row in enumerate(rows):
            r = to_record(row, i, with_soft=(split == "train" and args.soft_train))
            if r:
                recs.append(r)
            else:
                dropped += 1
        p = Path(out)
        if p.parent == Path("data"):
            p.parent.mkdir(exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_dec = sum(len(r["questions"]) for r in recs)
        n_soft = sum(1 for r in recs for q in r["questions"] if "soft" in q)
        print(f"{split}: {len(recs)}케이스 {n_dec}결정 (소프트 {n_soft}, 변환실패 {dropped})")


if __name__ == "__main__":
    main()
