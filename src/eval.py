"""K-Decision 평가 — 단일 forward 패스 제약 디코딩 + 캘리브레이션.

마커 위치 직전 로짓을 후보 심볼 토큰으로 제한해 softmax 확률을 얻고,
정확도 / Brier / ECE 와 (옵션) 검증셋 피팅 온도 T 를 적용한 지표를 계산한다.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

import common as C
from train import load_model


@torch.no_grad()
def score_records(model, tokenizer, records, marker_id, batch_size=4):
    """각 레코드의 필드별 확률 분포를 1회 forward 로 산출."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    results = []
    model.eval()
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        seqs, metas = [], []
        for rec in chunk:
            msgs = [
                {"role": "user", "content": C.build_user_content(rec)},
                {"role": "assistant", "content": C.build_assistant_skeleton(rec)},
            ]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            marker_pos = [i for i, t in enumerate(ids) if t == marker_id]
            assert len(marker_pos) == len(rec["questions"]), rec["id"]
            metas.append((rec, marker_pos))
            seqs.append(ids)
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s)
            attn[i, : len(s)] = 1
        input_ids, attn = input_ids.cuda(), attn.cuda()
        logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
        for i, (rec, marker_pos) in enumerate(metas):
            per_field = []
            for q, p in zip(rec["questions"], marker_pos):
                cand_ids = C.symbol_ids(tokenizer, C.field_symbols(q))
                logits_at = logits[i, p - 1, cand_ids]
                probs = torch.softmax(logits_at, dim=-1).cpu()
                per_field.append(
                    {"qid": q["id"], "type": q["type"], "domain": rec["domain"],
                     "gold": C.gold_symbol(rec, q),
                     "symbols": C.field_symbols(q),
                     "logits": logits_at.cpu().tolist(), "probs": probs.tolist()}
                )
            results.append({"id": rec["id"], "fields": per_field})
    return results


def metrics_of(results, T=1.0):
    def field_view(f):
        lg = torch.tensor(f["logits"]) / T
        p = torch.softmax(lg, dim=-1)
        pred = f["symbols"][int(p.argmax())]
        gold_idx = f["symbols"].index(f["gold"])
        onehot = torch.zeros_like(p)
        onehot[gold_idx] = 1.0
        return f, pred, gold_idx, p, float((p - onehot).pow(2).sum()), f["domain"], f["type"]

    accs, briers, confs, corrects = [], [], [], []
    by_domain = defaultdict(lambda: [0, 0])  # acc: [hit, total]
    by_type = defaultdict(lambda: [0, 0])
    nll_sum, nll_n = 0.0, 0
    for r in results:
        for f in results_f(r):
            _, pred, gold_idx, p, brier, dom, typ = field_view(f)
            hit = int(pred == f["gold"])
            accs.append(hit)
            corrects.append(hit)
            briers.append(brier)
            confs.append(float(p.max()))
            by_domain[dom][0] += hit
            by_domain[dom][1] += 1
            by_type[typ][0] += hit
            by_type[typ][1] += 1
            nll_sum += float(-torch.log(p[gold_idx] + 1e-12))
            nll_n += 1

    acc = sum(accs) / len(accs)
    brier = sum(briers) / len(briers)
    # ECE 15-bin
    bins = torch.tensor(confs), torch.tensor(corrects, dtype=torch.float)
    conf_t, corr_t = bins
    ece = 0.0
    for k in range(15):
        lo, hi = k / 15, (k + 1) / 15
        m = (conf_t > lo) & (conf_t <= hi) if k else (conf_t >= 0) & (conf_t <= hi)
        if m.any():
            ece += float(m.float().mean()) * abs(float(corr_t[m].mean()) - float(conf_t[m].mean()))
    return {
        "acc": acc, "brier": brier, "ece": ece, "nll": nll_sum / nll_n,
        "n": len(accs),
        "by_domain": {k: v[0] / v[1] for k, v in by_domain.items()},
        "by_type": {k: v[0] / v[1] for k, v in by_type.items()},
    }


def results_f(r):
    return r["fields"]


def fit_temperature(results):
    """타입별 NLL 최소화 온도 (choice/noul 각각) — Oumi 레시피."""
    Ts = {}
    for typ in ("choice", "noul"):
        sub = {"id": "v", "fields": [f for r in results for f in r["fields"]
                                     if f["type"] == typ]}
        if not sub["fields"]:
            Ts[typ] = 1.0
            continue
        best = (1.0, float("inf"))
        for T in [0.3 + 0.05 * i for i in range(100)]:
            m = metrics_of([sub], T)
            if m["nll"] < best[1]:
                best = (T, m["nll"])
        Ts[typ] = best[0]
    return Ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--meta", default=None, help="학습 시 저장된 kdecision_meta.json")
    ap.add_argument("--data", default="data/synth_v01_test.jsonl")
    ap.add_argument("--val", default="data/synth_v01_val.jsonl",
                    help="온도 피팅용(없으면 미사용)")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--int8", action="store_true",
                    help="9B 베이스 int8 평가 (장문 OOM 방지)")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if args.meta:
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        marker_id = meta["marker_id"]
    else:
        marker = C.resolve_marker(tokenizer)
        marker_id = tokenizer.encode(marker, add_special_tokens=False)[0]

    model = load_model(args.base, for_training=False,
                       dtype=torch.bfloat16 if args.int8 else None,
                       load_in_8bit=args.int8)
    if not args.int8:
        model = model.cuda()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        if not args.int8:  # int8은 병합·cuda 이동 불가 → 어댑터 추론 그대로
            model = model.merge_and_unload()
            print(f"[eval] 어댑터 병합: {args.adapter}")
        else:
            print(f"[eval] int8 어댑터 추론(병합 없음): {args.adapter}")
    if not args.int8:
        model.cuda()

    records = C.load_jsonl(args.data)[: args.limit]
    results = score_records(model, tokenizer, records, marker_id, args.bs)

    report = {"tag": args.tag, "base": args.base,
              "adapter": args.adapter, "n_records": len(records)}
    report["raw"] = metrics_of(results, T=1.0)

    if args.val and Path(args.val).exists():
        val_records = C.load_jsonl(args.val)[: 400]
        val_results = score_records(model, tokenizer, val_records, marker_id, args.bs)
        T_map = fit_temperature(val_results)
        report["fitted_T"] = T_map
        # 타입별 온도를 로짓에 선적용 후 통합 지표 산출
        scaled = {"id": "cal", "fields": []}
        for r in results:
            for f in r["fields"]:
                g = dict(f)
                t = T_map.get(f["type"], 1.0)
                g["logits"] = [l / t for l in f["logits"]]
                scaled["fields"].append(g)
        report["calibrated"] = metrics_of([scaled], T=1.0)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    path = out / f"{args.tag}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(m):
        return (f"acc={m['acc']*100:5.1f}%  brier={m['brier']:.4f}  "
                f"ece={m['ece']:.4f}  (n={m['n']})")

    print(f"\n== {args.tag} ==")
    print(f"raw        : {fmt(report['raw'])}")
    if "calibrated" in report:
        t_info = report["fitted_T"]
        t_str = "/".join(f"{k}:{v:.2f}" for k, v in t_info.items()) \
            if isinstance(t_info, dict) else f"{t_info:.2f}"
        print(f"calibrated : {fmt(report['calibrated'])}  T[{t_str}]")
    print("도메인별 정확도:", {k: f"{v*100:.0f}%" for k, v in report["raw"]["by_domain"].items()})
    print("타입별 정확도:", {k: f"{v*100:.0f}%" for k, v in report["raw"]["by_type"].items()})
    print(f"리포트: {path}")


if __name__ == "__main__":
    main()
