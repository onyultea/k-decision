"""격전지 풀메트릭 산출 — gold 확률 대비 KL/Brier/ECE + 정확도 + 서빙 지연.

리더보드 표 형식(acc/KL/Brier/ECE/latency)에 맞춰 #67 제출 보완용.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

import common as C
from train import load_model


def gold_probs_map():
    """test parquet에서 id별 질문별 gold 확률 추출."""
    f = glob.glob("/home/kitri/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/"
                  "snapshots/*/all/test-00000-of-00001.parquet")[0]
    out = {}
    for row in pq.read_table(f).to_pylist():
        gold = row["gold"]
        if isinstance(gold, str):
            gold = json.loads(gold)
        out[row["id"]] = gold
    return out


def norm(p, n):
    p = p + 1e-9
    return p / p.sum()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    marker_id = json.loads(Path(args.meta).read_text(encoding="utf-8"))["marker_id"]
    model = load_model("Qwen/Qwen3.5-9B", for_training=False,
                       dtype=torch.bfloat16, load_in_8bit=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    gold_map = gold_probs_map()
    recs = C.load_jsonl("data/typeddec_test.jsonl")[: args.limit]

    # 레코드 id ↔ 원본 gold 매핑은 순서 기반(typeddec 변환은 all/test 순서 보존)
    orig_ids = list(gold_map.keys())
    n_correct = n_total = 0
    kl_sum = brier_sum = 0.0
    latencies = []
    for i, rec in enumerate(recs):
        g = gold_map[orig_ids[i]] if i < len(orig_ids) else {}
        msgs = [{"role": "user", "content": C.build_user_content(rec)},
                {"role": "assistant", "content": C.build_assistant_skeleton(rec)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        t0 = time.time()
        logits = model(input_ids=ids).logits[0].float()
        torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000)
        pos = (ids[0] == marker_id).nonzero(as_tuple=True)[0].tolist()
        for q, p in zip(rec["questions"], pos):
            syms = C.CHOICE_SYMBOLS[: len(q["options"])] if q["type"] == "choice" else C.NOUL_SYMBOLS
            cand = C.symbol_ids(tok, syms)
            probs = torch.softmax(logits[p - 1, cand], dim=-1)
            pred = syms[int(probs.argmax())]
            gold_sym = C.gold_symbol(rec, q)
            n_correct += int(pred == gold_sym)
            n_total += 1
            # gold 분포 대비 KL/Brier
            gd = g.get(q["id"], {})
            gp = gd.get("probabilities") if isinstance(gd, dict) else None
            if q["type"] == "noul" and isinstance(gp, dict):
                keys = ["true", "false"]
                if set(gp) <= set(keys):
                    gold_vec = torch.tensor([gp.get("true", 0.5), gp.get("false", 0.5)])
                    pred_vec = torch.tensor([float(probs[0]), float(probs[1])])  # [P(Y),P(N)]
                    gold_vec = norm(gold_vec, 2); pred_vec = norm(pred_vec, 2)
                    kl_sum += float((gold_vec * (gold_vec / pred_vec).log()).sum())
                    brier_sum += float(((gold_vec - pred_vec) ** 2).sum())
            elif q["type"] == "choice" and isinstance(gp, dict):
                keys = q.get("_orig_keys") or list(gp)
                # 옵션이 'key: 설명' 조합으로 바뀌었으므로 매칭 불가 시 스킵
                if len(keys) == len(probs):
                    try:
                        gold_vec = torch.tensor([float(gp[k]) for k in keys])
                        gold_vec = norm(gold_vec, len(keys))
                        pred_vec = norm(probs, len(keys))
                        kl_sum += float((gold_vec * (gold_vec / pred_vec).log()).sum())
                        brier_sum += float(((gold_vec - pred_vec) ** 2).sum())
                    except Exception:
                        pass
    lat = sorted(latencies)
    out = {
        "accuracy": n_correct / n_total,
        "n": n_total,
        "kl_from_gold_approx": kl_sum / max(n_total, 1),
        "brier_approx": brier_sum / max(n_total, 1),
        "latency_ms_median": lat[len(lat) // 2],
        "latency_ms_p95": lat[int(len(lat) * 0.95)],
        "note": "KL/Brier는 noul+매칭가능 choice에 한한 근사(옵션 텍스트 변형으로 전수 매칭 불가)",
    }
    print(json.dumps(out, indent=2))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/final_metrics_v11.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
