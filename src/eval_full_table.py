"""v18a 전수 지표 — Accuracy/KL/Brier/ECE/Latency (리더보드 표 완전 매칭).

gold 확률과 예측 확률을 원본 criteria 키로 직접 매칭하여 정확히 계산.
"""
from __future__ import annotations

import glob
import json
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

import common as C
from train import load_model


def load_test_gold():
    """test parquet에서 id→gold, questions 추출 (원본 criteria 키 보존)."""
    f = glob.glob("/home/kitri/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/"
                  "snapshots/*/all/test-00000-of-00001.parquet")[0]
    rows = pq.read_table(f).to_pylist()
    out = []
    for row in rows:
        gold = row["gold"]
        if isinstance(gold, str):
            gold = json.loads(gold)
        qs = row["questions"]
        if isinstance(qs, str):
            qs = json.loads(qs)
        if isinstance(qs, dict):
            qs = [dict(v, id=k) for k, v in qs.items()]
        elif isinstance(qs, list) and qs and isinstance(qs[0], str):
            qs = [json.loads(q) for q in qs]
        out.append({"id": row["id"], "gold": gold, "orig_questions": qs})
    return out


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    meta = json.loads(Path("models/kd-9b-v18a-seed123/kdecision_meta.json").read_text())
    marker_id = meta["marker_id"]
    model = load_model("Qwen/Qwen3.5-9B", for_training=False,
                       dtype=torch.bfloat16, load_in_8bit=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, "models/kd-9b-v18a-seed123/adapter").eval()

    gold_rows = load_test_gold()
    recs = C.load_jsonl("data/typeddec_test.jsonl")[:400]

    n_correct = n_total = 0
    kl_sum = brier_sum = 0.0
    kl_count = brier_count = 0
    latencies = []
    # ECE용
    all_confidences = []
    all_corrects = []

    for i, rec in enumerate(recs):
        gr = gold_rows[i]
        gold = gr["gold"]
        orig_qs = gr["orig_questions"]

        msgs = [{"role": "user", "content": C.build_user_content(rec)},
                {"role": "assistant", "content": C.build_assistant_skeleton(rec)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        t0 = time.time()
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000)
        pos = (ids[0] == marker_id).nonzero(as_tuple=True)[0].tolist()

        for qi, (q, p) in enumerate(zip(rec["questions"], pos)):
            syms = C.CHOICE_SYMBOLS[:len(q["options"])] if q["type"] == "choice" else C.NOUL_SYMBOLS
            cand = C.symbol_ids(tok, syms)
            probs = torch.softmax(logits[p - 1, cand], dim=-1)
            pred_sym = syms[int(probs.argmax())]
            gold_sym = C.gold_symbol(rec, q)
            correct = pred_sym == gold_sym
            n_correct += correct
            n_total += 1
            all_confidences.append(float(probs.max()))
            all_corrects.append(correct)

            # 원본 질문에서 criteria 키 추출
            qid = q["id"]
            g = gold.get(qid, {})
            gp = g.get("probabilities", {})
            gtype = g.get("type", q["type"])
            oq = next((o for o in orig_qs if o.get("id") == qid), None)
            crit = oq.get("criteria") if oq else None

            if gtype == "noul" and isinstance(gp, dict) and "true" in gp:
                # gold: true/false → our: Y/N
                g_vec = [gp.get("true", 0.5), gp.get("false", 0.5)]
                p_vec = [float(probs[0]), float(probs[1])]  # [P(Y), P(N)]
                # KL
                kl = sum(g * math.log(max(g, 1e-10) / max(p, 1e-10))
                         for g, p in zip(g_vec, p_vec) if g > 0)
                kl_sum += kl; kl_count += 1
                # Brier
                br = sum((g - p) ** 2 for g, p in zip(g_vec, p_vec))
                brier_sum += br; brier_count += 1

            elif gtype in ("choice", "score") and isinstance(gp, dict) and len(gp) >= 2:
                if isinstance(crit, dict):
                    # 원본 criteria 키 순서로 gold와 pred 매칭
                    keys = list(crit.keys())
                    if len(keys) == len(probs):
                        g_vec = [float(gp.get(k, 0)) for k in keys]
                        s = sum(g_vec)
                        if s > 0:
                            g_vec = [v / s for v in g_vec]
                            p_vec = [float(probs[j]) for j in range(len(keys))]
                            # KL
                            kl = sum(g * math.log(max(g, 1e-10) / max(p, 1e-10))
                                     for g, p in zip(g_vec, p_vec) if g > 1e-10)
                            kl_sum += kl; kl_count += 1
                            # Brier
                            br = sum((g - p) ** 2 for g, p in zip(g_vec, p_vec))
                            brier_sum += br; brier_count += 1

    # ECE (15-bin)
    n_bins = 15
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        mask = [lo <= c < hi for c in all_confidences]
        if any(mask):
            bin_acc = sum(c for c, m in zip(all_corrects, mask) if m) / sum(mask)
            bin_conf = sum(c for c, m in zip(all_confidences, mask) if m) / sum(mask)
            ece += (sum(mask) / len(all_confidences)) * abs(bin_acc - bin_conf)

    lat_sorted = sorted(latencies)
    results = {
        "model": "ONYULTEA/K-Decision-9B-v2 (v18a)",
        "accuracy": n_correct / n_total,
        "n_decisions": n_total,
        "kl_from_gold": kl_sum / max(kl_count, 1),
        "kl_matched": kl_count,
        "brier": brier_sum / max(brier_count, 1),
        "brier_matched": brier_count,
        "ece": ece,
        "latency_ms_median": lat_sorted[len(lat_sorted) // 2],
        "latency_ms_p95": lat_sorted[int(len(lat_sorted) * 0.95)],
    }
    print(json.dumps(results, indent=2))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/v2_full_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
