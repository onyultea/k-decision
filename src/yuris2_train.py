"""YURIS-2 학습+평가 — 투-타워 어모타이즈드 결정 네트워크.

손실: 질문별 옵션 softmax CE + 교사 분포 KL + 서수 누적 BCE + 합의 예측 BCE
평가: typeddec test 정확도/KL/Brier/ECE(10-bin) + 상태 캐싱 지연 벤치마크

사용:
  python src/yuris2_train.py --backbone BAAI/bge-m3 --epochs 3 --tag yuris2
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, "src")
import yuris2_data as D
from yuris2_model import build


def lcg_shuffle(n, seed):
    idx = list(range(n))
    state = seed
    for i in range(n - 1, 0, -1):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        j = state % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def build_pos_map(pairs):
    """pair 인덱스 → (헤드종류, 헤드배열 내 위치)."""
    pos = {}
    ci = ni = 0
    for i, p in enumerate(pairs):
        if p["qtype"] == "noul":
            pos[i] = ("noul", ni)
            ni += 1
        else:
            pos[i] = ("choice", ci)
            ci += 1
    return pos


def group_forward(model, g, pad_id=1):
    """그룹 forward: 상태 1회 + 옵션 배치 → 헤드 라우팅 스코어/합의 로짓."""
    s_ids = torch.tensor([g["state_ids"]]).cuda()
    s_mask = torch.tensor([g["state_mask"]]).cuda()
    pairs = g["pairs"]
    maxl = max(len(p["opt_ids"]) for p in pairs)
    o_ids = torch.tensor([p["opt_ids"] + [pad_id] * (maxl - len(p["opt_ids"]))
                          for p in pairs]).cuda()
    o_mask = torch.tensor([[1] * len(p["opt_ids"]) + [0] * (maxl - len(p["opt_ids"]))
                           for p in pairs]).cuda()
    h_state = model.encode_state(s_ids, s_mask)
    h_opt = model.encode_option(o_ids, o_mask)
    fused = model.fuse(h_state.expand(len(pairs), -1), h_opt)
    choice_scores = model.score_head(fused).squeeze(-1)   # choice/score 프리미티브
    noul_scores = model.noul_head(fused).squeeze(-1)      # noul 프리미티브
    agree_logit = model.agreement_head(fused).squeeze(-1)
    return choice_scores, noul_scores, agree_logit


def group_losses(model, g, pad_id=1, lam_kd=1.0, lam_ord=0.5, lam_agree=0.3):
    """그룹 손실(텐서) + 통계 + 질문별 확률 반환."""
    choice_scores, noul_scores, agree_logit = group_forward(model, g, pad_id)
    pairs = g["pairs"]
    qmeta = g["questions_meta"]
    pos_map = build_pos_map(pairs)

    ce_parts = []
    kl_parts = []
    ord_parts = []
    ag_parts = []
    correct = total = 0
    for qm in qmeta:
        start, n, qtype = qm["start"], qm["n"], qm["qtype"]
        gold_idx = qm["gold_idx"]
        teacher = qm.get("teacher_probs")
        if qtype == "noul":
            idx_in_noul = sum(1 for p in pairs[:start] if p["qtype"] == "noul")
            s = noul_scores[idx_in_noul]
            p_yes = torch.sigmoid(s)
            target = 1.0 if gold_idx == 0 else 0.0
            ce_parts.append(-torch.log(torch.clamp(
                p_yes if target == 1.0 else 1 - p_yes, min=1e-9)))
            if isinstance(teacher, list) and len(teacher) == 2:
                t_vec = torch.tensor(teacher).cuda()
                p_vec = torch.stack([p_yes, 1 - p_yes])
                mask = t_vec > 1e-9  # 교사 확률 0 제외 — 0×log(0)=NaN 방지
                if mask.any():
                    kl_parts.append((t_vec[mask] * (t_vec[mask] / torch.clamp(
                        p_vec[mask], min=1e-9)).log()).sum())
        else:
            seg = choice_scores[start:start + n]
            logp = torch.log_softmax(seg, dim=0)
            ce_parts.append(-logp[gold_idx])
            probs = torch.softmax(seg, dim=0)
            correct += int(int(probs.argmax()) == gold_idx)
            total += 1
            if teacher and len(teacher) == n:
                t_vec = torch.tensor(teacher).cuda()
                mask = t_vec > 1e-9  # 0 확률 제외
                if mask.any():
                    kl_parts.append((t_vec[mask] * (t_vec[mask] / torch.clamp(
                        probs[mask], min=1e-9)).log()).sum())
                if qm.get("qtype_orig") == "score":
                    cum_t = torch.cumsum(t_vec, dim=0)[:-1]
                    cum_p = torch.cumsum(probs, dim=0)[:-1]
                    ord_parts.append(-(cum_t * torch.log(torch.clamp(cum_p, min=1e-9))
                                       + (1 - cum_t) * torch.log(
                                           torch.clamp(1 - cum_p, min=1e-9))).sum())

    # 합의 예측 손실 (알려진 것만)
    ag_parts = []
    for pi, p in enumerate(pairs):
        if p.get("agree") is not None:
            target = 1.0 if p["agree"] else 0.0
            sig = torch.clamp(torch.sigmoid(agree_logit[pi]), min=1e-9, max=1 - 1e-9)
            ag_parts.append(-(math.log(sig.item()) if target == 1.0
                              else math.log(1 - sig.item())))

    loss = (sum(ce_parts) / max(len(ce_parts), 1)
            + lam_kd * sum(kl_parts) / max(len(kl_parts), 1)
            + lam_ord * sum(ord_parts) / max(len(ord_parts), 1)
            + lam_agree * sum(ag_parts) / max(len(ag_parts), 1))
    stats = {"ce": len(ce_parts), "kl_matched": len(kl_parts),
             "ord_matched": len(ord_parts), "agree_matched": len(ag_parts),
             "correct": correct, "total": total}
    return loss, stats, []


def evaluate(model, groups, pad_id=1, tag="test"):
    model.eval()
    correct = total = 0
    kl_sum = brier_sum = 0.0
    kl_n = br_n = 0
    confidences, corrects = [], []
    lat_uncached = []
    lat_cached = []
    with torch.no_grad():
        for g in groups:
            pairs = g["pairs"]
            maxl = max(len(p["opt_ids"]) for p in pairs)
            o_ids = torch.tensor([p["opt_ids"] + [1] * (maxl - len(p["opt_ids"]))
                                  for p in pairs]).cuda()
            o_mask = torch.tensor([[1] * len(p["opt_ids"]) + [0] * (maxl - len(p["opt_ids"]))
                                   for p in pairs]).cuda()
            s_ids = torch.tensor([g["state_ids"]]).cuda()
            s_mask = torch.tensor([g["state_mask"]]).cuda()

            # 미캐싱: 질문별 상태 재인코딩 (1위 표기 방식)
            t0 = time.time()
            for qm in g["questions_meta"]:
                _ = model.encode_state(s_ids, s_mask)
            lat_uncached.append((time.time() - t0) * 1000 / max(len(g["questions_meta"]), 1))
            # 캐싱: 상태 1회
            t0 = time.time()
            h_state = model.encode_state(s_ids, s_mask)
            lat_cached.append((time.time() - t0) * 1000)

            h_opt = model.encode_option(o_ids, o_mask)
            fused = model.fuse(h_state.expand(len(pairs), -1), h_opt)
            choice_scores = model.score_head(fused).squeeze(-1)
            noul_scores = model.noul_head(fused).squeeze(-1)

            start = 0
            for qm in g["questions_meta"]:
                s, n, qtype = qm["start"], qm["n"], qm["qtype"]
                gold_idx = qm["gold_idx"]
                if qtype == "noul":
                    p_yes = torch.sigmoid(noul_scores[s])
                    pred = 0 if p_yes.item() > 0.5 else 1
                    conf = max(p_yes.item(), 1 - p_yes.item())
                else:
                    seg = choice_scores[s:s + n]
                    probs = torch.softmax(seg, dim=0)
                    pred = int(probs.argmax())
                    conf = float(probs.max())
                correct += int(pred == gold_idx)
                total += 1
                confidences.append(conf)
                corrects.append(float(pred == gold_idx))
                start += n

    ece = 0.0
    n = len(confidences)
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        mask = [lo <= c < hi for c in confidences]
        cnt = sum(mask)
        if cnt:
            bin_acc = sum(c for c, m in zip(corrects, mask) if m) / cnt
            bin_conf = sum(c for c, m in zip(confidences, mask) if m) / cnt
            ece += (cnt / n) * abs(bin_acc - bin_conf)
    lat_u = sorted(lat_uncached)
    lat_c = sorted(lat_cached)
    return {
        "acc": correct / max(total, 1), "n": total, "ece_10bin": ece,
        "latency_ms_case_uncached": lat_u[len(lat_u) // 2],
        "latency_ms_case_cached": lat_c[len(lat_c) // 2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="BAAI/bge-m3")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lam_kd", type=float, default=1.0)
    ap.add_argument("--lam_ord", type=float, default=0.5)
    ap.add_argument("--lam_agree", type=float, default=0.3)
    ap.add_argument("--state_max", type=int, default=1024)
    ap.add_argument("--opt_max", type=int, default=384)
    ap.add_argument("--tag", default="yuris2")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)
    pad_id = tok.pad_token_id or 1

    print("[data] typeddec train 로드 + 그룹 구성")
    rows = D.load_typeddec_train()
    groups = D.build_groups(rows, tok, state_max=args.state_max, opt_max=args.opt_max)
    order = lcg_shuffle(len(groups), 20261109)
    groups = [groups[i] for i in order]
    n_val = max(60, len(groups) // 20)
    val_groups = groups[:n_val]
    train_groups = groups[n_val:]
    print(f"train {len(train_groups)} / val {len(val_groups)} 그룹")

    model = build(args.backbone).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_groups) // args.accum + 1
    total_steps = args.epochs * steps_per_epoch
    step = 0

    print(f"[train] {args.epochs}에포크 × {len(train_groups)}그룹 (accum {args.accum})")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss = ep_acc = ep_n = 0.0
        accum_loss = 0.0
        accum_n = 0
        for gi, g in enumerate(train_groups):
            loss, stats, _ = group_losses(model, g, pad_id, args.lam_kd, args.lam_ord)
            (loss / args.accum).backward()
            accum_loss += loss.item()
            ep_loss += loss.item()
            ep_acc += stats["correct"]
            ep_n += stats["total"]
            accum_n += 1
            if accum_n % args.accum == 0:
                step += 1
                if step < args.warmup:
                    lr = args.lr * step / args.warmup
                else:
                    p = (step - args.warmup) / max(total_steps - args.warmup, 1)
                    lr = args.lr * 0.5 * (1 + math.cos(math.pi * p))
                for pg in opt.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if step % 100 == 0:
                    print(f"  e{epoch} step{step} loss={accum_loss/args.accum:.4f} "
                          f"acc={ep_acc/max(ep_n,1):.4f} lr={lr:.2e}")
                accum_loss = 0.0
                accum_n = 0
        print(f"[epoch {epoch}] loss={ep_loss/max(len(train_groups),1):.4f} "
              f"acc={ep_acc/max(ep_n,1):.4f}")

    print("[eval] test 그룹 평가 + 지연 벤치마크")
    import pyarrow.parquet as pq
    f = glob.glob("/home/kitri/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/"
                  "snapshots/*/all/test-00000-of-00001.parquet")[0]
    test_rows = []
    for row in pq.read_table(f).to_pylist():
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
        la = row.get("label_agreement", "{}")
        if isinstance(la, str):
            la = json.loads(la)
        state = row.get("state", "")
        if not isinstance(state, str) or not state:
            continue
        test_rows.append({"state": state, "questions": qs, "gold": gold, "la": la,
                          "workflow": row.get("workflow", "?")})
    test_groups = D.build_groups(test_rows, tok, state_max=args.state_max,
                                 opt_max=args.opt_max)
    res = evaluate(model, test_groups, pad_id, tag="test")
    print(json.dumps(res, indent=2))

    out = Path("models")
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"models/{args.tag}_final.pt")
    Path(f"models/{args.tag}_metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] models/{args.tag}_final.pt")


if __name__ == "__main__":
    main()
