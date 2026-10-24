"""모델 스프: v0.6 + v0.7 LoRA 어댑터 산술평균 → 통합 어댑터 (같은 베이스 전제)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="어댑터 A (예: v0.6)")
    ap.add_argument("--b", required=True, help="어댑터 B (예: v0.7)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="A 가중치")
    args = ap.parse_args()

    sa = load_file(str(Path(args.a) / "adapter_model.safetensors"))
    sb = load_file(str(Path(args.b) / "adapter_model.safetensors"))
    assert set(sa) == set(sb), "어댑터 키 불일치 — 다른 베이스/구성"
    out = {}
    for k in sa:
        ta, tb = sa[k], sb[k]
        if ta.dtype.is_floating_point:
            out[k] = (args.alpha * ta.float() + (1 - args.alpha) * tb.float()).to(ta.dtype)
        else:
            out[k] = ta
    outp = Path(args.out)
    outp.mkdir(parents=True, exist_ok=True)
    save_file(out, str(outp / "adapter_model.safetensors"))
    # 설정 복사(A 기준)
    for f in ("adapter_config.json",):
        src = Path(args.a) / f
        if src.exists():
            (outp / f).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[soup] {args.alpha:.2f}*A + {1-args.alpha:.2f}*B → {outp} ({len(out)} 텐서)")


if __name__ == "__main__":
    main()
