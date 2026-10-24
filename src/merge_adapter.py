"""LoRA 어댑터를 베이스에 병합해 독립 모델 디렉터리로 저장 (배포/리더보드 제출용)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--marker_meta", default=None,
                    help="kdecision_meta.json (마커 정보 복사용)")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from train import load_model

    model = load_model(args.base, for_training=False)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model = model.to(torch.float16)

    tok = AutoTokenizer.from_pretrained(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    if args.marker_meta:
        meta = json.loads(Path(args.marker_meta).read_text(encoding="utf-8"))
    else:
        import common as C

        meta = {"marker": C.resolve_marker(tok)}
    meta["base"] = args.base
    meta["merged_from_adapter"] = str(args.adapter)
    (out / "kdecision_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] 저장 완료: {out}")


if __name__ == "__main__":
    main()
