"""K-Decision LoRA SFT — <decision> 마커 단일패스 목적함수.

학습 입력의 어시스턴트 스킬레톤에는 각 필드 자리에 마커 토큰만 있고,
labels 의 마커 위치에만 정답 심볼 토큰 id 를 넣는다. 즉 마커 직전 위치의
로짓이 해당 필드의 결정을 예측하도록 학습 → 추론 시 스킬레톤 1회 forward
로 모든 필드가 동시에 결정된다 (Intern-Decision 방식).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import common as C


class DecisionDataset(Dataset):
    def __init__(self, records, tokenizer, marker_id, sym2id, max_len=768):
        self.samples = []
        skipped = 0
        for rec in records:
            msgs = [
                {"role": "user", "content": C.build_user_content(rec)},
                {"role": "assistant", "content": C.build_assistant_skeleton(rec)},
            ]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > max_len:
                skipped += 1
                continue
            marker_pos = [i for i, t in enumerate(ids) if t == marker_id]
            if len(marker_pos) != len(rec["questions"]):
                skipped += 1
                continue
            fields = []
            for pos, q in zip(marker_pos, rec["questions"]):
                syms = C.field_symbols(q)
                cand = [sym2id[s] for s in syms]
                soft = q.get("soft") if isinstance(q.get("soft"), list) else None
                if soft and len(soft) == len(syms) and sum(soft) > 0:
                    p = [max(float(x), 1e-6) for x in soft]
                    ssum = sum(p)
                    probs = [x / ssum for x in p]
                else:
                    gold = C.gold_symbol(rec, q)
                    probs = [1.0 if s == gold else 0.0 for s in syms]
                fields.append({"pos": pos, "cand_ids": cand, "probs": probs})
            self.samples.append({"input_ids": ids, "fields": fields})
        if skipped:
            print(f"[data] {len(self.samples)}사용 / {skipped}스킵(길이·마커)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, fields = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        attn.append([1] * n + [0] * pad)
        fields.append(b["fields"])
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "fields": fields,
    }


class SoftLabelTrainer(Trainer):
    """마커 위치별 후보 심볼 제약 로짓에 소프트 타깃 CE (Oumi 레시피).

    하드라벨 샘플은 one-hot 타깃으로 동일 경로를 탄다 — 추론(후보 제한
    softmax)과 정확히 같은 확률 분포를 최적화한다.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        fields = inputs.pop("fields")
        inputs.pop("labels", None)
        out = model(input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"])
        logits = out.logits.float()
        total = logits.new_tensor(0.0)
        count = 0
        for i, flds in enumerate(fields):
            for f in flds:
                l = logits[i, f["pos"] - 1, f["cand_ids"]]
                logp = torch.log_softmax(l, dim=-1)
                tgt = logits.new_tensor(f["probs"])
                total = total - (tgt * logp).sum()
                count += 1
        loss = total / max(count, 1)
        return (loss, out) if return_outputs else loss


def load_model(base: str, for_training: bool = True, dtype=None,
               load_in_4bit: bool = False, load_in_8bit: bool = False):
    """Qwen3.5 계열(텍스트 혹은 VL 통합)을 안전하게 로드. 비전 파라미터는 동결."""
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = dict(torch_dtype=dtype, attn_implementation="sdpa")
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        kind = "causal"
    except Exception:
        model = None
    if model is None:
        model = AutoModelForImageTextToText.from_pretrained(base, **kwargs)
        kind = "vl"
    n_frozen = 0
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in ("visual", "vision", "merger", "projector")):
            p.requires_grad_(False)
            n_frozen += 1
    if n_frozen:
        print(f"[model] 비전 관련 파라미터 {n_frozen}개 동결")
    print(f"[model] 로드: {base} ({kind})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--data", default="data/synth_v01_train.jsonl")
    ap.add_argument("--out", default="models/kd-0.8b-v0.1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--bf16", action="store_true",
                    help="Ampere(3090) 이상에서 bf16 학습 (Llama계/kanana 등 fp16 불안정 모델용)")
    ap.add_argument("--init_adapter", default=None,
                    help="기존 어댑터에서 이어학습 (경로 지정)")
    ap.add_argument("--max_len", type=int, default=768,
                    help="학습 시퀀스 상한 (버려지는 샘플 방지)")
    ap.add_argument("--qlora", action="store_true",
                    help="4비트 QLoRA (9B급을 3090에 올릴 때)")
    ap.add_argument("--int8", action="store_true",
                    help="8비트 LoRA — 4비트보다 16배 정밀, 확률 판단 태스크에 유리")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    marker = C.resolve_marker(tokenizer)
    marker_id = tokenizer.encode(marker, add_special_tokens=False)[0]
    symbols = sorted(set(C.CHOICE_SYMBOLS + C.NOUL_SYMBOLS))
    sym2id = {s: C.symbol_ids(tokenizer, [s])[0] for s in symbols}
    print(f"[marker] {marker!r} → id {marker_id}")

    records = C.load_jsonl(args.data)
    ds = DecisionDataset(records, tokenizer, marker_id, sym2id, max_len=args.max_len)

    model = load_model(args.base,
                       dtype=torch.bfloat16 if (args.bf16 or args.qlora or args.int8) else torch.float16,
                       load_in_4bit=args.qlora, load_in_8bit=args.int8)
    model.enable_input_require_grads()
    model.config.use_cache = False

    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"[init] 어댑터 이어학습: {args.init_adapter}")
    else:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    load_kwargs = dict(torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                       attn_implementation="sdpa")
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        fp16=not args.bf16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=42,
    )
    trainer = SoftLabelTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, pad_id),
    )
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "tokenizer")
    meta = {"base": args.base, "marker": marker, "marker_id": marker_id,
            "sym2id": sym2id, "data": str(args.data)}
    import json

    (out / "kdecision_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] 어댑터 저장: {out/'adapter'}")


if __name__ == "__main__":
    main()
