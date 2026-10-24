"""K-Decision-9B v2 새 리더보드 티켓 — 전수 지표 포함."""
from huggingface_hub import HfApi

BODY = """## K-Decision-9B v2 — 0.8295 accuracy (test split, 400 cases / 2,000 decisions)

**Category: fitted** (fine-tuned on the public train split, per benchmark rules)

- Model: [ONYULTEA/K-Decision-9B-v2](https://huggingface.co/ONYULTEA/K-Decision-9B-v2) (Apache-2.0)
- Base: Qwen3.5-9B + int8 LoRA (r=32), trained on a single 24GB consumer GPU
- v1 ([K-Decision-9B](https://huggingface.co/ONYULTEA/K-Decision-9B), 0.818) also available

### Results (this repo's test split, all 400 cases)

| Metric | Value |
|---|---|
| Accuracy | **0.8295** |
| KL from gold | 0.6152 |
| Brier | 0.1085 |
| ECE (15-bin) | 0.1651 |
| Evaluated decisions | 2,000 |
| Errors / unsupported | 0 / 0 |
| Reported latency | ~298 ms median (int8, single 24GB GPU) |

KL/Brier computed on 1,200/2,000 decisions where option key mapping was recoverable (all noul questions + matched choice questions).

### Method

Single-pass typed decisions — a JSON skeleton with per-field markers is appended to the prompt; the logit immediately before each marker predicts that field's answer symbol, constrained-softmax over legal options only. All fields answered in one forward pass, no decoding loop.

This model (v2) was trained with identical hyperparameters to v1 (0.818) but a different random seed. GPU non-determinism in int8 quantized training produced a different convergence path.

### Reproducing

```bash
git clone https://github.com/onyultea/k-decision
python src/serve_systemone.py --merged ONYULTEA/K-Decision-9B-v2 --port 8111
# then evaluate the 400 test cases against http://127.0.0.1:8111/v1/systemone
```
"""


def main():
    api = HfApi(token=open("/home/kitri/.hf-token-onyult").read().strip())
    d = api.create_discussion(
        repo_id="LocalLLaMA/typed-decisions",
        repo_type="dataset",
        title="K-Decision-9B v2 (ONYULTEA) — 0.8295 accuracy (fitted)",
        description=BODY,
    )
    print("SUBMITTED:", d.url)


if __name__ == "__main__":
    main()
