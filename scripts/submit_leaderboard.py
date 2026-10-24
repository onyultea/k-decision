"""LocalLLaMA/typed-decisions 리더보드 제출 (fitted 부문) — 데이터셋 토론창 게시."""
import os

from huggingface_hub import HfApi

BODY = """## K-Decision-9B — 0.818 accuracy (test split, 400 cases / 2,000 decisions)

**Category: fitted** (fine-tuned on the public train split, per benchmark rules)

- Model: [ONYULTEA/K-Decision-9B](https://huggingface.co/ONYULTEA/K-Decision-9B) (Apache-2.0)
- Base: Qwen3.5-9B + int8 LoRA (r=32), trained on a single 24GB consumer GPU
- Method: single-pass typed decisions — a JSON skeleton with per-field markers is appended to the prompt; the logit immediately before each marker predicts that field's answer symbol, constrained-softmax over legal options only. All fields answered in one forward pass, no decoding loop.
- Calibration: per-question-type temperature fitted on held-out data; full option probability distributions are returned for every choice.

### Results (this repo's test split, all 400 cases)

| Metric | Value |
|---|---|
| Accuracy | **0.818** |
| Evaluated decisions | 2,000 |
| Errors / unsupported | 0 / 0 |

### Reproducing

```
git clone https://github.com/onyultea/k-decision
python src/serve_systemone.py --merged ONYULTEA/K-Decision-9B --port 8111
# then evaluate the 400 test cases against http://127.0.0.1:8111/v1/systemone
```

The served model implements the System One request/response format (`state` + typed `questions` -> `probabilities`), compatible with this benchmark's per-question gold structure.
"""


def main():
    api = HfApi(token=open("/home/kitri/.hf-token-onyult").read().strip())
    d = api.create_discussion(
        repo_id="LocalLLaMA/typed-decisions",
        repo_type="dataset",
        title="K-Decision-9B (ONYULTEA) — 0.818 accuracy (fitted)",
        description=BODY,
    )
    print("SUBMITTED:", d.url)


if __name__ == "__main__":
    main()
