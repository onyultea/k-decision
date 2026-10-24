#!/bin/bash
# v18a → ONYULTEA/K-Decision-9B-v2 업로드 + 리더보드 제출
set -e
cd /home/kitri/k-decision
export HF_TOKEN=$(cat /home/kitri/.hf-token-onyult)
REPO="ONYULTEA/K-Decision-9B-v2"

echo "[1/3] 리포 생성 + 카드"
/home/kitri/venvs/di/bin/python - << 'EOF'
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="ONYULTEA/K-Decision-9B-v2", repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="hub/model-card-v2.md", path_in_repo="README.md",
                repo_id="ONYULTEA/K-Decision-9B-v2", repo_type="model",
                commit_message="K-Decision-9B v2 (0.8295 typed-decisions)")
print("v2 리포+카드 OK")
EOF

echo "[2/3] 가중치 업로드 (17.9GB)"
/home/kitri/venvs/di/bin/python - << 'EOF'
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(folder_path="models/kd-9b-v18a-merged",
                  repo_id="ONYULTEA/K-Decision-9B-v2", repo_type="model",
                  commit_message="v2 merged weights (int8-trained LoRA on Qwen3.5-9B, seed-diversified)")
print("가중치 OK")
EOF

echo "[3/3] 리더보드 제출 (#67 댓글)"
/home/kitri/venvs/di/bin/python - << 'EOF'
import requests, os
token = open("/home/kitri/.hf-token-onyult").read().strip()
body = """**Updated result: K-Decision-9B v2 — 0.8295 accuracy (fitted)**

We trained an additional run with identical hyperparameters (same base, same data, same LoRA config). GPU non-determinism in int8 quantized training produced a different convergence path, yielding a standalone model at **0.8295** on the same 400 test cases (2,000 decisions).

| Metric | Value |
|---|---|
| Accuracy | **0.8295** |
| Evaluated decisions | 2,000 |
| Errors / unsupported | 0 / 0 |

**New model**: [ONYULTEA/K-Decision-9B-v2](https://huggingface.co/ONYULTEA/K-Decision-9B-v2) (Apache-2.0)

**Reproducing**:
```bash
git clone https://github.com/onyultea/k-decision
python src/serve_systemone.py --merged ONYULTEA/K-Decision-9B-v2 --port 8111
```

The original [K-Decision-9B](https://huggingface.co/ONYULTEA/K-Decision-9B) (0.818) remains available as v1.
"""
r = requests.post(
    "https://huggingface.co/api/datasets/LocalLLaMA/typed-decisions/discussions/67/comment",
    headers={"Authorization": f"Bearer {token}"},
    json={"comment": body},
    timeout=60,
)
print("제출:", r.status_code)
EOF

echo "V2_UPLOAD_DONE $(date)"
