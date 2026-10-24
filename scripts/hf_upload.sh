#!/bin/bash
# K-Decision-9B 허깅페이스 업로드 (온율티 계정)
# 사용: bash scripts/hf_upload.sh [계정]   (기본 ONYULTEA)
set -e
ORG=${1:-ONYULTEA}
TOKEN_FILE=${HF_TOKEN_FILE:-/home/kitri/.hf-token-onyult}
REPO="$ORG/K-Decision-9B"
cd /home/kitri/k-decision
export HF_TOKEN=$(cat $TOKEN_FILE)

echo "[1/3] 리포 생성: $REPO"
/home/kitri/venvs/di/bin/python - << EOF
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="$REPO", repo_type="model", exist_ok=True)
print("리포 준비 OK:", "$REPO")
EOF

echo "[2/3] 모델카드 업로드"
/home/kitri/venvs/di/bin/python - << EOF
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj="hub/model-card.md",
    path_in_repo="README.md",
    repo_id="$REPO",
    repo_type="model",
)
print("카드 OK")
EOF

echo "[3/3] 병합 모델(17GB) + 어댑터 업로드 — 대역폭에 따라 30분~수시간"
/home/kitri/venvs/di/bin/python - << EOF
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path="models/kd-9b-v11-merged",
    repo_id="$REPO",
    repo_type="model",
    commit_message="K-Decision-9B v0.1 merged weights (int8-trained LoRA on Qwen3.5-9B)",
)
api.upload_folder(
    folder_path="models/kd-9b-v11-3090/adapter",
    path_in_repo="adapter/",
    repo_id="$REPO",
    repo_type="model",
    commit_message="LoRA adapter (r32)",
)
print("업로드 완료:", "$REPO")
EOF
echo "HF_UPLOAD_DONE $REPO"
