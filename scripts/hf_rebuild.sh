#!/bin/bash
# HF 완전 재생성: 기존 리포 삭제(이력 포함) → 재생성 → 상세 카드 + 가중치 클린 업로드
set -e
cd /home/kitri/k-decision
export HF_TOKEN=$(cat /home/kitri/.hf-token-onyult)
REPO=ONYULTEA/K-Decision-9B

echo "[1/4] 기존 리포 삭제 (이력까지)"
/home/kitri/venvs/di/bin/python - << EOF
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
try:
    api.delete_repo(repo_id="$REPO", repo_type="model")
    print("삭제 완료")
except Exception as e:
    print("삭제 스킵:", str(e)[:80])
EOF
sleep 10

echo "[2/4] 재생성 + 상세 카드"
/home/kitri/venvs/di/bin/python - << EOF
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id="$REPO", repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="hub/model-card.md", path_in_repo="README.md",
                repo_id="$REPO", repo_type="model",
                commit_message="K-Decision-9B v0.1")
print("카드 OK")
EOF

echo "[3/4] 가중치 업로드 (17.9GB — 대역폭에 따라 수십 분)"
/home/kitri/venvs/di/bin/python - << EOF
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(folder_path="models/kd-9b-v11-merged",
                  repo_id="$REPO", repo_type="model",
                  commit_message="merged weights (int8-trained LoRA on Qwen3.5-9B)")
api.upload_folder(folder_path="models/kd-9b-v11-3090/adapter", path_in_repo="adapter/",
                  repo_id="$REPO", repo_type="model",
                  commit_message="LoRA adapter (r32)")
print("가중치 OK")
EOF

echo "[4/4] 검증"
curl -s "https://huggingface.co/$REPO/raw/main/README.md" | head -3
echo ""
echo "HF_REBUILD_DONE $REPO $(date)"
