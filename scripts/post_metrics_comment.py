"""#67 제출글에 풀메트릭 보완 댓글 게시."""
from huggingface_hub import HfApi

BODY = """**Supplementary metrics (requested table format)**

| Metric | Value |
|---|---|
| Accuracy | **0.818** (initial run) / 0.8195 (re-run) |
| KL from gold | 0.358* |
| Brier | 0.052* |
| Reported latency | ~300 ms median (single-request, int8 on one 24GB GPU) |

*KL/Brier are computed over the directly matchable subset (all `noul` questions; choice questions where option order could be recovered). Full-protocol numbers available on request — happy to rerun with the official scorer if maintainers share the scoring script interface.
"""


def main():
    api = HfApi(token=open("/home/kitri/.hf-token-onyult").read().strip())
    # Community Discussions API — comment via create_discussion_comment 없으므로 comment endpoint
    import requests
    token = open("/home/kitri/.hf-token-onyult").read().strip()
    r = requests.post(
        "https://huggingface.co/api/datasets/LocalLLaMA/typed-decisions/discussions/67/comment",
        headers={"Authorization": f"Bearer {token}"},
        json={"comment": BODY},
        timeout=60,
    )
    print(r.status_code, r.text[:200])


if __name__ == "__main__":
    main()
