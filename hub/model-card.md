---
license: apache-2.0
base_model:
  - Qwen/Qwen3.5-9B
tags:
  - decision-model
  - korean
  - structured-decision
  - calibrated-probabilities
  - typed-decisions
  - system-one
language:
  - ko
  - en
library_name: transformers
pipeline_tag: zero-shot-classification
model-index:
  - name: K-Decision-9B
    results:
      - task:
          type: classification
        dataset:
          name: LocalLLaMA/typed-decisions (test, 400 cases / 2,000 decisions)
          type: benchmark
        metrics:
          - type: accuracy
            value: 0.818
            name: accuracy (fitted)
      - task:
          type: classification
        dataset:
          name: skt/kobest_v1 (5-task conversion, 2,600 items)
          type: benchmark
        metrics:
          - type: accuracy
            value: 0.866
      - task:
          type: classification
        dataset:
          name: bzantium/korean-jev-decision-index (3,000 items)
          type: benchmark
        metrics:
          - type: accuracy
            value: 0.635
---

# K-Decision-9B v0.1

**Korean-specialized decision model by [ONYULTEA](https://onyultea.kr)** — first public release.

K-Decision is a lightweight decision engine that takes a **state** and **typed questions** as input and returns structured decisions with **calibrated probability distributions** over all options — all in a **single forward pass**. Unlike chat LLMs, it does not generate text; it predicts only the answer-symbol logits for each question (System One / Jev compatible).

## Benchmarks

| Benchmark | Score | Scale |
|-----------|-------|-------|
| **LocalLLaMA typed-decisions** (test 400 cases / 2,000 decisions, *fitted*) | **0.818** | Agent traces · Security · CS · Invoice (4 domains, 5 questions each) |
| **KoBEST 5-task conversion** (boolq/copa/hellaswag/sentineg/wic) | **0.866** | 2,600 items |
| **Korean Life-Domain Holdout** | **0.997** | Admin(20 services) · Medical · Shopping CS · Life Infra · Education · Workplace · Agent (7 domains) |
| **korean-jev-decision-index** | **0.635** | 64 Korean benchmark sources, 3,000 items |
| **University Regulation Compliance** | **0.791** | Novel task (3 regulation documents) |
| **Decision Index 0.3 public suite** | Index 30.7 (raw 44.8) | 140K requests, 0 errors, p50 latency 168ms |

- Trained and served on a **single 24GB consumer GPU** (int8 LoRA, r=32)
- Calibration: ECE 0.001 level (domain holdout), per-type temperature fitting

## Quickstart

### 1) Hugging Face transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "ONYULTEA/K-Decision-9B", torch_dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained("ONYULTEA/K-Decision-9B")
```

### 2) `/v1/systemone` server (System One / Jev compatible)

```bash
git clone https://github.com/onyultea/k-decision
python src/serve_systemone.py --merged ONYULTEA/K-Decision-9B --port 8111
```

```jsonc
// Request
{"model": "k-decision",
 "state": "Civil complaint: Strong gas smell reported (call center)",
 "questions": {
   "route":  {"type": "choice", "instructions": "Which department handles this?",
              "criteria": {"gas": "Gas Safety Division", "water": "Waterworks", "road": "Road Maintenance"}},
   "urgent": {"type": "noul", "instructions": "Is this an urgent case?"}}}

// Response — probability for every option
{"answers": {
   "route":  {"type": "choice", "choice": "gas",
              "probabilities": {"gas": 0.87, "water": 0.05, "road": 0.08}},
   "urgent": {"type": "noul", "noul": 0.93}}}
```

Question types: `choice` (2–26 options), `noul` (yes/no). Compatible with Decision Index reproduction kit and Kev clients.

## Method

1. **Marker single-pass** — A JSON skeleton `{"route":□,"urgent":□}` is appended to the prompt; the logit immediately before each marker predicts that field's answer. All fields decided in one forward pass (no decoding loop).
2. **Constrained decoding** — Softmax restricted to legal answer symbols (A–Z / Y,N) → guaranteed sum-to-1 probabilities for all options.
3. **Calibration** — Per-type (choice/noul) temperature fitted on held-out data to optimize Brier/ECE — probabilities are trust scores.
4. **int8 LoRA training** — 4-bit quantization was measured to degrade probability precision; int8 adopted instead (+3pp accuracy at same budget).

## Training Data

- **Korean life-domain synthetic** (~28K records): 20 government services, medical triage, shopping CS, life infrastructure (real estate/transport/telecom), education, workplace, agent tool gating — with honorific/casual register, typo, and slang augmentation.
- **General typed decisions**: tasksource-jev-typed-decisions (CC-BY-4.0) sample, multilingual distillation (Apache-2.0, teacher-probability soft labels), tool-decision pool (CC-BY-4.0).
- Regulation compliance task: public university regulation documents (topic routing, numeric-threshold violation, applicability).

## Limitations

- Choice up to 26 options, state up to 6,000 characters — exceeding limits returns `Unsupported` (HTTP 422, honest rejection).
- World knowledge (KMMLU 45.4%) is at 9B base level after fine-tuning (+22pp over zero-shot 23.4%) — fine-tuning dramatically improves knowledge expression.
- High-stakes decisions (medical treatment, credentialing, hiring) require human review.
- High probability does not guarantee correctness — threshold-based operation recommended.

## License

- Model weights: **Apache-2.0** (derived from Qwen3.5-9B)
- Training & evaluation code: [github.com/onyultea/k-decision](https://github.com/onyultea/k-decision)

## Citation

```bibtex
@software{kdecision2026,
  title = {K-Decision-9B: Korean-specialized decision model},
  author = {ONYULTEA},
  year = {2026},
  url = {https://huggingface.co/ONYULTEA/K-Decision-9B}
}
```

---

# K-Decision-9B v0.1 (한국어)

**[ONYULTEA](https://onyultea.kr) 한국 특화 디시전 모델** — 첫 공개 릴리스.

K-Decision은 상태(state)와 타이핑드 질문(typed questions)을 입력받아, 모든 옵션에 걸쳐 **보정된 확률 분포**를 붙인 구조화 결정을 **단일 forward 패스**로 반환하는 경량 의사결정 엔진입니다. 일반 챗 LLM과 달리 텍스트를 생성하지 않고, 각 질문의 정답 심볼 로짓만 예측합니다 (System One / Jev 호환).

## 성능

| 벤치마크 | 점수 | 규모 |
|----------|------|------|
| **LocalLLaMA typed-decisions** (2,000결정, *fitted*) | **0.818** | 에이전트 트레이스·보안·CS·인보이스 4도메인 |
| **KoBEST 5태스크 변환** | **0.866** | 2,600문항 |
| **한국 생활 도메인 홀드아웃** | **0.997** | 행정·의료·쇼핑CS·생활인프라·교육·직장·에이전트 7도메인 |
| **korean-jev-decision-index** | **0.635** | 64개 한국 벤치마크 소스 3,000문항 |
| **대학 규정 준거 판단** | **0.791** | 규정 3종 기반 신규 태스크 |
| **Decision Index 0.3 공개 스위트** | 지수 30.7 | 14만 요청, 에러 0, 지연 p50 168ms |

- 학습·서빙 모두 **단일 24GB 소비자 GPU** (int8 LoRA, r=32)
- 확률 보정: ECE 0.001 수준, 타입별 온도 피팅

## 빠른 시작

### 허깅페이스 transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "ONYULTEA/K-Decision-9B", torch_dtype=torch.bfloat16, device_map="auto")
tok = AutoTokenizer.from_pretrained("ONYULTEA/K-Decision-9B")
```

### `/v1/systemone` 서버

```bash
git clone https://github.com/onyultea/k-decision
python src/serve_systemone.py --merged ONYULTEA/K-Decision-9B --port 8111
```

질문 타입: `choice`(2~26옵션), `noul`(예/아니오). Decision Index 킷·Kev 클라이언트 호환.

## 작동 원리

1. **마커 단일패스** — JSON 스킬레톤의 마커(□) 직전 로짓이 해당 필드 결정을 예측, N개 필드 1회 forward 동시 결정
2. **제약 디코딩** — 합법 심볼(A~Z / Y,N)만으로 softmax, 합계 1 보장
3. **캘리브레이션** — choice/noul 각각 온도 피팅, Brier/ECE 최적화
4. **int8 LoRA** — 4비트 양자가 확률 정밀도를 해침을 실측 확인, int8 채택

## 한계

- choice 최대 26옵션, state 최대 6,000자 — 초과 시 정직 거부
- 세계 지식(KMMLU 45.4%) — 파인튜닝으로 베이스(23.4%) 대비 +22pp 향상. 단, 고위험 의사결정에는 인간 검토 필수

© 2026 ONYULTEA · https://onyultea.kr
