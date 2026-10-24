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
  - name: K-Decision-9B-v2
    results:
      - task:
          type: classification
        dataset:
          name: LocalLLaMA/typed-decisions (test, 400 cases / 2,000 decisions)
          type: benchmark
        metrics:
          - type: accuracy
            value: 0.8295
            name: accuracy (fitted)
---

# K-Decision-9B v2

**Korean-specialized decision model by [ONYULTEA](https://onyultea.kr)**

K-Decision is a lightweight decision engine that takes a **state** and **typed questions** as input and returns structured decisions with **calibrated probability distributions** over all options — all in a **single forward pass**. System One / Jev compatible.

## Benchmarks

| Benchmark | Score |
|-----------|-------|
| **LocalLLaMA typed-decisions** (2,000 decisions, fitted) | **0.8295** |
| KoBEST 5-task conversion (2,600 items) | **0.866** |
| Korean Life-Domain Holdout (7 domains) | **0.997** |
| korean-jev-decision-index (3,000 items) | **0.635** |
| Decision Index 0.3 public suite (140K requests) | Index 30.7 · 0 errors · p50 168ms |

- Trained and served on a **single 24GB consumer GPU** (int8 LoRA, r=32)

## Quickstart

### `/v1/systemone` server

```bash
git clone https://github.com/onyultea/k-decision
python src/serve_systemone.py --merged ONYULTEA/K-Decision-9B-v2 --port 8111
```

```jsonc
// Request
{"state": "Civil complaint: Strong gas smell (call center)",
 "questions": {
   "route":  {"type": "choice", "instructions": "Which department?",
              "criteria": {"gas": "Gas Safety", "water": "Waterworks", "road": "Roads"}},
   "urgent": {"type": "noul", "instructions": "Urgent?"}}}
// Response — probability for every option
{"answers": {
   "route":  {"type": "choice", "choice": "gas",
              "probabilities": {"gas": 0.87, "water": 0.05, "road": 0.08}},
   "urgent": {"type": "noul", "noul": 0.93}}}
```

---

# K-Decision-9B v2 (한국어)

**[ONYULTEA](https://onyultea.kr) 한국 특화 디시전 모델**

상태(state)와 타이핑드 질문을 입력받아 모든 옵션에 **보정된 확률 분포**를 **단일 forward 패스**로 반환 (System One / Jev 호환).

## 성능

| 벤치마크 | 점수 |
|----------|------|
| **LocalLLaMA typed-decisions** (2,000결정) | **0.8295** |
| **KoBEST 5태스크** | **0.866** |
| **한국 생활 도메인** (7도메인) | **0.997** |
| **korean-jev-decision-index** | **0.635** |
| Decision Index 0.3 (14만 요청) | 지수 30.7 · 에러 0 |

## 빠른 시작

```bash
git clone https://github.com/onyultea/k-decision
python src/serve_systemone.py --merged ONYULTEA/K-Decision-9B-v2 --port 8111
```

© 2026 ONYULTEA · https://onyultea.kr
