# K-Decision

**한국 특화 디시전 모델 (Korean-specialized Decision Model)** — by [ONYULTEA](https://onyultea.kr)

상태(state)와 타이핑드 질문(typed questions)을 입력받아, 모든 옵션에 **보정된 확률 분포**를 붙인 구조화 결정을 **단일 forward 패스**로 반환하는 경량 의사결정 엔진의 학습·평가·서빙 파이프라인입니다. 일반 챗 LLM과 달리 텍스트를 생성하지 않고 정답 심볼 로짓만 예측합니다 (System One / Jev 호환).

## 모델

**[ONYULTEA/K-Decision-9B](https://huggingface.co/ONYULTEA/K-Decision-9B)** (Apache-2.0)
Qwen3.5-9B + int8 LoRA — 단일 24GB 소비자 GPU로 학습·서빙

| 벤치마크 | 점수 |
|----------|------|
| LocalLLaMA typed-decisions (2,000결정, fitted) | **0.818** |
| KoBEST 5태스크 변환 (2,600문항) | **0.866** |
| 한국 생활 도메인 홀드아웃 (7도메인) | **0.997** |
| korean-jev-decision-index (3,000문항) | **0.635** |
| 대학 규정 준거 판단 (규정 3종) | **0.791** |
| Decision Index 0.3 공개 스위트 (14만 요청) | 지수 30.7 · 에러 0 · 지연 p50 168ms |

## 빠른 시작

### 서빙 (System One 호환 API)

```bash
pip install -r requirements.txt
python src/serve_systemone.py --merged ONYULTEA/K-Decision-9B --port 8111
```

```jsonc
// POST /v1/systemone
{"state": "민원 접수: 가스 냄새가 심하게 나요 (콜센터)",
 "questions": {
   "route":  {"type": "choice", "instructions": "담당 부서는?",
              "criteria": {"gas": "가스안전과", "water": "상하수도과", "road": "도로과"}},
   "urgent": {"type": "noul", "instructions": "긴급 처리 대상인가?"}}}

// 응답 — 모든 옵션에 확률
{"answers": {
   "route":  {"type": "choice", "choice": "gas",
              "probabilities": {"gas": 0.87, "water": 0.05, "road": 0.08}},
   "urgent": {"type": "noul", "noul": 0.93}}}
```

### 학습 (자체 데이터로)

```bash
# 1) 데이터 생성 (한국 생활 7도메인 예시)
python src/gen_klife_data.py --out data

# 2) 마커 단일패스 LoRA 학습
python src/train.py --base Qwen/Qwen3.5-9B --data data/klife_v02_train.jsonl \
  --out models/my-kdecision --epochs 2 --bs 2 --accum 8 --int8 --bf16 --max_len 1088

# 3) 평가 (제약 디코딩 + 타입별 온도 보정)
python src/eval.py --base Qwen/Qwen3.5-9B --adapter models/my-kdecision/adapter \
  --meta models/my-kdecision/kdecision_meta.json --data data/klife_v02_test.jsonl
```

## 작동 원리

1. **마커 단일패스** — 프롬프트에 JSON 스킬레톤(`{"route":□,"urgent":□}`)을 붙이고, 마커 직전 로짓이 해당 필드의 결정을 예측. N개 필드가 1회 forward로 동시 결정 (디코딩 루프 없음)
2. **제약 디코딩** — 합법 심볼(A~Z / Y,N)만으로 softmax → 합계 1의 옵션 확률 보장
3. **캘리브레이션** — choice/noul 각각 홀드아웃 온도 피팅, ECE 0.001 수준
4. **int8 LoRA** — 4비트 양자가 확률 정밀도를 해침을 실측 확인(MNIST가 아니라 확률이 본체)하고 int8 채택

## 저장소 구성

```
src/
  common.py              # 스키마·프롬프트 빌더·마커 처리
  train.py               # 마커 단일패스 LoRA/SFT (fp16·bf16·int8·4bit)
  eval.py                # 제약 디코딩 평가 + Brier/ECE + 타입별 온도 보정
  serve_systemone.py     # /v1/systemone 서버 (Decision Index 킷·Kev 호환)
  gen_klife_data.py      # 한국 생활 7도메인 합성 데이터 생성기
  gen_synth_data.py      # 기초 4도메인 생성기
  gen_generic_data.py    # 일반 지식 믹스 (OBQA/ARC/한국 상식)
  gen_regulation_data.py # 규정 준거 판단 데이터 생성기 (규정 텍스트 → 타이핑드 결정)
  prep_kobest.py         # KoBEST → 타이핑드 결정 변환
  prep_kjdi.py           # korean-jev-decision-index → 평가셋 변환
  prep_typeddec.py       # LocalLLaMA/typed-decisions 변환 (소프트라벨 지원)
  convert_tasksource.py  # tasksource-jev-typed-decisions 대규모 변환
  merge_adapter.py       # LoRA 병합
data/                    # 생성·변환된 데이터셋
hub/model-card.md        # HF 모델카드 원본
research/                # 디시전 모델 시장 조사
```

## 지원하는 질문 타입

- `choice` — 2~26옵션, 모든 옵션에 확률 분포
- `noul` — 예/아니오, P(yes) 반환

## 라이선스

- 코드·데이터: 저장소 라이선스 참조
- 모델: Apache-2.0 ([ONYULTEA/K-Decision-9B](https://huggingface.co/ONYULTEA/K-Decision-9B))

## 인용

```bibtex
@software{kdecision2026,
  title = {K-Decision: Korean-specialized decision model},
  author = {ONYULTEA},
  year = {2026},
  url = {https://github.com/onyultea/k-decision}
}
```

© 2026 ONYULTEA
