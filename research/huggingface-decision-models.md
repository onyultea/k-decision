# 허깅페이스 디시전 모델 수집 결과 (2026-10-24 기준)

조사 방법: HF Models API (`search=decision`, `search=decision-transformer`) 다운로드순 정렬 + 웹 검색.
본 문서는 K-Decision 기획서(docs/k-decision-plan.md)의 근거 자료다.

---

## 1. 카테고리 개관: 두 갈래의 "디시전 모델"

### A. 고전 계열 — Decision Transformer (RL, 2021~)
- Chen et al. 2021, "Decision Transformer: Reinforcement Learning via Sequence Modeling" (arXiv:2106.01345, 인용 3,500+)
- RL을 조건부 시퀀스 모델링으로 재해석: (return-to-go, state, action) 시퀀스를 autoregressive 학습
- HF에는 주로 edbeeching의 Gym 실험 체크포인트(hopper/halfcheetah/walker2d)가 존재 — 규모·다운로드 모두 작음(최대 ~4K)
- 최신 서베이: Transformers in RL Survey (arXiv:2307.05979), opendilab/awesome-decision-transformer
- **위치**: 교육/연구용. 신규 개발의 베이스로는 부적합

### B. 신규 계열 — LLM 기반 의사결정 특화 모델 (2026년 하반기 급부상) ★
"상태 + 타이핑드 질문(typed questions) → 보정된 확률이 붙은 구조화 의사결정"을 단일 forward 패스로 출력하는
경량 특화 모델 패밀리. 대형 추론 LLM 앞단의 라우터/게이터/분류기 역할이 핵심 용도.

#### Intern-Decision (Shanghai AI Lab / InternLM) — 기술 레퍼런스 1순위
- 링크: github.com/internlm/Intern-Decision (Apache-2.0), arXiv:2610.22131, HF: internlm/Intern-Decision-{0.8B,2B,4B}
- 베이스: Qwen3.5 멀티모달 (비전타워/프로젝터 동결, LM만 파인튜닝), 사이즈 0.8B/2B/4B (+9B 학습설정)
- 목적함수: **autoregressive masked LM** — 입력에 `<decision>` 마커만 있는 JSON 스켈레톤을 넣고,
  각 마커 직전 로짓이 해당 필드의 정답 심볼을 예측 → 모든 필드가 **단일 causal forward 패스**로 결정됨
- 추론: 합법적 답 심볼로 로짓 제한 → softmax → 레이블 매핑, + 온도 스케일링 보정(체크포인트별 T 지정, 4B 기준 T=1.99)
- 질문 타입: **choice / score / noul(예·아니오)** 3종
- 성능: 평균 정확도 90.02% (7개 스위트, 10,751행), RTX 4090 BF16에서 44ms(4B), Brier 0.347 / ECE 0.065
- 대규모 LLM 대비 수십~수백 배 저렴·저지연이 승부수

#### StartLux-Decision — AutoThink 접근
- 링크: arXiv:2610.17309, HF: startlux-models/StartLux-Decision-v1/v1.1 (0.8B~35B), github.com/StartLuxLabs/Startlux-Decision
- **AutoThink**: 확신이 높으면 즉답(single-pass), 불확실할 때만 "think" 모드로 심층 추론 → 신뢰도 기반 적응 추론
- 3단계 RL 파이프라인 학습. Decision Index 0.3.2에서 174개 참가 모델 중 1위 (Full 69+)
- GGUF 변형 포함 다운로드 큰 흐름 — 로컬 수요 검증됨

#### vllm-sr Decision-2.0 / 3.0 시리즈
- Vega-27B, Lux-9B, Nox-4B, Kai-0.6B 등 (0.6B~27B, 멀티모달). "Decision 3.0"이 Jev Decision Index 양보드 1위 주장
- 파이프라인 태그가 zero-shot-classification으로 되어 있음 — 범용 zero-shot 분류기 포지션

#### 기타 동향
- Cloudflare "Clef" (상용, blog.cloudflare.com/clef-decision-models) — 상용 클라우드 진입 신호
- farbodtavakkoli/OTel-2.0-Decision-26B-A4B-4bit — 검색어 내 다운로드 1위(55만+)로 카테고리 수요 방증
- chaoliangUNSW/Jev-Style-* 시리즈 — 'Jev' 포맷 계열 커뮤니티 모델 (Jevbench MIT)

### C. 평가 인프라 — Jev Decision Index (리더보드)
- HF Space: huggingface.co/spaces/multimodalart/jev-decision-index (v0.4.1)
- Full score = **공개 스킬과 동일한 비공개 테스트 60% + 완전 비공개 태스크 40%** (오버피팅/게이밍 방지)
- chance-corrected 점수(0=랜덤, 100=만점), **proper scoring rules 기반 캘리브레이션 보상**, 지연시간(p50/p95) 추적
- 참고 기록: Decision Index 0.3 기준 예시 결과 — 149,432 요청, chance-corrected 17.43, median 60.8ms

---

## 2. HF 수집 원본 데이터 (search=decision, 다운로드순 상위 50)

| # | 모델 | 다운로드 | 좋아요 | 태스크 | 계열 비고 |
|---|------|---------|--------|--------|-----------|
| 1 | farbodtavakkoli/OTel-2.0-Decision-26B-A4B-4bit | 554,663 | 3 | text-generation | OTel 계열 |
| 2 | startlux-models/StartLux-Decision-v1.1-27B-AutoThink-GGUF | 32,544 | 10 | - | StartLux |
| 3 | startlux-models/StartLux-Decision-v1-9B-Q8_0-GGUF | 16,371 | 0 | - | StartLux |
| 4 | startlux-models/StartLux-Decision-v1.1-4B-Instant-GGUF | 14,738 | 4 | - | StartLux |
| 5 | startlux-models/StartLux-Decision-v1.1-27B-Instant-GGUF | 13,872 | 1 | - | StartLux |
| 6 | chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-GGUF | 12,852 | 27 | text-generation | Jev-Style |
| 7 | startlux-models/StartLux-Decision-v1-27B-Q8_0-GGUF | 12,102 | 6 | - | StartLux |
| 8 | startlux-models/StartLux-Decision-v1.1-4B-AutoThink-GGUF | 10,575 | 2 | - | StartLux |
| 9 | mradermacher/torchcast-decision-12b-GGUF | 10,002 | 0 | - | 서드파티 양자화 |
| 10 | startlux-models/StartLux-Decision-v1-4B-Q8_0-GGUF | 9,891 | 2 | - | StartLux |
| 11 | mradermacher/StartLux-Decision-35B-A3B-GGUF | 8,745 | 0 | - | 서드파티 양자화 |
| 12 | startlux-models/StartLux-Decision-v1-27B-Q4_K_M-GGUF | 8,614 | 0 | - | StartLux |
| 13 | startlux-models/StartLux-Decision-v1-9B-Q4_K_M-GGUF | 8,384 | 2 | - | StartLux |
| 14 | chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF | 8,275 | 22 | text-classification | Jev-Style |
| 15 | bombdefuser-124/Intern-Decision-4B-GGUF | 8,006 | 0 | image-text-to-text | Intern GGUF |
| 16 | startlux-models/StartLux-Decision-v1.1-9B-AutoThink-GGUF | 6,694 | 3 | - | StartLux |
| 17 | startlux-models/StartLux-Decision-v1-27B-BF16-GGUF | 5,835 | 0 | - | StartLux |
| 18 | Futurix-AI/FxAI-decision-model | 5,602 | 1 | text-classification | 기타 |
| 19 | chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16 | 5,245 | 5 | text-generation | Jev-Style(MLX) |
| 20 | **internlm/Intern-Decision-4B** | 5,220 | 90 | image-text-to-text | **공식 Intern** |
| 21 | chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF | 5,185 | 8 | text-classification | Jev-Style |
| 22 | startlux-models/StartLux-Decision-v1.1-9B-Instant-GGUF | 5,180 | 0 | - | StartLux |
| 23 | startlux-models/StartLux-Decision-v1-4B-Q4_K_M-GGUF | 4,930 | 1 | - | StartLux |
| 24 | mradermacher/torchcast-decision-27b-i1-GGUF | 4,859 | 2 | - | 서드파티 |
| 25 | startlux-models/StartLux-Decision-v1.1-9B-Instant | 4,600 | 1 | - | StartLux 원본 |
| 26 | startlux-models/StartLux-Decision-v1-35B-A3B | 4,552 | 1 | - | StartLux 원본(MoE) |
| 27 | startlux-models/StartLux-Decision-v1.1-27B-Instant | 4,405 | 12 | - | StartLux 원본 |
| 28 | edbeeching/decision-transformer-gym-hopper-medium | 3,937 | 7 | reinforcement-learning | **고전 RL DT** |
| 29 | convaiinnovations/laya-typed-decisions | 3,754 | 166 | text-classification | 타이핑드 결정 |
| 30 | **internlm/Intern-Decision-0.8B** | 3,672 | 37 | image-text-to-text | 공식 Intern |
| 31 | vllm-sr/Decision-2.0-Vega-27B-GGUF | 3,519 | 0 | zero-shot-classification | vllm-sr |
| 32 | startlux-models/StartLux-Decision-v1-0.8B-Q8_0-GGUF | 3,300 | 2 | - | StartLux |
| 33 | **vllm-sr/Decision-2.0-Vega-27B** | 3,105 | 77 | zero-shot-classification | vllm-sr 공식 |
| 34 | vllm-sr/Decision-2.0-Nox-4B | 2,825 | 32 | zero-shot-classification | vllm-sr |
| 35 | **internlm/Intern-Decision-2B** | 2,591 | 18 | image-text-to-text | 공식 Intern |
| 36 | chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2-GGUF | 2,396 | 7 | text-generation | Jev-Style |
| 37 | evalengine/decision-4b-gguf | 2,383 | 2 | text-classification | 기타 |
| 38 | vllm-sr/Decision-2.0-Kai-0.6B | 2,346 | 43 | zero-shot-classification | vllm-sr |
| 39 | startlux-models/StartLux-Decision-v1.1-0.8B-Instant-GGUF | 2,185 | 1 | - | StartLux |
| 40 | mradermacher/torchcast-decision-27b-GGUF | 2,159 | 1 | - | 서드파티 |
| 41 | startlux-models/StartLux-Decision-v1-2B-Q8_0-GGUF | 1,998 | 1 | - | StartLux |
| 42 | startlux-models/StartLux-Decision-v1-9B-BF16-GGUF | 1,984 | 0 | - | StartLux |
| 43 | mradermacher/docto-decision-gemma-4-e4b-fr-v0.1-GGUF | 1,983 | 0 | - | 프랑스어 특화 선례! |
| 44 | littlemex/prismyra-decision-qwen3.6-35b-a3b-fp8-36l | 1,960 | 1 | text-classification | 기타 |
| 45 | startlux-models/StartLux-Decision-v1-4B-BF16-GGUF | 1,887 | 0 | - | StartLux |
| 46 | startlux-models/StartLux-Decision-v1.1-2B-Instant-GGUF | 1,805 | 1 | - | StartLux |
| 47 | vllm-sr/Decision-2.0-Lux-9B | 1,763 | 43 | zero-shot-classification | vllm-sr |
| 48 | infosave/cmf-decision | 1,674 | 2 | text-classification | 기타(cortiq) |
| 49 | chaoliangUNSW/Jev-Style-2B-Decision-v3 | 1,481 | 5 | text-classification | Jev-Style 원본 |
| 50 | mradermacher/StartLux-Decision-v1-9B-GGUF | 1,479 | 1 | - | 서드파티 |

### search=decision-transformer (RL 계열, 상위 요약)
- edbeeching/decision-transformer-gym-hopper-medium 3,937 / -expert 853 / -halfcheetah-expert 85 외
- 대부분 Gym(고전 RL) 체크포인트 + onnx-community의 transformers.js 변환. 학술·교육 자산.

---

## 3. 핵심 인사이트

1. **빈 자리**: 수집 50개 중 한국어/한국 도메인 특화 디시전 모델은 **0개**. 프랑스어 특화(docto-decision-gemma)는 이미 존재 → 언어 특화 파생 모델의 유효성 입증.
2. **GGUF가 지배**: 다운로드 상위권 대부분이 양자화/GGUF 버전 → 로컬·엣지·저비용 추론 수요가 실수요의 핵심. K-Decision도 GGUF/ONNX 동시 배포 필수.
3. **포맷 표준 경쟁 중**: Jef 포맷(Jev-Style), Intern-Decision 포맷(choice/score/noul), StartLux(AutoThink) 등. 아직 확정 표준 없음 → 초기 진입자가 한국어 표준을 정의할 기회.
4. **평가 = 캘리브레이션**: Decision Index는 정답률만이 아니라 Brier/ECE 등 보정 품질과 지연시간을 평가. "확률이 믿을 수 있는가"가 이 카테고리의 핵심 경쟁력.
5. **사이즈 스펙트럼**: 0.6B~35B까지 존재하나 실용 core는 0.8B~9B. 한국어 특화는 2B(엣지)와 8B(서버) 양동작이 합리적.

## 4. 참고 링크
- Intern-Decision: https://github.com/internlm/Intern-Decision / https://arxiv.org/abs/2610.22131
- StartLux-Decision: https://arxiv.org/html/2610.17309v1 / https://github.com/StartLuxLabs/Startlux-Decision
- Decision Transformer 원논문: https://arxiv.org/abs/2106.01345 / https://github.com/kzl/decision-transformer
- Jev Decision Index: https://huggingface.co/spaces/multimodalart/jev-decision-index
- Cloudflare Clef: https://blog.cloudflare.com/clef-decision-models
- Transformers in RL 서베이: https://arxiv.org/abs/2307.05979
- 한국 기반모델: EXAONE 3.5 (https://www.lgresearch.ai/blog/view?seq=507), EXAONE 4.0 (https://www.lgresearch.ai/blog/view?seq=575), KMMLU-Pro (https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro)
- HF Deep RL Course Decision Transformers 유닛: https://huggingface.co/learn/deep-rl-course/en/unitbonus3/decision-transformers
