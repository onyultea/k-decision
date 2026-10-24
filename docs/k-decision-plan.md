# K-Decision 모델 기획서 (v0.1)

프로젝트명: **K-Decision — 한국 특화 의사결정 모델 패밀리**
주관: **주식회사 온율티 (OnyulT, 강원 춘천 — 의료·교육·공공 AI)**

문서 버전: v0.1 (초안) / 작성일: 2026-10-24 / 갱신: 2026-10-24 (1차 개발 착수)
근거 자료: [research/huggingface-decision-models.md](../research/huggingface-decision-models.md)
도메인 확정: [docs/onyult-domain-decision.md](onyult-domain-decision.md) — 1차 특화는
공공 민원 / 의료 응대 / CS 인텐트 / 에이전트 게이팅 4개 도메인.

---

## 1. 요약 (Executive Summary)

2026년 하반기, 허깅페이스에서는 **LLM 기반 '디시전 모델(Decision Model)'**이 신규 카테고리로 급부상하고 있다.
상태(state)와 타이핑드 질문(typed questions)을 입력받아 **보정된 확률(calibrated probability)이 붙은 구조화
의사결정을 단일 forward 패스로 출력**하는 경량 특화 모델로, Shanghai AI Lab의 Intern-Decision(0.8B~4B),
StartLux-Decision(0.8B~35B, AutoThink), vllm-sr Decision-2.0(0.6B~27B) 등이 경쟁 중이며
'Jev Decision Index' 리더보드가 표준 평가 인프라로 자리 잡는 중이다.

허깅페이스 수집 결과(다운로드 상위 50개 모델) **한국어/한국 도메인 특화 디시전 모델은 0개**다.
프랑스어 특화 파생 모델(docto-decision-gemma)이 이미 존재하므로 언어-도메인 특화 파생은 검증된 패턴이며,
**한국 자리는 현재 완전히 비어 있다.**

K-Decision의 목표는 이 자리를 선점하는 것이다:

> **"한국어 질문에 대해, 한국 도메인(행정·금융·CS·콘텐츠)에서, 믿을 수 있는 확률로 의사결정을
> 반환하는 오픈 디시전 모델 패밀리 + 그것을 재는 한국형 벤치마크(K-Decision Index)"**

1차 목표(6개월): 2B/8B 두 사이즈 모델을 오픈소스 공개하고, 국제 Decision Index 호환 +
한국 특화 태스크를 포함한 K-Decision Index 0.1을 함께 릴리스하여 카테고리 내 "한국 표준" 위치 확보.

---

## 2. 배경 분석

### 2.1 디시전 모델의 두 계열

| 구분 | Decision Transformer (RL) | LLM 디시전 모델 (신규) |
|------|---------------------------|------------------------|
| 발상 | RL을 시퀀스 모델링으로 재해석 (2021) | 분류·구조화 의사결정의 특화·경량화 (2026) |
| 입출력 | 상태/행동/보상 궤적 → 행동 | 상태 + 타이핑드 질문 → 결정+확률 |
| 규모 | 소형(Gym 실험 수준) | 0.6B~35B, 상용 서비스 대상 |
| HF 현황 | 레거시/교육 자산 | 다운로드 급증, 리더보드 경쟁 중 |
| K-Decision과의 관계 | 참고 이론 | **직접 벤치마크** |

### 2.2 왜 지금인가 — 이 카테고리가 존재하는 이유
- **에이전트 시대의 게이트키퍼 수요**: 대형 추론 LLM 앞단에서 라우팅/게이팅/분류를
  수십 ms·수십 분의 1 비용으로 처리해야 하는 니즈가 폭발
- **캘리브레이션이 본질**: 일반 LLM의 확률은 불안정. 디시전 모델은 Brier/ECE 같은
  proper scoring rule로 "확률을 믿을 수 있는가"를 직접 최적화·평가
- **표준 경쟁 공백**: Jev 포맷 / Intern-Decision 포맷(choice·score·noul) / StartLux(AutoThink)이
  경쟁 중이나 아직 표준 미확정 → 초기 진입자가 언어별 표준을 정의할 창

### 2.3 벤치마크할 핵심 기술 (Intern-Decision 중심)
1. **단일 패스 결정**: JSON 스켈레톤 + `<decision>` 마커. 마커 직전 로짓이 필드 정답을 예측 →
   모든 필드를 1회 causal forward로 출력 (4B 기준 44ms)
2. **출력 제약 추론**: 합법적 답 심볼로 로짓 제한 → softmax → 레이블 매핑
3. **온도 스케일링 캘리브레이션**: 체크포인트별 보정 온도로 Brier/ECE 개선
4. **AutoThink(StartLux)**: 확신 시 즉답, 불확실 시 선택적 심층 추론 — v2에서 도입
5. **평가**: 정확도 + chance-corrected 점수 + 캘리브레이션 + 지연(p50/p95)

---

## 3. 포지셔닝: "K-"가 의미하는 것

K-Decision의 특화는 4개 축으로 정의한다.

### 축 1 — 언어 (한국어 원시 상태)
- 존댓말/반말/존비어 예사어, 높임 수준, 방언(경상/전라/제주), 한자 혼용, 외래어 표기 변형
  (예: "노트북/노트북컴퓨터/랩탑", "카드/신용카드/체크카드")
- 인코딩·조사·띄어쓰기 오류가 섞인 실사용 텍스트(민원, 채팅, 리뷰)에 강인한 판정

### 축 2 — 도메인 (한국 산업·행정)
1급 타깃 도메인과 대표 결정 태스크:

| 도메인 | 대표 타이핑드 결정 태스크 |
|--------|--------------------------|
| 공공·행정 | 민원 분류(민원예규 체계), 긴급도, 부서 라우팅, 수수료 해당 여부(noul) |
| 금융 | 상담 인텐트, 투자권유 유도 질문 감지(noul), 고령자 대상 여부(noul), 상품 라우팅 |
| CS/커머스 | 인텐트 분류, 환불/교환 사유 판정, 악성 민원 에스컬레이션(score) |
| 콘텐츠 안전 | 한국형 유해성(명예훼손, 성별·장애 차별 표현, 불법 광고) 판정 |
| 에이전트 인프라 | 한국어 지시 → 서브에이전트/툴 라우팅, 대형모델 호출 게이팅(noul) |

### 축 3 — 규범 (한국 규제·가이드라인)
- AI 기본법(2026 시행), 개인정보보호법, 금융 분야 전자금융감독규정, 공공 부문 AI 가이드라인 반영
- 고위험 결정(채용·신용·행정 처분)에는 "판정 근거 태깅 + 인간 검토 권고" 출력 스키마 옵션 제공

### 축 4 — 문화·맥락
- 조직 위계·존중 표현이 결정에 영향을 주는 맥락(예: "급함/보통" 표현의 강도 해석)
- 한국 시간대·계절성 이벤트(정기재고지, 명절, 수능) 관련 맥락 태스크

### 경쟁 좌표
- Intern-Decision/StartLux: 영어·중국어 중심, 다국어 커버는 우연적 성능 → 한국어 태스크에서
  K-Decision이 결정적 우위를 가질 수 있는 구조
- Cloudflare Clef 등 상용: 한국 도메인 특화 없음, 온프레미스 불가 → 데이터 주권이 중요한
  공공·금융 고객에게 K-Decision(자체호스팅 가능)이 차별화 포인트

---

## 4. 제품 정의

### 4.1 모델 패밀리
| 모델 | 베이스 | 용도 | 배포 |
|------|--------|------|------|
| K-Decision-Edge 0.8~2B | Qwen3.5 계열 | 온디바이스/엣지, 초저지연 | GGUF/ONNX/MLX |
| K-Decision-Core 8B | Qwen3.5 또는 EXAONE 계열 | 서버·API 게이팅 | HF + vLLM + GGUF |
| (v2) K-Decision-AutoThink | Core 기반 | 저신뢰 구간 심층 추론 | HF |

### 4.2 I/O 계약 (v0.1 스키마 초안)
```jsonc
// 요청
{
  "state": "고객 채팅: '카드 해지하고 싶은데 공과금 자동이체 걸려있어도 돼요?'",
  "questions": [
    { "id": "intent",   "type": "choice", "options": ["해지문의","자동이체문의","해지가능여부질문","기타"] },
    { "id": "urgent",   "type": "noul",   "options": ["예","아니오"] },
    { "id": "escalate", "type": "score",  "scale": [0, 10] }
  ]
}
// 응답 — 각 결정에 보정된 확률
{
  "decisions": [
    { "id": "intent",   "value": "해지가능여부질문", "probabilities": { "해지가능여부질문": 0.71, "해지문의": 0.24, ... } },
    { "id": "urgent",   "value": "아니오",           "probabilities": { "예": 0.08, "아니오": 0.92 } },
    { "id": "escalate", "value": 3,                  "distribution": "..." }
  ],
  "calibration": { "temperature": 1.87, "ece_heldout": 0.041 }
}
```
- Intern-Decision 3타입(choice/score/noul) 호환 + 확장 타입 `multi`(복수선택), `rank`(순위)는 v1에서 검토
- 고위험 옵션: `require_rationale: true` 시 근거 태그 + `human_review_recommended` 플래그

### 4.3 API/서비스 목표
- p50 < 50ms (8B, 단일 A100/4090급 GPU), p95 < 150ms
- 엣지 2B: CPU 양자화 환경 p50 < 200ms
- 엔드포인트는 Intern-Decision 스타일 `POST /v1/decisions` 호환 채택(생태계 편승)

---

## 5. 기술 설계

### 5.1 베이스 모델 선정 (2트랙 병행)
| 트랙 | 베이스 | 근거 | 리스크 |
|------|--------|------|--------|
| A. 표준 | Qwen3.5 (0.8B/4B 계열) | Intern-Decision 실증 레시피 존재, Apache-2.0, 한국어 능력 양호 | 한국어 상한 |
| B. 프리미엄 | EXAONE 3.5 (2.4B/7.8B) | 한국어 최상급, 오픈 가중치 | **라이선스(비상업 조건) 반드시 사전 검토**, 커뮤니티 인지도 |

권장: Phase 1은 트랙 A로 빠르게 검증(레시피 이식이 쉬움), Phase 2에서 EXAONE 트랙을 병행해
한국어 성능 상한을 끌어올리고 라이선스 조건에 따라 공개 범위를 결정.

### 5.2 학습 파이프라인 (3단계)
1. **SFT**: 타이핑드 질문 JSONL → JSON 스켈레톤 + `<decision>` 마커 형식으로 변환해
   autoregressive masked LM 목적함수로 파인튜닝 (Intern-Decision 학습 설계 이식)
2. **캘리브레이션**: 홀드아웃에서 필드별/타입별 온도 스케일링 피팅 → 체크포인트 메타데이터로 저장
3. **(v2) 신뢰도 기반 RL**: StartLux의 3단계 RL 참고, 저신뢰 샘플에 대한 선택적 추론(AutoThink) 학습

### 5.3 데이터 파이프라인
- 변환기: 기존 한국어 데이터셋(KMMLU/KMMLU-Pro, KLUE, 한국어 지시튜닝 데이터, 공공 민원 스키마) →
  타이핑드 결정 JSONL로 자동 변환
- 합성 생성: 기성 한국어 LLM으로 도메인 시나리오·질문 생성 + 룰 기반 검증 + 교차 모델 합의 필터링
- 실데이터: 민원/상담 로그는 가명처리·재식별 통제 후 사용(개인정보보호법 준수 절차 문서화)

### 5.4 인프라
- 학습: XTuner(Intern-Decision 스택) 또는 transformers+TRL. LoRA → 병합(경량 예산) / 전체 파인튜닝(예산 허용 시)
- 추론: HF transformers(기본), vLLM(서비스), llama.cpp/GGUF(엣지), ONNX(윈도우 배포)
- 실험관리: 데이터·체크포인트·보정온도까지 버전 관리(재현성이 카테고리의 평가 조건)

---

## 6. 평가 전략 — K-Decision Index

### 6.1 벤치마크 구성 (0.1)
- **공개부(60%)**: 한국어 타이핑드 결정 태스크 — 민원 분류, CS 인텐트, 금융 게이팅(noul),
  유해성 판정, 긴급도(score), 에이전트 라우팅. 총 5천~1만 결정 규모로 시작
- **비공개 홀드아웃(40%)** + 일부는 비공개 신규 태스크(게이밍 방지, Decision Index 0.4.1 설계 차용)
- 기존 자산 변환: KMMLU-Pro(국가전문자격) → choice 타입 결정 태스크로 재구성

### 6.2 지표
| 지표 | 측정 대상 | 목표(1차) |
|------|-----------|-----------|
| chance-corrected score | 태스크 전체 | 범용 대형 LLM(70B급) 대비 ±5% 이내 |
| 한국어 태스크 정확도 | 도메인 태스크 | Intern-Decision-4B 대비 +10%p 이상 |
| Brier / ECE | 캘리브레이션 | ECE < 0.05 |
| 지연 p50/p95 | 서빙 | Core 8B: 50/150ms |
| 비용/1천 결정 | 경제성 | 범용 LLM 대비 1/50 이하 |

리더보드는 HF Space로 공개해 커뮤니티 제출을 받는 것이 생태계 장악의 핵심 동작.

---

## 7. 로드맵

| 단계 | 기간 | 목표 | 산출물 |
|------|------|------|--------|
| **P0 착수** | 1~2주 | Intern-Decision 재현, 베이스 벤치마크, 스키마 확정 | 재현 리포트, 데이터 스키마 v1, 베이스별 한국어 베이스라인 |
| **P1 프로토타입** | 4~6주 | Qwen3.5-4B 기반 SFT + 캘리브레이션, 합성데이터 10만 결정 | K-Decision-4B-proto (내부) |
| **P2 데이터·평가** | 6~8주 | K-Decision-Data v0.1 공개, K-Decision Index 0.1 오픈 | 데이터셋(HF), 리더보드 Space, 평가 보고서 |
| **P3 패밀리 공개** | 8~12주 | Edge 2B + Core 8B, GGUF/ONNX 배포, HF 공개 | K-Decision v1.0 모델 패밀리 + 기술보고서 |
| **P4 확장** | 12주~ | AutoThink, EXAONE 트랙, 도메인 파트너 파일럿(CS/금융/공공) | 파일럿 결과, 상용 라이선스 정책 |

마일스톤 판정기준(P1→P2): 한국어 타이핑드 태스크에서 proto가 범용 70B급 LLM 대비 정확도 열폭 5%p 이내 &
ECE < 0.08. 미달 시 데이터 품질·보정 구간 진단 후 P1 연장.

---

## 8. 리스크 & 대응

| 리스크 | 수준 | 대응 |
|--------|------|------|
| 카테고리 경쟁 가속(중국·미국 대형 랩의 다국어 확장) | 높음 | 6개월 내 v1.0 공개로 선점. "한국 도메인 실데이터 + 리더보드"라는 현지 해자 구축 |
| EXAONE 라이선스(비상업) | 중간 | 상업 버전은 Qwen/Apache 계열로, EXAONE은 연구 공개로 분리. 공개 전 법률 검토 |
| 개인정보(민원·상담 로그) | 높음 | 가명처리 파이프라인 + DPA 계약 + 합성데이터 우선 전략(실데이터는 후순위) |
| 캘리브레이션이 실서비스에서 붕괴(분포 이동) | 중간 | 도메인별 온도 리플레이(Intern-Decision 재피팅 구조 차용) + 온라인 모니터링 스키마 |
| 표준 포맷 전쟁에서 외톨이화 | 중간 | Intern-Decision I/O 호환 채택 + Decision Index 제출 병행(양쪽 생태계 편승) |
| 고위험 결정 오남용(채용·신용) | 중간 | 고위험 태스크는 근거·인간검토 플래그 강제, 이용가이드라인·모델카드 명시 |
| "한국어 특화" 성능 우위가 실제로 안 나옴 | 중간 | P0에서 베이스별 한국어 베이스라인을 먼저 측정해 가설 검증 후 진행(스테이지 게이트) |

---

## 9. 다음 액션 아이템 (즉시 실행 목록)

1. Intern-Decision 저장소 클론 → `request.json` 샘플 추론 재현, 학습 데이터 스키마(`docs/DATA.md`) 분석
2. 베이스 후보 다운로드 및 한국어 타이핑드 질문 베이스라인 측정 (Qwen3.5-4B vs EXAONE3.5-2.4B/7.8B)
3. StartLux 논문(arXiv:2610.17309) 3단계 RL 파이프라인 정독 → v2 설계에 반영할 요소 추출
4. Jev Decision Index 제출 킷 확인 → K-Decision 모델 사후 제출 및 방법론 차용
5. 데이터 스키마 v1 확정(본 문서 4.2 초안 기반) + KMMLU-Pro → 타이핑드 결정 변환 스크립트 작성
6. EXAONE 3.5 라이선스 전문 검토 (연구/상업 경계)
7. 합성데이터 생성 프롬프트 세트 초안 (도메인 5개 × 타입 3개 매트릭스)

---

## 10. 출처 (주요 참고)

- Intern-Decision — GitHub: <https://github.com/internlm/Intern-Decision>, 논문: <https://arxiv.org/abs/2610.22131>, HF: <https://huggingface.co/internlm/Intern-Decision-4B>
- StartLux-Decision — 논문: <https://arxiv.org/html/2610.17309v1>, GitHub: <https://github.com/StartLuxLabs/Startlux-Decision>
- Decision Transformer — <https://arxiv.org/abs/2106.01345>, 코드: <https://github.com/kzl/decision-transformer>
- Jev Decision Index — <https://huggingface.co/spaces/multimodalart/jev-decision-index>
- Cloudflare Clef — <https://blog.cloudflare.com/clef-decision-models>
- Transformers in RL Survey — <https://arxiv.org/abs/2307.05979>
- EXAONE 3.5 — <https://www.lgresearch.ai/blog/view?seq=507>, EXAONE 4.0 — <https://www.lgresearch.ai/blog/view?seq=575>
- KMMLU-Pro — <https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro>
- HF Deep RL Course(Decision Transformers) — <https://huggingface.co/learn/deep-rl-course/en/unitbonus3/decision-transformers>
