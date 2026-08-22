# Related Work 조사 — 필수 인용과 인접 실험 (2026-08-23)

작성일: 2026-08-23. 웹 검색으로 서지사항을 검증했다 (arXiv 번호, 게재처, 저자).
목적: (1) 논문에 반드시 인용해야 하는 논문 목록, (2) 의료·타 도메인에서 우리와
비슷한 실험을 한 논문과 우리의 차별점, (3) Related Work 절의 구성안.

---

## 0. 결론 먼저

**참신성 확인**: 2026-08-23 기준 검색으로는 NLA(자연어 오토인코더)/활성값
언어화(AV)를 **의료 도메인에 적용한 논문이 없다**. 가장 가까운 이웃은 둘로
갈라져 있고, 우리는 그 교차점에 선다:

- **의료 쪽 이웃** (BiasMedQA 계열): 임상 인지 편향을 프롬프트에 주입해
  정확도 하락을 재는 실험 — 전부 **행동만 본다**. 내부 표상도, 탐지 신호도,
  인과 설계(같은 케이스의 조건쌍)도 없다.
- **해석가능성 쪽 이웃** (내부-외부 불일치 계열, 2026): CoT 불충실을 내부
  신호로 탐지하는 방법들 — 전부 **비의료**이고, 신호가 확률 프로브나 회로
  거리라서 **사람이 읽을 수 있는 서술이 아니며**, 원인 귀속("무엇이
  움직였나")이 아니라 정오 예측("맞았나 틀렸나")을 한다.

우리 논문의 자리: **자연어 내부 판독 + 의료 인과 테스트베드 + 단일 실행
배포 가능 탐지(0.84) + 교정 사다리**의 조합은 어느 이웃도 갖고 있지 않다.

**주의해야 할 논문 2편** (심사위원이 반드시 꺼낼 것):
1. Li et al. (ICML 2026) — "AV 서술은 대상 모델이 아니라 언어화 모델의
   지식을 반영할 수 있다"는 비판. 우리의 답: 검증 배터리(스왑 추적 0.993,
   암기 0.000, 셔플 대조)와 인과 테스트베드가 바로 이 비판에 대한 방어다.
2. Yuan et al. (2026) "Hidden Error Awareness" — 내부 오류 신호는 "진단적일
   뿐 인과적이지 않다"(교정 개입 4종 전부 실패). 우리의 r5 사다리가 이
   주장을 자연어 판독으로 직접 재시험하는 셈 — 오늘 밤 결과가 어느 쪽이든
   논문에 자리가 있다 (성공하면 대조 결과, 실패하면 그들과 일치).

---

## 1. 필수 인용 목록 (shortlist)

도입·방법에서 반드시 인용해야 하는 논문. ★ = 본문에서 직접 논쟁해야 함.

| # | 논문 | 왜 필수인가 |
|---|------|------------|
| 1 | ★ Kantamneni, Fraser-Taliente, Ong, Marks et al. **"Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations"**, Transformer Circuits (Anthropic), 2026 | 우리가 쓰는 도구·체크포인트(`kitft/nla-gemma3-12b-L32-av`)의 출처 |
| 2 | ★ Li, Ceballos Arroyo, Rogers, Saphra, Wallace. **"Do Activation Verbalization Methods Convey Privileged Information?"**, arXiv:2509.13316, ICML 2026 | AV 방법론에 대한 대표 비판; 우리 검증 배터리의 존재 이유 |
| 3 | Ghandeharioun et al. **Patchscopes**, arXiv:2401.06102, ICML 2024 | 활성값→언어 계열의 시조 격 |
| 4 | Chen, Vondrick, Mao. **SelfIE**, arXiv:2403.10949, ICML 2024 | 같은 계열 (자기 임베딩 해석) |
| 5 | Pan et al. **LatentQA**, arXiv:2412.08686 | 같은 계열 (활성값에 질문하기) |
| 6 | ★ Turpin et al. **"Language Models Don't Always Say What They Think"**, arXiv:2305.04388, NeurIPS 2023 | 우리 개입 패러다임(답을 움직이지만 설명이 숨기는 원인)의 원형 |
| 7 | Lanham et al. **"Measuring Faithfulness in Chain-of-Thought Reasoning"**, arXiv:2307.13702, 2023 | CoT 충실성 측정 방법론의 표준 인용 |
| 8 | Chen, Benton et al. **"Reasoning Models Don't Always Say What They Think"**, arXiv:2505.05410, Anthropic 2025 | 힌트 공개율 <20%; 우리 "무관심이지 은폐가 아니다" 뉘앙스의 비교 대상 |
| 9 | ★ Schmidgall et al. **BiasMedQA** ("Evaluation and mitigation of cognitive biases in medical language models"), npj Digital Medicine 2024, arXiv:2402.08113 | 의료에서 가장 가까운 실험 — 행동만 본다는 것이 우리의 차별점 |
| 10 | Mahajan, Obermeyer, Daneshjou et al. **"Cognitive bias in clinical large language models"**, npj Digital Medicine 8:428, 2025 | 임상 LLM의 anchoring을 다룬 대표 관점 논문; AI in Medicine 독자층용 프레이밍 |
| 11 | Sharma et al. **"Towards Understanding Sycophancy in Language Models"**, arXiv:2310.13548, ICLR 2024 | 제안 추종의 일반 현상; 우리 wording 변형(동료/환자)의 배경 |
| 12 | Huang et al. **"Large Language Models Cannot Self-Correct Reasoning Yet"**, arXiv:2310.01798, ICLR 2024 | 사다리 r3(일반 self-refine)의 예측을 제공하는 논문 |
| 13 | Belinkov. **"Probing Classifiers: Promises, Shortcomings, and Advances"**, Computational Linguistics 2022 | 화이트박스 기준선(프로브)의 표준 인용 |
| 14 | Belrose et al. **Tuned Lens**, arXiv:2303.08112 (+ nostalgebraist의 logit lens, 2020) | 렌즈 계열 기준선 |
| 15 | Cunningham/Huben et al. **"Sparse Autoencoders Find Highly Interpretable Features"**, arXiv:2309.08600 (+ Bricken et al. "Towards Monosemanticity", Transformer Circuits 2023) | SAE 기준선; "probe/SAE로 되지 않냐"는 질문에 대한 인용 |
| 16 | Fansi Tchango et al. **DDXPlus**, arXiv:2205.09148, NeurIPS 2022 D&B | 데이터셋 |
| 17 | Croskerry. **"The Importance of Cognitive Errors in Diagnosis and Strategies to Minimize Them"**, Academic Medicine 2003 | 임상 anchoring bias의 고전; 소견서 개입의 임상적 실재성 근거 |

MCR을 쓰게 되면: Wu et al. **MedCaseReasoning**, arXiv:2505.11733 (Stanford/UCSF, 14,489 케이스).

---

## 2. 계열별 정리와 포지셔닝

### A. 활성값 → 자연어 (우리 도구의 계보)

| 논문 | 한 줄 요약 | 우리와의 관계 |
|------|-----------|--------------|
| **NLA** (Anthropic, transformer-circuits.pub/2026/nla) | AV(활성값→서술)와 AR(서술→활성값)을 RL로 공동 학습해 잔차 스트림을 비지도 설명. Claude Opus 4.6 배포 전 감사에서 미언어화 평가 인지를 표면화 | 도구의 출처. 그들은 방법과 안전 감사 사례를 제시; **도메인 과제(진단)에서 인과 검증된 응용은 없음** |
| **Patchscopes** (Google, ICML 2024) | 은닉 표상을 다른 프롬프트에 패치해 모델 스스로 언어로 풀게 하는 통일 프레임 | 방법 계열의 시조. 의료 없음, 인과 테스트베드 없음, 계기 검증 배터리 없음 |
| **SelfIE** (ICML 2024) | 임베딩을 forward pass 조작으로 문장 해석; 해석 기반 표상 편집까지 | 동일 계열. 위와 같은 차이 |
| **LatentQA** (2024) | 활성값에 대해 열린 질문에 답하도록 LLM을 학습 (자연어 출력 프로브) | 우리 v2 판독(구조화 XML)과 가장 형식이 비슷한 선행. 의료·인과 설계 없음 |
| ★ **Li et al.** (ICML 2026, arXiv:2509.13316) | 언어화 벤치마크는 내부 접근 없이도 잘 풀리고, 서술은 대상 모델이 아니라 **언어화 모델의 파라메트릭 지식**을 반영하곤 한다 | **우리가 정면으로 답해야 할 비판.** 답: ① 스왑 추적 0.993/암기 0.000/셔플 +0.64는 서술이 활성값에서 온다는 것을 보임 ② 개입 테스트베드에서는 "언어화기가 지어냈다면 나올 수 없는" 예측력(0.84)이 정답 기준 ③ 출력 기반 기준선 대비 우위(침묵 부분집합)가 곧 privileged information의 조작적 정의 |
| **Faithful-Patchscopes** (arXiv:2602.00300) | 은닉 표상 설명 자체의 모델 편향을 진단·완화 | 각주급 인용; 설명 방법의 편향 문제 인지 표시 |

### B. CoT 충실성 (개입 패러다임의 계보)

| 논문 | 한 줄 요약 | 우리와의 관계 |
|------|-----------|--------------|
| ★ **Turpin et al.** (NeurIPS 2023) | 답을 움직이는 편향 특징(예: 정답이 항상 A)을 넣어도 CoT는 그것을 언급하지 않는다 | 우리 설계의 원형. 차이: ① 우리 편향은 **임상적으로 실재하는** 소견서 ② 우리는 은폐를 재는 데서 멈추지 않고 **내부 판독으로 탐지까지** 간다 ③ 뉘앙스 반전 — 우리 체인은 소견서를 96% 언급한다. 숨기는 게 아니라 **stance가 답과 논리적으로 결합해** 신호가 없는 것 (판별 0.49–0.56) |
| **Lanham et al.** (2023) | 절단·오류 주입 등으로 CoT 의존도를 측정 | 충실성 측정 방법론 표준 인용 |
| **Chen et al.** (Anthropic 2025) | 추론 모델도 힌트 사용을 20% 미만으로만 공개; RL로도 포화 안 됨 | 최신 대형모델에서도 문제가 남아있다는 근거. 우리 결과와 같은 방향 |
| **FaithCoT-Bench** (arXiv:2510.04040) | 인스턴스 수준 CoT 충실성 벤치마크 | 탐지 과제의 벤치마크 존재 인지; 우리는 의료 + 자연어 판독이라는 점이 차이 |
| ★ **CIE-SCORER** (Shen et al., arXiv:2605.25603, 2026) | 문장 수준 회로로 내부 계산 그래프를 만들어 외부 추론 그래프와의 거리(Gromov-Wasserstein)로 불충실 탐지, FaithCoT-Bench SOTA | **방법적으로 가장 가까운 경쟁자.** 차이: ① 그들의 신호는 그래프 거리(숫자), 우리는 **읽을 수 있는 내부 결론**("내부는 Anemia라 결론") ② 정오/충실 이진 판정 vs 우리는 **원인 귀속**(무엇이 움직였는지 서술) ③ 비의료 ④ 우리 정답 레이블은 프롬프트 쌍의 **인과 설계**에서 나옴 |
| ★ **Yuan et al. "Hidden Error Awareness"** (arXiv:2605.09502, 2026) | 은닉 상태 선형 프로브가 추론 정오를 0.95 AUROC로 예측하지만(언어화 확신은 무변), 조향·패칭·self-correction 개입은 **전부 실패** — "진단적, 비인과적" | **사다리 실험의 직접 비교 대상.** 그들의 프로브는 정오만 읽고, 우리는 내부 결론의 **내용**을 읽어 그것을 피드백한다(r5). r5가 r4를 이기면 "내용이 있는 판독은 지렛대가 된다"는 대조 결과; 지면 그들의 결론이 자연어 판독에도 성립한다는 확장. 어느 쪽이든 보고 가치 있음 |
| **Mehrafarin et al.** (arXiv:2604.23351, 2026) | CoT가 틀려도 은닉 상태에는 정답이 있다 (활성값 패칭으로 회복) | "닻 내린 답과 회복 가능한 결론의 공존"이라는 우리 핵심 관찰의 비의료 평행 사례 |
| **arXiv:2603.17199** (2026) | 활성값 프로브로 동기화된 추론(rationalization)을 사전·사후 탐지 | 각주급; 내부 신호로 rationalization을 잡는 흐름의 존재 |

### C. 의료 LLM의 인지 편향·제안 취약성 (실험적으로 가장 가까운 이웃)

| 논문 | 한 줄 요약 | 우리와의 관계 |
|------|-----------|--------------|
| ★ **Schmidgall et al. BiasMedQA** (npj Digital Medicine 2024) | USMLE 1,273문항에 자기진단·최신성·확증 등 7개 임상 인지 편향을 주입; GPT-4는 견디고 Llama 2 계열은 크게 하락; 완화 프롬프트 3종은 부분 회복 | **의료에서 우리와 가장 가까운 실험.** 전부 행동 수준: 정확도 하락만 잰다. 우리가 더한 것: ① 인과 설계(같은 케이스 4조건, 위약 대조, 오답/정답 분리) ② **내부 판독에 의한 사례 단위 탐지**(0.84) ③ 탐지 기반 선택적 교정(사다리). 그들이 "편향에 약하다"에서 멈춘 곳에서 우리는 "어느 케이스가 지금 밀렸는지 안다"로 간다 |
| **"LLM Reasoning Does Not Protect Against Clinical Cognitive Biases"** (medRxiv 2025, BiasMedQA 사용) | 추론(reasoning) 모델도 임상 인지 편향에 취약 | 우리 CoT 이중 결과(완화하지만 귀속 못함)와 나란히 인용 |
| **Mahajan et al.** (npj Digital Medicine 2025) | 임상 LLM의 인지 편향 관점 논문; 자가회귀 처리에서 anchoring이 어떻게 생기는지; 추론 트레이스를 감사 고리로 쓰자고 제안 | AI in Medicine 독자용 프레이밍에 최적. 그들이 "추론 트레이스를 감사에 쓰자"고 **제안**한 것을 우리는 트레이스가 감사에 **불충분함**(귀속 0.49–0.56)을 보이고 내부 판독으로 대체 |
| **arXiv:2503.22746** (2025) | 의료 질의에서 사용자 발화 요인(강한 표현, 오정보 등)에 대한 LLM 민감성 | 환자-화자 wording 변형의 근거 인용 |
| **SycoEval-EM** (arXiv:2601.16529, 2026) | 응급 시뮬레이션 대화에서 의료 시코펀시 평가 | 의료 시코펀시가 활발한 주제라는 근거; 행동 수준 |
| **DiversityMedQA** (arXiv:2409.01497) | 인구학적 섭동으로 진단 편향 평가 | 선택 인용 (섭동 실험 계열) |

### D. 시코펀시·자기교정 (사다리의 배경)

- **Sharma et al.** (ICLR 2024): 시코펀시는 최신 어시스턴트의 일반 행동이고 인간 선호 학습이 부분 원인. 우리 소견서 추종은 이것의 임상 특수형 — 화자 3종(의뢰의/동료/환자)에서 살아남으면 "한 문장의 효과"가 아니라 "제안의 효과".
- **Huang et al.** (ICLR 2024): 외부 피드백 없는 자기교정은 추론을 못 고친다. **r3(일반 재고)의 예상 성적표.** r5(판독 피드백)는 "외부 아닌 내부-자기 신호"라는 제3의 범주 — 모델 밖 정보가 아니라 모델 안 정보를 밖으로 꺼내 되먹인다.

### E. 화이트박스 기준선 (표 4의 경쟁자 · ▢ᵇ probe-disagreement의 인용)

- **Belinkov** (CL 2022): 프로브 방법론 표준 + 프로브가 "표상에 있음"과 "사용함"을 구분 못한다는 고전적 한계 — 우리 개입 설계가 이 한계를 넘는 방식임을 설명할 때 인용.
- **logit lens** (nostalgebraist 2020) / **Tuned Lens** (Belrose et al., arXiv:2303.08112): 렌즈 계열. ▢ 신뢰도/렌즈 귀속 실험의 인용.
- **SAE**: Cunningham/Huben et al. (arXiv:2309.08600), Bricken et al. (Transformer Circuits 2023), Templeton et al. "Scaling Monosemanticity" (2024), 서베이 arXiv:2503.05613. "probe나 SAE만 써도 되지 않냐"에 대한 본문 답변(플래그 vs 서술; 미리 정한 개념 vs 열린 서술)에서 인용.
- 선택: Lindsey et al. "On the Biology of a Large Language Model" (Anthropic 2025) — 귀속 그래프로 내부-외부 불일치를 본 사례.

### F. 데이터셋·임상 배경

- **DDXPlus** (NeurIPS 2022 D&B): 합성 ~130만 환자, 49 병리, 감별진단 포함. 우리 표본(진단당 100 → 4,900 → 직접정답 1,747)의 출처. 합성 데이터라는 한계도 이 인용에서 정직하게.
- **MedCaseReasoning** (arXiv:2505.11733): 실제 증례 보고 14,489건 + 임상의 추론 문장. MCR 이전 실험이 확정되면 필수 인용으로 승격.
- **Croskerry** (Academic Medicine 2003; NEJM 2013 "From Mindless to Mindful Practice"): 진단 오류의 인지적 원인, anchoring. 소견서 개입이 "임상적으로 실재하는 교란"이라는 주장의 근거. npj 2025 관점 논문에 따르면 미국에서 진단·의료 오류로 연 4–8만 예방가능 사망, 그중 40–80%에 인지 편향 관여 — 도입부 동기 문장감.
- 의료 해석가능성 일반: "Cracking the clinical code" (의료 보고서 생성의 기계적 해석가능성 스코핑 리뷰, 2025), medRxiv 2026 "Why LLMs' Clinical Reasoning Fails" — Related Work의 의료 해석가능성 문단에서 1–2문장 인용.

---

## 3. Related Work 절 구성안 (4문단)

1. **Explaining model internals in natural language.** probe/lens/SAE(플래그와
   미리 정한 개념) → Patchscopes/SelfIE/LatentQA(열린 언어화) → NLA(비지도,
   재구성 목적함수) 순으로 좁힌다. Li et al. 비판을 여기서 소개하고 "본
   연구의 검증 배터리와 인과 테스트베드가 이에 답한다"로 마감.
2. **Chain-of-thought (un)faithfulness.** Turpin/Lanham/Chen: 설명이 원인을
   말하지 않는다 → 최근 내부 신호 탐지(CIE-SCORER, Hidden Error Awareness):
   전부 비의료·비언어적·정오 판정. 우리는 원인 귀속 + 읽을 수 있는 서술.
3. **Cognitive bias in medical LLMs.** Croskerry(임상 배경) → BiasMedQA와
   후속(행동 수준 취약성) → Mahajan(트레이스를 감사에 쓰자는 제안). 우리는
   같은 개입을 인과 설계로 다시 만들고, 트레이스 대신 내부 판독으로 감사.
4. **Correction and self-refinement.** Huang(자기교정 한계), Sharma(시코펀시),
   Yuan(내부 신호는 지렛대가 아니다) → 사다리가 이 세 주장을 한 표에서
   시험한다.

---

## 4. 검증 상태와 남은 확인

- 위 서지사항은 2026-08-23 웹 검색으로 확인 (arXiv 번호·게재처·핵심 주장).
- 미확인 항목: Schmidgall arXiv 번호(2402.08113)는 기억 기반 — 인용 직전
  재확인 필요. Croskerry 2003 서지도 최종 인용 시 재확인.
- 2026년 arXiv 논문들(CIE-SCORER, Hidden Error Awareness, Mehrafarin)은
  프리프린트 — 게재 여부를 투고 직전 한 번 더 확인.
- 참신성 문장은 "to our knowledge, no prior work applies activation
  verbalization to medical diagnosis"로 한정 서술 (검색 기준일 명시).

주요 출처 링크:
[NLA 논문](https://transformer-circuits.pub/2026/nla/) ·
[Li et al.](https://arxiv.org/abs/2509.13316) ·
[Patchscopes](https://arxiv.org/abs/2401.06102) ·
[SelfIE](https://arxiv.org/abs/2403.10949) ·
[LatentQA](https://arxiv.org/abs/2412.08686) ·
[Turpin](https://arxiv.org/abs/2305.04388) ·
[Chen 2025](https://arxiv.org/abs/2505.05410) ·
[CIE-SCORER](https://arxiv.org/abs/2605.25603) ·
[Hidden Error Awareness](https://arxiv.org/abs/2605.09502) ·
[Mehrafarin](https://arxiv.org/abs/2604.23351) ·
[BiasMedQA](https://www.nature.com/articles/s41746-024-01283-6) ·
[Mahajan](https://www.nature.com/articles/s41746-025-01790-0) ·
[Sharma](https://arxiv.org/abs/2310.13548) ·
[Tuned Lens](https://arxiv.org/abs/2303.08112) ·
[DDXPlus](https://arxiv.org/abs/2205.09148) ·
[MedCaseReasoning](https://arxiv.org/abs/2505.11733)
