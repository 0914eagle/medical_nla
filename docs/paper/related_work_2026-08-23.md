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

우리 논문의 자리: **자연어 내부 AV 판독 + 의료 인과 테스트베드 + 단일 실행
배포 가능 탐지(0.84) + 교정 사다리**의 조합은 어느 이웃도 갖고 있지 않다.

**주의해야 할 논문 2편** (심사위원이 반드시 꺼낼 것):
1. Li et al. (ICML 2026) — "AV 서술은 대상 모델이 아니라 언어화 모델의
   지식을 반영할 수 있다"는 비판. 우리의 답: 검증 배터리(스왑 추적 0.993,
   암기 0.000, 셔플 대조)와 인과 테스트베드가 바로 이 비판에 대한 방어다.
2. Yuan et al. (2026) "Hidden Error Awareness" — 내부 오류 신호는 "진단적일
   뿐 인과적이지 않다"(교정 개입 4종 전부 실패). 우리의 r5 사다리가 이
   주장을 자연어 AV 판독으로 직접 재시험하는 셈 — 오늘 밤 결과가 어느 쪽이든
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
| **LatentQA** (2024) | 활성값에 대해 열린 질문에 답하도록 LLM을 학습 (자연어 출력 프로브) | 우리 v2 AV 판독(구조화 XML)과 가장 형식이 비슷한 선행. 의료·인과 설계 없음 |
| ★ **Li et al.** (ICML 2026, arXiv:2509.13316) | 언어화 벤치마크는 내부 접근 없이도 잘 풀리고, 서술은 대상 모델이 아니라 **언어화 모델의 파라메트릭 지식**을 반영하곤 한다 | **우리가 정면으로 답해야 할 비판.** 답: ① 스왑 추적 0.993/암기 0.000/셔플 +0.64는 서술이 활성값에서 온다는 것을 보임 ② 개입 테스트베드에서는 "언어화기가 지어냈다면 나올 수 없는" 예측력(0.84)이 정답 기준 ③ 출력 기반 기준선 대비 우위(침묵 부분집합)가 곧 privileged information의 조작적 정의 |
| **Faithful-Patchscopes** (arXiv:2602.00300) | 은닉 표상 설명 자체의 모델 편향을 진단·완화 | 각주급 인용; 설명 방법의 편향 문제 인지 표시 |

### B. CoT 충실성 (개입 패러다임의 계보)

| 논문 | 한 줄 요약 | 우리와의 관계 |
|------|-----------|--------------|
| ★ **Turpin et al.** (NeurIPS 2023) | 답을 움직이는 편향 특징(예: 정답이 항상 A)을 넣어도 CoT는 그것을 언급하지 않는다 | 우리 설계의 원형. 차이: ① 우리 편향은 **임상적으로 실재하는** 소견서 ② 우리는 은폐를 재는 데서 멈추지 않고 **내부 AV 판독으로 탐지까지** 간다 ③ 뉘앙스 반전 — 우리 체인은 소견서를 96% 언급한다. 숨기는 게 아니라 **stance가 답과 논리적으로 결합해** 신호가 없는 것 (판별 0.49–0.56) |
| **Lanham et al.** (2023) | 절단·오류 주입 등으로 CoT 의존도를 측정 | 충실성 측정 방법론 표준 인용 |
| **Chen et al.** (Anthropic 2025) | 추론 모델도 힌트 사용을 20% 미만으로만 공개; RL로도 포화 안 됨 | 최신 대형모델에서도 문제가 남아있다는 근거. 우리 결과와 같은 방향 |
| **FaithCoT-Bench** (arXiv:2510.04040) | 인스턴스 수준 CoT 충실성 벤치마크 | 탐지 과제의 벤치마크 존재 인지; 우리는 의료 + 자연어 AV 판독이라는 점이 차이 |
| ★ **CIE-SCORER** (Shen et al., arXiv:2605.25603, 2026) | 문장 수준 회로로 내부 계산 그래프를 만들어 외부 추론 그래프와의 거리(Gromov-Wasserstein)로 불충실 탐지, FaithCoT-Bench SOTA | **방법적으로 가장 가까운 경쟁자.** 차이: ① 그들의 신호는 그래프 거리(숫자), 우리는 **읽을 수 있는 내부 결론**("내부는 Anemia라 결론") ② 정오/충실 이진 판정 vs 우리는 **원인 귀속**(무엇이 움직였는지 서술) ③ 비의료 ④ 우리 정답 레이블은 프롬프트 쌍의 **인과 설계**에서 나옴 |
| ★ **Yuan et al. "Hidden Error Awareness"** (arXiv:2605.09502, 2026) | 은닉 상태 선형 프로브가 추론 정오를 0.95 AUROC로 예측하지만(언어화 확신은 무변), 조향·패칭·self-correction 개입은 **전부 실패** — "진단적, 비인과적" | **사다리 실험의 직접 비교 대상.** 그들의 프로브는 정오만 읽고, 우리는 내부 결론의 **내용**을 읽어 그것을 피드백한다(r5). r5가 r4를 이기면 "내용이 있는 AV 판독은 지렛대가 된다"는 대조 결과; 지면 그들의 결론이 자연어 AV 판독에도 성립한다는 확장. 어느 쪽이든 보고 가치 있음 |
| **Mehrafarin et al.** (arXiv:2604.23351, 2026) | CoT가 틀려도 은닉 상태에는 정답이 있다 (활성값 패칭으로 회복) | "닻 내린 답과 회복 가능한 결론의 공존"이라는 우리 핵심 관찰의 비의료 평행 사례 |
| ★ **"Catching rationalization in the act"** (arXiv:2603.17199, 2026) | **MCQ에 힌트를 주입**하면 답이 힌트 쪽으로 밀리고 CoT는 인정 없이 합리화하는데, **활성값 프로브는 CoT가 못 잡는 이 motivated reasoning을 탐지** | **탐지 축의 가장 가까운 쌍둥이 (08-24 재분류: 각주급 → 정면 대응).** "힌트 주입 + 내부가 CoT를 이긴다"는 헤드라인이 겹침. 우리의 차이: ① 의료 + 임상적으로 실재하는 원인(소견서, MCQ 힌트 아님) ② 위약 대조 4조건 인과 설계와 케이스 단위 밀림 정답지 ③ 침묵 부분집합 프레임 ④ 프로브가 아니라 **읽을 수 있는 서술**(+열린 어휘) ⑤ 기전(궤적)과 ⑥ 교정까지 완주. 인용 필수, 차별점 명시 필수 |
| ★ **"When Truth Is Overridden"** (arXiv:2508.02087, AAAI 2026) | 시코펀시의 내부 기원: 사용자 의견이 **후기 레이어에서 학습된 지식을 억압**하고 로짓이 의견 쪽으로 급변 — activation patching으로 인과 검증. 화자의 권위는 내부에 인코딩되지 않는다고 보고 | **기전 축의 일반 도메인 선행.** "지식은 보존되고 출력 단계에서 뒤집힌다"는 우리 결렬 발견의 시코펀시판. 우리의 추가: ① 의료·임상 원인 ② 프롬프트 읽기 순서를 따라가는 **위치 궤적**(그들은 레이어 축) ③ 케이스 단위 배포 가능 탐지 ④ 서술 ⑤ 교정 ⑥ **화자 대화점**: 그들은 "권위 무관(의견 존재만 중요)"이라 했는데 우리는 행동에서 화자 기울기(임상 −17.7 vs 환자 −12.3)를 관찰 — 내부-행동의 긴장으로 논의 가치 |
| **Sycophancy Is Not One Thing** (arXiv:2509.21305) | 시코펀시 행동들이 내부에서 분리 가능한 별개 과정 | 화자·항복 논의의 보조 인용 |
| **Latent Introspection / Anthropic 개념 주입** (arXiv:2602.20031; Anthropic 2025) | 주입된 개념을 모델이 자기 보고로 탐지 | 내부 자기 보고 흐름의 존재; 우리는 외부 계기(AV 판독)로 접근 |
| **Lie to Me** (arXiv:2603.22582) | 추론 모델의 CoT 충실성 평가 | 보조 인용 |

### C. 의료 LLM의 인지 편향·제안 취약성 (실험적으로 가장 가까운 이웃)

| 논문 | 한 줄 요약 | 우리와의 관계 |
|------|-----------|--------------|
| ★ **Schmidgall et al. BiasMedQA** (npj Digital Medicine 2024) | USMLE 1,273문항에 자기진단·최신성·확증 등 7개 임상 인지 편향을 주입; GPT-4는 견디고 Llama 2 계열은 크게 하락; 완화 프롬프트 3종은 부분 회복 | **의료에서 우리와 가장 가까운 실험.** 전부 행동 수준: 정확도 하락만 잰다. 우리가 더한 것: ① 인과 설계(같은 케이스 4조건, 위약 대조, 오답/정답 분리) ② **내부 AV 판독에 의한 사례 단위 탐지**(0.84) ③ 탐지 기반 선택적 교정(사다리). 그들이 "편향에 약하다"에서 멈춘 곳에서 우리는 "어느 케이스가 지금 밀렸는지 안다"로 간다 |
| **"LLM Reasoning Does Not Protect Against Clinical Cognitive Biases"** (medRxiv 2025, BiasMedQA 사용) | 추론(reasoning) 모델도 임상 인지 편향에 취약 | 우리 CoT 이중 결과(완화하지만 귀속 못함)와 나란히 인용 |
| **Mahajan et al.** (npj Digital Medicine 2025) | 임상 LLM의 인지 편향 관점 논문; 자가회귀 처리에서 anchoring이 어떻게 생기는지; 추론 트레이스를 감사 고리로 쓰자고 제안 | AI in Medicine 독자용 프레이밍에 최적. 그들이 "추론 트레이스를 감사에 쓰자"고 **제안**한 것을 우리는 트레이스가 감사에 **불충분함**(귀속 0.49–0.56)을 보이고 내부 AV 판독으로 대체 |
| **arXiv:2503.22746** (2025) | 의료 질의에서 사용자 발화 요인(강한 표현, 오정보 등)에 대한 LLM 민감성 | 환자-화자 wording 변형의 근거 인용 |
| **SycoEval-EM** (arXiv:2601.16529, 2026) | 응급 시뮬레이션 대화에서 의료 시코펀시 평가 | 의료 시코펀시가 활발한 주제라는 근거; 행동 수준 |
| **DiversityMedQA** (arXiv:2409.01497) | 인구학적 perturbation으로 진단 편향 평가 | 선택 인용 (perturbation 실험 계열) |

### D. 시코펀시·자기교정 (사다리의 배경)

- **Sharma et al.** (ICLR 2024): 시코펀시는 최신 어시스턴트의 일반 행동이고 인간 선호 학습이 부분 원인. 우리 소견서 추종은 이것의 임상 특수형 — 화자 3종(의뢰의/동료/환자)에서 살아남으면 "한 문장의 효과"가 아니라 "제안의 효과".
- **Huang et al.** (ICLR 2024): 외부 피드백 없는 자기교정은 추론을 못 고친다. **r3(일반 재고)의 예상 성적표.** r5(AV 판독 피드백)는 "외부 아닌 내부-자기 신호"라는 제3의 범주 — 모델 밖 정보가 아니라 모델 안 정보를 밖으로 꺼내 되먹인다.

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

## 2.5 의료 해석가능성 지형도 (2026-08-23 추가 조사)

의료 도메인의 해석가능성 연구는 크게 두 흐름인데, **아직 서로 만나지 않았다**:
(a) 행동 perturbation으로 "설명을 믿어도 되는가"를 평가하는 흐름 — 내부를 안 봄,
(b) 프로브/SAE로 "지식이 내부 어디에 있는가"를 확인하는 흐름 — 지식의 소재만
보고 개별 사례의 오염은 안 봄. 우리는 내부를 읽어 **사례 단위 오염을
탐지·서술**하는 결합점에 선다.

### (a) 의료 CoT·설명 충실성 — 전부 행동 수준

| 논문 | 한 줄 요약 | 우리와의 관계 |
|------|-----------|--------------|
| ★ **"Faithful or Just Plausible?"** (arXiv:2603.13988, NeurIPS 2025 수락) | 폐쇄형 LLM(ChatGPT/Gemini)의 의료 추론에 인과 절제·위치 편향·**힌트 주입** 3종 perturbation; CoT 단계가 예측을 인과적으로 안 끌고, 모델은 외부 힌트를 **인정 없이 흡수** | **의료에서 힌트 주입까지 한, 가장 가까운 충실성 논문.** 폐쇄형이라 내부 접근이 원리적으로 불가 → "문제 확인"에서 멈춤. 우리는 열린 모델에서 같은 현상을 인과 설계로 만들고 내부 AV 판독으로 **탐지까지** 감. Related Work에서 "이들이 연 문제를 우리가 닫는다" 구도로 인용 |
| **Medical VLM 충실성** (arXiv:2510.11196) | 의료 비전-언어 모델에 멀티모달 perturbation으로 추론 충실성 평가 | 의료 충실성 평가가 멀티모달로도 확장 중이라는 근거; 역시 행동 수준 |
| **Clinical Reasoning Graphs** (arXiv:2606.29876) | LLM 진단 추론을 구조화 그래프로 채점 — "역량은 있으나 일관성 없음"; 추론 단계가 장식적(지워도 답 불변), 유사 케이스 간 비일관 | "체인은 장식"이라는 우리 T4 1행 결과의 의료판 방증. 탐지 신호는 없음 |
| **MR-Bench 서베이** (arXiv:2604.08559) | 의료 추론 LLM 서베이 + 벤치마크 | 도입부 지형 인용 |
| **FaithMed** (arXiv:2607.01440) | 근거 기반 의료 추론을 루브릭 RL로 학습 | "충실하게 만들기" 흐름; 우리는 "충실한지 감시하기" — 상보적 |

### (b) 의료 내부 표상 — 프로브·SAE (지식 소재 확인, 사례 단위 감사 아님)

| 논문 | 한 줄 요약 | 우리와의 관계 |
|------|-----------|--------------|
| **ADR 프로빙** (PMC11844579, 2025) | 은닉 상태 프로브로 약물 부작용 지식 확인 (AKI 0.957, MI 0.954 등) | "지식이 내부에 있다"의 의료 증거. 있는지를 볼 뿐, 특정 케이스에서 그 지식이 왜 안 쓰였는지는 못 봄 |
| **약리 지식 추적** (arXiv:2603.03407) | 약리 지식이 레이어·모듈 어디에 저장되는지 기계적 추적 | 소재 확인 계열 |
| **정렬-저항 프로빙** (medRxiv 2025.09.17.25336018) | 안전 정렬이 막은 의료 답을 은닉 상태에서 직접 읽어 복원; 프로브가 출력보다 보정도 좋음 | **"내부가 출력보다 많이 안다"의 의료 선례** — 우리 0.84(닻 내린 출력 옆에 살아있는 내부 결론)와 같은 방향, 다만 지식 복원이지 편향 탐지가 아님 |
| **Medical Knowledge Maps** (arXiv:2510.11390) | 의료 지식의 내부 지도화 | 소재 확인 계열 |
| **JMIR AI SAE-의료** (ai.jmir.org/2026/1/e81134) | 의료 LLM에 SAE 적용해 단의미 특징 추출 | "SAE로 되지 않냐" 질문의 의료판 인용 — SAE는 **미리 학습된 특징 사전**을 주지, 케이스별 "무엇이 답을 움직였나" 서술을 주지 않음 |
| **임상 시퀀스 SAE** (arXiv:2605.04072) | EHR 파운데이션 모델에 TopK SAE — 특징 복잡도·사망 예측 | 동일 계열 (EHR) |
| **의료 VLM SAE 조향** (arXiv:2605.24977) | SAE 특징 조향으로 의료 VLM 오류 유형 제어; 효과가 아키텍처 의존적 | 내부 개입의 의료 시도; Yuan의 "진단적, 비인과적"과 함께 사다리 논의에서 인용 가능 |
| **NeuroFaith** (arXiv:2506.09277, 비의료) | 자기 설명의 충실성을 내부 표상 정렬로 평가 | 방법적 이웃 (내부 vs 설명 비교); 비의료·프로브 기반 |

### (c) 리뷰·관점 (도입부용)

- **"Cracking the clinical code"** (ScienceDirect, 2025): 의료 보고서 생성의
  기계적 해석가능성 스코핑 리뷰 — "임상의 대면 인터페이스 부재가 실행 가치를
  제한" → 우리의 자연어 AV 판독이 정확히 그 인터페이스라는 연결.
- **"Why LLMs' Clinical Reasoning Fails"** (medRxiv 2026.01.26): 벤치마크
  고성능 의료 LLM의 설명 불가한 임상 변동성; 규제 관점(해석 불가한 CDS는
  기기 수준 감독 필요할 수 있음) — 규제 동기 문장감.
- **Mahajan et al.** (npj Digital Medicine 2025, §C 참조): 임상 인지 편향 관점.

**요약 포지셔닝 문장(논문용)**: 의료 해석가능성은 설명의 신뢰성을 행동으로
평가하는 흐름과 지식의 소재를 내부에서 확인하는 흐름으로 나뉘어 왔다. 전자는
설명이 믿을 수 없음을 보이지만 대안을 주지 않고, 후자는 지식이 존재함을
보이지만 개별 사례의 실패를 설명하지 않는다. 본 연구는 내부 표상을 자연어로
AV 판독해, 인과적으로 통제된 개별 사례에서 답을 움직인 원인을 탐지·서술한다.

---

## 3. Related Work 절 구성안 — 최종: 2절 (2026-08-23 재확정)

축은 "의료 / 방법"이 아니라 **"문제 / 도구"**. 처음 안(3.1 =
"Interpretability of medical LLMs")은 행동 평가(충실성·편향 주입)를
interpretability라는 제목 아래 넣어 제목이 내용의 절반과 안 맞았다 —
편향 주입은 모델을 밖에서 흔드는 행동 평가지 내부 해석이 아니다.
아래 구조는 2.1이 전부 행동 논문, 2.2가 전부 내부 논문이라 제목-내용
불일치가 없고, 교집합 구조(문제의 공백 × 도구의 공백 = 본 연구)는 유지.

### 3.1 Cognitive bias and unfaithful explanations in medical LLMs (문제)

- 문단 1 — 일반→임상: 설명이 답의 원인을 말하지 않는 현상(Turpin 한 줄) +
  임상 anchoring의 실재(Croskerry). → 의료 LLM 편향 주입: BiasMedQA,
  추론도 못 막음(medRxiv 2025), Mahajan(트레이스를 감사 고리로 제안).
- 문단 2 — 의료 설명 충실성 평가: Faithful or Just Plausible(인과 절제·
  힌트 주입; 폐쇄형이라 내부 접근 불가), Clinical Reasoning Graphs(추론
  단계는 장식적).
- 마감: 이 문헌 전체가 행동만 본다. 편향이 정확도를 떨어뜨린다는 것은
  알지만, 배포된 단일 실행에서 어느 케이스가 지금 밀렸는지는 알 수 없다 —
  출력 기반 탐지가 원리적으로 장님이 되는 부분집합이 존재한다(§결과).

### 3.2 Reading LLM internals: from probes to natural-language readouts (도구)

- 문단 1 — 일반 계보: probe/lens/SAE는 플래그와 미리 정한 개념을 준다
  (Belinkov, tuned lens, SAE 계열) → 열린 언어화: Patchscopes, SelfIE,
  LatentQA → NLA(비지도 재구성 목적함수; 본 연구의 도구) → Li et al.
  비판("서술이 언어화 모델의 지식일 수 있다")과 본 연구의 답(계기 검증
  배터리·인과 테스트베드).
- 문단 2 — 의료의 내부 접근(이 절로 이사): ADR·정렬-저항 프로브, AIIM
  2026 환각 프로브, JMIR AI SAE, EHR SAE — 의료에 내부 방법이 도달했지만
  지식 소재·정오 스칼라에서 멈춘다.
- 문단 3 — 내부-외부 불일치 탐지(비의료): CIE-SCORER(그래프 거리), Yuan
  ("진단적, 비인과적"), Mehrafarin(은닉 상태에 정답이 산다) — 전부
  비언어적·정오 판정·비의료.
- 마감: 내부를 읽는 도구는 성숙했지만, 의료 진단에서 개별 사례의 원인을
  자연어로 귀속한 적은 없다. 본 연구는 이 계열을 인과적으로 통제된 의료
  개입 위에서 계기로 검증해 그 자리를 채운다.

### 각 절의 마감 턴 ("기존은 이렇게 했다 → 근데 우리는", 08-24 확정)

- **2.1의 턴**: 이 문헌은 취약성을 집단 수준 정확도 하락으로 잰다. 우리는
  같은 임상적 개입을 **인과적으로 통제된 형태**(위약 대조, 케이스 단위 밀림
  정답지, 증거 표현 불변)로 다시 만들고 — 거기서 멈추지 않고, 트레이스가
  감사를 지탱하지 못함을 보인 뒤(귀속 0.5) **모델 내부로 들어간다.**
- **2.2의 턴**: 일반 도메인에서 조각조각 관측된 것들(힌트 영향은 프로브가
  CoT보다 잘 잡는다 · 의견은 후기 레이어에서 지식을 억압한다 · 은닉 상태는
  오류를 알지만 개입은 실패한다)을, 우리는 의료 진단에서 하나의 사슬로
  잇는다: **검증된 자연어 AV 판독**으로 — 귀속하고(0.84, 프로브 상한 병기),
  결렬임을 위치 궤적으로 보이고, 임상의가 읽을 문장으로 서술하고, 되먹여
  교정한다(+22.8pp, 선별 결합 시 순효과 양전) — 닫힌 라벨 없는 실제
  증례(MCR)까지.
- 두 턴이 찍는 공백이 다르다: 2.1 = 인과 설계·케이스 단위 앎의 부재,
  2.2 = 의료·서술·교정으로의 완주 부재. 합집합 = 서론 기여 5개.

### 두 절 밖으로 이사하는 인용

- Lanham/Chen(충실성 측정 방법론·최신 추론 모델) → **Introduction** 또는
  3.1 문단 1의 보조 인용 (Turpin은 3.1 도입 한 줄).
- npj 사망 통계(연 4–8만, 인지 편향 40–80% 관여) → **Introduction** 첫
  문단의 임상 동기 (Croskerry는 3.1에서 재인용).
- Sharma(시코펀시)·Huang(자기교정 한계) → **Table 5(사다리) 결과 논의** —
  related work에서 미리 소비하지 않는다.
- MedGemma TR·H-DDx·MedS-Bench(DDXPlus 성능 선행) → **Methods/데이터 절**
  ("35.7%는 공개 모델 표준 구간" 방어).
- DDXPlus·MedCaseReasoning 데이터셋 논문 → **Methods**.

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

---

## 부록 A. 의료 해석가능성 최신 논문 일람 (한 줄 정리, 2026-08-23 수집)

★ = 본문 인용 후보. 나머지는 지형 파악용.

### 내부 표상 읽기 — 프로브·은닉 상태 (의료)

- ★ **Readable but Not Controllable** (arXiv:2607.00158, 2026) — 의료 QA 환각의 뉴런 수준 분석: 탐지 특징은 읽히지만 개입 표적은 아니다 (Yuan의 "진단적, 비인과적"의 의료판).
- ★ **실시간 환각 탐지 은닉 프로브** (Artif Intell Med 2026, S153204642600105X) — 보정된 은닉 상태 프로브로 토큰 단위 Safe/AtRisk/Hallucinating 분류, FPR 제약 하 스트리밍 임상 배포 지향 — **AI in Medicine 게재 논문이라 저널 적합성 인용으로 유용**.
- ★ **ADR 프로빙** (PMC11844579, 2025) — 은닉 상태 프로브로 약물 부작용 지식 확인 (AKI 0.957 등).
- ★ **정렬-저항 프로빙** (medRxiv 2025.09.17.25336018) — 정렬이 막은 의료 답을 은닉 상태에서 복원; 내부가 출력보다 많이 알고 보정도 좋다.
- **약리 지식 추적** (arXiv:2603.03407) — 약리 지식의 레이어·모듈 소재 추적.
- **Medical Knowledge Maps** (arXiv:2510.11390) — 의료 지식의 내부 지도화.
- **MultiHaluDet** (arXiv:2605.24919) — 다국어 환각 탐지를 은닉 상태 프로브로 (비의료 포함).

### 내부 표상 분해 — SAE (의료)

- ★ **JMIR AI SAE-의료** (ai.jmir.org/2026/1/e81134) — 의료 LLM에 SAE 적용, 단의미 특징 추출.
- **임상 시퀀스 SAE** (arXiv:2605.04072) — EHR 파운데이션 모델에 TopK SAE; 사망 예측 신호 보존, 데이터셋 일반화는 불완전.
- **의료 VLM SAE 조향** (arXiv:2605.24977) — SAE 특징 조향으로 오류 유형 제어, 효과가 아키텍처 의존적.
- **의료 영상 SAE** (arXiv:2603.23794) — 의료 영상 표현 학습에 SAE.
- **단일세포 SAE** (PubMed 42155660) — 세포 파운데이션 모델에서 해석가능한 세포형 프로그램 (인접 분야).

### CoT·설명 충실성 (의료, 행동 수준)

- ★ **Faithful or Just Plausible?** (arXiv:2603.13988, NeurIPS 2025) — 폐쇄형 LLM 의료 추론에 인과 절제·위치 편향·힌트 주입; CoT는 예측을 안 끌고 힌트는 인정 없이 흡수됨.
- ★ **Clinical Reasoning Graphs** (arXiv:2606.29876) — 추론 단계가 장식적(지워도 답 불변), 유사 케이스 간 비일관 — "역량은 있으나 일관성 없음".
- **의료 VLM 충실성** (arXiv:2510.11196) — 멀티모달 perturbation으로 의료 VLM 추론 충실성 평가.
- **FaithMed** (arXiv:2607.01440) — 근거 기반 의료 추론을 루브릭 RL로 학습 (충실하게 만들기).
- **가이드라인 추론 학습·평가** (arXiv:2512.03838) — 진료지침 기반 추론의 학습과 평가.
- **MR-Bench** (arXiv:2604.08559) — 의료 추론 LLM 서베이 + 벤치마크.
- **전문가 수준 추론 자동 평가** (PMC12796170) — LLM 판정자로 의료 추론 품질 채점.
- **ART 임상 추론 신뢰성** (arXiv:2510.16095) — 보조생식술 도메인 맹검 비교 평가.
- **의료 인과 추론 주의 촉구** (OpenReview NuA9hnuxAG) — LLM의 의학 인과 추론에 대한 방법론적 경고.

### 인지 편향·제안 취약성 (의료, 행동 수준) — §C와 중복 요약

- ★ **BiasMedQA** (npj Digit Med 2024) — 7종 임상 인지 편향 주입, 정확도 하락; 행동만.
- ★ **추론은 편향을 못 막는다** (medRxiv 2025) — reasoning 모델도 임상 인지 편향에 취약.
- ★ **Mahajan 관점** (npj Digit Med 8:428, 2025) — 임상 LLM anchoring; 트레이스를 감사 고리로 제안.
- **사용자 요인 민감성** (arXiv:2503.22746) — 의료 질의의 사용자 발화 요인(강조·오정보)에 대한 민감성.
- **SycoEval-EM** (arXiv:2601.16529) — 응급 시뮬레이션 임상 대화의 시코펀시 평가.
- **병리 anchoring 인적 요인** (arXiv:2603.11821) — 계산병리에서 자동화 편향·닻 효과 (인간 요인).
- **의사-LLM 자동화 편향 RCT 프로토콜** (NCT07328815) — 행동 넛지로 자동화 편향 완화 임상시험.

### 개념 병목 (의료 영상 — 인접, "미리 정한 개념" 계열)

- **Concept Complement Bottleneck** (arXiv:2410.15446, PMLR 2026) — 교차 어텐션으로 개념별 특징 추출.
- **Radiologist-Guided Causal CBM** (arXiv:2605.07785) — 영상의학과 의사 유도 인과 개념 병목 (CXR).
- **멀티라벨 CBM 탐색 서베이** (OpenReview MeOQtY5kVM) — 의료 영상 다중 라벨 조건의 CBM.
- **하이퍼그래프 준지도 CBM** (arXiv:2606.01698) — 고차 개념 의존성 모델링.
- **Concept-Enhanced Multimodal RAG** (Inf Syst Front 2026) — 시각 표현을 임상 개념으로 분해 + RAG.
- **CheXOne** (arXiv:2604.00493) — 추론 가능 CXR 파운데이션 모델, 추론 트레이스의 임상 사실성 주장.

### 지식 편집 (의료)

- **의료 LLM 지식 편집** (arXiv:2402.18099) — 사실 지식과 설명 능력의 동시 편집 (MedLaSA).

### 리뷰·서베이 (도입부·한계 절용)

- ★ **Cracking the clinical code** (ScienceDirect 2025) — 의료 보고서 생성 기계적 해석 스코핑 리뷰; "임상의 대면 인터페이스 부재"를 핵심 공백으로 지목.
- ★ **Why LLMs' Clinical Reasoning Fails** (medRxiv 2026.01.26) — 벤치마크 고성능의 설명 불가 변동성; 규제 관점.
- **의료 LLM 신뢰성 종합 서베이** (arXiv:2502.15871) — truthfulness/privacy/safety/robustness/fairness/explainability.
- **임상 CDS 설명가능 NLP 체계적 리뷰** (Springer 2026, 42편) — 2023–2025 지형.
- **XAI 헬스케어 사용례 리뷰** (Frontiers AI 2026) — 영상·진단·재활 36편; 영상=saliency, 진단=SHAP/LIME 지배.
- **인간 중심 XAI 평가 서베이** (arXiv:2502.09849) — 임상 CDS에서 설명의 인간 평가.

**일람에서 읽히는 것**: 의료 쪽 내부 접근은 (1) 환각·지식 프로브(있다/없다·맞다/틀리다의 스칼라)와 (2) SAE 특징 사전(미리 학습된 개념), (3) 영상 쪽 개념 병목(미리 정한 개념)이 전부다. **개별 케이스에서 "무엇이 답을 움직였는지"를 자연어로 읽는 시도는 없다.** 행동 쪽은 힌트 주입까지 왔지만(2603.13988) 폐쇄형 모델이라 내부로 못 들어갔다.

---

## 부록 B. DDXPlus에서 LLM(·Gemma) 성능을 잰 선행 (2026-08-23 수집)

우리 직접 정답률(gemma-3-12b, 진단당 100케이스 균형 표본, 자유 서술 채점)
35.7%의 외부 맥락:

- ★ **MedGemma Technical Report** (Google, arXiv:2507.05201) — **Gemma 계열을
  DDXPlus로 직접 평가한 공식 선행.** 245케이스 부분집합, 49개 병리 폐쇄 목록
  프로토콜. 의료 파인튜닝 이득은 4B에서 +14pp로 유의하지만 27B에서는 사라지고
  일반 Gemma가 앞선다 (정확 수치는 인용 전 원문 표 재확인 ▢). 후속:
  MedGemma 1.5 TR (arXiv:2604.05081).
- ★ **H-DDx** (arXiv:2510.03700, NeurIPS 2025) — **22개 LLM(Gemma3 포함)을
  DDXPlus로 벤치마크.** free-text 진단을 ICD-10에 매핑하는 계층 지표(HDF1)
  제안 — flat top-k가 임상적 근접 오답을 놓친다는 비판. 사례 연구에서 base
  Gemma3가 정답을 놓치고 응급을 놓친 예 포함. 우리 별칭 인지 채점의 인용
  이웃이자, "채점 방식이 결과를 좌우한다"는 방증.
- **MedS-Bench** (arXiv:2408.12547) — DDXPlus 진단 과제 zero-shot: InternLM2
  35.20 / Mistral 34.80. **우리 35.7%와 거의 같은 구간** — 우리 소스 정답률이
  공개 벤치마크 수준과 일치한다는 외부 정합성 근거.
- 기타 DDXPlus-LLM 사용: **MEDDxAgent** (ACL 2025) 모듈형 진단 에이전트,
  **Second Opinion Matters** (arXiv:2505.23075) 전문가 앙상블 합의,
  **Inflated Excellence** (arXiv:2510.09275) 동적 평가로 벤치마크 거품 지적.

**논문에서의 용도**: ① 방법 절에서 "DDXPlus는 LLM 진단 벤치마크로 확립됨"의
근거 (MedGemma TR·H-DDx·MedS-Bench 인용) ② 우리 1,747/4,900(35.7%)이 낮은 게
아니라 공개 모델의 표준 구간임을 한 문장으로 방어 ③ 이들 전부 **성능 측정**에
그침 — 내부 표상·개입·귀속은 없음 → 참신성 문장 유지.

---

## 부록 C. 심사 대비 — "왜 SHAP/LIME이 아닌가" (AI in Medicine 독자 기본값)

의료 XAI의 지배적 패러다임은 입력 기여도 분석(LIME: 국소 선형 대리모델,
SHAP: Shapley 값 배분)이다 — Frontiers 2026 리뷰 기준 진단 분야는
SHAP/LIME이 지배. 예상 질문에 대한 3단 답:

1. **작동 조건 불일치**: 두 방법 모두 스칼라 출력과 케이스당 수백 회
   재실행을 전제 — 자유 서술 생성 LLM 진단에는 기여도를 배분할 스칼라가
   없고, 다회 재실행은 배포 조건과 충돌.
2. **적용하면 우리의 [상한] 행이 된다**: 소견서의 기여도를 perturbation으로 재는
   것은 곧 "소견서 제거 후 재실행 비교" = 짝지은 조건쌍. 본 연구에서 그
   신호는 배포 불가이며(0.57), 단일 실행 내부 AV 판독(0.84)이 이를 능가.
3. **답의 종류가 다르다**: 입력 가중치 목록 vs 내부 결론의 내용 — 교정
   사다리 r5에 되먹일 수 있는 것은 후자뿐.

인용: Ribeiro et al., LIME (KDD 2016, arXiv:1602.04938) · Lundberg & Lee,
SHAP (NeurIPS 2017, arXiv:1705.07874). Related Work 3.2 문단 1의 "플래그와
고정 개념" 옆에 입력 기여도 계열로 한 줄 언급 후 이 논리로 처리.

---

## 부록 D. 배포 전제의 근거 — "임상 LLM의 입력에는 선행 의견이 붙어 온다" (08-24)

서론·방법의 전제 문장("의뢰 소견서는 배포 상황의 실재하는 입력이다")을
받치는 인용 사슬. 3단:

1. **LLM은 이미 임상의가 쓴 문서를 입력으로 받는다**:
   - 의뢰서를 직접 입력으로 쓴 연구 — "Capability of LLMs in assisting
     GPs with diagnoses" (Applied Intelligence 2025, 의뢰서 기반 진단 보조
     평가).
   - 실제 응급실 임상 노트를 입력으로 권고 생성 — Nature Communications
     2024 (s41467-024-52415-1; 트리아지·선행 평가가 노트에 포함).
   - 실배포: Epic×Microsoft GPT-4 통합 — 차트 요약·수신함 답신, 입력은
     EHR 노트 전문 (UCSD·UW·Stanford 파일럿; DAX Copilot 노트 15만+ 건).
2. **그 문서들에는 구조적으로 선행 진단·의심이 들어 있다**:
   - copy-forward/복붙 문헌: 경과기록 텍스트의 >50%가 이전 노트에서 유래,
     입원 경과기록의 82%가 복사·템플릿 (PMC8861699); 감사 차트의 7.4%가
     복붙 포함, 그중 36%가 진단 오류에 기여 (ScienceDirect
     S1386505622002489).
   - diagnostic momentum: 선행 진단 라벨이 검증 없이 릴레이됨 — 선행 심장
     검사 언급만으로 적절 의뢰율 31.4%→12.5% (물리치료사 실험).
3. **따라서**: 임상 LLM의 컨텍스트에는 선행 임상의의 의심이 실려 오는
   것이 기본값이며, 소견서 개입은 그 실재 입력의 최소 모델이다.

서론 전제 문장 + 방법 3.2의 첫 문단에서 인용. 표절 주의: 위 수치 인용 시
원문 확인 후 재서술.
