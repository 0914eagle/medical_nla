# 실험 계획 v3 — 데이터 정의부터 (2026-08-18)

원칙: 모든 실험은 §0에 정의된 데이터 산출물 위에서만 정의된다. 각 실험은
"어느 데이터의 어느 가공물 → 어느 표의 어느 행"으로 주소를 갖는다.

---

# 0. 데이터 정의

## D1. DDXPlus — 주 실험대 (계측 가능한 세계)

**원본**: 합성 감별진단 벤치마크. 케이스 = 인구학 정보 + evidence(문진
문항·항병력) 집합 + 감별진단/최종진단. 우리는 적격 진단 26클래스 사용.

**왜 주 실험대인가**: ① 케이스마다 gold 증거(cue) 문자열이 구조화 —
판독·설명의 **기계 채점**이 가능한 유일 조건 ② 진단 공간이 닫혀 있어
probe·likelihood 채널 성립 ③ 프롬프트가 템플릿 기반이라 **construction-exact
counterfactual**(cue 하나만 바꾼 쌍) 제작 가능.

**가공 산출물** (a–c 구축 완료, d–e 신규):

| ID | 가공물 | 규모 | 쓰임 |
|---|---|---|---|
| D1-a | 진단-heldout split: train 18클래스 / heldout 8클래스 (seed 17). train 884 / val 189 / test_seen 727 / test_heldout 800 | 평가 풀 1058 (source-correct만 학습에 쓴 결과 오답 농축, base rate 74.2% — 표기 의무) | v1 붕괴 실증(표2-A), probe/likelihood 학습·평가(표1) |
| D1-b | cue-position rows: 3,200케이스 × 케이스당 ≤4 cue = 12,800행. 각 행 = cue의 토큰 span에서 활성값 추출, 타깃 = 그 cue 하나. **cue-string heldout**: 유니크 164개 중 41개(25%) 학습 완전 제외. train 7,515 / val 1,086 / test_seen 2,122 / test_heldout 438. layer 16/24/32 × (LoRA/vanilla) | 438행×3layer 전수 수동 A/B/C/D | 판독 검증(표2), layer 지도(Fig2) |
| D1-c | counterfactual 쌍: 150 swap쌍(한 슬롯 교체) + 300 retained(orig/swap/removed 3변형). 재구성 프롬프트가 원본과 문자열 일치함을 검증 | 450행 | 충실성 관문(표3), CoT 1-A′의 재료 |
| D1-d **[신규]** | 자연 분포 test split: 오답 필터를 거치지 않은 새 케이스 표본 (~500-1000) | 해결 대결표 전용 | 표4-C (base/CoT/self-refine/MedPrompt/ours/SFT) |
| D1-e **[신규]** | CoT 생성물: D1-c와 D1-d의 프롬프트에 "진단 + 결정적 근거 설명" 지시를 붙여 생성한 설명 텍스트 | 150쌍 + 힌트주입 ~200 | 가설1 2-arm 실험(표3 CoT열) |
| D1-f **[신규]** | 감사 테이블: 평가 풀 케이스별 (E: 전체 gold cue의 L24 판독, S1: 26-way likelihood 분포, S2: probe, O: 출력) | 오답 785 + 정답 273, cue행 ~6천 | 오류 해부학(표4-A/B), 비순환 예측(표1) |

### D1 가공물 상세

**D1-a (진단-heldout)**: 적격 26클래스를 **클래스 단위**로 train 18 / heldout
8 분할(seed 17) → train 클래스의 **source-correct 케이스만** 학습(884/189;
판독기가 "옳게 아는 상태"를 배우게) → 평가 pool: test_seen 727(학습 클래스
새 환자) + test_heldout 800(미학습 8클래스×100) + leakage guard. 활성값은
format 위치(마지막 토큰). 부산물: source-correct 필터로 평가 풀 1058의
base rate가 오답 74.2% — 표기 의무.

**D1-b (cue-position)**: 케이스 evidence → gold cue 문자열(3~12개/케이스)
→ 케이스당 ≤4개 샘플링(12,800행) → 각 cue를 프롬프트에서 문자열 매칭 →
tokenizer offset span → **span 마지막 subword 토큰** 위치(반복 등장은
occurrence index) → 케이스 1회 forward로 L16/24/32 hidden state 저장 →
행 = (벡터 1, 타깃 = 그 cue 하나). **cue-string heldout**: 유니크 164개 중
41개(25%) 추첨, 해당 행을 train/val에서 완전 제거(진단-heldout이 아닌 이유:
cue가 진단 간 공유되어 진단만 빼면 누출). train 7,515 / val 1,086 /
test_seen 2,122 / test_heldout 438. 학습 = 동결 L32-AV + layer별 rank-16
LoRA(L24 2ep, L32 3ep); 동일 벡터의 vanilla 판독 병행. 채점 = heldout
438×3layer 전수 수동 A/B/C/D.

**D1-c (counterfactual)**: 프롬프트의 템플릿 재구성이 원본과 **문자열 완전
일치**함을 검증(construction-exact) → test 150케이스에서 슬롯 1개 교체
(swap)/제거(removed) + retained 슬롯 2개×{orig,swap,removed} → 각 변형
새로 forward → 슬롯 위치 벡터 → L24 판독. swap 150쌍 전수 수동(T/D/O/X).
1-A′의 CoT 검증에 동일 쌍 재사용(같은 개입을 CoT와 판독이 공유).

**D1-d (자연 분포 split, 신규)**: 기존 평가 풀은 오답 74% 농축이라 개입
전후 정확도 측정에 부적합 → **어떤 가공에도 쓰인 적 없는** 케이스에서
진단별 층화 무작위 ~500–1,000개. 개입 비교(표4-C) 외에는 사용 금지.

**D1-e (CoT 생성물, 신규)**: D1-c의 orig/swap/removed 프롬프트 + "진단과
결정적 근거 설명" 지시 → CoT 생성. **판정 방향 주의**: 설명에서 인용 cue를
추출하는 방식(매칭 실패 위험, 선행 연구에 없는 방식)은 쓰지 않는다. 대신
선행 프로토콜대로 **"우리가 개입한 cue를 설명이 언급하는가"** 를 판정한다
(판정 대상이 특정 cue 하나라 LLM judge + 수동 검증으로 신뢰 확보 가능).
추가 arm: Lanham식 CoT 절단(설명 앞부분만 주고 답하게 → 답 불변이면 장식).
별도: source-정답 ~200케이스에 힌트 문장 삽입판(Turpin/의료판 프로토콜).

**D1-f (감사 테이블, 신규)**: 평가 풀 1058 전 케이스의 gold cue **전부**
(~6천 행)를 L24 cue-위치에서 신규 추출 → 판독으로 cue별 **E**(원형/왜곡/
부재; 자동 + 표본 200 수동 검증) → **S1** = 26 진단명 시퀀스 likelihood
분포(margin/entropy) → **S2** = v2-train 재학습 probe 예측+마진 → **O** =
source 출력 → 케이스당 {E, S1, S2, O, gold} 레코드 → 4유형 결정 트리.

## D2. MedQA-USMLE — 이식/해결 확장 (gold 증거 없음)

**역할**: "감사 기반 개입이 표준 벤치마크의 정확도를 올리는가"의 외부 타당성.
**가공**: ① 지문 내 증거 문장의 마지막 토큰에서 zero-shot 판독 → 표본
~100행 수동 점검(판독 이식성 확인; 불충분 시 light LoRA 재튜닝) ② 오답노트
파이프라인(D1과 동일 구조) 적용. **지표는 정확도 델타만** — 증거 채점은
D1의 몫, D2는 결과 지표의 세계.
**쓰임**: 표4-C의 확장 열. 9월 작업.

## D3. 약물 트랙 — 방법 일반화 (cue = 증상 → 약물/이상반응) [신규 조사]

교수님 제안(약물 데이터) 대응. 선택 기준: **① 텍스트 내 span 주석**(우리
cue-위치 방법의 전제) ② 공개 접근 ③ 과제가 판별형(정답 채점 가능).

| 후보 | 구조 | 판정 |
|---|---|---|
| **PHEE** (권장 1순위) | ~5,000 문장(MEDLINE 사례보고), 약물·이상반응·이벤트 **span 주석**, 공개 | ①②③ 충족 |
| ADE corpus v2 | 1,644 abstract 유래 문장, drug–effect 관계 주석, 공개 | 충족 (규모 작음) |
| DDI corpus | 약물 entity span + 상호작용 유형 라벨 | 충족 (과제가 26지선다형과 유사해 구조 재활용 용이) |
| CT-ADE / OpenDDI | KB·표 형태 (텍스트 span 없음) | **부적합 — 제외** |
| CADEC | 환자 포럼 텍스트 span 주석 | 예비 (구어체 — 이식성 시험용) |

**가공(PHEE 기준)**: 문장 → "이 서술에서 보고되는 유해사건/원인 약물은?"
프롬프트; drug/effect span → cue-위치 행; **span 치환 counterfactual**(약물
A↔B 교체 — 템플릿이 아니어도 span 단위 치환은 construction-exact로 가능);
판독 타깃 = span 문자열. 축소판 프로토콜: zero-shot 판독 + 소규모 재튜닝 +
counterfactual 관문만 (해부학·개입은 D1 전용).
**쓰임**: "cue가 증상이 아니어도 같은 방법이 선다" — 일반화 절 + 표2 확장
행. 9월, D2와 우선순위 경합 시 교수님 결정 사항.

## D4. DiReCT — 실제 임상노트 (조건부)

MIMIC-IV 유래 511개 실제 임상노트에 **의사가 관찰→진단 추론을 span 수준
주석** (NeurIPS 2024 D&B). 우리 감사의 "사람 gold"라 개념적으로 최적이나
**PhysioNet credentialing 필요** — 신청은 즉시 걸어두고, 논문에는 조건부
(승인 시 소규모 검증, 아니면 future work 명시).

## 제외와 사유 (한계 절에 그대로)

- MIMIC 원본: 접근·PHI·일정. → DiReCT 승인으로 부분 대체 시도
- 중국어 문진 대화(MuZhi/Dxy류): 백본·파이프라인 언어 불일치
- KB형 약물 DB: 텍스트 span이 없어 cue-위치 방법 정의 불가

**범위 권고**: 1차 완결 = **D1 단독**. 이후 D2 이식 + D3(PHEE) 중 가능분.
D4는 승인 대기.

---

# 1. 가설1 실험 — 선행 방법론 채택 + NLA arm의 2-arm 쌍

공통: D1-c/d/e, 동일 백본(Gemma-3-12b), CoT와 NLA가 같은 케이스·같은 지표.

**1-A′ 개입 반영 (Turpin/Lanham 프로토콜 + NLA arm)** — 데이터 D1-c+e
- arm CoT-1 (Turpin식): swap/removal로 **우리가 바꾼 cue**를 설명이 언급하는
  비율. 답이 바뀐 쌍에서 특히. (인용-추출 방식이 아니라 개입-언급 판정)
- arm CoT-2 (Lanham식): CoT 절단/오염 후 답 불변율 = 설명이 답의 원인이
  아니었던 비율
- arm NLA: 같은 쌍에서 판독 추적 0.993 / 잔존 0 (완료)
- 확장 권고: D1-c를 500케이스 × 슬롯 2개 ≈ 1,000 swap쌍으로 확대(추출·판독
  비용만, 학습 없음) — 표3의 통계력과 CoT arm 표본을 동시에 강화
- 지지 결과: "개입 반영: CoT n% vs NLA 99.3%" 한 행 → 표3
- 반대 결과: CoT가 충실하면 가설1 축소, 가설2·표4가 부담 (명시)

**1-B′ 숨은 원인 가시화 (힌트 주입 + NLA arm)** — 데이터 D1-a 정답 케이스 ~200
- arm CoT: "동료 의사는 [오답]을 시사" 삽입 → flip률, 힌트 미인정률
- arm NLA: ① 힌트 문장 span 판독 → 숨은 원인이 내부에서 보이는가 ② flip
  케이스의 cue 증거 판독은 온전한가 (flip이 증거 아닌 힌트 탓임을 내부에서 입증)
- 지지 결과: 출력이 은폐한 원인을 내부 판독이 노출 → 표3 마지막 행

**1-C′ 오답 예고 격차** — 데이터 D1-a(1058) + D1-f
- output 최선 3종(confidence 완료 0.67–0.70 / 26-way likelihood
  margin·entropy [신규] / CoT self-consistency [신규]) vs 내부(증거
  인코딩률·probe) — 전 신호 gold 불사용 행끼리 공정 비교 → 표1

---

# 2. 가설2 실험 — 내부 도구 로스터 전체를 같은 벡터에

로스터 (실행 가능성 순): linear probe [D1-a로 재학습, E1] / **logit lens
[신규, 비용 0]** / training-free verbalization = vanilla AV (Patchscopes·
SelfIE 계열로 명명, 완료) / SAE — **Gemma-3-12b용 공개 SAE 부재로 실험
불가**: 비교표에 구조 행(원자 feature, 조합·문장 불가) + 한계 절 명시.

**2-A probe 전수 + 해리** — D1-a: AUROC(1058) + 오답 중 probe=gold 비율
(Orgad 해리의 의료판) → 표1, 표4-A 예고
**2-B 사다리 실증** — D1-b heldout 438 벡터에서: probe(26지선다) → logit
lens(top-k 토큰: 단어 파편은 나오나 조합 문장 불가) → vanilla(문장은 되나
액자 confabulation) → ours(A+B 73.1%) — "닫힌 라벨 → 열린 토큰 → 열린
문장(불신) → 열린 문장(검증)" 4단 사다리 → 표2
**2-C (선택) 속성 probe** — ankle vs calf 이진 probe로 왜곡의 표상/판독
귀속 → discussion

---

# 3. 가설3 실험 — 설계 서사 6단계 (Method 섹션의 뼈대)

1. **왜 cue 타깃**: v1(진단 타깃) heldout 0/800 붕괴 — 26지선다는 클러스터→
   라벨 암기로 뚫림. cue는 고엔트로피 정답지 [D1-a, 완료] → 표2-A
2. **왜 cue 위치**: v3(cue 타깃·format 위치) 0.19 실패 — 답 직전 벡터엔
   증거가 결론으로 접힘. v4(cue 자기 span) 성공 [D1-b, 완료] → 표2-A, Fig2
3. **layer별 어떻게**: L32-AV 디코더 동결(공유 몸통) + layer별 rank-16
   LoRA(q/k/v/o/gate/up/down 7 projection)만 학습. L24 = operating point
   (73.1%, D 0.9%) [완료] → 표2-B
4. **vanilla vs 튜닝**: vanilla도 내용은 담지만 지어낸 액자에 흩뿌림 —
   LoRA는 창조가 아닌 **증류**(confabulation 제거, 단일-cue 정밀화). 좋아진
   것 = 신뢰 가능성 [완료] → 표2 행 + 정성2
5. **faithful함의 증명**: 4중 관문 — OOD 수동 438×3 / 반암기(gold-최근접 +
   layer 대조 34/73/56) / vanilla 귀속 / counterfactual(0.993, 0/150,
   phantom 0.003) [완료] + LLM-judge 제2 평가 κ [J, 신규] → 표2·3, appendix
6. **어떻게 이용**: 감사 테이블 [E1, D1-f] → 오류 4유형 해부학 [E2] →
   유형×개입 3-arm [E3] → 오답노트(내부-감사 vs 출력-만 vs 무처치) [E4] →
   베이스라인 대결 [B1, D1-d] → (스트레치) AR 텍스트 패칭 [E6] → 표4

---
