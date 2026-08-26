# E0. DiReCT data and evaluator audit

## 질문

제한 배포본을 누수 없이 분할하고 공식 평가 코드를 재현할 수 있는가?

## 결과

- sample JSON 511, KG JSON 24, invalid sample 0
- disease category 25, official canonical PDD 61
- deduction 5,109개, observation exact-substring grounding 4,965/5,109 = 0.9718
- 469 patient groups, patient ID unparsed 4행
- exact duplicate copy 1행
- canonical label conflict 10행
- 최종 eligible 496행
- raw note에서 normalized gold-label phrase 직접 노출 28/511 = 0.0548

## Split

| split | rows | patient groups | PDDs | categories |
|---|---:|---:|---:|---:|
| train | 263 | 244 | 56 | 25 |
| val_seen | 62 | 56 | 27 | 18 |
| test_seen | 71 | 63 | 28 | 23 |
| test_pdd_heldout | 100 | 95 | 5 | 4 |

Held-out PDD는 HFrEF, HFpEF, NSTEMI, Low-risk PE, Non-Allergic Asthma다. 환자 그룹은
네 split 사이에서 겹치지 않는다. 같은 환자가 연결한 PDD는 connected component로
묶어 train과 heldout에 나뉘지 않게 했다.

## Official evaluator smoke

Oracle 10행을 official Llama-3-8B semantic matcher에 통과시켰다. 10/10 evaluation JSON이
생성됐고 missing/invalid는 0이었다. Acccat, Accdiag, Obscomp, Expcom, Expall은 1.0이었다.
Official Obspre/Obsrec은 `+1` denominator 때문에 oracle에서도 평균 0.8104이며 버그가
아니다. Unsmooothed observation precision/recall은 1.0이나 공식 점수로 보고하지 않는다.

## 판정

데이터 구조, 환자 분리, 중복 제거와 official evaluator smoke는 완료됐다. Note 본문에
canonical PDD 또는 annotation-root 진단명이 정규화된 완전 구문으로 직접 등장한 행은
28/511(5.48%)였다. 이 flag는 split별 label-leakage sensitivity cohort에 사용하며,
primary eligibility를 결과에 맞춰 다시 바꾸는 데 사용하지 않는다.

현재 71+100행은 E1/E2 설계 점검에 이미 사용됐으므로 exploratory pilot이다. E3 이후의
최종 주표는 별도로 동결한 confirmatory protocol을 사용한다.
