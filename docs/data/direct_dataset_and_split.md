# DiReCT 데이터셋과 정본 split

## DiReCT가 제공하는 것

DiReCT restricted release는 511개의 임상 note JSON과 24개의 diagnostic KG JSON을
포함한다. 각 sample에는 원문 입력 필드와 하나의 diagnosis root가 있으며, 주석 node는
관찰(Input), 근거(Cause), 중간/최종 진단(Intermedia) 구조를 이룬다.

`PDD`는 Primary Discharge Diagnosis다. Disease category보다 세분화된 최종 진단 label이다.
예를 들어 category가 Heart Failure일 때 PDD는 HFrEF, HFpEF처럼 나뉠 수 있다.

## 감사 결과

| 항목 | 값 |
|---|---:|
| Raw notes | 511 |
| Valid JSON | 511 |
| Disease categories | 25 |
| Official canonical PDDs | 61 |
| KG files | 24 |
| Deduction triples/chains | 5,109 |
| Exact-substring grounded observations | 4,965/5,109 (0.9718) |
| Parsed patient groups | 469 |

최초 폴더 경로 기반 집계의 PDD 62개는 정본이 아니다. 공식 `data_list.csv`와 annotation
root를 정규화한 뒤 official canonical PDD 61개를 사용한다.

## 제외 15행

- Canonical PDD 의미 충돌 10행: `STEMI` folder와 `NSTE-ACS` annotation root
- Patient ID parse 실패 4행
- Exact duplicate copy 1행

따라서 primary split에는 496행이 남는다. 원본 511행을 모델 선택이나 주표의 분모로
섞지 않는다.

## Patient/PDD-disjoint split

| split | notes | patient groups | 역할 |
|---|---:|---:|---|
| train | 263 | 244 | Medical-NLA 학습 |
| val_seen | 62 | 56 | layer/epoch/hyperparameter 선택 |
| test_seen | 71 | 63 | seen-PDD 최종 평가 |
| test_pdd_heldout | 100 | 95 | unseen-PDD 최종 평가 |

합계는 496 notes와 458 patient groups다. 원 audit의 469 groups에서 제외 15행에 포함된
group이 빠지므로 두 숫자는 모순이 아니다.

같은 환자가 여러 PDD에 걸치는 경우 PDD를 connected component로 묶었다. Held-out PDD는
HFrEF, HFpEF, NSTEMI, Low-risk PE, Non-Allergic Asthma이며 train에는 등장하지 않는다.

## 263 train rows로 가능한 범위

LoRA 기반 domain adaptation과 feasibility test는 가능하다. 그러나 263행만으로 범용 의료
판독기를 학습했다고 주장할 수 없다. 그래서 다음을 함께 사용한다.

- 환자 분리와 PDD-heldout으로 암기 여부 확인
- 3 seeds와 strong regularization
- DDXPlus의 더 큰 통제 데이터로 grounding objective 보완
- MCR frozen OOD로 외적 일반화 확인

## 데이터셋별 역할 분리

| 데이터 | 정답으로 쓰는 것 | 평가 |
|---|---|---|
| DiReCT | physician observation-rationale-diagnosis annotation | clinical alignment |
| DDXPlus | pathology, evidence ID/value, differential | activation grounding, patching |
| MCR | case report diagnosis/reasoning | frozen natural-text OOD |

DiReCT annotation은 activation의 ground truth가 아니다. Physician gold와 source model state가
다를 수 있으므로 source-wrong 행은 decision fidelity 분석에서 따로 다룬다.

## 제한 데이터 경로

- Server 62: `/data/heejae/restricted/direct/`
- Server 125: `/data1/heejae/restricted/direct/`

두 서버의 root 차이를 manifest에 하드코딩하지 않는다. `DATA_ROOT`를 사용하고, 다른
서버로 복사한 private manifest의 `source_path`는 destination root로 다시 쓴다.

원문, private manifest, patient ID, raw predictions는 Git에 커밋하지 않는다.
