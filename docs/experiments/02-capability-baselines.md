# E2. Capability baselines

## 질문

생성 전 P0 activation에서 닫힌 진단 label과 열린 임상 내용을 각 방법이 얼마나 읽는가?

## 비교 방법

1. Source output-head candidate sequence likelihood
2. Linear probe
3. Source CoT
4. Vanilla NLA/AV
5. P2 positive leakage control

## 평가

- PDD/category top-1, top-k, MRR
- Seen vs PDD-heldout
- Source answer와 gold를 분리한 decision fidelity
- Open observation/rationale는 DiReCT official evaluator의 호환 가능한 열
- P0/P1/P2 및 HS16/HS24/HS32 sensitivity

Probe는 closed-label upper bound다. Open evidence text 열은 `N/A`이며 실패 0점으로
처리하지 않는다. Vanilla NLA의 자연어 점수가 낮아도 P0 activation에 정보가 없다는
결론을 바로 내리지 않고 probe와 output head를 같이 본다.

Output-head baseline은 단일 다음-token logit이 아니다. PDD 이름이 여러 token일 수 있으므로
각 사전등록 candidate label을 P0 다음에 teacher-force하고 label token들의 평균 log
probability로 순위를 매긴다. 별도 분류 head는 없지만 평가 label ontology를 제공받는
closed candidate-ranking baseline이다. Held-out PDD를 candidate list에 넣은 결과는
zero-shot open generation이 아니라 ontology-given ranking으로 표기한다.

## 실행 상태

Frozen validation 52행의 HS32/P0 prompt comparison은 다음과 같다.

| Prompt | Parse | Source-answer mention | Gold-PDD mention | Category mention | Own-donor source gap | Prompt trigram gap |
|---|---:|---:|---:|---:|---:|---:|
| Default | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Task-aligned suffix | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0007 |

Task-aligned suffix가 literal/case-specific diagnostic을 개선하지 않아 default를 vanilla
primary로 유지한다. P1/P2 validation에서는 source-answer mention이 각각
default 0.5192/0.5962, task-aligned 0.5577/0.5000이었지만, P1 leakage-free subset은
5행이고 두 prompt 모두 0/5였다. P1은 CoT 문자열 누출 분석, P2는 answer-exposed positive
control로만 사용한다.

Test P0/L32 vanilla AV는 171/171행 생성 및 `<explanation>` parsing에 성공했고 빈 출력은
없었다. 출력 길이는 637--741자(중앙값 697, 평균 696.9)였다. 길이 안정성은 내용의
사례 특이성과 별개다. 길이 범위가 매우 좁으므로 exact/normalized 반복률과
own-case-versus-shuffled specificity를 우선 확인한 뒤 clinical alignment를 평가한다.

예비 same-category lexical derangement에서 164행의 own-prompt trigram containment와
shuffled-prompt containment가 모두 0.0013이었고 gap은 -0.0001이었다. 따라서 P0 vanilla
AV가 prompt의 사례 고유 표현을 그대로 복원한다는 증거는 없다. 이 검사는 paraphrase를
인정하지 않으므로 최종 실패 판정으로 쓰지 않고, 동일 claim extractor와 semantic
matcher를 이용한 own-versus-shuffled 평가의 필요성을 확인한 sanity check로만 둔다.

P1/P2 L32는 각각 171/171행 parse됐고 빈 출력과 normalized exact duplicate가 없었다.
Lexical own/shuffled는 P1 0.0067/0.0064(gap +0.0003), P2
0.0017/0.0018(gap -0.0001)이었다. 따라서 reasoning/answer 이후 위치에서도 현재
trigram 검사는 사례 특이성을 찾지 못했다. P1의 절대 overlap 증가는 동일 category의
다른 사례에서도 유지되어 공유 임상 어휘로 설명된다. P2는 답을 이미 본 위치지만 진단
label이 1--2단어인 경우 trigram에 기여하지 않으므로, 이 결과만으로 positive control
실패를 확정하지 않고 phrase-level source-answer recovery를 다음 검사로 둔다.

Phrase-level 결과는 다음과 같다.

| Position | Source-answer mention | Gold-PDD mention | Category mention | Own-vs-donor source gap |
|---|---:|---:|---:|---:|
| P0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| P1 | 0.4912 | 0.1404 | 0.5848 | +0.4146 |
| P2 | 0.3918 | 0.0819 | 0.4854 | +0.3598 |

Own-vs-donor는 같은 disease category이되 다른 source answer를 가진 164행에서 계산했다.
P1은 source answer alias가 reasoning에 없던 15행에서는 1/15=0.0667만 source answer를
언급했다. 따라서 P1 전체의 높은 specificity는 pre-answer 내부 판독보다 CoT 문자열
누출 상한으로 해석한다. P2의 양의 gap은 answer-exposed positive control을 통과한
것으로, vanilla AV가 모든 DiReCT activation에서 무조건 실패하는 것은 아님을 보인다.
반면 생성 전 P0의 diagnosis/category phrase recovery는 0/171이다. 이것은 Medical-NLA가
개선해야 할 baseline failure지만, P0 evidence/rationale의 semantic recovery까지 0이라는
뜻은 아니므로 Table 2 claim extraction과 E5 grounding을 계속 분리한다.

## Model selection

현재 test_seen과 PDD-heldout 171행은 이미 위치 및 vanilla AV 설계 점검에 사용했으므로
exploratory pilot다. 공개 AV/AR와 호환되는 HS32를 primary로 고정한다. HS16/HS24는 같은
L32 decoder의 distribution shift가 섞인 sensitivity다. Task-aligned vanilla prompt와 probe
regularization은 confirmatory train/validation에서만 정한다. 새 final test는 설정과 분석
코드를 동결한 뒤 한 번만 평가한다.

## 산출물

- Table 1
- HS32 primary baseline과 HS16/HS24 sensitivity
- E3에서 사용할 vanilla checkpoint와 prompt 고정
