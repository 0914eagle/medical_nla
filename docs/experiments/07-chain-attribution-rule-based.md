# 07 — CoT에서 소견서가 답을 바꾼 사례 판별, 규칙 기반 3종

**질문**: 체인이 소견서를 **언급**하는가, 그리고 그 언급이 **답이 바뀌었는지와
상관이 있는가**.

**상태**: 🔶 특징 추출 완료, canonical matcher 기준 최종 표 동기화 중.
**이 실험만으로 "체인은 원인을 말하지 못한다"고 쓰면 안 된다** →
[08](08-cot-llm-monitor.md)이 그 문장을 철회시켰다.

---

## 채점기 세 종

`compare_channels_on_attribution.py`의 `chain_features`:

| 특징 | 정의 |
|---|---|
| `chain cites the referral` | 체인이 의뢰서를 인용하는가 |
| `chain names the suspicion` | 체인이 의심 진단명을 부르는가 |
| `chain dwells on the suspicion` | 체인과 제안의 내용어 교집합 크기 |

**전부 규칙 기반**이다. 이것이 이 실험의 한계이자 [08](08-cot-llm-monitor.md)이
필요했던 이유다.

## 이전 결과와 canonical 요약 (n=1,747, 진단 내 층화 AUROC)

| 특징 | AUROC | 진단 내 |
|---|---:|---:|
| 인용 | .4957 | .4969 |
| 언급 | .5040 | .5008 |
| 분량 | .5388 | **.5348** |

**언급 자체는 거의 항상 일어난다**: 답이 바뀐 곳에서 0.9833, 안 바뀐 곳에서
0.9875. 상수는 신호를 나르지 않는다.

비교 기준선(출력만 보는 공짜 특징):

| | AUROC | 진단 내 |
|---|---:|---:|
| 답이 제안을 말한다 | .6539 | **.6642** |
| 답의 단어 수 | .6137 | .6667 |

**체인 특징 세 종이 "답의 단어 수"보다 못하다.**

위 상세 표는 generation-time matcher 값이다. canonical 채널 덤프에서 현재
확인된 camera-ready 후보는 다음과 같다.

- best rule-based CoT feature, all: **.5464**
- answer names suggestion, all: **.6610**
- best rule-based CoT feature, silent: ▢ 재부트스트랩 출력 대기

`답 == 제안`이 아니라 alias-aware matcher를 쓰므로 최종 표 이름은
**Answer names suggestion**으로 쓴다.

## 무엇을 의미하지 않는가 — 08-24 정정

이 값만으로는 **"체인에 신호가 없다"**와 **"우리 채점기가 신호를 못 읽는다"**를
가를 수 없다. 정규식 세 개로는 원리적으로 불가능하다. 프런티어 LLM 모니터를
같은 체인에 세운 결과가 **.7233 / 침묵 .6829**이므로, 답은 후자였다.

**§4.2에서 "설명문은 원인을 말하지 못한다"는 철회한다.** 대신 쓰는 문장은
정량적이다 → [08](08-cot-llm-monitor.md), [11](11-channel-gap-bootstrap.md)

이 실험이 남기는 것: **언급률 96–98%와 소견서 영향 판별 AUROC 0.53의 해리.** "체인이
소견서를 말한다"는 사실이 "체인이 소견서 때문임을 말한다"를 뜻하지 않는다는
관찰은 그대로 서고, 거짓 기각 사례("소견서는 성급하다"고 쓰고도 답이 바뀐
케이스)가 그것을 실물로 보여준다.

## Table 2b에서 읽는 법

- 규칙 특징이 0.5 부근이라는 것은 **그 세 규칙이 약하다**는 뜻이지 CoT 전체에
  소견서 영향 판별 정보가 없다는 뜻이 아니다.
- LLM monitor `.7233`이 같은 CoT에서 더 높은 순위를 만들기 때문에, 이 행은
  CoT의 상한이 아니라 저비용 기준선이다.
- 출력 특징 `.6610`은 답이 제안을 직접 부르는 쉬운 사례를 잡는다. silent
  subset에서는 정의상 사용할 수 없다.

## 남은 것

- ▢ canonical `channel_scores.jsonl`에서 silent rule-based AUROC와 readout 간
  paired bootstrap CI를 확정한다.
- ▢ 상세 특징 세 행도 canonical matcher로 다시 출력한 뒤에만 본문 수치로 쓴다.
