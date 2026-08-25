# 10 — AV 판독으로 소견서 유발 답변 이동 판별 (Table 2b · Figure 4a)

**질문**: 자연어 판독이 **한 번의 실행에서** 소견서가 답을 바꿨음을 짚는가.

**상태**: ✅ 완료 (DDXPlus). 구간은 [11](11-channel-gap-bootstrap.md)

---

## 질문을 두 채널에 똑같이 던진다

각 채널은 **오답 소견서 arm 하나**만 본다. 정답은 소견서 없는 arm에서 오는데
**어느 채널도 그것을 못 본다.** 이것이 배포 상황이기도 하다 — 추론 시점에
참조할 반사실이 없다.

## 판독 채널의 특징들

`readout_features` (`compare_channels_on_attribution.py`):

| 특징 | 뜻 |
|---|---|
| `readout before the answer names the suspicion` | 결론 판독이 제안 진단명을 부르는가 |
| `readout cues cite the referral` | **근거 슬롯**이 의뢰서를 인용하는가 |
| `readout cues name the suspicion` | 근거 슬롯이 제안을 부르는가 |
| `internal conclusion contradicts the answer` | 1 − 토큰 겹침(내부 결론, 출력 답) |
| **`answer omits the internal conclusion (containment)`** | 위와 같은 신호의 **길이 강건형** |
| `[paired] conclusion readouts diverge` | 두 arm을 다 봄 — **상한이지 배포 신호가 아님** |

**cue 위치는 도울 수 없다.** 소견서가 그 뒤에 있으므로 인과 어텐션 하에서
두 arm 사이에 비트 단위로 같고 같은 말을 두 번 한다. 신호는 소견서 자신의
위치나 최종 토큰에서만 나올 수 있다.

## 결과 (n=1,747, 진단 내 층화)

아래 상세 특징표는 최초 matcher 결과를 보존한 감사 기록이다. canonical
camera-ready 핵심값은 **전체 .7506**, **silent .8302**다.

**전체**

| 채널 | AUROC | 진단 내 |
|---|---:|---:|
| 답 == 제안 (출력만) | .6539 | .6642 |
| 체인 특징(최강) | .5388 | .5348 |
| 내부 결론이 답과 모순 | .7145 | .6910 |
| **답이 내부 결론을 누락 (포함 검사)** | **.7528** | **.7545** |

**침묵 구역** (canonical n=1,641, moved 218 — 답이 제안과 다른 케이스. 출력만 보는
기준선은 여기서 정의상 0.5)

| 채널 | AUROC | 진단 내 |
|---|---:|---:|
| 결론 판독이 제안을 부름 | .5487 | .5404 |
| 근거가 의뢰서를 인용 | .5000 | .5000 |
| 내부 결론이 답과 모순 | .7800 | .7833 |
| **답이 내부 결론을 누락** | **.8214** | ~~.8415~~ → **.8302 canonical** |

## 읽는 법 — 어느 특징이 이기는지가 결론을 바꾼다

이기는 특징은 **"출력이 내부 결론을 담지 않는다"**이지 **"판독이 소견서를
말한다"**가 아니다. 근거 슬롯이 의뢰서를 인용하는 특징은 정확히 **0.5000**,
즉 아무 정보도 없다.

**그러므로 이 채널이 하는 일은 "소견서 탓임을 말하기"가 아니라 "내부와 출력이
어긋났음을 보이기"다.** 논문 문장을 그 범위로 써야 한다. `compare_channels`의
docstring이 실행 전에 적어둔 반증 조건이 이것이었고, 결과는 그 조건의 경계에
가깝다 — 원인을 직접 설명하는 것이 아니라 **국소화 + 불일치 탐지**다.

`[paired]` 행(.5286/.5696)은 두 arm을 다 보고도 이보다 낮다. 반사실을 준다고
저절로 좋아지지 않는다.

## 프로브와의 관계

같은 정본 침묵 구역(n=1,641)에서 LLM 모니터는 **.6829**, 판독은 **.8302**,
프로브는 **.9840**이다(전체는 .7233/.7506/.9280). 정본 표가 주장하는 것은
AV의 우승이 아니라 경계다 — 내부를 안 보는 채널보다 내부 채널이 강하고,
닫힌 49-class에서는 지도 프로브가 가장 강하다.

판독은 고정 클래스 목록 없이 문장을 생성할 수 있다. 그러나 이것만으로
open-vocabulary probe/retrieval baseline이 불가능하거나 판독이 우월하다는 뜻은
아니다. **"출력 형식이 정의된다 ≠ 실제로 된다"**이며 후자는 아직 못 보였다
→ [13](13-mcr-conclusion-adapter.md)

## Table 2b에서 읽는 법

- 이기는 신호는 `answer omits the internal conclusion`이다. 즉 자연어 판독이
  소견서라는 원인을 직접 설명했다기보다 **내부 결론과 출력의 불일치**를
  드러냈다.
- `.8302`는 진단 내 AUROC다. 진단별 취약성 차이를 이용한 pooled 성능이 아니다.
- silent subset에서 출력 복사 신호는 정의상 상수다. 이 구간에서 readout이
  monitor보다 높은 것이 내부 채널의 실질적 추가 정보다.
- `[paired]` 특징은 no-note arm도 보므로 배포 가능한 한 번 실행 신호가 아니다.

## 남은 것

- ▢ 상세 readout 특징 전체를 canonical matcher로 다시 출력해 감사 기록과
  정본 표를 분리한다.
- ▢ MCR wrong-note activation에서 source-aligned readout을 검증하기 전에는
  open-vocabulary 일반화를 주장하지 않는다.

## 재현

```bash
python scripts/compare_channels_on_attribution.py \
  --answers $ART/results/ddxplus_hint_answers_v2.jsonl \
  --cases $DATA/ddxplus_hint_cases_v2.jsonl \
  --cot-answers $ART/results/ddxplus_hint_answers_cot_full.jsonl \
  --readouts $ART/results/readout_hint_final_L32_v2.jsonl \
  --readout-manifests $ART/activations/hint_positions_L32/layer32/last_token/manifest.jsonl \
  --dump $ART/results/channel_scores.jsonl
```
