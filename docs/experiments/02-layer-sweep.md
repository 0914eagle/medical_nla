# 02 — 레이어 스윕 (cue 위치, L16 / L24 / L32)

**질문**: 어느 층의 활성값이 소견을 가장 잘 담고 있는가.

**상태**: ✅ 완료 (2026-08-17). 원본: `docs/results/results_2026-08-17_layer_sweep.md`

---

## 설정

- 동일 레시피(케이스별 활성값, cue 문자열 heldout, 소견 하나짜리 타깃,
  시드 17 — **레이어 간 분할이 동일**)를 L16 / L24 / L32에서
- **모든 레이어가 같은 L32-AV 체크포인트를 통과**하고 레이어별 LoRA만 다르다
  (공유 디코더 + 레이어별 어댑터). 어댑터: `medical_nla_cue_position_L{16,24}_v5_lora_e2`
- ⚠️ **이 스윕의 결함**: L16/L24는 2에폭, L32는 3에폭이었다. 그래서 레이어
  효과에 에폭 효과가 섞여 있고, 지금 `run_readout_training.sh`가 모든 레이어에
  같은 에폭을 강제하는 이유가 이것이다.

## 결과 — 역U자, 24에서 정점

438개 heldout-cue 행 전수 의미 채점 (A/B/C/D):

| 레이어 | A 정확 | B 패러프레이즈 | **A+B 의미 판독** | C 계열/속성 오류 | D 완전 오답 |
|---|---:|---:|---:|---:|---:|
| 16 | 10.7% | 23.3% | **34.0%** | 52.1% | 13.9% |
| **24** | 16.7% | 56.4% | **73.1%** | 26.0% | **0.9%** |
| 32 | 17.8% | 37.9% | **55.7%** | 35.8% | 8.4% |

자동 지표도 순서가 같다 (heldout strict / soft@0.5 / mean token recall):
L16 .107/.607/.510, **L24 .167/.813/.658**, L32 .178/.699/.589.

**L24는 에폭이 하나 적은데도 이긴다.**

## 무엇이 갈리는가

> seen-cue 풀은 어느 레이어에서나 0.97–0.99다 — **분포 내 판독은 어디서나
> 쉽고, 레이어는 열린 어휘 일반화에서 갈린다.**

이 한 줄이 [13](13-mcr-conclusion-adapter.md)에 직결된다. MCR은 우리 코퍼스
중 유일한 열린 어휘 코퍼스인데 **L32에서만 쟀다.**

## 질적 구조

- **L24 (정점)**: 패러프레이즈가 정밀하고 최소 편집이다 — "how severe is the
  itching" → "how bad/intense is the itching", bloated abdomen → "swelling of
  the abdomen (this is called ascites)"(올바른 임상 해석을 덧붙임). 완전 오답
  D가 438중 4개로 거의 없고, L32의 극성 뒤집힘이 일어나지 않는다.
- **L32**: 괜찮지만 무디다. 템플릿 계열로 흘러가고 극성이 뒤집힌다
  ("no shortness of breath...", 안정/운동 반전), D 8.4%.
- **L16 (바닥)**: 대략의 주제는 읽지만 내용 결합이 약하다.

## 한정어 — 답 위치는 스윕한 적이 없다

이 스윕은 **cue 위치**에서만 했다. 논문의 주장이 사는 **답 위치**는 L32로
고정돼 있는데, 그건 선택이 아니라 **상속**이다(AV 체크포인트가 `...-L32-av`).
§4.1에 이 한정어를 적어야 한다: *"레이어는 cue 위치에서 스윕했고 답 위치는
L32로 고정했다."*

▢ 답 위치 L24 판독을 올바른 템플릿(`medical_nla_v2_readout.txt`)으로 실행.
08-24에 시도했으나 기존 `readout_hint_final_L24.jsonl`이 cue용 템플릿으로
생성돼 있어(진단명을 대지 말라는 블록 안에서 진단명을 대고 있다) 비교가
성립하지 않았다.

## Appendix Figure A1을 읽는 법

Appendix Figure A1은 하나의 position ablation이 아니라 **서로 다른 두 layer sweep을
나란히 둔 그림**이다.

- **(a) Cue-token reader, held-out cue strings**: cue 위치별 판독을 학습하고
  학습에서 보지 않은 cue 문자열 438개를 읽는다. `L16 .510 → L24 .658 →
  L32 .589`로 L24가 가장 높다.
- **(b) Final-prompt-token reader, diagnosis-held-out split**: cue-first 타깃으로
  학습한 답-형성 위치 판독이다. seen 진단은 `.360/.684/.625`, held-out
  진단은 `.188/.249/.188`이다. 모든 층에서 큰 seen–heldout 격차가 남는다.

**말할 수 있는 것**: 두 sweep 모두 L24에서 정점을 보이며, cue-token 판독은
held-out cue 문자열에도 상당한 lexical recall을 보인다. 답-형성 위치의
cue-first 판독은 보지 않은 진단으로의 전이가 약하다.

**말하면 안 되는 것**: (a)의 `.658`과 (b)의 `.249` 차이를 순수한
`cue position > final position` 효과로 해석하지 않는다. 두 패널은 위치뿐 아니라
학습 recipe, held-out 축, 표본이 모두 다르다. 위치 효과를 주장하려면 같은
adapter·같은 split·같은 target으로 통제된 ablation이 별도로 필요하다.

## 남은 것

- Appendix Figure A1 캡션에 `different recipes and held-out axes; compare within panels`
  를 명시한다.
- 현재 그림 숫자는 lexical scorer다. 보류 중인 외부 의미 판정 결과와 같은
  값처럼 섞지 않는다.
- 통제된 cue-vs-final position ablation은 현재 논문의 필수 관문은 아니지만,
  위치 자체의 인과 효과를 주장하려면 필요하다.
