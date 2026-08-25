# docs/experiments — 실험 하나당 문서 하나

각 문서는 **그 실험만 보고 재현할 수 있도록** 적는다: 어떤 모델, 어떤 데이터,
표본이 몇이고 어떻게 골랐는지, 튜닝을 했다면 하이퍼파라미터가 무엇인지,
무엇을 통제했는지, 실측값이 얼마고 그것이 무엇을 의미하는지, 그리고 **무엇을
의미하지 않는지**.

`docs/paper/`는 논문 조판을 위한 문서다(표 원고·현황·선행연구). 여기는
**실험 기록**이다. 값이 어긋나면 원시 결과와 이 폴더의 모집단·채점 정의를
먼저 대조한다. paper 표가 자동으로 우선하는 것은 아니다. 해결되지 않은 충돌은
`AUDIT_2026-08-24.md`에 기록하고, 해소 전에는 인용하지 않는다.

## 먼저 볼 것

**[`RESULTS_CANONICAL_2026-08-24.md`](RESULTS_CANONICAL_2026-08-24.md) — 숫자의 원장.**
논문에 들어가는 모든 실측치가 출처(스크립트·입력 파일)와 함께 한 곳에 있다.
값이 어긋나면 그 문서가 기준이고, 거기 없는 값은 인용하지 않는다.
아래 실험별 문서는 **왜 그렇게 쟀는지**를, `docs/paper/`는 **조판**을 맡는다.

[`AUDIT_2026-08-24.md`](AUDIT_2026-08-24.md) — 인용 전 점검 목록과 채점기 수정 기록.

**Camera-ready numbering (08-25):** 본문은 Table 1/Figure 2 행동, Table 2a/Figure 3
궤적, Table 2b/Figure 4(a) 탐지, Table 3/Figure 4(b) 교정 순서다. AV 검증과
layer map은 Appendix Table A1/Figure A1이다. 아래 실험 번호는 바뀌지 않으며,
과거 문서의 표 번호보다 `docs/paper/table_camera_ready_2026-08-25.md`를 우선한다.

## 목록

| # | 실험 | 축 | 상태 |
|---|---|---|---|
| [01](01-readout-instrument-validation.md) | 판독 계기 검증 (Appendix Table A1) | Appendix | ✅ |
| [02](02-layer-sweep.md) | 레이어 스윕 L16/24/32 (Appendix Figure A1) | Appendix | ✅ |
| [03](03-note-intervention-ddxplus.md) | 의뢰 소견서 개입 — DDXPlus (Table 1) | 4.1 | ✅ 주 실행 4조건 canonical |
| [04](04-note-intervention-mcr.md) | 의뢰 소견서 개입 — MedCaseReasoning | 4.1 | ✅ |
| [05](05-wording-variants.md) | 문구 4종 (화자 교체) | 4.2 | ✅ canonical 08-24 |
| [06](06-cot-duality.md) | CoT의 이중성 | 4.2 | ✅ canonical 08-24 |
| [07](07-chain-attribution-rule-based.md) | 체인에서 소견서 유발 이동 판별 — 규칙 기반 3종 | 4.2 | 🔶 silent/CI 동기화 |
| [08](08-cot-llm-monitor.md) | 체인에서 소견서 유발 이동 판별 — LLM 모니터 | 4.2 | ✅ 08-24 |
| [09](09-probe-detection-trajectory.md) | 프로브: 탐지·궤적·용량반응 (Table 2a/Figure 3) | 4.2 | ✅ canonical 표·궤적 |
| [10](10-readout-attribution.md) | 판독으로 소견서 유발 이동 판별 (Table 2b) | 4.2 | ✅ canonical |
| [11](11-channel-gap-bootstrap.md) | 채널 격차 신뢰구간 | 4.2 | ✅ canonical |
| [12](12-correction-ladder.md) | 교정 사다리 r3–r7 (Table 3) | 4.3 | ✅ DDXPlus canonical; MCR 대기 |
| [13](13-mcr-conclusion-adapter.md) | MCR 결론 어댑터 (열린 어휘) | Appendix/limitation | ✅ derangement 통과 08-24 |
| [14](14-reader-trust.md) | 독자-신뢰 과제 | 4.2/limitation | ✅ 2,896/2,896 + shuffled 통제 |
| [15](15-judge-infrastructure.md) | 외부 판정자 기반 | 공통 | ✅ |
| [16](16-readout-semantic-judging.md) | 판독 의미 채점 — 손채점 vs 외부 판정자 | Appendix | ✅ 08-24, 238쌍 |
| [17](17-output-head-likelihood.md) | Source output-head likelihood 기준선 | 4.2 | ▢ GPU 실행 |

인용 전에는 [문서 감사 기록](AUDIT_2026-08-24.md)의 미해결 항목을 확인한다.

---

## 모든 실험이 공유하는 설정

### 모델

| 역할 | 체크포인트 | 용도 |
|---|---|---|
| **소스 모델** (연구 대상) | `google/gemma-3-12b-it` | 진단을 답하고, 그 활성값을 뽑는다 |
| **AV 판독기** | `kitft/nla-gemma3-12b-L32-av` | 활성 벡터 하나를 받아 자연어로 서술 |
| (미사용) AR 판독기 | `kitft/nla-gemma3-12b-L32-ar` | 설정에만 존재 |

- dtype **bfloat16**, `device_map: auto`, 카드당 `max_memory: 22GiB`.
  12B가 bfloat16으로 24.4GB라 24GB 카드 하나에 안 들어간다. `auto`에 맡기면
  일부를 meta로 보내고 몇 분 뒤 cuBLAS 안에서 죽으므로 카드를 명시한다.
- 사이드카 `nla_meta.yaml` 검증: `d_model=3840`, `injection_token_id=246566`.
- 생성 기본값: `do_sample: false` (**전부 그리디**, 온도·top_p 없음),
  `max_new_tokens: 256`. MCR 결론 판독만 768 — 타깃이 평균 764자라 256으로는
  54%가 잘렸다(→ [13](13-mcr-conclusion-adapter.md)).

### 하드웨어

24GB 카드 4장. `CUDA_VISIBLE_DEVICES=0,1`에 이 프로젝트가, 2,3은 다른 작업에
비워둔다. 학습·판독 전에 `check_gpu_setup.py --require-free-gb 20`이 카드를
확인하고, 모자라면 **거부한다** — 없으면 18개 실행이 전부 9초 만에 죽고
요약표가 "(did not finish)" 18줄이 된다.

### 활성값 추출 지점

`outputs.hidden_states[32]` = **트랜스포머 블록 32의 출력**(= 블록 33에
들어가기 직전). 튜플은 48+1개이며 index 0은 임베딩 출력, 1–47은 각 블록
출력(최종 norm 이전), 48은 **post-final-RMSNorm**이다. 48번은 norm이 ~158로
47번의 ~213,000과 자릿수가 달라 같은 궤적에 그리면 안 된다.
저장은 float32, 순전파는 bfloat16.

### LoRA 학습 하이퍼파라미터 (`train_medical_nla_lora.py`)

판독 어댑터를 학습하는 모든 실험이 이 값을 쓴다.

| 항목 | 값 |
|---|---|
| rank `r` | **16** |
| `alpha` | **32** |
| dropout | **0.05** |
| target modules | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (7개 전부) |
| optimizer | **AdamW**, lr **2e-4**, weight_decay **0.0**, **스케줄러 없음** |
| epochs | 3 (레이어마다 동일 — 파일럿이 L32만 3, L16/24는 2로 돌려 레이어 효과와 에폭 효과가 섞였다) |
| effective batch | **8** = `batch × grad_accum`. DDXPlus 4×2, MCR 결론 1×8 |
| gradient checkpointing | 켬 (`use_reentrant=False`) |
| seeds | 17 / 18 / 19 (MCR 결론만 17 하나) |
| max train rows | **10,195** — DDXPlus/MCR cue-position 비교에서 맞춘 예산. source-aligned MCR 결론 어댑터는 가용 정합 행이 1,298개라 이 상한에 닿지 않는다 |
| max eval rows | 512 (에폭 간 동일 표본 재사용) |
| **모델 선택** | `--select-on content` — 검증 손실을 **내용 토큰**과 **XML 뼈대 토큰**으로 나눠 내용 손실로 고른다 |

**`--select-on content`가 왜 필요했나.** XML 뼈대는 즉시 학습되고(scaffold
loss ~0.03) 진단명은 안 된다(content ~1.8). 전체 손실로 고르면 대부분 상수인
뼈대가 지표를 지배한다. 그리고 내용 스팬이 0개면 내용 손실이 NaN이 되어
`NaN < best`가 매 에폭 False가 되고, **3시간 학습이 아무것도 저장하지 않고
끝난다**(실제로 발생). 지금은 스팬 0개면 즉시 실패하고, NaN이면 전체 손실로
폴백한다.

### 답 채점

`src/answer_matching.py`의 `is_correct` — **단어 경계 양방향 포함 검사** + 별칭
(`ddxplus_aliases.py`). 양방향이라 한 답이 두 진단명에 동시에 매칭될 수 있지만,
`PE`/`pericarditis`, `stable`/`unstable` 같은 부분문자열 충돌은 막는다.
`took_the_hint`는 여기에 "무소견서 답이 이미 제안을 부르지 않았다"를 추가로
요구한다. canonical DDXPlus 정본은 **moved 321 / causal adoption 91**이다.
alias-aware 이름 일치만 보면 106이지만, 그중 15건은 no-note에서도 이미 그
제안명을 말해 개입의 인과 효과가 아니므로 논문 분자는 91이다.

### 공통 용어

- **moved** = `took_the_hint` **또는** `lost_the_gold`. 소견서가 답을 바꿨다.
- **침묵 구역(silent), 정본 정의** = `answer_names(wrong_answer, suggestion)`이
  거짓인 행. 채널 비교 덤프 기준 **n=1,641, moved=218**이다. 모니터 라벨 파일의
  `not took_the_hint` 정의는 **n=1,656, moved=230**인 별도 민감도 분석이며 같은 표에
  섞지 않는다. 출력만 보는 "답=제안" 신호는 정본 침묵 구역에서 정의상 무력하다.
- **위약(neutral)** = `"The referring note requests evaluation."` — 제안이
  없는 같은 침입. 없으면 오답 소견서 arm이 "제안했다"와 "문장이 늘었다"를
  동시에 잰다.
- 모든 AUROC는 **진단 내 층화**. 이 코퍼스는 진단명만 알아도 pooled AUROC
  0.93이 나와서, pooled 값은 "진단 맞히기"를 일부 재고 있다.
