# Legacy wrong-note study archive

이 디렉터리는 2026-08-17부터 2026-08-25까지 수행한 `wrong referral note` 연구의
문서 원본을 보존한다. 파일을 삭제하지 않은 이유는 코드·수치·의사결정의 provenance를
남기기 위해서다.

## 현재 논문에서 분리한 이유

이 연구는 인위적으로 만든 오진 소견서가 모델 답을 얼마나 움직이는지 측정했다.
그러나 실제 임상 사용에서 그런 입력이 자연스럽고 대표적인지에 대한 외적 타당성이
충분하지 않았다. 따라서 다음 결과는 흥미로운 기전 pilot이지만 현재 Medical-NLA
논문의 주 근거가 아니다.

- wrong-note 조건의 정확도 하락과 answer movement
- CoT, probe, AV를 이용한 moved-case 탐지
- 재고 사다리와 reader-trust 실험
- wrong-note trajectory와 selective correction 구상

현재 연구는 인공 소견서에 의존하지 않고 DiReCT의 의사 주석 설명과 DDXPlus의
데이터셋 고유 증거 반사실을 사용한다.

## 사용 규칙

- 이 디렉터리의 표와 `canonical` 수치는 현재 논문 표에 복사하지 않는다.
- 재사용할 수 있는 것은 activation 추출, probe, AV/AR, bootstrap, figure 생성 등
  인프라와 실패 분석이다.
- 내부 링크는 이동 전 경로를 가리킬 수 있다. 역사 기록으로만 읽는다.
