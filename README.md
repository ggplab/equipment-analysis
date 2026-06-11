# equipment-analysis

교육용 공공데이터(장비 센서) 분석 프로젝트 — **목적 → 가설 → 검증 → 해석** 구조로 진행한 두 개의 분석을 정적 웹사이트로 정리했습니다.

🔗 **사이트**: https://ggplab.github.io/equipment-analysis/

## 두 개의 미션

| 미션 | 데이터 | 질문 | 핵심 결과 |
|------|--------|------|-----------|
| 🚜 Mission A 건설장비 예지보전 | 부품 1,000개 센서·잔존수명 | 어떤 부품부터 점검해야 할까? | 점검 임박 부품 174개(17.4%), 복합 위험군 평균 잔존수명 76시간 |
| ⚙️ Mission B 제조장비 이상탐지 | 장비 7,672건 센서·고장여부 | 고장 나기 전에 미리 알 수 있을까? | 경보선 "진동 ≥ 2.6" — 알람 적중률 79.9% |

## 분석 원칙

수강생이 머신러닝·통계검정을 배우기 전 단계이므로, **개수 세기 · 비율(%) · 그룹별 평균 · 기본 차트(히스토그램/막대/산점도)** 만 사용했습니다. 예측 모델링은 사용하지 않습니다.

## 구조

```
data/      원본 CSV 2종
analysis/  분석 스크립트 (mission_a.py, mission_b.py)
docs/      정적 사이트 (GitHub Pages 소스)
  ├── index.html        메인
  ├── mission-a.html    건설장비 예지보전
  ├── mission-b.html    제조장비 이상탐지
  ├── charts/           분석 차트 PNG
  └── assets/style.css  공용 스타일
```

## 재현 방법

```bash
python3 analysis/mission_a.py   # docs/charts/a/ 에 차트 생성
python3 analysis/mission_b.py   # docs/charts/b/ 에 차트 생성
```

요구 패키지: pandas, matplotlib (macOS 기준 한글 폰트 AppleGothic 사용)
