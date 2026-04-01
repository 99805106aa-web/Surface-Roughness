# MSA ISO 22514-7 Web Application

이 애플리케이션은 기존 데스크톱 기반의 MSA(측정 시스템 분석) 프로그램을 웹 기반으로 변환한 도구입니다. ISO 22514-7 및 VDA 5 표준을 준수합니다.

## 주요 기능
- **Study 1 (측정능력 분석):** Cg, Cgk, 치우침(Bias), 선형성 및 불확도 분석
- **Study 2/3 (Gage R&R):** ANOVA 기반 반복성, 재현성, NDC 분석
- **데이터 로드:** CSV 및 Excel 파일 지원
- **시각화:** Run Chart, 불확도 구성도, Gage R&R 변동 그래프
- **내보내기:** 분석 결과 엑셀 다운로드 지원

## 설치 방법

1. Python이 설치되어 있어야 합니다 (3.8+ 권장).
2. 필요한 라이브러리를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

## 실행 방법

터미널에서 아래 명령어를 실행하세요:
```bash
streamlit run app.py
```

## 프로젝트 구조
- `app.py`: 메인 웹 UI 및 페이지 구성
- `logic/`: 비즈니스 로직 (분석 및 계산 로직 분리)
  - `study1_core.py`: Study 1 계산 로직
  - `study2_core.py`: Study 2/3 ANOVA 및 Gage R&R 로직
  - `common.py`: 공통 유틸리티
