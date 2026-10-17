# 강판 표면 결함 유형 분류

~~프로젝트에서 해결하고자 하는 문제와 핵심 내용을 한 문장으로 소개합니다.~~
강판 표면 측정 정보(결함의 위치, 크기, 형상, 밝기 측정값, 강판 정보)를 이용해 결함을 7가지 유형 중 하나로 분류하는 다중 분류 모델 프로젝트이다.

~~프로젝트의 배경, 목적, 주요 결과를 2~3문장으로 요약합니다.~~
강판 표면 결함 7종을 분류하기 위해 결함 유형별 관측 수가 최대 약 12배 차이 나는 불균형 데이터를 분석하고, Macro F1을 주요 평가 지표로 설정했다. 기본 설정의 Baseline 모델 4종을 시작으로 트리 기반 앙상블 모델의 하이퍼파라미터 튜닝, 클래스 가중치 조정, 소프트 보팅 앙상블 및 클래스별 예측 확률 보정을 적용했다. 최종 선정된 모델(LightGBM + Extra Trees)은 테스트 데이터 389건에서 Macro F1 0.835, Accuracy 0.820을 기록했다.

![프로젝트 대표 이미지](figures/overview.png)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 소개

**프로젝트 배경**

~~어떤 문제를 해결하기 위한 프로젝트인지 설명합니다.~~
- 결함 검출과 결함 유형 분류는 별개의 작업이다. 결함이 발견되더라도 정확한 유형을 판단하는 과정이 추가로 필요하다. 결함 유형에 따라 점검해야 할 공정과 조치 방법이 달라질 수 있으므로 유형을 잘못 분류하면 불필요한 점검이 발생하여 시간과 비용이 증가할 수 있다.
~~해당 문제를 머신러닝으로 접근한 이유를 설명합니다.~~
- 데이터에는 결함의 위치, 면적, 둘레, 밝기, 형상 지표 등 다양한 수치형 측정값이 기록되어 있다. 그러나 고려해야 할 변수가 많고, 결함 유형별 특성이 여러 변수의 조합으로 나타날 수 있어 사람이 일관된 분류 규칙을 직접 정의하기 어렵다. 따라서 여러 변수 간 관계와 패턴을 데이터로부터 학습하여 결함 유형을 분류할 수 있는 머신러닝을 활용한다.

**프로젝트 목표**

~~예측 대상(Target)과 목표를 정의합니다.~~
- **예측 대상(Target)**: 강판 표면 결함 유형(Fault_Type)의 7개 클래스(Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults)를 분류한다. 데이터셋에 정상 강판 데이터는 포함되어 있지 않으므로 양품/불량 판별은 대상이 아니다.

~~프로젝트에서 중요하게 고려한 평가 지표를 설명합니다.~~
- 주 평가 지표: Macro F1. 결함 유형별 관측 수에 차이가 있는 불균형 데이터의 특성을 고려하여 관측 수가 적은 결함 유형도 동일한 비중으로 평가하기 위해 선택했다.
- 평가 보조 지표: Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, 클래스별 F1-score 및 혼동행렬을 활용하여 전체 예측 정확도와 결함 유형별 분류 성능을 함께 확인한다.
- 모델 평가 방향: 기본 설정의 Baseline 모델과 클래스 가중치 조정, 클래스별 예측 확률 보정 등을 적용한 모델의 성능을 평가하고 Macro F1을 중심으로 결과를 분석한다.

### 1.2 프로젝트 정보

| 항목 | 내용 |
|---|---|
| 프로젝트 기간 | 2026.10.13 ~ 2026.10.19 |
| 프로젝트 유형 | 팀 프로젝트 |
| 데이터 출처 | [Steel Plates Faults (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/198) |
| 주요 기술 | Python, pandas, NumPy, scikit-learn, LightGBM, CatBoost, SciPy, Matplotlib |

### 1.3 주요 성과

- **모델 성능:** ~~예) Recall 0.62 → 0.81 개선~~ 최종 모델(LightGBM + Extra Trees 소프트 보팅)은 테스트 데이터 389건에서 Macro F1 0.8349(부트스트랩 95% 신뢰구간: 0.786~0.876), Accuracy 0.8201, Balanced Accuracy 0.8527
→ 예시처럼 줄여서 쓰면 > 최종 모델(LightGBM + Extra Trees)의 테스트 Macro F1 0.8349, Accuracy 0.8201 달성 < 정도면 될 것 같아요.

- **불균형 처리:** ~~예) SMOTE 적용 후 소수 클래스 탐지 성능 개선~~ 클래스 가중치 탐색과 클래스별 예측 확률 보정을 적용했다. 최종 앙상블 모델의 확률 보정 전후 Macro F1은 OOF 기준 0.8236 → 0.8358, 동일 테스트 데이터 기준 0.8264 → 0.8349로 상승했다.
→ 예시처럼 줄여서 쓰면 > 클래스별 예측 확률 보정을 통해 동일 테스트 데이터의 Macro F1 0.8264 → 0.8349 개선 <

- **주요 인사이트:** ~~예) 특정 변수가 예측에 중요한 영향을 미치는 것으로 확인~~ 강종과 결함 유형 사이에 강한 연관성이 나타났다(χ² = 667.0, p < 0.001, Cramér's V = 0.586). K_Scatch, Stains, Z_Scratch는 F1-score 0.92 이상을 기록한 반면 Bumps와 Other_Faults 간 혼동은 전체 오분류 70건 중 33건(47.1%)을 차지한다. 순열 변수 중요도 분석에서는 Length_of_Conveyer, Steel_Type, Steel_Plate_Thickness가 주요 변수로 나타났다.
→ 간략하게 작성하면 > 강종과 결함 유형 간 강한 연관성을 확인했으며, Bumps와 Other_Faults 간 혼동이 전체 오분류의 47.1%를 차지했다. 순열 변수 중요도 분석에서는 Length_of_Conveyer, Steel_Type, Steel_Plate_Thickness가 주요 변수로 나타났다. <도 괜찮을 것 같아요.

---

## 2. 프로젝트 구조

실제 프로젝트 디렉터리 구조에 맞게 수정합니다.

```text
project/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
│
├── data/
│   └── raw/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── models/
│   └── best_model.pkl
│
├── figures/
│   ├── overview.png
│   ├── target_distribution.png
│   ├── eda_result.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── dashboard.png
│
└── app.py
```

**주요 디렉터리 및 파일 설명**

| 파일 / 디렉터리 | 설명 |
|---|---|
| `pyproject.toml` | 프로젝트 의존성 및 설정 |
| `uv.lock` | 의존성 버전 고정 |
| `data/` | 분석에 사용되는 데이터 |
| `notebooks/` | 데이터 분석 및 머신러닝 실험 코드 |
| `models/` | 학습된 모델 저장 |
| `figures/` | 분석 결과 및 시각화 이미지 |
| `app.py` | Streamlit 대시보드 |

*실제 사용하는 파일과 디렉터리만 작성합니다.*

---

## 3. 설치 및 실행 방법

본 프로젝트는 **uv + Jupyter Notebook** 환경을 기반으로 수행되었습니다.

### 3.1 실행 환경

- Python 3.14.4
- uv
- Jupyter Notebook
- Streamlit

### 3.2 Repository 다운로드

GitHub Repository의 HTTPS 주소를 사용합니다.

```bash
git clone https://github.com/happening12321/steel-defect-classification
cd <프로젝트 폴더 만들고 지정 필요>
```

### 3.3 uv 설치 및 환경 구성

먼저 [uv 공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)를 참고하여 uv를 설치합니다.

`pyproject.toml`과 `uv.lock`에 정의된 프로젝트 의존성을 설치합니다.

```bash
uv sync
```

프로젝트 의존성에는 Notebook 실행에 필요한 `ipykernel` 등이 포함되어 있어야 합니다.

### 3.4 데이터 준비

데이터가 Repository에 포함되지 않은 경우 아래 방법으로 준비합니다.

1. [데이터셋 다운로드](URL)
2. 다운로드한 데이터를 `data/raw/` 디렉터리에 저장
3. Notebook에서 데이터를 불러와 분석 진행

### 3.5 Jupyter Notebook 실행

**VS Code 사용 시**

1. VS Code에서 프로젝트 폴더를 엽니다.
2. 실행할 `.ipynb` 파일을 엽니다.
3. 우측 상단의 **Select Kernel**을 클릭합니다.
4. 프로젝트의 `.venv`에 있는 Python 환경을 선택합니다.
5. Notebook 셀을 순서대로 실행합니다.

### 3.6 분석 및 모델링 실행 순서

| 순서 | 파일 | 주요 내용 |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | 데이터 탐색 및 EDA |
| 2 | `notebooks/02_preprocessing.ipynb` | 데이터 전처리 |
| 3 | `notebooks/03_modeling.ipynb` | 모델 학습 및 평가 |

*실제 Notebook 파일명과 실행 순서에 맞게 수정합니다.*

### 3.7 Streamlit 실행

Notebook을 통해 필요한 모델 및 결과 파일을 준비한 후 다음 명령어를 실행합니다.

```bash
uv run streamlit run app.py
```

실행 후 브라우저에서 아래 주소로 접속합니다.

`http://localhost:8501`

별도 웹 배포 없이 로컬 환경에서 실행할 수 있습니다.

---

## 4. 기술 스택

| 구분 | 사용 기술 |
|---|---|
| Language | Python |
| Environment & Package Management | uv |
| Data Processing | Pandas, NumPy |
| Statistical Analysis | SciPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn, LightGBM, CatBoost |
| Imbalanced Learning | Class Imbalance Handling |
| Model Interpretation | Permutation Importance (scikit-learn) |
| Dashboard | Streamlit |
| Development | Jupyter Notebook, VS Code, Git |

---

## 5. 데이터셋

### 5.1 데이터 개요

**데이터 출처:** [Steel Plates Faults (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/198)

| 항목 | 내용 |
|---|---|
| 데이터 수 | 1,941건 (결함 1건 = 1행), 결측치 0건, 중복 행 0건 |
| Feature 수 | 원본 27개 → 강종 2열(TypeOfSteel_A300/A400)을 Steel_Type 1열로 통합해 최종 입력 26개 (수치형 25개, 범주형 1개) |
| Target 변수 | Fault_Type — 결함 유형 원-핫 7열을 1열로 병합 |
| 문제 유형 | 다중 분류 (7개 클래스) |
| 클래스 비율 | Other_Faults 34.7% / Bumps 20.7% / K_Scatch 20.1% / Z_Scratch 9.8% / Pastry 8.1% / Stains 3.7% / Dirtiness 2.8% |

### 5.2 데이터 특징

~~- 주요 Feature와 데이터 특성~~
~~- Target 클래스 분포~~
~~- 분석 시 고려해야 할 데이터 특성~~

- **주요 Feature**: 결함 영역의 위치(X·Y 좌표 최소·최대), 크기(Pixels_Areas, X·Y_Perimeter), 밝기(Sum·Minimum·Maximum_of_Luminosity), 형상 지표(Edges_Index, Empty_Index, Square_Index, Orientation_Index 등), 변환 변수(LogOfAreas, Log_X·Log_Y_Index, SigmoidOfAreas), 강판 정보(Steel_Type, Steel_Plate_Thickness, Length_of_Conveyer)

- **이상치 및 분포**: IQR 1.5배 기준 이상치 후보 비율이 Sum_of_Luminosity 20.6%, Pixels_Areas 20.4%, Outside_X_Index 19.1%, X_Perimeter 18.1%로 높게 나타났다. 해당 값들은 실제 결함 특성을 반영할 가능성이 있으므로 제거하지 않았다. 반면 LogOfAreas(0.3%)와 Log_Y_Index(0.2%)는 이상치 후보 비율이 상대적으로 낮았다.

- **변수 간 범위 차이**: Y 좌표는 최대 약 1,300만인 반면 일부 형상 지표는 0~1 범위로 변수 간 값의 크기 차이가 크다. 최종 모델은 변수의 스케일에 상대적으로 영향을 적게 받는 트리 기반 모델을 사용하므로 별도의 스케일링은 적용하지 않았다.

- **강종과 결함 유형의 연관성**: 카이제곱 검정 결과 강종과 결함 유형 간 유의한 연관성이 확인되었다(p < 0.001, Cramér's V = 0.586). 결함 유형별 강종 분포에 차이가 나타났다. K_Scatch(390/391)와 Stains(71/72)는 대부분 A400에서, Z_Scratch(172/190)는 주로 A300에서 발생한다.

**Target 클래스 분포**

![Target Distribution](figures/target_distribution.png)

~~불균형 데이터의 특징과 모델링에서 고려해야 할 사항을 설명합니다.~~
가장 많은 Other_Faults(673건)와 가장 적은 Dirtiness(55건)의 관측 수가 약 12.2배 차이 나므로 모델링 시 다음 사항을 고려했다.

- **평가 지표**: 클래스 불균형으로 인한 Accuracy의 한계를 보완하기 위해 Macro F1을 주요 평가 지표로 선정했다.
- **데이터 분할**: 클래스 비율을 유지하는 층화 분할을 적용했다. (테스트 389건 중 Stains 14건, Dirtiness 11건)
- **불균형 처리**: Baseline(Gradient Boosting)에서 Pastry(Recall 0.594)와 Dirtiness(Recall 0.636)의 재현율이 상대적으로 낮게 나타나 이를 보완하기 위해 클래스 가중치 튜닝과 클래스별 예측 확률 보정을 시도했다.
- **평가 시 주의사항**: 소수 클래스는 테스트 표본이 적어 1~2건의 오분류에도 지표가 크게 변할 수 있으므로 최종 모델의 Macro F1에 대한 부트스트랩 95% 신뢰구간을 함께 확인했다.

---

## 6. 데이터 분석 및 전처리

### 6.1 탐색적 데이터 분석 (EDA)

**주요 분석 내용**

- 데이터 구조 및 변수 유형 확인
- 결측치 및 중복 데이터 확인
- 주요 변수의 분포 확인
- Target 클래스 분포 분석
- 변수 간 관계 분석

**주요 발견 사항**

1. 발견 사항 및 해석
2. 발견 사항 및 해석
3. 발견 사항 및 해석

**분석 결과 시각화**

![EDA Result](figures/eda_result.png)

### 6.2 데이터 전처리

| 처리 항목 | 적용 방법 | 선택 이유 |
|---|---|---|
| 결측치 처리 | | |
| 이상치 처리 | | |
| 범주형 인코딩 | | |
| 스케일링 | | |
| Feature Engineering | | |

**전처리 과정에서 고려한 사항**

- Train / Validation / Test 데이터 분할 방법
- 데이터 누수(Data Leakage) 방지 방법
- 주요 전처리 방법을 선택한 이유

---

## 7. 머신러닝 모델링

### 7.1 Baseline 모델

**사용 모델:** 모델명

**선택 이유**

- Baseline으로 해당 모델을 선택한 이유
- 기본 모델 성능 및 개선이 필요한 부분

### 7.2 모델 비교 및 성능 개선

**적용한 모델**

- 모델 1
- 모델 2
- 모델 3

**주요 실험**

- Baseline 모델 학습
- 다양한 머신러닝 모델 비교
- 불균형 데이터 처리
- 하이퍼파라미터 튜닝

### 7.3 불균형 데이터 처리

**적용한 방법**

- 처리 방법 1
- 처리 방법 2

**선택 이유**

각 불균형 처리 방법을 선택한 배경과 기대 효과를 작성합니다.

**실험 시 고려 사항**

- Resampling 등의 불균형 처리는 학습 데이터에만 적용
- 동일한 평가 데이터에서 처리 전후 성능 비교
- 소수 클래스의 Precision, Recall, F1-score 등 비교

### 7.4 모델 성능 비교

| 모델 / 실험 | Precision | Recall | F1-score | PR-AUC |
|---|---|---|---|---|
| Baseline | | | | |
| 모델 A | | | | |
| 모델 A + 불균형 처리 | | | | |
| 모델 B + 불균형 처리 | | | | |
| 최종 모델 | | | | |

*프로젝트 목적에 적합한 평가 지표를 사용합니다.*

**성능 비교 결과**

![Model Comparison](figures/model_comparison.png)

**주요 분석 결과**

- 성능이 가장 우수했던 모델:
- 불균형 데이터 처리 전후 성능 변화:
- 모델별 성능 차이가 발생한 원인에 대한 해석:

---

## 8. 최종 모델 및 결과 분석

### 8.1 최종 모델 선정

- **최종 모델:**
- **적용한 불균형 처리 방법:**
- **주요 하이퍼파라미터:**
- **선정 이유:**

### 8.2 최종 모델 평가

**Confusion Matrix**

![Confusion Matrix](figures/confusion_matrix.png)

**평가 결과 해석**

- False Positive 및 False Negative 분석
- 실제 문제에서 중요하게 고려해야 할 오류
- 최종 모델의 강점과 한계

### 8.3 Feature Importance 및 모델 해석

![Feature Importance](figures/feature_importance.png)

**주요 분석 결과**

- 예측에 중요한 영향을 미친 Feature
- 해당 Feature의 영향에 대한 해석
- 분석 결과를 통해 도출한 인사이트

SHAP을 활용한 경우 해당 시각화와 분석 결과를 추가합니다.

---

## 9. Streamlit 대시보드

머신러닝 프로젝트의 주요 분석 결과와 모델 성능을 확인하기 위한 Streamlit 대시보드입니다.

### 9.1 주요 기능

- 데이터 및 Target 클래스 분포 시각화
- 불균형 처리 전후 모델 성능 비교
- 최종 모델 평가 결과 확인

### 9.2 실행 화면

![Streamlit Dashboard](figures/dashboard.png)

*실제 Streamlit 실행 화면을 캡처하여 첨부합니다.*

---

## 10. 결론 및 회고

### 10.1 프로젝트 주요 결과

- 프로젝트 목표 달성 여부
- 최종 모델의 성능 및 주요 개선 사항
- 불균형 데이터 처리 과정에서 확인한 결과
- 분석을 통해 도출한 주요 인사이트

### 10.2 한계점 및 개선 방향

- 데이터 측면의 한계
- 모델 성능 및 일반화 측면의 한계
- 추가로 시도하고 싶은 실험이나 개선 방법

### 10.3 프로젝트를 통해 배운 점

- 데이터 분석 및 모델링 과정에서 새롭게 이해한 내용
- 프로젝트 진행 중 발생한 문제와 해결 경험
- 향후 유사한 프로젝트에서 개선하고 싶은 점

---

## 11. 팀원 및 역할 (팀 프로젝트인 경우)

| 이름 | 담당 내용 | GitHub |
|---|---|---|
| 팀원 1 | | |
| 팀원 2 | | |
| 팀원 3 | | |

팀 프로젝트인 경우 각 팀원의 주요 기여 내용을 작성합니다.
