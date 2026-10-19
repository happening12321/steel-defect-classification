# 강판 표면 결함 유형 분류

강판 표면 측정 정보(결함의 위치, 크기, 형상, 밝기 측정값, 강판 정보)를 이용해 결함을 7가지 유형 중 하나로 분류하는 다중 분류 모델 프로젝트이다.

강판 표면 결함 7종을 분류하기 위해 결함 유형별 관측 수가 최대 약 12배 차이 나는 불균형 데이터를 분석하고, Macro F1을 주요 평가 지표로 설정했다. 기본 설정의 Baseline 모델 4종을 시작으로 트리 기반 앙상블 모델의 하이퍼파라미터 튜닝, 클래스 가중치 조정, 소프트 보팅 앙상블 및 클래스별 예측 확률 보정을 적용했다. 최종 선정된 모델(LightGBM + Extra Trees)은 테스트 데이터 389건에서 Macro F1 0.835, Accuracy 0.820을 기록했다.

![프로젝트 대표 이미지](figures/overview.png)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 소개

**프로젝트 배경**

- 결함 검출과 결함 유형 분류는 별개의 작업이다. 결함이 발견되더라도 정확한 유형을 판단하는 과정이 추가로 필요하다. 결함 유형에 따라 점검해야 할 공정과 조치 방법이 달라질 수 있으므로 유형을 잘못 분류하면 불필요한 점검이 발생하여 시간과 비용이 증가할 수 있다.
- 데이터에는 결함의 위치, 면적, 둘레, 밝기, 형상 지표 등 다양한 수치형 측정값이 기록되어 있다. 그러나 고려해야 할 변수가 많고, 결함 유형별 특성이 여러 변수의 조합으로 나타날 수 있어 사람이 일관된 분류 규칙을 직접 정의하기 어렵다. 따라서 여러 변수 간 관계와 패턴을 데이터로부터 학습하여 결함 유형을 분류할 수 있는 머신러닝을 활용한다.

**프로젝트 목표**

- **예측 대상(Target)**: 강판 표면 결함 유형(Fault_Type)의 7개 클래스(Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults)를 분류한다. 데이터셋에 정상 강판 데이터는 포함되어 있지 않으므로 양품/불량 판별은 대상이 아니다.
- **주 평가 지표**: Macro F1. 결함 유형별 관측 수에 차이가 있는 불균형 데이터의 특성을 고려하여 관측 수가 적은 결함 유형도 동일한 비중으로 평가하기 위해 선택했다.
- **평가 보조 지표**: Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, 클래스별 F1-score 및 혼동행렬을 활용하여 전체 예측 정확도와 결함 유형별 분류 성능을 함께 확인한다.
- **모델 평가 방향**: 기본 설정의 Baseline 모델과 클래스 가중치 조정, 클래스별 예측 확률 보정 등을 적용한 모델의 성능을 같은 테스트 데이터에서 평가하고 Macro F1을 중심으로 결과를 분석한다.

### 1.2 프로젝트 정보

| 항목 | 내용 |
|---|---|
| 프로젝트 기간 | 2026.10.13 ~ 2026.10.19 |
| 프로젝트 유형 | 팀 프로젝트 |
| 데이터 출처 | [Steel Plates Faults (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/198) |
| 주요 기술 | Python, pandas, NumPy, scikit-learn, LightGBM, CatBoost, SciPy, Matplotlib, Streamlit |

### 1.3 주요 성과

- **모델 성능:** 최종 모델(LightGBM + Extra Trees)의 테스트 Macro F1 0.8349, Accuracy 0.8201 달성 (Baseline Gradient Boosting 0.7941 대비 +0.041)
- **불균형 처리:** 클래스별 예측 확률 보정을 통해 동일 테스트 데이터의 Macro F1 0.8264 → 0.8349 개선. Baseline 대비 소수 유형 Pastry의 Recall은 0.594 → 0.906으로 올랐다.
- **주요 인사이트:** 강종과 결함 유형 간 강한 연관성을 확인했으며, Bumps와 Other_Faults 간 혼동이 전체 오분류의 47.1%를 차지했다. 순열 변수 중요도 분석에서는 Length_of_Conveyer, Steel_Type, Steel_Plate_Thickness가 주요 변수로 나타났다.

---

## 2. 프로젝트 구조

Notebook은 데이터 파일과 같은 폴더에서 실행하도록 작성되어 있다(`DATA_PATH = "steel_plates_faults.csv"`). Streamlit 앱은 실행에 필요한 파일을 모두 `steel_streamlit_app/` 폴더에 모아 두었다.

```text
project/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
│
├── steel_plates_faults.csv              # 원본 데이터
├── steel_fault_baseline.ipynb           # EDA + Baseline 4종 비교
├── 최종 모델.ipynb                        # 전처리·튜닝·앙상블·평가·모델 저장
│
├── steel_streamlit_app/                 # Streamlit 대시보드
│   ├── app.py                           # 대시보드 화면
│   ├── steel_model.py                   # 전처리 함수·모델 클래스
│   ├── steel_fault_model_combined.joblib  # 최종 모델 (Notebook 실행 시 생성)
│   └── steel_plates_faults.csv
│
├── data/                                # 원본 데이터 및 변수 분석
├── notebooks/                           # EDA, 코드 리뷰 등 실험 노트북
├── model 0.8349/                        # 최종 모델 노트북 (팀 공유본)
├── reports/                             # 프로젝트 보고서
├── submission/                          # 일자별 과제 제출물
│
└── figures/
    ├── overview.png
    ├── target_distribution.png
    ├── eda_result.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── dashboard.png
```

**주요 디렉터리 및 파일 설명**

| 파일 / 디렉터리 | 설명 |
|---|---|
| `pyproject.toml` | 프로젝트 의존성 및 설정 |
| `uv.lock` | 의존성 버전 고정 |
| `steel_plates_faults.csv` | 분석에 사용한 원본 데이터 (1,941행 × 34열) |
| `steel_fault_baseline.ipynb` | 데이터 구조 점검, EDA, Baseline 4종 비교 |
| `최종 모델.ipynb` | 파생 변수 검증, 4개 모델 튜닝, 앙상블·확률 보정, 테스트 평가, 변수 중요도, 모델 저장 |
| `steel_fault_model_combined.joblib` | 전체 데이터로 재학습한 최종 모델과 설정값 |
| `steel_streamlit_app/app.py` | Streamlit 대시보드 화면 (CSV 일괄 예측, 단일 결함 예측, 모델 상세) |
| `steel_streamlit_app/steel_model.py` | Notebook과 같은 전처리 함수(`merge_columns`, `build_features`)와 모델 클래스(`ScaledSoftVoting`). 저장된 모델을 앱에서 불러올 때 필요하다. |
| `figures/` | 분석 결과 및 시각화 이미지 |

---

## 3. 설치 및 실행 방법

본 프로젝트는 **uv + Jupyter Notebook** 환경을 기반으로 수행되었습니다.

### 3.1 실행 환경

- Python 3.14.4
- uv
- Jupyter Notebook / VS Code
- Streamlit

### 3.2 Repository 다운로드

GitHub Repository의 HTTPS 주소를 사용합니다.

```bash
git clone https://github.com/happening12321/steel-defect-classification.git
cd steel-defect-classification
```

### 3.3 uv 설치 및 환경 구성

먼저 [uv 공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)를 참고하여 uv를 설치합니다.

`pyproject.toml`과 `uv.lock`에 정의된 프로젝트 의존성을 설치합니다.

```bash
uv sync
```

프로젝트 의존성에는 pandas, numpy, matplotlib, scipy, scikit-learn, lightgbm, catboost, joblib, streamlit, ipykernel이 포함되어 있어야 합니다.

### 3.4 데이터 준비

데이터가 Repository에 포함되지 않은 경우 아래 방법으로 준비합니다.

1. [UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198)에서 데이터 다운로드
2. 파일명을 `steel_plates_faults.csv`로 맞춰 Notebook과 같은 폴더에 저장
3. Notebook에서 데이터를 불러와 분석 진행

### 3.5 Jupyter Notebook 실행

**VS Code 사용 시**

1. VS Code에서 프로젝트 폴더를 엽니다.
2. 실행할 `.ipynb` 파일을 엽니다.
3. 우측 상단의 **Select Kernel**을 클릭합니다.
4. 프로젝트의 `.venv`에 있는 Python 환경을 선택합니다.
5. Notebook 셀을 순서대로 실행합니다.

> 그래프 한글 폰트가 `Malgun Gothic`(Windows)으로 설정되어 있습니다. macOS에서는 `AppleGothic`으로 바꿔 실행합니다.

### 3.6 분석 및 모델링 실행 순서

| 순서 | 파일 | 주요 내용 |
|---|---|---|
| 1 | `steel_fault_baseline.ipynb` | 데이터 검증, EDA, 60/20/20 분할, Baseline 4종 학습·평가 |
| 2 | `최종 모델.ipynb` | 80/20 분할, 파생 변수 검증, 튜닝, 앙상블·확률 보정, 테스트 평가, 모델 저장 |

### 3.7 Streamlit 실행

`최종 모델.ipynb`를 실행해 만든 `steel_fault_model_combined.joblib`과 원본 데이터 `steel_plates_faults.csv`를 `steel_streamlit_app/` 폴더에 넣은 뒤, 해당 폴더에서 다음 명령어를 실행합니다.

```bash
cd steel_streamlit_app
uv run streamlit run app.py
```

> 앱과 같은 폴더에 `app.py`, `steel_model.py`, `steel_fault_model_combined.joblib`이 있어야 합니다. 다른 폴더에서 실행하면 `File does not exist: app.py` 오류가 납니다. `steel_plates_faults.csv`는 예시 데이터와 단일 예측 시작값에 쓰이며, 없어도 앱은 실행됩니다.

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
| Statistical Analysis | SciPy (카이제곱 검정) |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn, LightGBM, CatBoost |
| Imbalanced Learning | 클래스 가중치(class_weight, auto_class_weights), 클래스별 예측 확률 보정 |
| Model Interpretation | Permutation Importance (scikit-learn) |
| Model Persistence | joblib |
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

- **주요 Feature**: 결함 영역의 위치(X·Y 좌표 최소·최대), 크기(Pixels_Areas, X·Y_Perimeter), 밝기(Sum·Minimum·Maximum_of_Luminosity), 형상 지표(Edges_Index, Empty_Index, Square_Index, Orientation_Index 등), 변환 변수(LogOfAreas, Log_X·Log_Y_Index, SigmoidOfAreas), 강판 정보(Steel_Type, Steel_Plate_Thickness, Length_of_Conveyer)

- **이상치 및 분포**: IQR 1.5배 기준 이상치 후보 비율이 Sum_of_Luminosity 20.6%, Pixels_Areas 20.4%, Outside_X_Index 19.1%, X_Perimeter 18.1%로 높게 나타났다. 해당 값들은 실제 결함 특성을 반영할 가능성이 있으므로 제거하지 않았다. 반면 LogOfAreas(0.3%)와 Log_Y_Index(0.2%)는 이상치 후보 비율이 상대적으로 낮았다.

- **변수 간 범위 차이**: Y 좌표는 최대 약 1,300만인 반면 일부 형상 지표는 0~1 범위로 변수 간 값의 크기 차이가 크다. 최종 모델은 변수의 스케일에 상대적으로 영향을 적게 받는 트리 기반 모델을 사용하므로 별도의 스케일링은 적용하지 않았다.

- **강종과 결함 유형의 연관성**: 카이제곱 검정 결과 강종과 결함 유형 간 유의한 연관성이 확인되었다(p < 0.001, Cramér's V = 0.586). 결함 유형별 강종 분포에 차이가 나타났다. K_Scatch(390/391)와 Stains(71/72)는 대부분 A400에서, Z_Scratch(172/190)는 주로 A300에서 발생한다.

**Target 클래스 분포**

![Target Distribution](figures/target_distribution.png)

가장 많은 Other_Faults(673건)와 가장 적은 Dirtiness(55건)의 관측 수가 약 12.2배 차이 나므로 모델링 시 다음 사항을 고려했다.

- **평가 지표**: 클래스 불균형으로 인한 Accuracy의 한계를 보완하기 위해 Macro F1을 주요 평가 지표로 선정했다.
- **데이터 분할**: 클래스 비율을 유지하는 층화 분할을 적용했다. (테스트 389건 중 Stains 14건, Dirtiness 11건)
- **불균형 처리**: Baseline(Gradient Boosting)에서 Pastry(Recall 0.594)와 Dirtiness(Recall 0.636)의 재현율이 상대적으로 낮게 나타나 이를 보완하기 위해 클래스 가중치 튜닝과 클래스별 예측 확률 보정을 시도했다.
- **평가 시 주의사항**: 소수 클래스는 테스트 표본이 적어 1~2건의 오분류에도 지표가 크게 변할 수 있으므로 최종 모델의 Macro F1에 대한 부트스트랩 95% 신뢰구간을 함께 확인했다.

---

## 6. 데이터 분석 및 전처리

### 6.1 탐색적 데이터 분석 (EDA)

**주요 분석 내용**

- 데이터 구조 및 변수 유형 확인: 34개 열이 모두 정수형·실수형이며, 결함 유형과 강종은 원-핫 형태로 저장되어 있다.
- 결측치 및 중복 데이터 확인: 결측치와 중복 행은 없고, 좌표 최솟값이 최댓값보다 큰 행처럼 논리적으로 맞지 않는 값도 없었다.
- 주요 변수의 분포 확인: IQR 기준 이상치 후보 비율과 로그 변환 변수의 분포를 비교했다.
- Target 클래스 분포 분석: 유형별 건수와 비율을 확인했다.
- 변수 간 관계 분석: 강종과 결함 유형의 교차표 및 카이제곱 검정, 정의상 연결된 변수 묶음을 확인했다.

**주요 발견 사항**

1. **강종만으로 나뉘는 유형과 그렇지 않은 유형이 있다.** K_Scatch·Stains·Z_Scratch는 한쪽 강종에 몰려 있어 강종이 강한 단서가 된다. 반면 Bumps(A300 약 69%)와 Other_Faults(A400 약 60%)는 두 강종에 고르게 걸쳐 있으므로 이 두 유형은 강종 외의 측정값으로 구분해야 한다.
2. **원본 변수와 변환 변수가 함께 들어 있다.** 면적은 Pixels_Areas·LogOfAreas·SigmoidOfAreas, 길이는 좌표 범위·Log_X/Y_Index, 밝기는 평균 밝기·Luminosity_Index로 같은 정보를 여러 형태로 담고 있다. 트리 모델은 값의 순서로 분기하므로 이런 변수에서 비슷한 파생 변수를 더 만드는 것은 의미가 적다고 판단했다.
3. **강종 두 열은 완전히 반대 값이다.** TypeOfSteel_A300과 TypeOfSteel_A400은 항상 합이 1이라 하나의 정보를 두 번 표현한다. 그래서 하나의 열(Steel_Type)로 합쳤다.

**분석 결과 시각화**

![EDA Result](figures/eda_result.png)

### 6.2 데이터 전처리

| 처리 항목 | 적용 방법 | 선택 이유 |
|---|---|---|
| 결측치 처리 | 별도 처리 없음 | 원본에 결측치가 없다. Baseline 파이프라인에는 새 데이터에 대비해 중앙값·최빈값 대체 단계를 넣어 두었다. |
| 이상치 처리 | 그대로 유지 | 면적·둘레가 큰 값은 측정 오류가 아니라 큰 결함의 특성이다. 트리 모델은 극단값의 영향을 적게 받는다. |
| 범주형 인코딩 | 강종 2열 → `Steel_Type`(A300 = 0, A400 = 1), 결함 7열 → `Fault_Type`(0~6) | 중복 정보를 하나로 합치고 7열로 나뉜 정답을 하나의 다중 분류 타깃으로 만들었다. |
| 스케일링 | 최종 모델에는 적용하지 않음 (Baseline 로지스틱 회귀에만 StandardScaler) | 최종 후보가 모두 트리 계열이라 변수 단위 차이의 영향을 받지 않는다. |
| Feature Engineering | 밝기 범위, 가로세로 비, 둘레/면적 비 3개를 검증했으나 사용하지 않음 | 5-Fold × 3회 반복 교차검증에서 Macro F1이 0.7967 → 0.7961로 오히려 조금 낮아졌다. |

**전처리 과정에서 고려한 사항**

- **데이터 분할**: 두 Notebook 모두 `random_state=42` 층화 분할로 테스트 20%(389건)를 가장 먼저 떼어 냈다. 그래서 Baseline과 최종 모델을 같은 테스트 데이터로 비교할 수 있다. 나머지 80%는 Baseline에서는 Train 60% / Validation 20%로 나눴고, 최종 모델에서는 나누지 않고 5-Fold 교차검증에 사용했다.
- **데이터 누수 방지**: 정답 7열은 입력에서 뺐다. 파생 변수 채택, 하이퍼파라미터, 앙상블 구성, 확률 보정 배율은 모두 개발 데이터 안에서 정했고 테스트 데이터는 마지막 평가에만 사용했다. 학습과 예측에 같은 전처리 함수(`build_features`)를 써서 두 단계의 변환이 달라지지 않도록 했다.
- **전처리를 최소화한 이유**: 데이터에 결측치가 없고 이미 변환 변수가 포함되어 있어 전처리를 늘리기보다 모델과 불균형 처리 쪽에서 개선을 시도하는 것이 낫다고 판단했다.

---

## 7. 머신러닝 모델링

### 7.1 Baseline 모델

**사용 모델:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (기본 설정, 불균형 처리 없음)

**선택 이유**

- 가장 단순한 선형 모델부터 대표적인 트리 앙상블까지 기본 설정으로 학습해서 별도의 튜닝 없이 어느 정도 성능이 나오는지 기준점을 잡았다.
- Validation Macro F1이 가장 높은 Gradient Boosting(0.7865)을 Baseline 대표 모델로 정했다. 테스트 Macro F1은 0.7941이다.

| 모델 | Accuracy | Macro Precision | Macro Recall | Macro F1 (Test) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7198 | 0.7387 | 0.7171 | 0.7258 |
| Decision Tree | 0.7147 | 0.7067 | 0.7041 | 0.7036 |
| Random Forest | 0.7763 | 0.8315 | 0.7696 | 0.7967 |
| Gradient Boosting | 0.7738 | 0.8437 | 0.7598 | 0.7941 |

**기본 모델 성능 및 개선이 필요한 부분**

- 트리 계열 모델이 선형 모델보다 Macro F1이 약 0.07 높아서 이후 실험은 트리 계열 위주로 진행했다.
- 트리 계열의 학습 점수는 0.99 이상이지만 Validation은 약 0.78이라 과적합이 크다.
- Macro Recall이 Macro Precision보다 낮다. 즉, 소수 유형을 다른 유형으로 놓치는 경우가 많았다. 특히 Pastry의 Recall이 0.594로 가장 낮았다.

### 7.2 모델 비교 및 성능 개선

**적용한 모델**

- Random Forest
- Extra Trees
- LightGBM
- CatBoost

Baseline 결과를 바탕으로 트리 계열 모델 4종을 후보로 정했다. SVM·KNN 같은 거리 기반 모델도 사전에 확인했지만 Macro F1이 트리 계열보다 크게 낮아 제외했다.

**주요 실험**

1. **파생 변수 검증**: 반복 교차검증으로 파생 변수 3개의 효과를 확인했고 효과가 없어 원본 26개 변수만 사용했다.
2. **하이퍼파라미터 튜닝**: 모델마다 `RandomizedSearchCV`로 20개 조합을 5-Fold 교차검증해 Macro F1이 가장 높은 설정을 골랐다. 클래스 가중치도 탐색 대상에 넣었다.
3. **앙상블**: 튜닝된 모델들의 예측 확률을 평균하는 소프트 보팅을 상위 2개 / 3개 / 4개 조합으로 비교했다. 비교에는 개발 데이터의 OOF(Out-of-Fold) 확률만 사용했다.
4. **확률 보정**: 앙상블의 예측 확률에 클래스별 배율을 곱해 소수 유형의 예측 기준을 조정했다.
5. **일반화 점검**: 중첩 교차검증과 10회 반복 분할로 성능이 특정 분할에서만 높게 나온 것인지 확인했다.

### 7.3 불균형 데이터 처리

**적용한 방법**

- **클래스 가중치**: 모델마다 클래스 가중치 사용 여부와 방식을 튜닝으로 정했다. Random Forest·Extra Trees는 `balanced`, CatBoost는 `SqrtBalanced`가 선택되었고, LightGBM은 가중치를 쓰지 않을 때 성능이 가장 좋았다.
- **클래스별 예측 확률 보정**: 각 유형의 예측 확률에 0.5~3.0 사이 배율을 곱한 뒤 가장 높은 확률의 유형을 고르는 방식이다. 배율이 1보다 크면 그 유형을 더 자주 예측한다. OOF 확률에서 Macro F1이 가장 높아지는 배율을 찾은 결과 Pastry 2.2, Dirtiness 1.9로 소수 유형은 높이고 Z_Scratch·Stains는 0.5로 낮췄다.

**선택 이유**

SMOTE 같은 오버샘플링은 가짜 샘플을 만들고 언더샘플링은 데이터를 버린다. 데이터가 1,941건으로 적어 원본을 바꾸지 않는 방법을 우선했다. 클래스 가중치는 학습 단계에서 소수 유형의 오류를 더 크게 반영하고 확률 보정은 예측 단계에서 판단 기준을 옮긴다. 두 방법을 함께 쓰면 소수 유형의 Recall을 높일 수 있을 것으로 기대했다. 확률 보정은 과하게 맞춰질 위험이 있어 OOF 기준 개선 폭이 0.005 이상일 때만 쓰도록 미리 정했다(실제 개선 폭 0.0122).

**실험 시 고려 사항**

- 클래스 가중치와 확률 보정 배율은 개발 데이터 안에서만 결정
- 처리 전후 성능은 같은 테스트 데이터 389건으로 비교
- 전체 Macro F1뿐 아니라 소수 유형의 Precision, Recall, F1 변화를 함께 확인

### 7.4 모델 성능 비교

같은 테스트 데이터 389건 기준이다. 튜닝 모델과 앙상블의 Precision·Recall은 Notebook에서 따로 계산하지 않아 F1만 적었다.

| 모델 / 실험 | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
|---|---:|---:|---:|
| Baseline (Gradient Boosting) | 0.8437 | 0.7598 | 0.7941 |
| Random Forest + 클래스 가중치 | | | 0.7850 |
| Extra Trees + 클래스 가중치 | | | 0.7988 |
| CatBoost + 클래스 가중치 | | | 0.8078 |
| LightGBM (튜닝) | | | 0.8305 |
| LightGBM + Extra Trees 앙상블 | | | 0.8264 |
| **최종 모델** (앙상블 + 확률 보정) | **0.8283** | **0.8527** | **0.8349** |

다중 분류 문제라 PR-AUC 대신 Macro 지표로 비교했다. 최종 모델 외의 테스트 점수는 참고용이며 모델 선택은 개발 데이터의 OOF 점수로 했다.

**성능 비교 결과**

![Model Comparison](figures/model_comparison.png)

**주요 분석 결과**

- **성능이 가장 우수했던 모델**: LightGBM + Extra Trees 앙상블에 확률 보정을 적용한 최종 모델이다. 단일 모델로는 LightGBM이 가장 높았다.
- **불균형 데이터 처리 전후 성능 변화**: Baseline과 비교하면 Macro Recall이 0.760에서 0.853으로 올라 놓치는 결함이 줄었다. 그 대신 Macro Precision은 0.844에서 0.828로 조금 낮아졌다. 다만 확률 보정만 따로 떼어 보면 테스트에서는 +0.0085였지만 중첩 교차검증에서는 평균 −0.0028로 효과가 일정하지 않았다.
- **모델별 성능 차이의 원인**: 이 데이터는 강종·두께·컨베이어 길이처럼 특정 구간 값이 결함 유형과 연결된 변수가 많아서 구간을 나눠 판단하는 트리 모델이 선형 모델보다 유리했던 것으로 보인다. 튜닝한 모델들과 최종 모델의 차이는 0.05 이내로 테스트 데이터의 95% 신뢰구간 폭(약 0.09)보다 작아 우열을 확정하기는 어렵다.

---

## 8. 최종 모델 및 결과 분석

### 8.1 최종 모델 선정

- **최종 모델**: LightGBM과 Extra Trees의 예측 확률을 평균하는 소프트 보팅 앙상블
- **적용한 불균형 처리 방법**: Extra Trees의 클래스 가중치(`balanced`) + 클래스별 예측 확률 보정(Pastry 2.2, Dirtiness 1.9, Z_Scratch 0.5, Stains 0.5, 나머지 1.0)
- **주요 하이퍼파라미터**

| 모델 | 설정 |
|---|---|
| LightGBM | n_estimators 400, learning_rate 0.03, num_leaves 31, max_depth 5, min_child_samples 10, subsample 0.8 |
| Extra Trees | n_estimators 300, max_depth 20, min_samples_split 5, max_features 0.5, class_weight balanced |

- **선정 이유**: 단일 모델 4개와 앙상블 3개 중 개발 데이터 OOF Macro F1이 0.8358로 가장 높았다. CatBoost까지 넣은 3~4개 조합보다 점수가 높으면서 구조도 단순하다.

### 8.2 최종 모델 평가

| Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro F1 95% 신뢰구간 |
|---:|---:|---:|---:|---:|---|
| 0.8201 | 0.8527 | 0.8283 | 0.8527 | 0.8349 | 0.786 ~ 0.876 |

| 유형 | Precision | Recall | F1 | 테스트 건수 |
|---|---:|---:|---:|---:|
| K_Scatch | 0.9744 | 0.9744 | 0.9744 | 78 |
| Z_Scratch | 1.0000 | 0.8947 | 0.9444 | 38 |
| Stains | 0.9286 | 0.9286 | 0.9286 | 14 |
| Dirtiness | 0.7500 | 0.8182 | 0.7826 | 11 |
| Other_Faults | 0.7969 | 0.7556 | 0.7757 | 135 |
| Bumps | 0.7568 | 0.6914 | 0.7226 | 81 |
| Pastry | 0.5918 | 0.9062 | 0.7160 | 32 |

**Confusion Matrix**

![Confusion Matrix](figures/confusion_matrix.png)

**평가 결과 해석**

- **False Positive 및 False Negative 분석**: 389건 중 70건이 틀렸고 그중 33건이 Bumps와 Other_Faults를 서로 바꿔 예측한 경우다. Pastry는 확률 보정으로 Recall이 0.906까지 올랐지만 그만큼 다른 유형(주로 Other_Faults·Bumps)을 Pastry로 잘못 예측한 경우가 늘어 Precision이 0.592로 낮다. 소수 유형을 놓치지 않는 대신 오탐을 일부 감수한 결과다.
- **실제 문제에서 중요하게 고려해야 할 오류**: 결함 유형을 잘못 판정하면 원인과 관계없는 공정을 점검하게 된다. 따라서 Bumps로 예측된 결함이나 확신도가 낮은 예측은 바로 확정하지 않고 검사원이 다시 확인하는 방식으로 쓰는 것이 적절하다.
- **최종 모델의 강점과 한계**: K_Scatch·Z_Scratch·Stains는 F1이 0.93 이상이라 자동 분류에 활용할 수 있는 수준이다. 반면 Bumps·Other_Faults·Pastry 세 유형은 F1이 0.72~0.78에 머물렀다. 또한 Dirtiness(11건)·Stains(14건)는 테스트 건수가 적어 한두 건으로도 점수가 크게 달라진다.

### 8.3 Feature Importance 및 모델 해석

변수 하나의 값을 무작위로 섞었을 때 테스트 Macro F1이 얼마나 떨어지는지(Permutation Importance, 20회 반복)로 중요도를 측정했다.

![Feature Importance](figures/feature_importance.png)

| 순위 | 변수 | Macro F1 감소량 |
|---:|---|---:|
| 1 | Length_of_Conveyer | 0.095 |
| 2 | Steel_Type | 0.082 |
| 3 | Steel_Plate_Thickness | 0.073 |
| 4 | Y_Minimum | 0.037 |
| 5 | Orientation_Index | 0.017 |

**주요 분석 결과**

- **예측에 중요한 영향을 미친 Feature**: 컨베이어 길이, 강종, 강판 두께 세 변수가 다른 변수보다 2배 이상 중요했다. 결함의 크기·밝기·형상 지표는 그 아래였다.
- **Feature 영향 해석**: 모델은 결함 모양보다 "어떤 강판, 어떤 라인에서 나온 결함인가"를 더 크게 참고한다. 이는 데이터 안에서 나타난 연관성이지, 해당 조건이 결함의 원인이라는 뜻은 아니다. 또 서로 비슷한 정보를 가진 변수는 한쪽을 섞어도 다른 쪽이 대신하므로, 중요도가 낮은 변수가 꼭 필요 없는 변수라고 볼 수는 없다.
- **분석 결과를 통해 도출한 인사이트**: 생산 조건에 크게 의존한다는 것은 새로운 라인이나 두께 범위에서는 성능이 떨어질 수 있다는 뜻이다. 또한 혼동이 많은 Bumps·Other_Faults·Pastry를 구분하려면 표면 높이나 결함 이미지처럼 현재 데이터에 없는 정보가 필요할 것으로 보인다.

SHAP은 사용하지 않았다.

---

## 9. Streamlit 대시보드

최종 모델로 새로운 결함 데이터의 유형을 예측하고, 모델 정보와 성능을 확인할 수 있는 Streamlit 대시보드다. `최종 모델.ipynb`에서 저장한 `steel_fault_model_combined.joblib`을 불러와 사용한다.

### 9.1 주요 기능

| 화면 | 기능 |
|---|---|
| 사이드바 | 최종 모델 구성(LightGBM + Extra Trees), 확률 보정 사용 여부, 테스트 Macro F1(0.8349)·Accuracy(0.8201) 표시 |
| CSV 일괄 예측 | 원본과 같은 열 구조의 CSV를 업로드하면 결함별 예측 유형, 확신도, 7개 유형별 예측 확률을 표로 보여준다. 예측 결함 분포 그래프와 결과 CSV 다운로드를 제공한다. |
| 재검토 대상 표시 | 확신도가 기준값(기본 0.5, 슬라이더로 조정)보다 낮은 예측을 "재검토"로 표시해 검사원이 다시 확인할 결함을 구분한다. |
| 정답 비교 | 업로드한 CSV에 결함 유형 7열이 있으면 Macro F1, Accuracy, 혼동행렬, 유형별 Precision·Recall·F1을 계산한다. |
| 단일 결함 예측 | 강종과 25개 측정값을 직접 입력하거나 예시 데이터의 행 번호를 불러와 결함 1건의 유형과 유형별 확률을 확인한다. 기본 입력값은 원본 데이터의 중앙값이다. |
| 모델 상세 | 테스트 성능 지표, 클래스별 확률 보정 배율, 두 모델의 하이퍼파라미터, 입력 변수 목록을 보여준다. |

> 앱의 모델은 테스트 평가 후 전체 데이터(1,941건)로 다시 학습한 버전이다. 따라서 원본 데이터로 정답 비교를 하면 학습에 사용한 데이터를 다시 맞히는 셈이라 점수가 실제보다 높게 나온다(앱에도 경고 문구를 표시한다). 실제 성능은 테스트 Macro F1 0.8349를 기준으로 본다.

### 9.2 실행 화면

![Streamlit Dashboard](figures/dashboard.png)

---

## 10. 결론 및 회고

### 10.1 프로젝트 주요 결과

- **프로젝트 목표 달성 여부**: 불균형 데이터에서 Macro F1을 기준으로 Baseline과 불균형 처리 모델을 같은 테스트 데이터로 비교했고, 최종 모델이 모든 Baseline보다 높은 성능을 보였다.
- **최종 모델의 성능 및 주요 개선 사항**: 테스트 Macro F1은 Baseline 0.794에서 최종 0.835로 올랐다. 테스트 분할을 바꿔 10번 다시 평가해도 평균 0.829(최저 0.809)로 특정 분할에서만 잘 나온 결과는 아니었다.
- **불균형 데이터 처리 과정에서 확인한 결과**: 클래스 가중치와 확률 보정으로 소수 유형을 놓치는 경우가 줄었다(Pastry Recall 0.594 → 0.906). 대신 Pastry로 잘못 예측하는 경우가 늘었고 확률 보정의 효과는 평가 방법에 따라 달라 신중하게 해석해야 한다.
- **분석을 통해 도출한 주요 인사이트**: 모델은 생산 조건 변수에 크게 의존하며 오분류는 대부분 Bumps·Other_Faults·Pastry 세 유형 사이에서 발생한다.

### 10.2 한계점 및 개선 방향

**데이터 측면의 한계**

- Other_Faults는 정해진 유형에 속하지 않는 결함을 모은 유형이라 특징이 일정하지 않고 이 때문에 Bumps·Pastry와 경계가 흐리다.
- 전체 데이터가 2,000건이 안 되고 일부 유형은 테스트 건수가 10여 건이라 점수가 표본 구성에 따라 크게 흔들린다.
- 결함 이미지, 생산 시점, 공정 기록 같은 정보가 없어 혼동이 많은 유형을 더 구분할 단서가 부족하다.

**모델 성능 및 일반화 측면의 한계**

- 학습 데이터에서는 거의 완벽하게 맞히지만 교차검증 점수와는 차이가 커, 과적합이 남아 있다.
- 생산 조건 변수에 의존하므로 다른 공장이나 새로운 제품에 그대로 적용하기는 어렵다.
- 확률 보정 배율은 개발 데이터에 맞춰 찾은 값이라 새로운 데이터에서도 같은 효과가 난다고 보장하기 어렵다.

**추가로 시도하고 싶은 실험이나 개선 방법**

- 결함 이미지나 공정 정보를 결합해 Bumps·Other_Faults·Pastry 구분력 높이기
- 트리 깊이·최소 잎 크기를 더 제한해 과적합 줄이기
- 하이퍼파라미터 탐색까지 중첩 교차검증 안에 넣어 성능 추정의 낙관적 편향 줄이기
- 튜닝, 클래스 가중치, 앙상블, 확률 보정을 하나씩 추가하며 각각의 효과를 따로 측정하기

### 10.3 프로젝트를 통해 배운 점

- 다양한 머신러닝 모델을 비교하고 성능 개선을 시도하면서 모델을 복잡하게 만든다고 성능이 무조건 오르지는 않는다는 것을 배웠다.
- 교차검증을 통해 데이터 분할에 따라 성능이 달라질 수 있다는 점을 알게 되었다. 그래서 테스트 점수 하나보다 여러 분할의 평균과 신뢰구간을 함께 보는 것이 중요하다.

---

## 11. 팀원 및 역할

| 이름 | 담당 내용 | GitHub |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
