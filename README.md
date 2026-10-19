# 강판 표면 결함 유형 분류

강판 표면에서 검출된 결함의 위치·크기·밝기·형상 측정값과 강판 속성으로, 해당 결함을 7가지 유형 중 하나로 자동 분류하는 머신러닝 프로젝트입니다.

공개 데이터인 Steel Plates Faults(1,941건)를 사용해 검사원의 결함 유형 판정을 보조하는 모델을 개발했습니다. 유형 간 표본 수 차이가 최대 12.2배인 불균형 데이터이므로 Macro F1을 주 지표로 삼았고, LightGBM + Extra Trees 소프트 보팅 앙상블에 클래스별 확률 배율을 적용한 최종 모델이 테스트 Macro F1 **0.835**(Baseline 0.794)를 기록했습니다.

![프로젝트 대표 이미지](figures/overview.png)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 소개

**프로젝트 배경**

- 강판은 자동차 차체, 가전 외장재, 선박, 건축 자재 등에 쓰이는 기초 소재로, 압연·이송 과정에서 롤·설비 접촉, 냉각수·윤활유·이물질 노출 등으로 다양한 표면 결함이 생깁니다.
- 결함 **검출**은 자동 표면 검사 시스템으로 자동화되고 있지만, 검출된 결함의 **유형 분류**에는 별도 판단이 필요합니다. 결함 유형에 따라 점검할 공정이 달라지므로, 유형을 잘못 판정하면 엉뚱한 공정을 점검하게 되고 실제 원인은 방치될 수 있습니다.
- 발생 빈도가 낮은 유형은 검사원이 접할 기회가 적어 숙련도에 따라 판정 일관성이 달라질 수 있습니다.
- **머신러닝으로 접근한 이유:** 결함마다 27개의 수치형 측정값이 있고 일부 유형은 값의 범위가 겹쳐 사람이 규칙으로 정리하기 어렵습니다. 측정값으로 유형을 자동 분류하면 판정 기준을 일관되게 유지하고, 어떤 변수가 유형 구분에 유용한지·어떤 유형이 기존 변수로 구분하기 어려운지도 함께 확인할 수 있습니다.

**프로젝트 목표**

- **예측 대상(Target):** 검출된 결함 1건의 유형 7가지(Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults) 중 하나. 데이터에 정상 강판이 없으므로 정상/불량 판별 모델이 아닙니다.
- **평가 지표:** 유형별 표본 수가 고르지 않으므로 각 유형의 F1을 같은 비중으로 평균한 **Macro F1**을 주 지표로 사용했습니다. 보조 지표로 Accuracy, Balanced Accuracy, 유형별 Precision·Recall·F1, 혼동행렬을 사용했습니다.
- **성공 기준:** ① Macro F1 0.70 이상 및 다수 클래스 기준 모델(전부 Other_Faults로 예측 시 Macro F1 0.074) 대비 우위, ② 7개 유형 모두 Recall 0.60 이상, ③ 변수 중요도로 현재 입력 변수의 충분성 판단
- **불균형 처리 전후 비교:** 불균형 처리 없는 Baseline과 클래스 가중치·클래스별 확률 배율을 적용한 모델을 같은 테스트셋(389건)에서 비교했습니다.

### 1.2 프로젝트 정보

| 항목 | 내용 |
|---|---|
| 프로젝트 기간 | 2026.10.13 ~ 2026.10.19 |
| 프로젝트 유형 | 팀 프로젝트 (DS 1기 4조) |
| 데이터 출처 | [UCI Machine Learning Repository – Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) |
| 주요 기술 | Python, pandas, scikit-learn, LightGBM, CatBoost |

### 1.3 주요 성과

- **모델 성능:** 테스트 Macro F1 0.794 (Baseline, Gradient Boosting) → **0.835** (최종 앙상블). 부트스트랩(1,000회) 95% 신뢰구간 0.786~0.876
- **불균형 처리:** Baseline 대비 소수 유형 Pastry의 Recall 0.594 → 0.906, F1 0.585 → 0.716으로 개선. 클래스별 확률 배율 단독 효과는 학습 데이터 OOF +0.0122(0.8236 → 0.8358), 테스트 +0.0085(0.8264 → 0.8349)입니다.
- **주요 인사이트:** 순열 중요도 상위 3개가 컨베이어 길이·강종·강판 두께로, 모델이 결함 형태보다 생산 조건 정보에 크게 의존합니다. 오분류의 78.6%가 Bumps·Other_Faults·Pastry 세 유형 사이에서 발생해, 이 유형들은 현재 측정값만으로 구분이 어렵습니다.

---

## 2. 프로젝트 구조

현재 Notebook은 데이터 파일과 같은 폴더에서 실행하도록 작성되어 있습니다(`DATA_PATH = "steel_plates_faults.csv"`).

```text
project/
├── README.md
├── steel_plates_faults.csv              # 원본 데이터
├── steel_fault_baseline.ipynb           # EDA + Baseline 4종 비교
├── 최종 모델.ipynb                        # 전처리·튜닝·앙상블·평가·저장
├── app.py                               # Streamlit 대시보드
│
├── split_indices.joblib                 # (Baseline 실행 시 생성) 분할 인덱스
├── baseline_scores.csv                  # (Baseline 실행 시 생성)
├── baseline_per_class_test.csv          # (Baseline 실행 시 생성)
├── steel_fault_model_combined.joblib    # (최종 모델 실행 시 생성) 최종 모델
│
└── figures/                             # README에 사용한 결과 그래프 이미지
```

**주요 디렉터리 및 파일 설명**

| 파일 / 디렉터리 | 설명 |
|---|---|
| `steel_plates_faults.csv` | 분석에 사용한 원본 데이터 (1,941행 × 34열) |
| `steel_fault_baseline.ipynb` | 데이터 구조 점검, EDA, Baseline 4종 비교 |
| `최종 모델.ipynb` | 파생 변수 검증, 4개 모델 튜닝, 앙상블·확률 보정, 테스트 평가, 변수 중요도, 모델 저장 |
| `steel_fault_model_combined.joblib` | 전체 데이터로 재학습한 최종 모델과 설정값 |
| `figures/` | README에 삽입한 그래프 이미지 (`target_distribution.png`, `confusion_matrix.png`, `feature_importance.png`, `model_comparison.png`) |
| `pyproject.toml`, `uv.lock` | 프로젝트 의존성 정의와 버전 고정 (uv) |
| `app.py` | Streamlit 대시보드 (샘플 예측, CSV 업로드 예측, 모델 정보) |

---

## 3. 설치 및 실행 방법

### 3.1 실행 환경

- Python 3.13.14
- Jupyter Notebook / VS Code
- Windows 11
- uv: 패키지·가상환경 관리 (`pyproject.toml` 기준 Python 3.13 이상)
- Streamlit: 대시보드 실행에 사용

### 3.2 Repository 다운로드

```bash
git clone <repository-url>
cd <repository-name>
```

> **[작성 필요]** GitHub Repository 주소

### 3.3 uv 설치 및 환경 구성

[uv 공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)를 참고해 uv를 설치한 뒤 의존성을 설치합니다.

```bash
uv sync
```

`pyproject.toml`에 pandas, numpy, matplotlib, scikit-learn, lightgbm, catboost, streamlit, ipykernel 등이 정의되어 있고, scipy와 joblib은 scikit-learn과 함께 설치됩니다.

### 3.4 데이터 준비

1. [UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)에서 데이터 다운로드
2. 파일명을 `steel_plates_faults.csv`로 맞춰 Notebook과 같은 폴더에 저장
   (`data/raw/`에 두려면 각 Notebook의 `DATA_PATH`를 수정)
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
| 1 | `steel_fault_baseline.ipynb` | 데이터 검증, EDA, 60/20/20 분할, Baseline 4종 학습·평가 |
| 2 | `최종 모델.ipynb` | 80/20 분할, 파생 변수 검증, 튜닝, 앙상블·확률 보정, 테스트 평가, 모델 저장 |
| 3 | `app.py` | 저장된 모델로 결함 유형 예측 (Streamlit 대시보드) |

### 3.7 Streamlit 실행

1. `최종 모델.ipynb`를 끝까지 실행해 `steel_fault_model_combined.joblib`을 생성합니다.
2. `app.py`, `steel_fault_model_combined.joblib`, `steel_plates_faults.csv`, `figures/`를 같은 폴더에 둡니다.
3. 터미널에서 실행합니다.

```bash
uv run streamlit run app.py
```

uv를 쓰지 않으면 `streamlit run app.py`로 실행합니다. 브라우저에서 `http://localhost:8501`이 열립니다.


---

## 4. 기술 스택

| 구분 | 사용 기술 |
|---|---|
| Language | Python 3.13.14 |
| Environment & Package Management | uv (`pyproject.toml`, `uv.lock`) |
| Data Processing | pandas 3.0.6, NumPy 2.5.4 |
| Visualization | Matplotlib 3.11.2 |
| Machine Learning | scikit-learn 1.9.1, LightGBM 4.7.0, CatBoost 1.2.10 |
| Statistics | SciPy 1.18.1 (카이제곱 검정) |
| Imbalanced Learning | 클래스 가중치(class_weight, auto_class_weights), 클래스별 확률 배율 조정 (imbalanced-learn은 환경에 설치되어 있으나 Baseline·최종 모델에서는 사용하지 않음) |
| Model Interpretation | Permutation Importance (SHAP 미사용) |
| Model Persistence | joblib 1.6.0 |
| Dashboard | Streamlit 1.66.0 |
| Development | Jupyter Notebook, VS Code |

---

## 5. 데이터셋

### 5.1 데이터 개요

**데이터 출처:** [UCI Machine Learning Repository – Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) (Semeion 연구센터 제공, CC BY 4.0)

| 항목 | 내용 |
|---|---|
| 데이터 수 | 1,941건 (결함 1건 = 1행) |
| Feature 수 | 원본 27개 → 강종 2열을 1열로 병합해 모델 입력 26개 |
| Target 변수 | `Fault_Type` (원본의 결함 유형 원-핫 7열을 1열로 병합) |
| 문제 유형 | 다중 클래스 분류 (7개 클래스) |
| 클래스 비율 | Other_Faults 34.7%, Bumps 20.7%, K_Scatch 20.1%, Z_Scratch 9.8%, Pastry 8.1%, Stains 3.7%, Dirtiness 2.8% |

### 5.2 데이터 특징

- **주요 Feature:** 결함의 위치·크기·둘레(7개), 밝기(3개), 강판 속성(강종, 두께, 컨베이어 길이), 형태 지표(8개), 로그·시그모이드 변환 변수(4개), Outside_Global_Index
- 결함 이미지가 아닌 **이미지에서 추출한 수치형 특성**이며, 수집 시점·강판 ID·생산 라인 정보는 없습니다.
- 결측치 0건, 중복 행 0건. 정답 열 검증 통과(강종 2열·결함 7열 모두 0/1이며 행 합계 1)
- `K_Scatch`는 원본 열 이름의 철자를 그대로 사용했습니다.

**Target 클래스 분포**

| 유형 | 건수 | 비율 |
|---|---:|---:|
| Other_Faults | 673 | 34.7% |
| Bumps | 402 | 20.7% |
| K_Scatch | 391 | 20.1% |
| Z_Scratch | 190 | 9.8% |
| Pastry | 158 | 8.1% |
| Stains | 72 | 3.7% |
| Dirtiness | 55 | 2.8% |

![Target Distribution](figures/target_distribution.png)

최다 유형 Other_Faults(673건)가 최소 유형 Dirtiness(55건)의 12.2배입니다. 전체 정확도만 보면 다수 유형 성능에 가려 소수 유형 성능을 파악하기 어렵기 때문에 Macro F1로 평가했고, 층화 분할로 모든 분할에서 유형 비율을 유지했습니다. 테스트셋의 Stains(14건)·Dirtiness(11건)는 1~2건 차이로도 지표가 크게 흔들립니다.

---

## 6. 데이터 분석 및 전처리

### 6.1 탐색적 데이터 분석 (EDA)

**주요 분석 내용**

- 데이터 구조 및 변수 유형 확인 (모두 int64/float64)
- 결측치 및 중복 데이터 확인 (모두 0건)
- 주요 변수의 분포 확인 (IQR 기준 이상치 후보 비율)
- Target 클래스 분포 분석
- 변수 간 관계 분석 (강종 × 결함 유형 카이제곱 검정)

**주요 발견 사항**

1. **강종과 결함 유형의 연관성이 강합니다** (χ² = 667.0, 자유도 6, p < 0.001, Cramér's V = 0.586). K_Scatch(390/391)·Stains(71/72)는 거의 A400, Z_Scratch(172/190)는 대부분 A300입니다. 반면 Bumps(A300 69%)와 Other_Faults(A400 60%)는 강종이 섞여 있어 강종만으로는 구분되지 않습니다.
2. **면적·둘레·밝기 합 변수는 오른쪽 꼬리가 깁니다.** Pixels_Areas의 IQR 이상치 비율은 20.35%지만 로그 변환된 LogOfAreas에서는 0.31%로 줄어, 데이터에 이미 포함된 변환 변수가 치우침을 완화합니다.
3. **정의상 강하게 연결된 변수 묶음이 있습니다.** 면적(Pixels_Areas–LogOfAreas–SigmoidOfAreas), 길이(좌표 범위–Log_X/Y_Index), 밝기(평균 밝기–Luminosity_Index)는 단조 관계이고, 강종 2열은 완전히 반대 관계입니다. 그래서 강종은 1열로 합치고 단조 관계 파생 변수는 만들지 않았습니다.

**분석 결과 시각화**

![EDA Result](figures/eda_result.png)

※ 유형별 강종 구성 비율과 건수 (χ² = 667.0, Cramér's V = 0.586)

### 6.2 데이터 전처리

| 처리 항목 | 적용 방법 | 선택 이유 |
|---|---|---|
| 결측치 처리 | 처리 없음 (Baseline 파이프라인에는 중앙값·최빈값 대체 포함) | 결측 0건. Baseline은 새 데이터에 결측이 생겨도 동작하도록 대체 단계를 넣음 |
| 이상치 처리 | 제거·대체 없이 유지 | 큰 면적·긴 둘레는 실제 결함 특성이며 원자료 오류가 없음. 트리 모델은 값의 순서로 분기해 극단값 영향이 작음 |
| 범주형 인코딩 | 강종 2열 → `Steel_Type`(A300=0, A400=1), 타깃 7열 → `Fault_Type`(정수 0~6) | 두 강종 열이 하나의 이진 정보를 중복 표현. 매핑 밖 값이 들어오면 오류 발생 |
| 스케일링 | 최종 모델 미적용 (Baseline 로지스틱 회귀만 StandardScaler) | 최종 후보가 모두 트리 계열이라 단위·범위 차이 영향 없음 |
| Feature Engineering | 파생 변수 3개(밝기 범위, 가로세로 비, 둘레/면적 비) 검증 후 **미채택** | 반복 교차검증 Macro F1 0.7967(원본) vs 0.7961(파생 추가)로 향상 없음 |

**전처리 과정에서 고려한 사항**

- **데이터 분할:** Baseline은 Train/Val/Test 60/20/20, 최종 모델은 학습/테스트 80/20 층화 분할입니다. 최종 모델은 별도 검증셋 대신 학습 데이터의 층화 5-겹 교차검증을 사용해 학습 데이터를 늘렸습니다. 두 코드는 같은 방식(`random_state=42`)으로 테스트 20%를 먼저 분리하므로 **테스트 389건이 동일**합니다.
- **데이터 누수 방지:** 타깃 7열은 입력에서 제거했고, 테스트 데이터를 가장 먼저 분리했습니다. 파생 변수 채택, 튜닝, 앙상블 구성, 확률 배율 탐색은 모두 학습 데이터 안에서만 결정했고, 테스트는 최종 모델에 1회만 적용했습니다. 학습·예측이 같은 `build_features` 함수를 사용해 변환 불일치를 막았습니다.
- **생산 조건 변수 의존 위험:** 시점 누수는 아니지만, 중요도 상위 변수가 컨베이어 길이·강종·두께라 새로운 라인·제품에서는 성능이 떨어질 수 있습니다.

---

## 7. 머신러닝 모델링

### 7.1 Baseline 모델

**사용 모델:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (모두 기본 설정, 불균형 처리 없음)

**선택 이유**

- 선형 모델 1종과 트리 계열 3종을 기본 설정으로 비교해, 필수 전처리만 했을 때의 성능 수준과 어떤 모델 계열이 이 데이터에 맞는지 확인했습니다.
- Validation Macro F1 기준 최고 모델은 **Gradient Boosting**(Val 0.7865, Test 0.7941)입니다.

| 모델 | Accuracy | Macro Precision | Macro Recall | Macro F1 (Test) | 95% CI |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.7198 | 0.7387 | 0.7171 | 0.7258 | 0.666 ~ 0.774 |
| Decision Tree | 0.7147 | 0.7067 | 0.7041 | 0.7036 | 0.642 ~ 0.759 |
| Random Forest | 0.7763 | 0.8315 | 0.7696 | 0.7967 | 0.742 ~ 0.844 |
| Gradient Boosting | 0.7738 | 0.8437 | 0.7598 | 0.7941 | 0.740 ~ 0.839 |

**개선이 필요한 부분**

- 트리 계열은 Train Macro F1이 0.99~1.00인 반면 Val은 Decision Tree 0.708, Random Forest 0.784, Gradient Boosting 0.787로 과적합 경향이 있습니다. Train과 Val의 차이는 Decision Tree 0.292, Random Forest 0.216, Gradient Boosting 0.202입니다.
- Train+Val(80%)의 5-겹 × 3회 반복 교차검증 Macro F1은 Gradient Boosting 0.7990 ± 0.0208, Random Forest 0.7859 ± 0.0254, Decision Tree 0.7213 ± 0.0294, Logistic Regression 0.7161 ± 0.0340입니다.
- Pastry(GB F1 0.585, Recall 0.594)와 Bumps(F1 0.684)의 성능이 낮고, 가장 많은 오분류는 Bumps ↔ Other_Faults(GB 41건)입니다.
- Macro Precision이 Macro Recall보다 높아, 소수 유형을 놓치는(Recall이 낮은) 경향이 있습니다.

### 7.2 모델 비교 및 성능 개선

**적용한 모델**

- Random Forest
- Extra Trees
- LightGBM
- CatBoost

선형 모델(로지스틱 회귀, 테스트 Macro F1 0.726)과 단일 Decision Tree(0.704)는 Baseline에서 트리 앙상블(0.794~0.797)보다 낮아 최종 단계 후보에서 제외했습니다. 로지스틱 회귀·SVM·KNN 같은 선형·거리 기반 모델은 팀 내 기본 비교에서 트리 계열보다 Macro F1이 0.07~0.17 낮아 처음부터 제외했습니다.

**주요 실험**

- Baseline 모델 학습 (4종 기본 설정)
- 파생 변수 추가 여부 검증 (5-겹 × 3회 반복 교차검증)
- 하이퍼파라미터 튜닝: `RandomizedSearchCV`, 모델별 20개 조합 × 5-fold, 평가 지표 Macro F1, 클래스 가중치를 탐색 대상에 포함
- 앙상블: 상위 2/3/4개 모델의 예측 확률 평균(소프트 보팅)을 OOF 확률로 비교
- 불균형 데이터 처리: 클래스 가중치 + 클래스별 확률 배율 조정
- 일반화 점검: 테스트 Macro F1 부트스트랩 95% 신뢰구간(1,000회), 단일 모델·앙상블 후보별 OOF 비교

### 7.3 불균형 데이터 처리

**적용한 방법**

- **클래스 가중치 튜닝:** 가중치를 고정하지 않고 튜닝 대상에 포함했습니다. 선택 결과 Random Forest·Extra Trees는 `balanced`, CatBoost는 `SqrtBalanced`, LightGBM은 가중치 없음이 선택되었습니다.
- **클래스별 확률 배율 조정:** 앙상블 예측 확률에 클래스별 배율(0.5~3.0, 0.1 간격)을 곱해 Macro F1이 최대가 되는 값을
학습 데이터 OOF 확률로 찾았습니다. 선택 배율은 Pastry 2.2, Dirtiness 1.9, Z_Scratch 0.5, Stains 0.5, 나머지 1.0입니다.
- 오버샘플링(SMOTE)·언더샘플링은 사용하지 않았습니다.

**선택 이유**

데이터 자체를 늘리거나 줄이지 않고 학습 시 가중치와 예측 단계의 결정 기준만 조정해, 원본 분포를 유지하면서 소수 유형의 재현율을 높이는 것을 기대했습니다. 배율 > 1이면 해당 유형을 더 자주 예측(Recall↑, Precision↓)하므로, 정밀도와 재현율 사이의 균형점을 사람이 정하지 않고 데이터로 정했습니다. 배율 적용 후 OOF 이득이 0.005 이상일 때만 채택하도록 규칙을 두었습니다(실제 이득 0.0122).

**실험 시 고려 사항**

- 가중치 튜닝과 배율 탐색은 학습 데이터(OOF 확률)에서만 수행
- 동일한 테스트 데이터(389건)에서 처리 전후 성능 비교
- 소수 클래스의 Precision, Recall, F1 비교

### 7.4 모델 성능 비교

모든 행은 같은 테스트 데이터 389건 기준입니다. Baseline은 Train 1,164건, 나머지는 학습 데이터 1,552건으로 학습했습니다. 비어 있는 칸은 산출하지 않은 값입니다.

| 모델 / 실험 | Precision (Macro) | Recall (Macro) | F1-score (Macro) | PR-AUC |
|---|---:|---:|---:|---|
| Baseline (Gradient Boosting, Val 기준 선택) | 0.8437 | 0.7598 | 0.7941 | |
| Baseline (Random Forest) | 0.8315 | 0.7696 | 0.7967 | |
| Random Forest + 튜닝(balanced 가중치) | | | 0.7850 | |
| Extra Trees + 튜닝(balanced 가중치) | | | 0.7988 | |
| CatBoost + 튜닝(SqrtBalanced 가중치) | | | 0.8078 | |
| LightGBM + 튜닝(가중치 없음) | | | 0.8305 | |
| 앙상블(상위 3: LightGBM + ET + CatBoost), 배율 미적용 | | | 0.8220 | |
| 앙상블(전체 4), 배율 미적용 | | | 0.8236 | |
| 앙상블(LightGBM + ET), 배율 미적용 | | | 0.8264 | |
| **최종 모델** (앙상블 + 클래스별 배율) | **0.8283** | **0.8527** | **0.8349** | |

※ 최종 모델 외 후보의 테스트 점수는 참고값이며 모델 선택에는 사용하지 않았습니다(선택은 학습 데이터 OOF 기준).

**성능 비교 결과**

![Model Comparison](figures/model_comparison.png)

※ 세로축은 0.70부터 시작합니다. 상위 후보 간 차이(0.80~0.83)는 테스트 95% 신뢰구간 폭(약 0.09)보다 작습니다.

**주요 분석 결과**

- **성능이 가장 우수했던 모델:** LightGBM + Extra Trees 소프트 보팅 앙상블 + 클래스별 배율 (OOF 0.8358, 테스트 0.8349). 단일 모델 중에서는 LightGBM이 CV(0.8168 ± 0.0154)와 테스트(0.8305) 모두 가장 높았습니다.
- **불균형 데이터 처리 전후 성능 변화:** Baseline 대비 Macro Recall이 0.760 → 0.853으로 크게 올랐고 Macro Precision은 0.844 → 0.828로 소폭 내려, 소수 유형을 놓치는 경향이 줄었습니다. 배율 단독 효과는 학습 OOF +0.0122, 테스트 +0.0085입니다. 다만 배율은 OOF 확률에 맞춰 찾은 값이라 학습 데이터 기준 이득은 다소 낙관적이며, 테스트 이득도 신뢰구간 폭(약 0.09)보다 훨씬 작아 효과를 단정하기 어렵습니다.
- **모델별 성능 차이의 원인 해석:** 트리 계열이 선형 모델보다 높은 것은 강종·두께·컨베이어 길이처럼 특정 값 구간이 유형과 연결된 변수를 분기로 잘 활용하기 때문으로 보입니다. 다만 상위 후보 간 차이(0.80~0.83)는 테스트 신뢰구간 폭(약 0.09)보다 작아 우열을 단정하기 어렵습니다.

---

## 8. 최종 모델 및 결과 분석

### 8.1 최종 모델 선정

- **최종 모델:** LightGBM + Extra Trees 소프트 보팅 앙상블 (`ScaledSoftVoting`)
- **적용한 불균형 처리 방법:** Extra Trees `class_weight=balanced` + 클래스별 확률 배율 (Pastry 2.2, Z_Scratch 0.5, K_Scatch 1.0, Stains 0.5, Dirtiness 1.9, Bumps 1.0, Other_Faults 1.0)
- **주요 하이퍼파라미터:**
  - LightGBM: 트리 400개, 학습률 0.03, num_leaves 31, 최대 깊이 5, min_child_samples 10, subsample 0.8, colsample_bytree 1.0, reg_lambda 0, 가중치 없음
  - Extra Trees: 트리 300개, 최대 깊이 20, min_samples_split 5, min_samples_leaf 1, max_features 0.5, class_weight balanced
- **선정 이유:** 단일 모델 4개와 앙상블 3개, 총 7개 후보 중 학습 데이터 OOF Macro F1(배율 적용)이 0.8358로 가장 높았습니다. CatBoost를 포함한 3~4개 구성보다 단순하고, 두 모델 모두 트리 계열이라 순열 중요도로 해석할 수 있습니다.

### 8.2 최종 모델 평가

| Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro F1 95% CI |
|---:|---:|---:|---:|---:|---|
| 0.8201 | 0.8527 | 0.8283 | 0.8527 | 0.8349 | 0.786 ~ 0.876 |

| 유형 | Precision | Recall | F1 | 테스트 건수 |
|---|---:|---:|---:|---:|
| Pastry | 0.5918 | 0.9062 | 0.7160 | 32 |
| Z_Scratch | 1.0000 | 0.8947 | 0.9444 | 38 |
| K_Scatch | 0.9744 | 0.9744 | 0.9744 | 78 |
| Stains | 0.9286 | 0.9286 | 0.9286 | 14 |
| Dirtiness | 0.7500 | 0.8182 | 0.7826 | 11 |
| Bumps | 0.7568 | 0.6914 | 0.7226 | 81 |
| Other_Faults | 0.7969 | 0.7556 | 0.7757 | 135 |

**Confusion Matrix**

![Confusion Matrix](figures/confusion_matrix.png)

※ 왼쪽: 건수, 오른쪽: 실제 유형 기준 비율 (테스트 389건)

**평가 결과 해석**

- **오분류 분석:** 테스트 389건 중 70건(18.0%)을 잘못 분류했습니다. 그중 55건(78.6%)이 Bumps·Other_Faults·Pastry 사이에서 발생했고, Bumps → Other_Faults 18건, Other_Faults → Bumps 15건으로 이 두 유형 간 혼동이 33건(47.1%)입니다. Pastry로 잘못 예측된 20건 중 19건이 Other_Faults(12건)·Bumps(7건)에서 와서 Pastry의 Precision이 0.59로 낮습니다.
- **실제 문제에서 중요한 오류:** 유형을 잘못 판정하면 관련 없는 공정을 점검하게 되므로, Recall이 가장 낮은 Bumps(0.691)와 Precision이 낮은 Pastry(0.592)는 자동 판정보다 검사원 확인 대상으로 두는 것이 안전합니다.
- **강점:** 7개 유형 모두 Recall 0.60 이상으로 성공 기준을 충족했고, K_Scatch·Z_Scratch·Stains는 F1 0.93 이상입니다.
- **한계:** 테스트 Macro F1의 95% 신뢰구간 폭이 약 0.09로 넓고, Stains(14건)·Dirtiness(11건)는 표본이 적어 지표 불확실성이 큽니다.

### 8.3 Feature Importance 및 모델 해석

테스트 데이터에서 변수 값을 하나씩 섞었을 때 Macro F1이 얼마나 떨어지는지(Permutation Importance, 20회 반복)로 기여도를 측정했습니다.

| 순위 | 변수 | Macro F1 감소량 |
|---:|---|---|
| 1 | Length_of_Conveyer | 0.0947 ± 0.0135 |
| 2 | Steel_Type | 0.0817 ± 0.0135 |
| 3 | Steel_Plate_Thickness | 0.0733 ± 0.0133 |
| 4 | Y_Minimum | 0.0370 ± 0.0080 |
| 5 | Orientation_Index | 0.0171 ± 0.0093 |
| 6 | Maximum_of_Luminosity | 0.0147 ± 0.0060 |
| 7 | Pixels_Areas | 0.0129 ± 0.0044 |
| 8 | X_Minimum | 0.0126 ± 0.0101 |
| 9 | Minimum_of_Luminosity | 0.0120 ± 0.0090 |
| 10 | Edges_Y_Index | 0.0118 ± 0.0075 |

![Feature Importance](figures/feature_importance.png)

**주요 분석 결과**

- **예측에 중요한 Feature:** 상위 3개(컨베이어 길이·강종·두께)의 감소량이 4위 Y_Minimum의 2.0~2.6배입니다.
- **해석:** 모델이 결함 형태보다 "어느 라인·어떤 제품에서 나온 결함인가"를 주요 단서로 사용합니다. 이는 이 데이터 안의 연관성일 뿐 결함 발생 원인을 입증하지는 않습니다. 또한 상관된 변수끼리는 정보를 대신할 수 있어, 중요도가 낮다고 해당 측정값이 불필요하다고 단정할 수 없습니다.
- **인사이트:** K_Scatch·Z_Scratch·Stains는 현재 변수로 충분히 구분되지만, Bumps·Other_Faults·Pastry는 표면 높이·깊이, 결함 영상의 질감, Other_Faults 세부 라벨 등 추가 정보가 필요할 가능성이 있습니다.

SHAP은 사용하지 않았습니다.

---

## 9. Streamlit 대시보드

### 9.1 주요 기능

- **샘플 예측:** 데이터에서 결함 1건을 행 번호 또는 무작위로 골라 예측 유형, 신뢰도, 실제 유형, 7개 유형별 확률을 보여줍니다.
- **CSV 업로드 예측:** 여러 건이 담긴 CSV를 올리면 건별 예측 결과, 유형별 예측 건수를 보여주고 결과를 CSV로 내려받을 수 있습니다. 필요한 열 누락, 강종 값 오류, 빈 값을 검사합니다.
- **검사원 확인 권장 표시:** 예측 유형이 Pastry·Bumps이거나 신뢰도가 기준(사이드바 슬라이더, 기본 0.50)보다 낮으면 "확인 권장"으로 표시합니다. Pastry는 Precision, Bumps는 Recall이 낮기 때문입니다(8.2 참고).
- **모델 정보:** 최종 모델 구성, 클래스별 확률 배율, 테스트 성능, 유형별 성능표, 모델 비교·혼동행렬·변수 중요도 그래프를 보여줍니다.

> 저장된 모델은 전체 데이터로 다시 학습한 것이므로, 샘플 예측 결과는 성능 지표가 아닙니다.

### 9.2 실행 화면

![Streamlit Dashboard](figures/dashboard.png)

※ 샘플 예측 탭 화면

---

## 10. 결론 및 회고

### 10.1 프로젝트 주요 결과

- **목표 달성 여부:**

  | 성공 기준 | 결과 | 판정 |
  |---|---|---|
  | Macro F1 ≥ 0.70, 기준 모델 대비 우위 | 테스트 Macro F1 0.835 (다수 클래스 기준 0.074) | 달성 |
  | 유형별 Recall ≥ 0.60 | 최저 Bumps 0.691 | 달성 |
  | 변수 충분성 결론 | K_Scatch·Z_Scratch·Stains 충분, Bumps·Other_Faults·Pastry 추가 측정 검토 필요 | 달성 |

- **최종 모델 성능 및 개선 사항:** Baseline(GB) 0.794 → 최종 0.835. 학습 데이터 OOF Macro F1 0.836, 테스트 0.835로 서로 비슷했고, 테스트 95% 신뢰구간(0.786~0.876)의 하한도 목표치 0.70을 넘었습니다.
- **불균형 데이터 처리 결과:** Macro Recall 0.760 → 0.853, Pastry Recall 0.594 → 0.906으로 소수 유형 탐지가 개선되었습니다. 확률 배율은 OOF 이득이 기준(0.005) 이상일 때만 채택하는 조건부 방식이었고, 그 단독 효과는 테스트 +0.0085로 작습니다.
- **주요 인사이트:** 모델은 생산 조건 변수(컨베이어 길이·강종·두께)에 크게 의존하며, Bumps·Other_Faults·Pastry는 현재 측정값만으로 구분하기 어렵습니다.

### 10.2 한계점 및 개선 방향

**데이터 측면의 한계**

- Other_Faults는 여러 형태가 섞인 "기타" 유형이라 Bumps·Pastry와 특징이 겹치며, 테스트 오답의 78.6%가 이 세 유형 사이에서 발생했습니다.
- 전체 표본이 1,941건으로 적고, 테스트의 Dirtiness 11건·Stains 14건은 성능 추정의 불확실성이 큽니다.
- 시점·공정 정보가 없는 공개 데이터 1개로만 검증했고, 결함 이미지나 공정 기록 같은 외부 데이터를 활용하지 않았습니다.

**모델 성능 및 일반화 측면의 한계**

- 학습 점수 1.000과 검증 점수(0.836) 사이 격차가 남아 있습니다.
- 생산 조건 변수 의존도가 높아, 새로운 라인·강종·두께 범위에서는 성능이 보장되지 않습니다.
- 확률 배율의 효과가 중첩 교차검증에서 평균 −0.003으로 불안정합니다.

**추가로 시도하고 싶은 실험**

- Other_Faults 여부를 먼저 이진 분류한 뒤 나머지를 6개 유형으로 분류하는 2단계 구조
- 6개 유형별 판별 모델을 결합해 모두 "해당 없음"이면 Other_Faults로 분류하는 방식
- 희소 유형 데이터 추가 확보, 외부 데이터(이미지·공정 정보) 결합
- 최소 잎 크기·깊이 제한을 강화한 설정과 비교해 과적합 완화
- 하이퍼파라미터 선택에도 중첩·반복 교차검증을 적용해 낙관적 편향 감소
- 모델 교체·튜닝·가중치·앙상블·배율 적용을 순차 비교해 개선 원인 분리

### 10.3 프로젝트를 통해 배운 점

- 최원우: 프로젝트를 통해 배운 점: 적절한 파생 변수를 만드는 것은 쉽지 않은 작업임을 배웠다.

- 손규리: 데이터 전처리부터 모델링까지의 전 과정을 직접 수행하며, 분석 흐름을 익힐 수 있어 뜻깊었다. 여러 모델의 조합을 반복적으로 실험하고 결과를 비교하면서, 성능을 점진적으로 개선하는 방법을 배웠다.

- 강민정: 이번 모델링 프로젝트를 통해 단순히 정확도가 높은 모델을 만드는 것만으로는 충분하지 않다는 점을 배웠다. 처음에는 다양한 알고리즘을 적용하고 하이퍼파라미터를 조정하면 성능을 높일 수 있을 것이라고 생각했지만 실제로는 모델이나 파생변수를 추가하더라도 성능이 오히려 떨어지는 경우가 있었다.
이러한 과정을 통해 모델의 성능을 개선하기 위해서는 무작정 새로운 방법을 적용하기보다 데이터의 특성과 변수 간 관계를 먼저 이해하고 실험 결과를 객관적으로 비교하는 것이 중요하다는 것을 느꼈다.

- 황류경: 다양한 머신러닝 모델을 비교하고 성능 개선을 시도하면서 모델을 복잡하게 만든다고 성능이 무조건 오르지는 않는다는 것을 배웠다. 또한 교차검증을 통해 데이터 분할에 따라 성능이 달라질 수 있다는 점을 알게 되었다.

- 신유정:EDA에서 파악한 변수 특성을 조합해, 결함 유형 구분에 도움이 되는 파생변수를 설계해 볼 수 있었다. 다만 모델링 개선 과정에서 시도가 많아져 과정이 복잡해졌으므로, 다음 프로젝트에서는 범위를 한정하고 개선 방안을 미리 계획한 뒤 진행하고자 한다.
 

---

## 11. 팀원 및 역할

| 이름 | 담당 내용 | GitHub |
|---|---|---|
| 팀원 1 |최원우 |발표 |https://github.com/SucutSkunk/team_project.git |
| 팀원 2 |손규리 |보고서 작성 |sgr100405-arch/kant-ds |
| 팀원 3 |강민정 |코드리뷰, GitHub관리 | |
| 팀원 4 |황류경 |보고서작성 |Kant_DS/Team Project at main · das46578/Kant_DS |
| 팀원 5 |신유정 |README.md작성 |https://github.com/sinyu-dev/Team_project_01.git |
