"""
강판 표면 결함 분류 — Streamlit 앱
실행: streamlit run app.py
필요 파일(같은 폴더): steel_model.py, steel_fault_model_combined.joblib
선택 파일: steel_plates_faults.csv (예시 데이터 / 단일 예측 시작값)
"""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import steel_model
from steel_model import FAULT_COLS, STEEL_COLS, RAW_FEATURES, predict_with_bundle

MODEL_PATH = Path("steel_fault_model_combined.joblib")
DATA_PATH = Path("steel_plates_faults.csv")

# 노트북에서 저장한 모델은 ScaledSoftVoting을 '__main__' 모듈 소속으로 기록한다.
# 앱에서 불러올 수 있도록 같은 이름으로 연결해 둔다 (노트북 수정 불필요).
sys.modules["__main__"].ScaledSoftVoting = steel_model.ScaledSoftVoting

# 단일 예측 입력의 기본값 (원본 데이터 중앙값)
MEDIANS = {
    "X_Minimum": 435, "X_Maximum": 467, "Y_Minimum": 1204128, "Y_Maximum": 1204136,
    "Pixels_Areas": 174, "X_Perimeter": 26, "Y_Perimeter": 25, "Sum_of_Luminosity": 19202,
    "Minimum_of_Luminosity": 90, "Maximum_of_Luminosity": 127, "Length_of_Conveyer": 1364,
    "Steel_Plate_Thickness": 70, "Edges_Index": 0.2273, "Empty_Index": 0.4121,
    "Square_Index": 0.5556, "Outside_X_Index": 0.0101, "Edges_X_Index": 0.6364,
    "Edges_Y_Index": 0.9474, "Outside_Global_Index": 1.0, "LogOfAreas": 2.2406,
    "Log_X_Index": 1.1761, "Log_Y_Index": 1.3222, "Orientation_Index": 0.0952,
    "Luminosity_Index": -0.133, "SigmoidOfAreas": 0.5063,
}
INT_COLS = ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
            "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
            "Maximum_of_Luminosity", "Length_of_Conveyer", "Steel_Plate_Thickness"]
INPUT_GROUPS = {
    "위치·크기": ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
               "X_Perimeter", "Y_Perimeter", "LogOfAreas", "Log_X_Index", "Log_Y_Index", "SigmoidOfAreas"],
    "밝기": ["Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Luminosity_Index"],
    "강판·형태 지표": ["Length_of_Conveyer", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
                  "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
                  "Outside_Global_Index", "Orientation_Index"],
}

st.set_page_config(page_title="강판 결함 분류", page_icon="🔍", layout="wide")


@st.cache_resource
def load_bundle(path: Path):
    return joblib.load(path)


@st.cache_data
def load_sample(path: Path):
    return pd.read_csv(path)


def check_columns(df: pd.DataFrame):
    """원본 형식(강종 2열) 또는 병합 형식(Steel_Type) 중 하나인지 확인"""
    base = [c for c in RAW_FEATURES if c not in STEEL_COLS]
    missing = [c for c in base if c not in df.columns]
    has_steel = set(STEEL_COLS).issubset(df.columns) or "Steel_Type" in df.columns
    if not has_steel:
        missing.append("TypeOfSteel_A300/A400 또는 Steel_Type")
    return missing


# ------------------------------------------------------------
# 모델 불러오기
# ------------------------------------------------------------
st.title("강판 표면 결함 분류")
if not MODEL_PATH.exists():
    st.error(f"`{MODEL_PATH}` 파일이 없습니다. `최종 모델.ipynb`를 끝까지 실행해 모델을 저장한 뒤 "
             "이 앱과 같은 폴더에 두세요.")
    st.stop()

bundle = load_bundle(MODEL_PATH)
names = bundle["class_names"]
sample = load_sample(DATA_PATH) if DATA_PATH.exists() else None

with st.sidebar:
    st.header("모델 정보")
    st.write(f"**구성:** {bundle['candidate']}")
    st.write(" + ".join(bundle["members"]))
    st.write(f"**확률 보정:** {'사용' if bundle.get('class_scale') else '미사용'}")
    scores = bundle.get("test_scores", {})
    if scores:
        st.metric("Test Macro F1", f"{scores['Macro F1']:.4f}")
        st.metric("Test Accuracy", f"{scores['Accuracy']:.4f}")
        st.caption("개발 80%로 학습한 모델의 테스트 20% 성능. 앱의 모델은 전체 데이터로 다시 학습한 버전입니다.")

tab_batch, tab_single, tab_info = st.tabs(["CSV 일괄 예측", "단일 결함 예측", "모델 상세"])

# ------------------------------------------------------------
# 1) CSV 일괄 예측
# ------------------------------------------------------------
with tab_batch:
    st.write("원본과 같은 열 구조의 CSV를 올리면 결함 유형을 예측합니다. 결함 7열이 있으면 정답과 비교합니다.")
    up = st.file_uploader("CSV 파일", type="csv")
    use_sample = sample is not None and st.checkbox("예시 데이터(steel_plates_faults.csv) 사용", value=up is None)

    data = pd.read_csv(up) if up is not None else (sample if use_sample else None)
    if data is not None:
        missing = check_columns(data)
        if missing:
            st.error("필요한 열이 없습니다: " + ", ".join(missing))
            st.stop()

        result = predict_with_bundle(bundle, data)
        threshold = st.slider("낮은 확신도 기준", 0.3, 0.9, 0.5, 0.05,
                              help="이 값보다 확신도가 낮은 예측은 재검토 대상으로 표시합니다.")
        result["review"] = np.where(result["confidence"] < threshold, "재검토", "")

        c1, c2, c3 = st.columns(3)
        c1.metric("예측 건수", f"{len(result):,}")
        c2.metric("평균 확신도", f"{result['confidence'].mean():.2f}")
        c3.metric("재검토 대상", f"{(result['review'] == '재검토').sum():,}")

        st.subheader("예측 결함 분포")
        st.bar_chart(result["predicted_fault"].value_counts().reindex(names, fill_value=0))

        has_label = set(FAULT_COLS).issubset(data.columns)
        view = result.copy()
        if has_label:
            view.insert(0, "actual_fault", data[FAULT_COLS].idxmax(axis=1).values)
        st.subheader("예측 결과")
        st.dataframe(view.round(3), height=320)
        st.download_button("예측 결과 CSV 다운로드",
                           pd.concat([data, result], axis=1).to_csv(index=False).encode("utf-8-sig"),
                           file_name="steel_predictions.csv", mime="text/csv")

        if has_label:
            st.subheader("정답과 비교")
            if use_sample and up is None:
                st.warning("예시 데이터는 모델 학습에 사용된 데이터라 점수가 실제 성능보다 높게 나옵니다. "
                           "실제 성능은 사이드바의 Test 점수를 보세요.")
            y_true, y_pred = view["actual_fault"], view["predicted_fault"]
            m1, m2 = st.columns(2)
            m1.metric("Macro F1", f"{f1_score(y_true, y_pred, labels=names, average='macro', zero_division=0):.4f}")
            m2.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
            cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=names),
                              index=[f"실제 {n}" for n in names], columns=[f"예측 {n}" for n in names])
            st.dataframe(cm.style.background_gradient(cmap="Blues", axis=None))
            rep = pd.DataFrame(classification_report(y_true, y_pred, labels=names, output_dict=True,
                                                     zero_division=0)).T.loc[names]
            st.dataframe(rep.round(3))

# ------------------------------------------------------------
# 2) 단일 결함 예측
# ------------------------------------------------------------
with tab_single:
    st.write("결함 하나의 측정값을 입력해 결함 유형을 예측합니다.")
    start = MEDIANS.copy()
    start_steel = "A400"
    if sample is not None:
        pick = st.number_input("시작값으로 불러올 예시 데이터 행 번호 (-1 = 중앙값)",
                               min_value=-1, max_value=len(sample) - 1, value=-1, step=1)
        if pick >= 0:
            row = sample.iloc[int(pick)]
            start = {c: row[c] for c in MEDIANS}
            start_steel = "A300" if row["TypeOfSteel_A300"] == 1 else "A400"
            st.caption(f"이 행의 실제 결함: **{sample[FAULT_COLS].iloc[int(pick)].idxmax()}**")
    else:
        pick = -1

    with st.form("single"):
        steel = st.radio("강종", ["A300", "A400"], index=["A300", "A400"].index(start_steel), horizontal=True)
        values = {}
        for group, cols in INPUT_GROUPS.items():
            st.markdown(f"**{group}**")
            grid = st.columns(4)
            for i, col in enumerate(cols):
                v = start[col]
                if col in INT_COLS:
                    values[col] = grid[i % 4].number_input(col, value=int(v), step=1, key=f"{col}_{pick}")
                else:
                    values[col] = grid[i % 4].number_input(col, value=float(v), format="%.4f", key=f"{col}_{pick}")
        submitted = st.form_submit_button("예측하기", type="primary")

    if submitted:
        one = pd.DataFrame([values])
        one["TypeOfSteel_A300"] = int(steel == "A300")
        one["TypeOfSteel_A400"] = int(steel == "A400")
        one = one[RAW_FEATURES]
        res = predict_with_bundle(bundle, one).iloc[0]
        st.success(f"예측 결함: **{res['predicted_fault']}** (확신도 {res['confidence']:.1%})")
        probs = (pd.DataFrame({"결함 유형": names, "확률": [res[f"prob_{n}"] for n in names]})
                 .sort_values("확률", ascending=False).reset_index(drop=True))
        st.dataframe(probs, hide_index=True,
                     column_config={"확률": st.column_config.ProgressColumn("확률", format="%.3f",
                                                                          min_value=0.0, max_value=1.0)})

# ------------------------------------------------------------
# 3) 모델 상세
# ------------------------------------------------------------
with tab_info:
    st.subheader("테스트 성능 (개발 80% 학습 → 테스트 20%)")
    if scores:
        st.dataframe(pd.DataFrame([scores]).round(4))
    if bundle.get("class_scale"):
        st.subheader("클래스별 확률 배율")
        st.caption("1보다 크면 그 결함을 더 자주 예측하도록 보정합니다.")
        st.dataframe(pd.Series(bundle["class_scale"], index=names, name="배율").round(2).to_frame().T)
    st.subheader("하이퍼파라미터")
    for m, p in bundle["params"].items():
        with st.expander(m):
            st.json({k: v for k, v in p.items() if v is not None and not callable(v)})
    st.subheader("입력 변수")
    st.write(", ".join(bundle["feature_cols"]))
