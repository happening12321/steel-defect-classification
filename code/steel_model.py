"""
강판 결함 분류 모델 공용 코드
- 노트북(최종 모델.ipynb)과 Streamlit 앱이 같은 전처리·모델 클래스를 쓰도록 분리한 모듈
- 노트북의 merge_columns / build_features / ScaledSoftVoting 과 동일한 내용
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

STEEL_COLS = ["TypeOfSteel_A300", "TypeOfSteel_A400"]
FAULT_COLS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains",
              "Dirtiness", "Bumps", "Other_Faults"]
TARGET = "Fault_Type"
STEEL_CODE = {"A300": 0, "A400": 1}

# 원본 CSV의 입력 변수 27개 (결함 7열 제외)
RAW_FEATURES = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400",
    "Steel_Plate_Thickness", "Edges_Index", "Empty_Index", "Square_Index",
    "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index", "Outside_Global_Index",
    "LogOfAreas", "Log_X_Index", "Log_Y_Index", "Orientation_Index",
    "Luminosity_Index", "SigmoidOfAreas",
]


def merge_columns(data: pd.DataFrame) -> pd.DataFrame:
    """강종 2열 → Steel_Type, 결함 7열 → Fault_Type (결함 열이 없으면 강종만)"""
    out = data.copy()
    if set(STEEL_COLS).issubset(out.columns):
        pos = out.columns.get_loc("TypeOfSteel_A300")
        steel = np.where(out["TypeOfSteel_A300"] == 1, "A300", "A400")
        out = out.drop(columns=STEEL_COLS)
        out.insert(pos, "Steel_Type", steel)
    if set(FAULT_COLS).issubset(out.columns):
        out[TARGET] = out[FAULT_COLS].idxmax(axis=1)
        out = out.drop(columns=FAULT_COLS)
    return out


def build_features(data: pd.DataFrame, use_extra: bool) -> pd.DataFrame:
    out = data.drop(columns=[TARGET], errors="ignore").copy()
    out["Steel_Type"] = out["Steel_Type"].map(STEEL_CODE)
    if out["Steel_Type"].isna().any():
        raise ValueError("Steel_Type에 A300 / A400 이외의 값이 있습니다.")
    if use_extra:
        x_len = out["X_Maximum"] - out["X_Minimum"] + 1
        y_len = out["Y_Maximum"] - out["Y_Minimum"] + 1
        out["Luminosity_Range"] = out["Maximum_of_Luminosity"] - out["Minimum_of_Luminosity"]
        out["Aspect_Ratio"] = x_len / y_len
        out["Perimeter_Area_Ratio"] = (out["X_Perimeter"] + out["Y_Perimeter"]) / out["Pixels_Areas"]
    return out


class ScaledSoftVoting(BaseEstimator, ClassifierMixin):
    """여러 모델의 확률 평균 + 클래스별 배율 보정"""

    def __init__(self, estimators, class_scale=None):
        self.estimators = estimators
        self.class_scale = class_scale

    def fit(self, X, y):
        self.estimators_ = [clone(e).fit(X, y) for e in self.estimators]
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        P = np.mean([e.predict_proba(X) for e in self.estimators_], axis=0)
        if self.class_scale is not None:
            P = P * np.asarray(self.class_scale)
            P = P / P.sum(axis=1, keepdims=True)
        return P

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def predict_with_bundle(bundle: dict, new_data: pd.DataFrame) -> pd.DataFrame:
    """원본 형식(TypeOfSteel_A300/A400) 또는 병합 형식(Steel_Type) 입력 → 예측 결과"""
    names = bundle["class_names"]
    merged = merge_columns(new_data.drop(columns=names, errors="ignore"))
    X_new = build_features(merged, bundle["use_extra_features"])[bundle["feature_cols"]]
    proba = bundle["model"].predict_proba(X_new)
    out = pd.DataFrame(proba, columns=[f"prob_{n}" for n in names], index=new_data.index)
    out.insert(0, "predicted_fault", [names[i] for i in proba.argmax(axis=1)])
    out.insert(1, "confidence", proba.max(axis=1))
    return out
