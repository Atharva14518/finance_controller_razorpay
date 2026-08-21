"""XGBoost confidence scoring with SHAP explanations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from pipeline.ingest import DATA_DIR, load_labeled_pairs

FEATURE_COLS = [
    "amount_diff",
    "amount_diff_pct",
    "date_diff_days",
    "method_match",
    "desc_similarity",
]


class MLScorer:
    def __init__(self, model: XGBClassifier | None = None, explainer=None):
        if model is None:
            model, explainer = self._train_default()
        self.model = model
        self.explainer = explainer

    @staticmethod
    def _train_default() -> tuple[XGBClassifier, shap.TreeExplainer]:
        df = load_labeled_pairs()
        X = df[FEATURE_COLS]
        y = df["label"]
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        return model, explainer

    def score(self, features: dict) -> tuple[float, dict[str, float]]:
        row = pd.DataFrame([{k: features[k] for k in FEATURE_COLS}])
        proba = float(self.model.predict_proba(row)[0][1])
        shap_vals = self.explainer.shap_values(row)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_dict = {
            col: round(float(val), 4)
            for col, val in zip(FEATURE_COLS, shap_vals[0])
        }
        return proba, shap_dict

    def score_batch(self, feature_rows: list[dict]) -> list[tuple[float, dict[str, float]]]:
        return [self.score(f) for f in feature_rows]
