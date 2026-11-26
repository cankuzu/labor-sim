"""Machine-learning baseline: predict each occupation's 2022->2025 employment change
from AI exposure features, with the same target the simulation is scored on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

DEFAULT_FEATURES = ["pctl_anthropic", "pctl_eloundou", "pctl_microsoft", "pctl_aioe",
                    "automation_share", "n_tasks", "log_wage"]


def prepare(occ: pd.DataFrame, target: str = "emp_change_2022_2025") -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = occ.copy()
    wage_cols = sorted(c for c in df.columns if c.startswith("wage_"))
    if wage_cols:
        df["log_wage"] = np.log(df[wage_cols[0]])
    feats = [f for f in DEFAULT_FEATURES if f in df.columns and df[f].notna().mean() > 0.5]
    if target not in df:
        raise KeyError(f"target '{target}' missing: download OEWS May 2022 and May 2025 and rebuild occupations.csv")
    df = df.dropna(subset=feats + [target])
    df["major_group"] = df["soc"].astype(str).str[:2]
    X = pd.get_dummies(df[feats + ["major_group"]], columns=["major_group"], dtype=float)
    return X, df[target], feats


def evaluate(occ: pd.DataFrame, target: str = "emp_change_2022_2025", seed: int = 0, n_splits: int = 5) -> dict:
    X, y, feats = prepare(occ, target)
    model = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    pred = cross_val_predict(model, X, y, cv=KFold(n_splits, shuffle=True, random_state=seed))
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {"n": len(y), "features": feats,
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "spearman": float(spearmanr(y, pred).statistic),
            "importance": imp.head(10).round(4).to_dict(),
            "predictions": pd.DataFrame({"soc": occ.loc[y.index, "soc"].values, "observed": y.values, "predicted": pred})}


def score_simulation(sim_change: pd.DataFrame, observed: pd.DataFrame) -> dict:
    """Same metrics for the simulation's occupation-level output (columns soc, predicted)."""
    m = observed.merge(sim_change, on="soc", how="inner")
    return {"n": len(m), "mae": float(mean_absolute_error(m["observed"], m["predicted"])),
            "rmse": float(np.sqrt(mean_squared_error(m["observed"], m["predicted"]))),
            "spearman": float(spearmanr(m["observed"], m["predicted"]).statistic)}
