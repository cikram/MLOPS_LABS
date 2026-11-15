"""Monitoring: extension 4 (chi-square, MAE by hour) + steps 8 and 9 moved from the notebook."""
from functools import lru_cache                                   # NEW

import joblib                                                     # NEW
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mean_absolute_error                   # NEW

from .clean import clean
from .common import ART, FEATURES, R, REF_MONTH, TARGET, load_month   # NEW: ART, REF_MONTH
from .validate import validate                                    # NEW: only if your monitor() uses validate()
from .version import build_datasets                               # NEW

CAT_COLS = ["PULocationID", "payment_type"]


def chi2_categorical(ref, cur, col, top=20):
    """Chi-square test of homogeneity between the reference and the current month, with Cramer's V."""
    cats = ref[col].value_counts().head(top).index           # top categories of the reference
    r = ref[col].where(ref[col].isin(cats), -1)              # everything else goes to the bucket -1
    c = cur[col].where(cur[col].isin(cats), -1)
    table = pd.concat([r.value_counts(), c.value_counts()],
                      axis=1, keys=["ref", "cur"]).fillna(0)  # contingency table (K x 2)
    chi2, p, dof, _ = chi2_contingency(table.values)
    cramers_v = np.sqrt(chi2 / table.values.sum())           # 2 groups -> min(r-1, c-1) = 1
    return {"chi2": chi2, "p_value": p, "dof": dof, "cramers_v": cramers_v}


def mae_by_hour(df, model):
    """MAE in minutes for each pickup hour (0 to 23)."""
    err = (df[TARGET] - model.predict(df[FEATURES])).abs()
    return err.groupby(df["tpep_pickup_datetime"].dt.hour).mean()


def hourly_report(ref, cur, model):
    """MAE by hour for the reference and the current month, and their ratio (> 1 = worse than reference)."""
    out = pd.DataFrame({"ref": mae_by_hour(ref, model), "cur": mae_by_hour(cur, model)})
    out["ratio"] = out["cur"] / out["ref"]
    return out.round(2)


def report_drift(ref_test, model, ym, cat_cols=CAT_COLS):
    """Load and clean month ym, then print the chi-square results and the 5 worst hours."""
    cur, _ = clean(load_month(ym), ym, R)
    print(f"\n=== {ym} ===")
    for col in cat_cols:
        r = chi2_categorical(ref_test, cur, col)
        print(f"{col}: chi2={r['chi2']:.1f}  dof={r['dof']}  p={r['p_value']:.3g}  Cramer's V={r['cramers_v']:.4f}")
    rep = hourly_report(ref_test, cur, model)
    print("5 worst hours (MAE ratio vs reference):")
    print(rep.sort_values("ratio", ascending=False).head(5).to_string())


def diagnose(ref_test, model, ym, cat_cols=CAT_COLS, top_hours=5):
    """Context for an alert: Cramer's V per categorical column and the worst pickup hours (no printing)."""
    cur, _ = clean(load_month(ym), ym, R)
    cramers = {col: round(float(chi2_categorical(ref_test, cur, col)["cramers_v"]), 3) for col in cat_cols}
    rep = hourly_report(ref_test, cur, model).sort_values("ratio", ascending=False).head(top_hours)
    worst = [{"hour": int(h), "mae_ref": float(r["ref"]), "mae_cur": float(r["cur"]), "ratio": float(r["ratio"])}
             for h, r in rep.iterrows()]
    return {"cramers_v": cramers, "worst_hours": worst}


# ---------------------------------------------------------------------------
# NEW: moved from the notebook (steps 8 and 9)
# ---------------------------------------------------------------------------
PSI_WARN, PSI_ALERT = 0.10, 0.25
MAE_WARN, MAE_ALERT = 1.10, 1.25


@lru_cache(maxsize=1)
def load_reference():
    """Model v1 and the reference datasets, loaded once (they replace the notebook globals)."""
    model = joblib.load(ART / "model_v1.joblib")
    _, _, _, train_df, test_df = build_datasets(REF_MONTH)
    return model, train_df, test_df


def _psi(r, c):
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)      # avoid log(0) and division by 0
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref, cur, bins=10):
    ref, cur = ref.dropna(), cur.dropna()
    q = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges = np.concatenate([[-np.inf], q[1:-1], [np.inf]])
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    c = np.histogram(cur, bins=edges)[0] / len(cur)
    return _psi(r, c)


def psi_categorical(ref, cur, top=20):
    ref, cur = ref.dropna(), cur.dropna()
    keep = ref.value_counts().head(top).index
    ref_g, cur_g = ref.where(ref.isin(keep), -1), cur.where(cur.isin(keep), -1)
    cats = list(keep) + [-1]
    r = ref_g.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    c = cur_g.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    return _psi(r, c)


NUM_FEATS, CAT_FEATS = ["trip_distance", "fare_amount", "pickup_hour", "passenger_count"], ["payment_type", "PULocationID"]


def monitor(ym):
    model, ref, _ = load_reference()          # ref = the training data of model v1 (as in the notebook)
    b = load_month(ym); val = validate(b, ym, R); c, _ = clean(b, ym, R)
    row = {"month": ym, "rows": len(c), "pct_dropped": round(100 * (1 - len(c) / len(b)), 2), "rules_failing": int((val.status == "FAIL").sum())}
    for f in NUM_FEATS: row[f"psi_{f}"] = round(psi_numeric(ref[f], c[f]), 3)
    for f in CAT_FEATS: row[f"psi_{f}"] = round(psi_categorical(ref[f], c[f]), 3)
    row["MAE_min"] = round(mean_absolute_error(c[TARGET], model.predict(c[FEATURES])), 3)
    return row


def decide(row, ref_mae):
    psi_cols = [c for c in row.index if c.startswith("psi_")]
    max_psi = row[psi_cols].max()
    mae_ratio = row["MAE_min"] / ref_mae
    if mae_ratio >= MAE_ALERT or (max_psi >= PSI_ALERT and mae_ratio >= MAE_WARN):
        return "RETRAIN"
    if mae_ratio >= MAE_WARN or max_psi >= PSI_WARN or row["rules_failing"] > 0:
        return "INVESTIGATE"
    return "OK"