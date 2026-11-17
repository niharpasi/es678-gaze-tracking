import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.models import (
    get_candidate_models,
    select_spatial_calibration_points,
    LowShotAffineCalibrator,
)


def compute_metrics(y_true_mm: np.ndarray, y_pred_mm: np.ndarray) -> dict:
    errors = np.linalg.norm(y_true_mm - y_pred_mm, axis=1)
    return {
        "mean_mm": float(np.mean(errors)),
        "median_mm": float(np.median(errors)),
        "std_mm": float(np.std(errors)),
        "r2_x": float(r2_score(y_true_mm[:, 0], y_pred_mm[:, 0])),
        "r2_y": float(r2_score(y_true_mm[:, 1], y_pred_mm[:, 1])),
    }


def run_within_subject_cv(df: pd.DataFrame, X_raw: np.ndarray, X_geom: np.ndarray) -> pd.DataFrame:
    y_mm = np.stack(df["target_mm"].values)
    subjects = df["subject"].unique()
    model_names = list(get_candidate_models().keys())
    records = {m: [] for m in model_names}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for subj in subjects:
        mask = (df["subject"] == subj).values
        X_r_s, X_g_s, y_s = X_raw[mask], X_geom[mask], y_mm[mask]

        for train_idx, test_idx in kf.split(X_g_s):
            scaler_g = StandardScaler().fit(X_g_s[train_idx])
            scaler_r = StandardScaler().fit(X_r_s[train_idx])

            models = get_candidate_models()
            for name, model in models.items():
                if "OLS" in name:
                    model.fit(scaler_r.transform(X_r_s[train_idx]), y_s[train_idx])
                    preds = model.predict(scaler_r.transform(X_r_s[test_idx]))
                else:
                    model.fit(scaler_g.transform(X_g_s[train_idx]), y_s[train_idx])
                    preds = model.predict(scaler_g.transform(X_g_s[test_idx]))

                metrics = compute_metrics(y_s[test_idx], preds)
                records[name].append(metrics["mean_mm"])

    summary = []
    for name in model_names:
        summary.append(
            {
                "Model": name,
                "Within_Subject_Mean_mm": round(float(np.mean(records[name])), 2),
                "Within_Subject_Std_mm": round(float(np.std(records[name])), 2),
            }
        )
    return pd.DataFrame(summary)


def run_loso_and_depth_analysis(
    df: pd.DataFrame, X_raw: np.ndarray, X_geom: np.ndarray, k_list=(1, 3, 5, 9)
):
    y_mm = np.stack(df["target_mm"].values)
    subjects = df["subject"].unique()
    model_names = list(get_candidate_models().keys())

    loso_metrics = {m: [] for m in model_names}
    calib_curves = {m: {k: [] for k in k_list} for m in model_names if "Model" in m}
    spatial_records = []

    for subj in subjects:
        train_mask = (df["subject"] != subj).values
        test_mask = (df["subject"] == subj).values

        scaler_g = StandardScaler().fit(X_geom[train_mask])
        scaler_r = StandardScaler().fit(X_raw[train_mask])

        X_g_tr = scaler_g.transform(X_geom[train_mask])
        X_g_te = scaler_g.transform(X_geom[test_mask])
        X_r_tr = scaler_r.transform(X_raw[train_mask])
        X_r_te = scaler_r.transform(X_raw[test_mask])

        y_tr, y_te = y_mm[train_mask], y_mm[test_mask]
        w_mm = df.loc[test_mask, "w_mm"].iloc[0]
        h_mm = df.loc[test_mask, "h_mm"].iloc[0]

        models = get_candidate_models()
        for name, model in models.items():
            if "OLS" in name:
                model.fit(X_r_tr, y_tr)
                y_pred = model.predict(X_r_te)
            else:
                model.fit(X_g_tr, y_tr)
                y_pred = model.predict(X_g_te)

            m = compute_metrics(y_te, y_pred)
            loso_metrics[name].append(m)

            if name in calib_curves:
                for k in k_list:
                    cal_idx = select_spatial_calibration_points(y_te, k=k, frames_per_point=3)
                    eval_idx = np.setdiff1d(np.arange(len(y_te)), cal_idx)

                    calibrator = LowShotAffineCalibrator(k_points=k)
                    calibrator.fit(y_pred[cal_idx], y_te[cal_idx])
                    y_pred_cal = calibrator.predict(y_pred[eval_idx])

                    cal_m = compute_metrics(y_te[eval_idx], y_pred_cal)
                    calib_curves[name][k].append(cal_m["mean_mm"])

                    if name == "Model 2: Kernel Ridge (RBF)" and k == 5:
                        norm_x = y_te[eval_idx, 0] / w_mm
                        norm_y = y_te[eval_idx, 1] / h_mm
                        err_zero = np.linalg.norm(y_te[eval_idx] - y_pred[eval_idx], axis=1)
                        err_cal = np.linalg.norm(y_te[eval_idx] - y_pred_cal, axis=1)
                        for nx_i, ny_i, ez, ec in zip(norm_x, norm_y, err_zero, err_cal):
                            spatial_records.append(
                                {
                                    "subject": subj,
                                    "norm_x": nx_i,
                                    "norm_y": ny_i,
                                    "err_zero_mm": ez,
                                    "err_cal_mm": ec,
                                }
                            )

    loso_rows = []
    for name in model_names:
        mean_errs = [x["mean_mm"] for x in loso_metrics[name]]
        r2_xs = [x["r2_x"] for x in loso_metrics[name]]
        r2_ys = [x["r2_y"] for x in loso_metrics[name]]
        row = {
            "Model": name,
            "LOSO_Mean_mm": round(float(np.mean(mean_errs)), 2),
            "LOSO_Std_mm": round(float(np.std(mean_errs)), 2),
            "R2_X": round(float(np.mean(r2_xs)), 3),
            "R2_Y": round(float(np.mean(r2_ys)), 3),
        }
        if name in calib_curves:
            for k in k_list:
                row[f"Cal_k{k}_mm"] = round(float(np.mean(calib_curves[name][k])), 2)
        loso_rows.append(row)

    spatial_df = pd.DataFrame(spatial_records)
    fitts_audit = compute_fitts_law_ui_audit(spatial_df)

    return pd.DataFrame(loso_rows), calib_curves, spatial_df, fitts_audit


def compute_fitts_law_ui_audit(spatial_df: pd.DataFrame) -> pd.DataFrame:
    dx = spatial_df["norm_x"] - 0.5
    dy = spatial_df["norm_y"] - 0.5
    eccentricity = np.sqrt(dx ** 2 + dy ** 2)

    zones = pd.cut(
        eccentricity,
        bins=[-0.01, 0.22, 0.42, 1.0],
        labels=["Inner Center (r <= 0.22)", "Mid Screen (0.22 < r <= 0.42)", "Outer Corners (r > 0.42)"],
    )
    spatial_df = spatial_df.copy()
    spatial_df["Screen_Zone"] = zones

    audit_rows = []
    for zone, group in spatial_df.groupby("Screen_Zone", observed=False):
        if len(group) == 0:
            continue
        audit_rows.append(
            {
                "Screen_Zone": str(zone),
                "Samples": len(group),
                "ZeroShot_Mean_mm": round(float(group["err_zero_mm"].mean()), 2),
                "Calibrated_k5_Mean_mm": round(float(group["err_cal_mm"].mean()), 2),
                "Min_Safe_Btn_Radius_ZeroShot_mm": round(float(np.percentile(group["err_zero_mm"], 95)), 2),
                "Min_Safe_Btn_Radius_Cal5_mm": round(float(np.percentile(group["err_cal_mm"], 95)), 2),
            }
        )
    return pd.DataFrame(audit_rows)
