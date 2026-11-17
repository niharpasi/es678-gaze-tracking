import numpy as np
import pandas as pd


def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes:
      1. X_raw: Raw 2D pupil pixel coordinates (4D) for the unregularized OLS baseline.
      2. X_geom: Scale-, translation-, and roll-normalized binocular geometric features + 3D head pose (20D).
    """
    landmarks = np.stack(df["landmarks"].values)
    head_pose = np.stack(df["head_pose"].values)
    n = landmarks.shape[0]

    X_raw = landmarks[:, [4, 5, 10, 11]].copy()
    X_geom = np.zeros((n, 20), dtype=np.float64)

    for i in range(n):
        lm = landmarks[i]
        l_outer = lm[0:2]
        l_inner = lm[2:4]
        l_pupil = lm[4:6]
        r_inner = lm[6:8]
        r_outer = lm[8:10]
        r_pupil = lm[10:12]

        inter_ocular_vec = r_outer - l_outer
        scale_base = np.linalg.norm(inter_ocular_vec) + 1e-8
        roll_angle = np.arctan2(inter_ocular_vec[1], inter_ocular_vec[0])
        c, s = np.cos(-roll_angle), np.sin(-roll_angle)
        R_align = np.array([[c, -s], [s, c]])

        l_center = 0.5 * (l_outer + l_inner)
        r_center = 0.5 * (r_outer + r_inner)

        l_width = np.linalg.norm(l_inner - l_outer) + 1e-8
        r_width = np.linalg.norm(r_outer - r_inner) + 1e-8

        l_disp = (R_align @ (l_pupil - l_center)) / l_width
        r_disp = (R_align @ (r_pupil - r_center)) / r_width

        l_disp_global = (R_align @ (l_pupil - l_center)) / scale_base
        r_disp_global = (R_align @ (r_pupil - r_center)) / scale_base

        binoc_mean = 0.5 * (l_disp + r_disp)
        binoc_diff = l_disp - r_disp

        # Normalized eye position in camera frame (captures vertical/horizontal head shift)
        mid_eyes = 0.5 * (l_center + r_center) / scale_base

        hp = head_pose[i]  # 6D: 3D rotation (rodrigues pitch/yaw/roll) + 3D translation (tx, ty, tz)

        X_geom[i] = [
            l_disp[0], l_disp[1],
            r_disp[0], r_disp[1],
            l_disp_global[0], l_disp_global[1],
            r_disp_global[0], r_disp_global[1],
            binoc_mean[0], binoc_mean[1],
            binoc_diff[0], binoc_diff[1],
            mid_eyes[0], mid_eyes[1],
            hp[0], hp[1], hp[2],
            hp[3] * 0.01, hp[4] * 0.01, hp[5] * 0.001,
        ]

    return X_raw, X_geom
