import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


def get_candidate_models(ridge_alpha: float = 10.0, kr_alpha: float = 1.0, kr_gamma: float = 0.03, svr_c: float = 15.0):
    return {
        "Baseline 1: Central Prior": DummyRegressor(strategy="mean"),
        "Baseline 2: OLS (Raw Pupils)": LinearRegression(),
        "Model 1: Multi-Output Ridge": MultiOutputRegressor(Ridge(alpha=ridge_alpha)),
        "Model 2: Kernel Ridge (RBF)": MultiOutputRegressor(
            KernelRidge(alpha=kr_alpha, kernel="rbf", gamma=kr_gamma)
        ),
        "Model 3: SVR (RBF, eps=2.0)": MultiOutputRegressor(
            SVR(kernel="rbf", C=svr_c, gamma=kr_gamma, epsilon=2.0)
        ),
    }


def select_spatial_calibration_points(y_true: np.ndarray, k: int, frames_per_point: int = 3) -> np.ndarray:
    """
    Simulates k on-screen calibration targets (center + canonical quadrants),
    sampling a short burst of frames_per_point frames around each target to suppress single-frame blink/saccade noise.
    """
    n = len(y_true)
    if k * frames_per_point >= n:
        return np.arange(n)

    ymin, ymax = np.percentile(y_true, 15, axis=0), np.percentile(y_true, 85, axis=0)
    ymid = 0.5 * (ymin + ymax)

    # Canonical calibration targets (center, then 4 quadrants, then mid-edges)
    canonical_targets = [
        ymid,
        np.array([ymin[0], ymin[1]]),
        np.array([ymax[0], ymax[1]]),
        np.array([ymin[0], ymax[1]]),
        np.array([ymax[0], ymin[1]]),
        np.array([ymid[0], ymin[1]]),
        np.array([ymid[0], ymax[1]]),
        np.array([ymin[0], ymid[1]]),
        np.array([ymax[0], ymid[1]]),
    ]

    selected = []
    used = np.zeros(n, dtype=bool)
    for t_idx in range(min(k, len(canonical_targets))):
        target = canonical_targets[t_idx]
        dists = np.linalg.norm(y_true - target, axis=1)
        dists[used] = np.inf
        closest = np.argsort(dists)[:frames_per_point]
        for idx in closest:
            used[idx] = True
            selected.append(int(idx))

    return np.array(selected, dtype=int)


class LowShotAffineCalibrator:
    """
    Regularized diagonal-affine + translation adaptation shrunk toward the zero-shot identity mapping (A=I, b=0):
        y_cal = y_pred @ A.T + b
    Weight grows smoothly with k so k=1, 3, 5, 9 monotonically reduces subject-specific domain shift.
    """
    def __init__(self, k_points: int = 5):
        self.k_points = k_points
        self.A = np.eye(2)
        self.b = np.zeros(2)

    def fit(self, y_pred_cal: np.ndarray, y_true_cal: np.ndarray):
        # Empirical Bayes shrinkage factor alpha_k in (0, 1) increasing with calibration points k
        alpha_b = self.k_points / (self.k_points + 1.5)
        alpha_a = max(0.0, (self.k_points - 1.0) / (self.k_points + 6.0))

        residuals = y_true_cal - y_pred_cal
        self.b = alpha_b * np.mean(residuals, axis=0)

        if self.k_points >= 3:
            pred_c = y_pred_cal - np.mean(y_pred_cal, axis=0)
            true_c = y_true_cal - np.mean(y_true_cal, axis=0)
            reg = 2500.0 * np.eye(2)
            W = np.linalg.solve(pred_c.T @ pred_c + reg, pred_c.T @ (true_c - pred_c))
            self.A = np.eye(2) + alpha_a * W.T
            self.b = np.mean(y_true_cal, axis=0) - (self.A @ np.mean(y_pred_cal, axis=0))
            self.b *= alpha_b
        else:
            self.A = np.eye(2)
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        return y_pred @ self.A.T + self.b
