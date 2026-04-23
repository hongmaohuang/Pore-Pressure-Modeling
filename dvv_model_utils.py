import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


def to_elapsed_seconds(index):
    index = pd.to_datetime(index)
    return (index - index[0]).total_seconds().astype(float)


def fit_annual_harmonic(index, values, period_s=SECONDS_PER_YEAR):
    values = np.asarray(values, dtype=float)
    t_s = to_elapsed_seconds(index)
    omega = 2.0 * np.pi / period_s
    X = np.column_stack(
        [
            np.ones(len(values), dtype=float),
            np.cos(omega * t_s),
            np.sin(omega * t_s),
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(X, values, rcond=None)
    mean_value, a_cos, b_sin = coeffs
    amplitude = float(np.hypot(a_cos, b_sin))
    phase_rad = float(np.arctan2(b_sin, a_cos))
    fitted = X @ coeffs
    return {
        "mean": float(mean_value),
        "a_cos": float(a_cos),
        "b_sin": float(b_sin),
        "amplitude": amplitude,
        "phase_rad": phase_rad,
        "omega": float(omega),
        "fitted": fitted,
    }


def harmonic_phase_to_day(phase_rad, period_days=365.25):
    return float((phase_rad / (2.0 * np.pi)) * period_days)


def fit_linear_model(X, y, feature_names, fit_intercept=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if fit_intercept:
        X_design = np.column_stack([np.ones(len(X), dtype=float), X])
        names = ["intercept"] + list(feature_names)
    else:
        X_design = X
        names = list(feature_names)
    coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ coeffs
    residuals = y - y_hat
    return {
        "feature_names": names,
        "coefficients": coeffs,
        "predicted": y_hat,
        "residuals": residuals,
        "n_parameters": int(len(coeffs)),
    }


def predict_linear_model(X, coefficients, fit_intercept=True):
    X = np.asarray(X, dtype=float)
    coeffs = np.asarray(coefficients, dtype=float)
    if fit_intercept:
        X_design = np.column_stack([np.ones(len(X), dtype=float), X])
    else:
        X_design = X
    return X_design @ coeffs


def compute_information_criteria(y_true, y_pred, n_parameters):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    n_obs = len(y_true)
    rss = float(np.sum(residuals ** 2))
    if n_obs == 0:
        return {"rss": np.nan, "aic": np.nan, "bic": np.nan}
    rss = max(rss, np.finfo(float).eps)
    aic = n_obs * np.log(rss / n_obs) + 2.0 * n_parameters
    bic = n_obs * np.log(rss / n_obs) + np.log(n_obs) * n_parameters
    return {"rss": rss, "aic": float(aic), "bic": float(bic)}


def compute_basic_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "corr": corr, "r2": r2}


def rolling_origin_cross_validation(X, y, feature_names, fit_intercept=True, n_splits=5):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_obs = len(y)
    test_size = max(1, n_obs // (n_splits + 1))
    results = []
    for split_idx in range(n_splits):
        train_end = test_size * (split_idx + 1)
        test_end = min(train_end + test_size, n_obs)
        if test_end <= train_end or train_end < 2:
            continue
        fit = fit_linear_model(
            X[:train_end],
            y[:train_end],
            feature_names=feature_names,
            fit_intercept=fit_intercept,
        )
        y_test_pred = predict_linear_model(
            X[train_end:test_end],
            coefficients=fit["coefficients"],
            fit_intercept=fit_intercept,
        )
        metrics = compute_basic_metrics(y[train_end:test_end], y_test_pred)
        results.append(
            {
                "split": split_idx + 1,
                "train_size": int(train_end),
                "test_size": int(test_end - train_end),
                **metrics,
            }
        )
    return pd.DataFrame(results)
