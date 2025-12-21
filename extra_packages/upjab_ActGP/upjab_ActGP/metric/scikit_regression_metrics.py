import numpy as np

from sklearn.metrics import (
    explained_variance_score,
    r2_score,
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_absolute_percentage_error,
    max_error,
    mean_pinball_loss,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
)

# Optional root metrics (newer sklearn). We'll compute fallbacks if missing.
try:
    from sklearn.metrics import root_mean_squared_error
except Exception:
    root_mean_squared_error = None

try:
    from sklearn.metrics import root_mean_squared_log_error
except Exception:
    root_mean_squared_log_error = None


def _as_1d_float(x):
    a = np.asarray(x, dtype=float).ravel()
    if a.ndim != 1:
        raise ValueError("Inputs must be 1D.")
    return a


def _safe(metric_fn, *args, **kwargs):
    """Call a metric and return np.nan if it errors (domain/shape issues)."""
    try:
        return float(metric_fn(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_sklearn_regression_metrics(
    y_true,
    y_pred,
    *,
    pinball_alpha=0.5,          # for mean_pinball_loss & d2_pinball_score
    tweedie_power=0.0,          # for d2_tweedie_score & mean_tweedie_deviance
    sample_weight=None
):
    """
    y_true, y_pred: 1D lists/arrays of equal length.

    Notes on domains:
    - MSLE/RMSLE require y_true >= 0 and y_pred >= 0.
    - Poisson deviance: y_true >= 0, y_pred > 0.
    - Gamma deviance:   y_true > 0,  y_pred > 0.
    - Tweedie deviance: depends on power; power=0 (Normal) has no nonneg. constraint.
    - Pinball & D2 pinball use 'alpha' (quantile). With point preds, alpha=0.5 (median) is typical.
    """
    y_true = _as_1d_float(y_true)
    y_pred = _as_1d_float(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    metrics = {}

    # Variance / goodness-of-fit
    metrics["explained_variance_score"] = _safe(explained_variance_score, y_true, y_pred, sample_weight=sample_weight)
    metrics["r2_score"] = _safe(r2_score, y_true, y_pred, sample_weight=sample_weight)
    metrics["d2_absolute_error_score"] = _safe(d2_absolute_error_score, y_true, y_pred, sample_weight=sample_weight)
    metrics["d2_pinball_score"] = _safe(d2_pinball_score, y_true, y_pred, alpha=pinball_alpha, sample_weight=sample_weight)
    metrics["d2_tweedie_score"] = _safe(d2_tweedie_score, y_true, y_pred, power=tweedie_power, sample_weight=sample_weight)

    # Error metrics
    metrics["mean_absolute_error"] = _safe(mean_absolute_error, y_true, y_pred, sample_weight=sample_weight)
    metrics["median_absolute_error"] = _safe(median_absolute_error, y_true, y_pred)
    # metrics["mean_squared_error"] = _safe(mean_squared_error, y_true, y_pred, sample_weight=sample_weight, squared=True)
    metrics["mean_squared_error"] = _safe(mean_squared_error, y_true, y_pred, sample_weight=sample_weight)
    metrics["rmse"] = (
        _safe(root_mean_squared_error, y_true, y_pred, sample_weight=sample_weight)
        if root_mean_squared_error is not None
        else _safe(mean_squared_error, y_true, y_pred, sample_weight=sample_weight, squared=False)
    )
    metrics["mean_squared_log_error"] = _safe(mean_squared_log_error, y_true, y_pred, sample_weight=sample_weight)
    if root_mean_squared_log_error is not None:
        metrics["rmsle"] = _safe(root_mean_squared_log_error, y_true, y_pred, sample_weight=sample_weight)
    else:
        msle_val = metrics["mean_squared_log_error"]
        metrics["rmsle"] = float(np.sqrt(msle_val)) if np.isfinite(msle_val) else float("nan")
    metrics["mean_absolute_percentage_error"] = _safe(mean_absolute_percentage_error, y_true, y_pred, sample_weight=sample_weight)
    metrics["max_error"] = _safe(max_error, y_true, y_pred)

    # Quantile / Pinball
    metrics["mean_pinball_loss(alpha={})".format(pinball_alpha)] = _safe(
        mean_pinball_loss, y_true, y_pred, alpha=pinball_alpha, sample_weight=sample_weight
    )

    # Deviance family
    metrics["mean_poisson_deviance"] = _safe(mean_poisson_deviance, y_true, y_pred, sample_weight=sample_weight)
    metrics["mean_gamma_deviance"] = _safe(mean_gamma_deviance, y_true, y_pred, sample_weight=sample_weight)
    metrics["mean_tweedie_deviance(power={})".format(tweedie_power)] = _safe(
        mean_tweedie_deviance, y_true, y_pred, power=tweedie_power, sample_weight=sample_weight
    )

    return metrics


# --- Example ---
if __name__ == "__main__":
    y_true = [3.0, 5.0, 2.5, 7.0, 10.0]
    y_pred = [2.5, 5.1, 2.0, 8.0,  9.5]

    out = compute_sklearn_regression_metrics(
        y_true, y_pred,
        pinball_alpha=0.5,   # median
        tweedie_power=0.0    # Normal
    )
    for k, v in out.items():
        print(f"{k}: {v}")


    from upjab_ActGP.metric.MAPE import MAPE
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(f"MAPE: {MAPE(y_pred, y_true)}%")