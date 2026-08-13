import json
import math
import warnings
from datetime import date, timedelta
from typing import Any, Optional

from arch import arch_model
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from app.config import DEFAULT_FORECAST_RANGE, RANGE_LABEL_TO_DAYS, ZERO_SHOT_MODEL_ID
from app.state import CommodityDataset


def safe_pct_change(current: float, previous: float) -> Optional[float]:
    if previous in (0.0, None) or np.isnan(previous) or np.isnan(current):
        return None
    return ((current / previous) - 1.0) * 100.0


def value_at_or_after(dates: list[date], values: np.ndarray, cutoff: date) -> Optional[float]:
    for idx, current_date in enumerate(dates):
        if current_date >= cutoff and not np.isnan(values[idx]):
            return float(values[idx])
    return None


def serialize_series(dates: list[date], values: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"date": current_date.isoformat(), "price": round(float(value), 6)}
        for current_date, value in zip(dates, values)
        if not np.isnan(value)
    ]


def rolling_volatility(returns_pct: np.ndarray, window: int = 30) -> list[Optional[float]]:
    result: list[Optional[float]] = [None] * len(returns_pct)
    for idx in range(window - 1, len(returns_pct)):
        window_values = returns_pct[idx - window + 1 : idx + 1]
        valid = window_values[~np.isnan(window_values)]
        result[idx] = round(float(np.std(valid, ddof=0)), 6) if len(valid) else None
    return result


def drawdown(values: np.ndarray) -> list[Optional[float]]:
    running_peak = -np.inf
    output: list[Optional[float]] = []
    for value in values:
        if np.isnan(value):
            output.append(None)
            continue
        running_peak = max(running_peak, float(value))
        if running_peak <= 0:
            output.append(None)
            continue
        output.append(round(((float(value) / running_peak) - 1.0) * 100.0, 6))
    return output


def infer_series_frequency(dates: list[date]) -> tuple[str, str, int]:
    if len(dates) < 3:
        return "calendar-daily", "D", 7
    deltas = np.diff(np.array([current.toordinal() for current in dates], dtype=int))
    deltas = deltas[deltas > 0]
    if not len(deltas):
        return "calendar-daily", "D", 7
    median_gap = float(np.median(deltas))
    if median_gap <= 1.5:
        return "calendar-daily", "D", 7
    if median_gap <= 3.5:
        return "business-daily", "B", 5
    if median_gap <= 10:
        return "weekly", "W", 1
    if median_gap <= 35:
        return "monthly", "MS", 1
    return "quarterly", "QS", 1


def forecast_horizon_label(alias: str, steps: int) -> str:
    if alias == "D":
        return f"{steps} days"
    if alias in {"D", "B"}:
        return f"{steps} sessions"
    if alias == "W":
        return f"{steps} weeks"
    if alias == "MS":
        return f"{steps} months"
    if alias == "QS":
        return f"{steps} quarters"
    return f"{steps} periods"


def series_arrays_from_points(points: list[dict[str, Any]]) -> tuple[list[date], np.ndarray]:
    dates = [date.fromisoformat(point["date"]) for point in points]
    values = np.array([float(point["price"]) for point in points], dtype=float)
    return dates, values


def filter_series_points_by_range(points: list[dict[str, Any]], range_label: str) -> list[dict[str, Any]]:
    days = RANGE_LABEL_TO_DAYS.get(range_label)
    if days is None or not points:
        return points
    last_date = date.fromisoformat(points[-1]["date"])
    cutoff = last_date - timedelta(days=days)
    filtered = [point for point in points if date.fromisoformat(point["date"]) >= cutoff]
    return filtered if filtered else points


def seasonal_period_for_alias(alias: str, series_length: int) -> Optional[int]:
    candidates = {"D": 7, "B": 5, "W": 52, "MS": 12, "QS": 4}
    seasonal_period = candidates.get(alias)
    if seasonal_period is None:
        return None
    return seasonal_period if series_length >= seasonal_period * 2 else None


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    aligned_actual, aligned_pred = actual.align(predicted, join="inner")
    if len(aligned_actual) == 0:
        return float("inf")
    diff = aligned_actual.astype(float) - aligned_pred.astype(float)
    return float(np.sqrt(np.mean(np.square(diff))))


def rolling_residual_std(series: pd.Series) -> float:
    diffs = series.diff().dropna()
    if len(diffs) < 5:
        return max(abs(float(series.iloc[-1])) * 0.01, 0.01)
    return max(float(diffs.std(ddof=0)), 0.01)


def forecast_naive(train: pd.Series, future_index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series([float(train.iloc[-1])] * len(future_index), index=future_index)


def forecast_drift(train: pd.Series, future_index: pd.DatetimeIndex) -> pd.Series:
    if len(train) < 2:
        return forecast_naive(train, future_index)
    step = (float(train.iloc[-1]) - float(train.iloc[0])) / max(len(train) - 1, 1)
    values = [float(train.iloc[-1]) + step * idx for idx in range(1, len(future_index) + 1)]
    return pd.Series(values, index=future_index)


def forecast_ets(train: pd.Series, future_index: pd.DatetimeIndex, seasonal: Optional[str], damped: bool, seasonal_periods: Optional[int]) -> Optional[pd.Series]:
    if seasonal and not seasonal_periods:
        return None
    if len(train) < 12:
        return None
    if seasonal == "mul" and float(train.min()) <= 0:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(
                train.astype(float),
                trend="add",
                damped_trend=damped,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
        output = fitted.forecast(len(future_index))
        output.index = future_index
        return output.astype(float)
    except Exception:
        return None


def fit_best_arima(train: pd.Series) -> tuple[Optional[Any], Optional[tuple[int, int, int]], Optional[float]]:
    volatility = float(train.pct_change().dropna().std()) if len(train) > 2 else 0.0
    d_candidates = [0, 1, 2] if volatility > 0.018 else [0, 1]
    p_candidates = [0, 1, 2]
    q_candidates = [0, 1, 2]
    best_fit = None
    best_order = None
    best_aic = None
    for d_value in d_candidates:
        for p_value in p_candidates:
            for q_value in q_candidates:
                if p_value == 0 and d_value == 0 and q_value == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        fitted = ARIMA(
                            train,
                            order=(p_value, d_value, q_value),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit()
                    if not bool(getattr(fitted, "mle_retvals", {}).get("converged", True)):
                        continue
                    aic = float(fitted.aic)
                    if np.isfinite(aic) and (best_aic is None or aic < best_aic):
                        best_fit = fitted
                        best_order = (p_value, d_value, q_value)
                        best_aic = aic
                except Exception:
                    continue
    return best_fit, best_order, best_aic


def forecast_arima(train: pd.Series, future_index: pd.DatetimeIndex) -> tuple[Optional[pd.Series], Optional[tuple[int, int, int]], Optional[float]]:
    if len(train) < 25:
        return None, None, None
    fitted, order, aic = fit_best_arima(train.astype(float))
    if fitted is None:
        return None, None, None
    try:
        prediction = fitted.forecast(steps=len(future_index))
        prediction.index = future_index
        return prediction.astype(float), order, aic
    except Exception:
        return None, None, None


def forecast_arch_family(train: pd.Series, future_index: pd.DatetimeIndex, vol: str) -> tuple[Optional[pd.Series], Optional[float]]:
    if len(train) < 60:
        return None, None
    returns = (100.0 * train.pct_change()).dropna()
    if len(returns) < 40:
        return None, None
    fit_returns = returns.iloc[-min(len(returns), 500):].astype(float)
    mean_type = "ARX" if len(fit_returns) >= 80 else "Constant"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = arch_model(
                fit_returns,
                mean=mean_type,
                lags=1 if mean_type == "ARX" else 0,
                vol=vol,
                p=1,
                q=1 if vol == "GARCH" else 0,
                dist="t",
                rescale=False,
            ).fit(disp="off", show_warning=False)
        forecast = fitted.forecast(horizon=len(future_index), reindex=False)
        mean_returns = np.asarray(forecast.mean.iloc[-1], dtype=float)
        if mean_returns.size != len(future_index):
            return None, None
        level_values = []
        current = float(train.iloc[-1])
        for mean_ret in mean_returns:
            current = current * (1.0 + (float(mean_ret) / 100.0))
            level_values.append(current)
        aic = float(fitted.aic) if np.isfinite(getattr(fitted, "aic", np.nan)) else None
        return pd.Series(level_values, index=future_index), aic
    except Exception:
        return None, None


def evaluate_candidates(series: pd.Series, future_index: pd.DatetimeIndex, pandas_alias: str) -> dict[str, Any]:
    horizon = len(future_index)
    holdout = max(horizon, 3)
    max_origins = min(2, max(1, (len(series) - 20) // holdout))
    residual_std = rolling_residual_std(series)
    seasonal_period = seasonal_period_for_alias(pandas_alias, len(series))
    recent_volatility = float(series.pct_change().dropna().iloc[-120:].std()) if len(series) > 20 else 0.0

    candidates: list[dict[str, Any]] = []
    candidate_defs = [("Naive", "naive"), ("Drift", "drift"), ("Holt Damped", "holt_damped"), ("ARIMA", "arima")]
    if pandas_alias in {"D", "B"} and recent_volatility >= 0.015:
        candidate_defs.extend([("ARCH", "arch"), ("GARCH", "garch")])
    if seasonal_period:
        candidate_defs.extend([("ETS Additive", "ets_add"), ("ETS Multiplicative", "ets_mul")])

    def forecast_by_kind(kind: str, train: pd.Series, idx: pd.DatetimeIndex) -> tuple[Optional[pd.Series], Optional[list[int]], Optional[float]]:
        if kind == "naive":
            return forecast_naive(train, idx), None, None
        if kind == "drift":
            return forecast_drift(train, idx), None, None
        if kind == "holt_damped":
            return forecast_ets(train, idx, seasonal=None, damped=True, seasonal_periods=None), None, None
        if kind == "ets_add":
            return forecast_ets(train, idx, seasonal="add", damped=True, seasonal_periods=seasonal_period), None, None
        if kind == "ets_mul":
            return forecast_ets(train, idx, seasonal="mul", damped=True, seasonal_periods=seasonal_period), None, None
        if kind == "arch":
            forecast, aic = forecast_arch_family(train, idx, "ARCH")
            return forecast, None, aic
        if kind == "garch":
            forecast, aic = forecast_arch_family(train, idx, "GARCH")
            return forecast, None, aic
        forecast, order, aic = forecast_arima(train, idx)
        return forecast, list(order) if order else None, aic

    for name, kind in candidate_defs:
        errors = []
        last_order = None
        last_aic = None
        for origin in range(max_origins, 0, -1):
            split_idx = len(series) - origin * holdout
            if split_idx < 20:
                continue
            train = series.iloc[:split_idx]
            test = series.iloc[split_idx: split_idx + holdout]
            pred, order, aic = forecast_by_kind(kind, train, test.index)
            if pred is None or len(pred) != len(test):
                continue
            errors.append(rmse(test, pred))
            if order is not None:
                last_order = order
            if aic is not None and np.isfinite(aic):
                last_aic = round(float(aic), 3)
        if errors:
            candidates.append({"name": name, "kind": kind, "score_rmse": float(np.mean(errors)), "order": last_order, "aic": last_aic})

    if not candidates:
        final_forecast = forecast_naive(series, future_index)
        return {
            "best_name": "Naive",
            "best_kind": "naive",
            "best_order": None,
            "best_aic": None,
            "best_rmse": None,
            "forecast": final_forecast,
            "residual_std": residual_std,
            "candidate_scores": [],
        }

    best = min(candidates, key=lambda item: item["score_rmse"])
    naive_candidate = next((item for item in candidates if item["kind"] == "naive"), None)
    if naive_candidate and best["kind"] != "naive":
        improvement = (naive_candidate["score_rmse"] - best["score_rmse"]) / naive_candidate["score_rmse"]
        if improvement < 0.005:
            best = naive_candidate

    final_forecast, _, _ = forecast_by_kind(best["kind"], series, future_index)
    if final_forecast is None:
        final_forecast = forecast_naive(series, future_index)

    return {
        "best_name": best["name"],
        "best_kind": best["kind"],
        "best_order": best["order"],
        "best_aic": best["aic"],
        "best_rmse": round(best["score_rmse"], 6) if np.isfinite(best["score_rmse"]) else None,
        "forecast": final_forecast,
        "residual_std": residual_std,
        "candidate_scores": [
            {"model": candidate["name"], "rmse": round(candidate["score_rmse"], 6) if np.isfinite(candidate["score_rmse"]) else None}
            for candidate in sorted(candidates, key=lambda item: item["score_rmse"])
        ],
    }


def build_forecast(dates: list[date], values: np.ndarray) -> dict[str, Any]:
    frequency_name, pandas_alias, horizon = infer_series_frequency(dates)
    series_index = pd.DatetimeIndex(pd.to_datetime([current.isoformat() for current in dates]))
    series = pd.Series(values, index=series_index).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    regular = series.asfreq(pandas_alias).interpolate(method="time").ffill().bfill()

    if regular.isna().all() or len(regular) < 15:
        return {
            "model": "unavailable",
            "frequency": frequency_name,
            "frequency_alias": pandas_alias,
            "horizon_steps": horizon,
            "horizon_label": forecast_horizon_label(pandas_alias, horizon),
            "order": None,
            "aic": None,
            "history": [],
            "forecast": [],
            "fallback_used": True,
        }

    fit_series = regular.iloc[-min(len(regular), 720):].astype(float)
    future_index = pd.date_range(
        fit_series.index[-1] + pd.tseries.frequencies.to_offset(pandas_alias),
        periods=horizon,
        freq=pandas_alias,
    )
    selection = evaluate_candidates(fit_series, future_index, pandas_alias)
    final_forecast = selection["forecast"]
    residual_std = float(selection["residual_std"])
    forecast_points = []
    for step_idx, (stamp, estimate) in enumerate(final_forecast.items(), start=1):
        band = residual_std * math.sqrt(step_idx)
        forecast_points.append(
            {
                "date": stamp.date().isoformat(),
                "value": round(float(estimate), 6),
                "lower": round(float(estimate - 1.96 * band), 6),
                "upper": round(float(estimate + 1.96 * band), 6),
            }
        )

    history = [{"date": stamp.date().isoformat(), "value": round(float(value), 6)} for stamp, value in fit_series.iloc[-120:].items()]
    latest_value = float(fit_series.iloc[-1])
    forecast_end = float(forecast_points[-1]["value"]) if forecast_points else latest_value
    implied_change = safe_pct_change(forecast_end, latest_value)
    return {
        "model": selection["best_name"],
        "model_kind": selection["best_kind"],
        "frequency": frequency_name,
        "frequency_alias": pandas_alias,
        "horizon_steps": horizon,
        "horizon_label": forecast_horizon_label(pandas_alias, horizon),
        "order": selection["best_order"],
        "aic": selection["best_aic"],
        "validation_rmse": selection["best_rmse"],
        "candidate_scores": selection["candidate_scores"],
        "history": history,
        "forecast": forecast_points,
        "fallback_used": selection["best_kind"] == "drift" and not selection["candidate_scores"],
        "implied_change_pct": round(implied_change, 4) if implied_change is not None else None,
    }


def build_forecast_for_range(series_points: list[dict[str, Any]], range_label: str) -> dict[str, Any]:
    filtered_points = filter_series_points_by_range(series_points, range_label)
    dates, values = series_arrays_from_points(filtered_points)
    forecast = build_forecast(dates, values)
    forecast["training_range"] = range_label
    forecast["training_points"] = len(filtered_points)
    forecast["window_start"] = filtered_points[0]["date"] if filtered_points else None
    forecast["window_end"] = filtered_points[-1]["date"] if filtered_points else None
    return forecast


def apply_news_risk_to_forecast(forecast: dict[str, Any], news_context: dict[str, Any]) -> dict[str, Any]:
    adjusted = json.loads(json.dumps(forecast))
    risk_score = float(news_context.get("risk_score", 0.0) or 0.0)
    event_score = float(news_context.get("event_score", 0.0) or 0.0)
    tone_score = float(news_context.get("tone_score", news_context.get("sentiment_score", 0.0)) or 0.0)
    multiplier = 1.0 + min(risk_score, 3.0) * 0.2
    event_shift_pct = event_score * (0.9 + min(risk_score, 2.0) * 0.28)
    adjusted_points = adjusted.get("forecast", [])
    for idx, point in enumerate(adjusted_points, start=1):
        center = float(point["value"])
        lower = float(point["lower"])
        upper = float(point["upper"])
        decay = max(0.45, 1.0 - (idx - 1) * 0.1)
        shifted_center = center * (1.0 + (event_shift_pct / 100.0) * decay)
        lower_radius = center - lower
        upper_radius = upper - center
        point["value"] = round(shifted_center, 6)
        point["lower"] = round(shifted_center - lower_radius * multiplier, 6)
        point["upper"] = round(shifted_center + upper_radius * multiplier, 6)
    if adjusted_points:
        latest_value = adjusted_points[-1]["value"]
        start_value = adjusted.get("history", [])[-1]["value"] if adjusted.get("history") else latest_value
        implied_change = safe_pct_change(float(latest_value), float(start_value))
        adjusted["implied_change_pct"] = round(implied_change, 4) if implied_change is not None else adjusted.get("implied_change_pct")
    adjusted["news_risk_level"] = news_context.get("risk_level", "unknown")
    adjusted["news_risk_score"] = news_context.get("risk_score", 0.0)
    adjusted["news_sentiment_score"] = news_context.get("sentiment_score", 0.0)
    adjusted["news_sentiment_label"] = news_context.get("sentiment_label", "neutral")
    adjusted["news_tone_score"] = round(tone_score, 3)
    adjusted["news_tone_label"] = news_context.get("tone_label", news_context.get("sentiment_label", "neutral"))
    adjusted["news_event_score"] = round(event_score, 3)
    adjusted["news_event_label"] = news_context.get("event_label", "neutral")
    adjusted["news_transformer_score"] = news_context.get("transformer_score")
    adjusted["news_transformer_available"] = news_context.get("transformer_available", False)
    adjusted["news_transformer_model"] = news_context.get("transformer_model", ZERO_SHOT_MODEL_ID)
    adjusted["news_transformer_applied"] = news_context.get("transformer_applied", False)
    adjusted["news_headlines_count"] = len(news_context.get("items", []))
    adjusted["news_adjustment_pct"] = round(event_shift_pct, 3)
    adjusted["news_adjustment_applied"] = bool(news_context.get("items"))
    return adjusted


def compute_dataset(meta: dict[str, Any], rows: list[tuple[date, float]]) -> CommodityDataset:
    rows = sorted((row for row in rows if not np.isnan(row[1])), key=lambda item: item[0])
    dates = [row[0] for row in rows]
    values = np.array([row[1] for row in rows], dtype=float)
    if not dates:
        raise ValueError(f"No data rows found for {meta['name']}.")

    returns_pct = np.full(len(values), np.nan)
    if len(values) > 1:
        previous = values[:-1]
        current = values[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            returns_pct[1:] = ((current / previous) - 1.0) * 100.0

    latest_date = dates[-1]
    latest_value = float(values[-1])
    high_idx = int(np.nanargmax(values))
    low_idx = int(np.nanargmin(values))
    ytd_start = value_at_or_after(dates, values, date(latest_date.year, 1, 1))
    one_year_start = value_at_or_after(dates, values, latest_date - timedelta(days=365))
    trailing_year_mask = np.array([current_date >= latest_date - timedelta(days=365) for current_date in dates])
    trailing_year_values = values[trailing_year_mask]

    stats = {
        "current_price": round(latest_value, 6),
        "latest_date": latest_date.isoformat(),
        "52_week_high": round(float(np.nanmax(trailing_year_values)), 6) if trailing_year_values.size else None,
        "52_week_low": round(float(np.nanmin(trailing_year_values)), 6) if trailing_year_values.size else None,
        "all_time_high": round(float(np.nanmax(values)), 6),
        "all_time_high_date": dates[high_idx].isoformat(),
        "all_time_low": round(float(np.nanmin(values)), 6),
        "all_time_low_date": dates[low_idx].isoformat(),
        "pct_change_ytd": round(safe_pct_change(latest_value, ytd_start), 4) if ytd_start is not None else None,
        "pct_change_1y": round(safe_pct_change(latest_value, one_year_start), 4) if one_year_start is not None else None,
        "points": len(rows),
        "start_date": dates[0].isoformat(),
    }
    analytics = {
        "returns": [{"date": current_date.isoformat(), "value": round(float(ret), 6)} for current_date, ret in zip(dates, returns_pct) if not np.isnan(ret)],
        "rolling_volatility_30d": [{"date": current_date.isoformat(), "value": value} for current_date, value in zip(dates, rolling_volatility(returns_pct)) if value is not None],
        "drawdown": [{"date": current_date.isoformat(), "value": value} for current_date, value in zip(dates, drawdown(values)) if value is not None],
    }
    series = serialize_series(dates, values)
    return CommodityDataset(meta=meta, series=series, stats=stats, analytics=analytics, forecast=build_forecast_for_range(series, DEFAULT_FORECAST_RANGE))
