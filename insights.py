"""
Rule-based auto-insights engine.

Takes a cleaned DataFrame plus the detected numeric/categorical/date columns
and produces a list of plain-English findings: missing data warnings,
strong correlations, outliers, trend direction, and top/bottom category
performance. No ML -- just thresholded statistics, so it's fast, transparent,
and works on any tabular dataset.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Insight:
    severity: str  # "info" | "warning" | "success"
    text: str


def missing_data_insights(df: pd.DataFrame, threshold: float = 0.05) -> list[Insight]:
    insights = []
    miss_pct = df.isna().mean().sort_values(ascending=False)
    flagged = miss_pct[miss_pct > threshold]
    for col, pct in flagged.items():
        insights.append(Insight(
            "warning",
            f"**{col}** has {pct:.1%} missing values ({df[col].isna().sum()} rows) — "
            f"consider imputing or excluding it from models."
        ))
    if flagged.empty and df.isna().sum().sum() == 0:
        insights.append(Insight("success", "No missing values detected across any column."))
    return insights


def correlation_insights(df: pd.DataFrame, numeric_cols: list[str], strong: float = 0.6) -> list[Insight]:
    insights = []
    if len(numeric_cols) < 2:
        return insights
    corr = df[numeric_cols].corr(numeric_only=True)
    seen = set()
    pairs = []
    for c1 in corr.columns:
        for c2 in corr.columns:
            if c1 == c2 or (c2, c1) in seen:
                continue
            seen.add((c1, c2))
            val = corr.loc[c1, c2]
            if pd.notna(val) and abs(val) >= strong:
                pairs.append((c1, c2, val))
    pairs.sort(key=lambda x: -abs(x[2]))
    for c1, c2, val in pairs[:5]:
        direction = "positive" if val > 0 else "negative"
        insights.append(Insight(
            "info",
            f"**{c1}** and **{c2}** show a strong {direction} correlation (r = {val:.2f})."
        ))
    return insights


def outlier_insights(df: pd.DataFrame, numeric_cols: list[str], iqr_multiplier: float = 3.0) -> list[Insight]:
    insights = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) > 0:
            pct = len(outliers) / len(series)
            insights.append(Insight(
                "warning",
                f"**{col}** has {len(outliers)} extreme outlier(s) ({pct:.1%} of rows), "
                f"e.g. values beyond [{lower:,.1f}, {upper:,.1f}]."
            ))
    return insights


def trend_insights(df: pd.DataFrame, date_col: str, metric_col: str) -> list[Insight]:
    insights = []
    if date_col is None or metric_col is None:
        return insights
    ts = df[[date_col, metric_col]].dropna().copy()
    if ts.empty:
        return insights
    ts["_period"] = ts[date_col].dt.to_period("M")
    monthly = ts.groupby("_period")[metric_col].sum()
    if len(monthly) < 2:
        return insights
    first_half = monthly.iloc[: len(monthly) // 2].mean()
    second_half = monthly.iloc[len(monthly) // 2:].mean()
    if first_half == 0:
        return insights
    pct_change = (second_half - first_half) / abs(first_half)
    if abs(pct_change) >= 0.05:
        direction = "grown" if pct_change > 0 else "declined"
        insights.append(Insight(
            "success" if pct_change > 0 else "warning",
            f"**{metric_col}** has {direction} by {abs(pct_change):.1%} comparing the "
            f"first half to the second half of the time period."
        ))

    # Best / worst single month
    best_month = monthly.idxmax()
    worst_month = monthly.idxmin()
    insights.append(Insight(
        "info",
        f"Peak month for **{metric_col}** was {best_month} ({monthly.max():,.0f}); "
        f"lowest was {worst_month} ({monthly.min():,.0f})."
    ))
    return insights


def categorical_breakdown_insights(df: pd.DataFrame, cat_col: str, metric_col: str) -> list[Insight]:
    insights = []
    if cat_col is None or metric_col is None:
        return insights
    grouped = df.groupby(cat_col)[metric_col].sum().sort_values(ascending=False)
    if grouped.empty or len(grouped) < 2:
        return insights
    total = grouped.sum()
    top_cat, top_val = grouped.index[0], grouped.iloc[0]
    bottom_cat, bottom_val = grouped.index[-1], grouped.iloc[-1]
    insights.append(Insight(
        "info",
        f"**{top_cat}** leads {cat_col} with {top_val:,.0f} total {metric_col} "
        f"({top_val/total:.1%} of the total)."
    ))
    insights.append(Insight(
        "info",
        f"**{bottom_cat}** is the lowest-performing {cat_col} with {bottom_val:,.0f} total {metric_col}."
    ))
    return insights


def skew_insights(df: pd.DataFrame, numeric_cols: list[str], threshold: float = 1.0) -> list[Insight]:
    insights = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        skew = series.skew()
        if abs(skew) >= threshold:
            direction = "right (long tail of high values)" if skew > 0 else "left (long tail of low values)"
            insights.append(Insight(
                "info",
                f"**{col}** is skewed {direction} (skewness = {skew:.2f})."
            ))
    return insights


def generate_all_insights(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    date_col: str | None,
    primary_metric: str | None,
) -> list[Insight]:
    insights: list[Insight] = []
    insights += missing_data_insights(df)
    insights += correlation_insights(df, numeric_cols)
    insights += outlier_insights(df, numeric_cols)
    if date_col and primary_metric:
        insights += trend_insights(df, date_col, primary_metric)
    if categorical_cols and primary_metric:
        insights += categorical_breakdown_insights(df, categorical_cols[0], primary_metric)
    insights += skew_insights(df, numeric_cols)
    return insights
