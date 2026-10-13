"""Data preprocessing helpers.

Every function is pure (takes a DataFrame, returns a new DataFrame) so it can be
tested without Streamlit. The Streamlit page lives in pages/7_Data_Preprocessing.py.
"""

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


DEFAULT_MISSING_TOKENS = ["?", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "none", "-", ""]

MISSING_METHODS = [
    "Mean",
    "Median",
    "Mode (most frequent)",
    "Constant value",
    "Drop rows with missing values",
    "Drop columns with missing values",
]


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def numeric_cols(df):
    return [c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]


def categorical_cols(df):
    return [
        c for c in df.columns
        if not is_numeric_dtype(df[c]) or is_bool_dtype(df[c])
        if not is_datetime64_any_dtype(df[c])
    ]


# ---------------------------------------------------------------------------
# 1. Placeholder clean-up ("?", "NA", ... -> real missing values)
# ---------------------------------------------------------------------------

def standardize_missing(df, tokens, convert_numeric=True):
    out = df.copy()
    before = int(out.isna().sum().sum())
    converted = []

    for column in categorical_cols(out):
        series = out[column].astype(object)
        series = series.map(lambda v: v.strip() if isinstance(v, str) else v)
        series = series.where(~series.isin(tokens), np.nan)
        out[column] = series

        if convert_numeric:
            non_null = int(series.notna().sum())
            if non_null and not is_bool_dtype(df[column]):
                numeric = pd.to_numeric(series, errors="coerce")
                # Only convert when *every* real value is numeric, so no text is lost.
                if int(numeric.notna().sum()) == non_null:
                    out[column] = numeric
                    converted.append(column)

    after = int(out.isna().sum().sum())
    message = f"Standardized placeholders: {after - before:,} new missing cell(s)."
    if converted:
        message += f" Converted to numeric: {', '.join(map(str, converted))}."
    return out, message


# ---------------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------------

def handle_missing(df, columns, method, constant=None):
    out = df.copy()
    columns = [c for c in columns if c in out.columns]
    skipped = []

    if method == "Drop rows with missing values":
        n = len(out)
        out = out.dropna(subset=columns)
        return out, f"Dropped {n - len(out):,} row(s) with missing values in {len(columns)} column(s)."

    if method == "Drop columns with missing values":
        to_drop = [c for c in columns if out[c].isna().any()]
        out = out.drop(columns=to_drop)
        return out, f"Dropped {len(to_drop)} column(s) containing missing values."

    filled = 0
    for column in columns:
        missing = int(out[column].isna().sum())
        if missing == 0:
            continue

        if method in ("Mean", "Median"):
            if not is_numeric_dtype(out[column]) or is_bool_dtype(out[column]):
                skipped.append(str(column))
                continue
            value = out[column].mean() if method == "Mean" else out[column].median()
        elif method == "Mode (most frequent)":
            modes = out[column].mode(dropna=True)
            if modes.empty:
                skipped.append(str(column))
                continue
            value = modes.iloc[0]
        elif method == "Constant value":
            value = constant
            if is_numeric_dtype(out[column]) and not is_bool_dtype(out[column]):
                try:
                    value = float(constant)
                except (TypeError, ValueError):
                    raise ValueError(f"'{constant}' is not a number, but column '{column}' is numeric.")
        else:
            raise ValueError(f"Unknown method: {method}")

        out[column] = out[column].fillna(value)
        filled += missing

    message = f"Filled {filled:,} missing cell(s) using {method.lower()}."
    if skipped:
        message += f" Skipped (not applicable): {', '.join(skipped)}."
    return out, message


# ---------------------------------------------------------------------------
# 3. Duplicates
# ---------------------------------------------------------------------------

KEEP_OPTIONS = {
    "Keep first occurrence": "first",
    "Keep last occurrence": "last",
    "Remove all copies": False,
}


def duplicate_mask(df, subset=None, keep="first"):
    return df.duplicated(subset=subset or None, keep=keep)


def remove_duplicates(df, subset=None, keep="first"):
    out = df.drop_duplicates(subset=subset or None, keep=keep)
    return out, f"Removed {len(df) - len(out):,} duplicate row(s)."


# ---------------------------------------------------------------------------
# 4. Encoding
# ---------------------------------------------------------------------------

def encode_columns(df, columns, method, drop_first=False):
    """Returns (DataFrame, mappings) - mappings is {column: {code: category}} for label encoding."""
    out = df.copy()
    columns = [c for c in columns if c in out.columns]
    mappings = {}

    if method == "Label encoding":
        for column in columns:
            category = out[column].astype("category")
            codes = category.cat.codes
            mappings[column] = {int(i): str(v) for i, v in enumerate(category.cat.categories)}
            out[column] = codes.where(codes >= 0, np.nan) if (codes < 0).any() else codes
        return out, mappings

    if method == "One-hot encoding":
        out = pd.get_dummies(out, columns=columns, drop_first=drop_first, dtype=int)
        return out, mappings

    raise ValueError(f"Unknown encoding method: {method}")


# ---------------------------------------------------------------------------
# 5. Scaling / normalization
# ---------------------------------------------------------------------------

SCALING_METHODS = [
    "Standardization (z-score)",
    "Min-Max normalization",
    "Robust scaling (median / IQR)",
]


def scale_columns(df, columns, method, feature_range=(0.0, 1.0)):
    out = df.copy()
    columns = [c for c in columns if c in numeric_cols(out)]
    if not columns:
        raise ValueError("Select at least one numeric column.")

    if method == SCALING_METHODS[0]:
        scaler = StandardScaler()
    elif method == SCALING_METHODS[1]:
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == SCALING_METHODS[2]:
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    out[columns] = scaler.fit_transform(out[columns].astype(float))  # NaNs are preserved
    return out, f"Applied {method} to {len(columns)} column(s)."


def describe_columns(df, columns):
    if not columns:
        return pd.DataFrame()
    stats = df[columns].agg(["mean", "std", "min", "max"]).T
    return stats.round(4)


# ---------------------------------------------------------------------------
# 6. Outliers
# ---------------------------------------------------------------------------

OUTLIER_METHODS = ["IQR rule", "Z-score"]
OUTLIER_ACTIONS = ["Remove rows", "Cap values (winsorize)"]


def _bounds(series, method, threshold):
    if method == "IQR rule":
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return q1 - threshold * iqr, q3 + threshold * iqr
    mean, std = series.mean(), series.std()
    if pd.isna(std) or std == 0:
        return -np.inf, np.inf
    return mean - threshold * std, mean + threshold * std


def detect_outliers(df, columns, method, threshold):
    """Returns (summary DataFrame, combined boolean mask of rows with any outlier)."""
    rows = []
    combined = pd.Series(False, index=df.index)
    for column in columns:
        series = df[column]
        low, high = _bounds(series, method, threshold)
        mask = (series < low) | (series > high)
        combined |= mask
        rows.append({
            "Column": column,
            "Lower bound": round(float(low), 4) if np.isfinite(low) else None,
            "Upper bound": round(float(high), 4) if np.isfinite(high) else None,
            "Outliers": int(mask.sum()),
            "Outliers %": round(float(mask.mean() * 100), 2),
        })
    return pd.DataFrame(rows), combined


def handle_outliers(df, columns, method, threshold, action):
    columns = [c for c in columns if c in numeric_cols(df)]
    if not columns:
        raise ValueError("Select at least one numeric column.")

    summary, mask = detect_outliers(df, columns, method, threshold)

    if action == "Remove rows":
        out = df.loc[~mask]
        return out, f"Removed {int(mask.sum()):,} row(s) containing outliers ({method}, threshold {threshold})."

    out = df.copy()
    capped = 0
    for column in columns:
        low, high = _bounds(out[column], method, threshold)
        before = out[column].copy()
        out[column] = out[column].clip(lower=low if np.isfinite(low) else None,
                                       upper=high if np.isfinite(high) else None)
        capped += int((before != out[column]).sum())
    return out, f"Capped {capped:,} outlier value(s) to the {method} bounds (threshold {threshold})."
