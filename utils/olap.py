"""OLAP operations on a flat dataset (pure pandas, no Streamlit).

The uploaded table is treated as a small in-memory cube:
  * dimensions = categorical / low-cardinality columns (plus derived bins and date parts)
  * measures   = numeric columns

Operations: slice, dice, drill-down / roll-up along a hierarchy, and pivot.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

AGG_OPTIONS = {
    "Sum": "sum",
    "Average": "mean",
    "Count": "count",
    "Median": "median",
    "Min": "min",
    "Max": "max",
}

MISSING_LABEL = "(missing)"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEP = " › "


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def measure_candidates(df):
    return [c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]


def dimension_candidates(df, max_unique=100):
    dims = []
    for column in df.columns:
        if df[column].nunique(dropna=True) <= max_unique:
            dims.append(column)
    return dims


def labels(series):
    """String labels for a column, with a visible label for missing values."""
    return series.astype(object).where(series.notna(), MISSING_LABEL).astype(str)


def members(series):
    """Distinct labels of a dimension in a natural order (category / numeric / alphabetical)."""
    present = labels(series)
    if isinstance(series.dtype, pd.CategoricalDtype):
        seen = set(present)
        out = [str(c) for c in series.cat.categories if str(c) in seen]
    else:
        values = series.dropna().unique().tolist()
        try:
            values = sorted(values)
        except TypeError:
            values = sorted(values, key=str)
        out = [str(v) for v in values]
    if series.isna().any():
        out.append(MISSING_LABEL)
    return out


# ---------------------------------------------------------------------------
# Derived dimensions
# ---------------------------------------------------------------------------

def derive_dimensions(df, bin_columns=(), n_bins=4, bin_method="Equal width", date_column=None):
    """Return (new DataFrame, list of derived dimension names, warnings)."""
    out = df.copy()
    derived, warnings = [], []

    for column in bin_columns:
        name = f"{column} (bins)"
        values = pd.to_numeric(out[column], errors="coerce")
        try:
            if bin_method == "Equal frequency":
                out[name] = pd.qcut(values, q=n_bins, duplicates="drop")
            else:
                out[name] = pd.cut(values, bins=n_bins)
            out[name] = out[name].astype("category")
            derived.append(name)
        except ValueError as error:
            warnings.append(f"Could not bin '{column}': {error}")

    if date_column:
        parsed = pd.to_datetime(out[date_column], errors="coerce")
        if parsed.notna().sum() == 0:
            warnings.append(f"'{date_column}' could not be read as dates.")
        else:
            year = f"{date_column} · Year"
            quarter = f"{date_column} · Quarter"
            month = f"{date_column} · Month"
            day = f"{date_column} · Day"
            out[year] = parsed.dt.year.astype("Int64")
            out[quarter] = parsed.dt.quarter.map(lambda v: f"Q{int(v)}" if pd.notna(v) else np.nan)
            out[month] = pd.Categorical(
                parsed.dt.month.map(lambda v: MONTHS[int(v) - 1] if pd.notna(v) else np.nan),
                categories=MONTHS, ordered=True,
            )
            out[day] = parsed.dt.day.astype("Int64")
            derived += [year, quarter, month, day]

    return out, derived, warnings


# ---------------------------------------------------------------------------
# Slice / dice
# ---------------------------------------------------------------------------

def slice_cube(df, dimension, value):
    """Fix ONE dimension to ONE value."""
    return df[labels(df[dimension]) == str(value)]


def dice_cube(df, selections):
    """Select a sub-cube: selections = {dimension: [allowed values, ...]} (AND across dimensions)."""
    mask = pd.Series(True, index=df.index)
    for dimension, allowed in selections.items():
        mask &= labels(df[dimension]).isin([str(v) for v in allowed])
    return df[mask]


# ---------------------------------------------------------------------------
# Aggregation, drill-down / roll-up
# ---------------------------------------------------------------------------

def value_column_name(measure, agg):
    return f"{agg} of {measure}"


def aggregate(df, group_columns, measure, agg):
    """Group by dimensions and aggregate one measure. group_columns=[] gives the grand total."""
    func = AGG_OPTIONS[agg]
    value_name = value_column_name(measure, agg)
    values = pd.to_numeric(df[measure], errors="coerce")

    if not group_columns:
        return pd.DataFrame({"Level": ["All (grand total)"], value_name: [getattr(values, func)()],
                             "Rows": [len(df)]})

    temp = df[list(group_columns)].copy()
    temp["__v__"] = values
    grouped = (
        temp.groupby(list(group_columns), dropna=False, observed=True)
        .agg(**{value_name: ("__v__", func), "Rows": ("__v__", "size")})
        .reset_index()
    )
    for column in group_columns:
        grouped[column] = labels(grouped[column])
    return grouped


def parent_options(df, hierarchy, level):
    """Members of the level above `level` (used to drill into a single parent)."""
    if level < 2:
        return []
    parents = hierarchy[: level - 1]
    combos = pd.DataFrame({c: labels(df[c]) for c in parents}).drop_duplicates()
    # keep a natural order by sorting each level using member order
    for c in reversed(parents):
        order = {m: i for i, m in enumerate(members(df[c]))}
        combos = combos.sort_values(c, key=lambda s: s.map(order), kind="stable")
    return [tuple(row) for row in combos.itertuples(index=False, name=None)]


def drill(df, hierarchy, level, measure, agg, parent=None):
    """
    level 0  -> grand total (fully rolled up)
    level k  -> grouped by the first k dimensions of the hierarchy
    parent   -> tuple of member labels for the first k-1 dimensions (drill into one parent)
    """
    data = df
    if parent and level >= 2:
        data = dice_cube(df, {dim: [val] for dim, val in zip(hierarchy[: level - 1], parent)})
    return aggregate(data, list(hierarchy[:level]), measure, agg)


# ---------------------------------------------------------------------------
# Pivot
# ---------------------------------------------------------------------------

def pivot(df, rows, column, measure, agg, margins=True):
    func = AGG_OPTIONS[agg]
    dims = list(rows) + ([column] if column else [])
    temp = pd.DataFrame({c: labels(df[c]) for c in dims})
    temp["__v__"] = pd.to_numeric(df[measure], errors="coerce")

    table = pd.pivot_table(
        temp, index=list(rows), columns=column if column else None, values="__v__",
        aggfunc=func, margins=margins, margins_name="Total",
    )
    if not column:
        table.columns = [value_column_name(measure, agg)]

    def reorder(index, series):
        order = members(series) + ["Total"]
        return [o for o in order if o in index]

    if len(rows) == 1:
        table = table.reindex(reorder(table.index, df[rows[0]]))
    if column:
        table = table[reorder(table.columns, df[column])]
    return table


def flatten_index(table):
    """Make index/columns plain strings (needed for charts and PDF)."""
    out = table.copy()
    if isinstance(out.index, pd.MultiIndex):
        out.index = [SEP.join(map(str, t)) for t in out.index]
    else:
        out.index = out.index.map(str)
    out.columns = [str(c) for c in out.columns]
    return out
