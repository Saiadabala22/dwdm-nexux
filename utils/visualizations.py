"""Reusable interactive visualizations for Dataset Analyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in numeric_columns(df)]


def safe_plot_df(df: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    """Bound chart rendering work without changing the underlying dataset."""
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=42)


def make_box_plot(df: pd.DataFrame, column: str, group: str | None = None):
    data = safe_plot_df(df)
    return px.box(data, x=group if group else None, y=column,
                  color=group if group else None, points="outliers",
                  title=f"Box plot — {column}", template="plotly_white")


def make_scatter_plot(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    data = safe_plot_df(df)
    return px.scatter(data, x=x, y=y, color=color if color else None,
                      hover_data=list(data.columns[: min(4, len(data.columns))]),
                      title=f"Scatter plot — {x} vs {y}", template="plotly_white")


def make_histogram(df: pd.DataFrame, column: str):
    return px.histogram(safe_plot_df(df), x=column, marginal="box", nbins=30,
                        title=f"Distribution — {column}", template="plotly_white")


def make_correlation_heatmap(df: pd.DataFrame):
    cols = numeric_columns(df)
    if len(cols) < 2:
        return None
    corr = df[cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto",
                    title="Numerical attribute correlation", template="plotly_white")
    return fig


def make_missing_plot(df: pd.DataFrame):
    missing = df.isna().sum().sort_values(ascending=False)
    data = missing.rename_axis("Attribute").reset_index(name="Missing values")
    data["Missing %"] = (data["Missing values"] / max(len(df), 1) * 100).round(2)
    fig = px.bar(data, x="Attribute", y="Missing values", hover_data=["Missing %"],
                 title="Missing values by attribute", template="plotly_white")
    fig.update_layout(xaxis_tickangle=-35)
    return fig


def make_bar_chart(df: pd.DataFrame, column: str, top_n: int = 20):
    counts = df[column].astype("string").fillna("(Missing)").value_counts().head(top_n)
    data = counts.rename_axis(column).reset_index(name="Count")
    return px.bar(data, x=column, y="Count", title=f"Top values — {column}",
                  template="plotly_white")


def make_line_chart(df: pd.DataFrame, column: str):
    data = safe_plot_df(df).reset_index(drop=True)
    data["Row index"] = np.arange(1, len(data) + 1)
    return px.line(data, x="Row index", y=column, title=f"{column} by row order",
                   template="plotly_white")


def make_pair_plot(df: pd.DataFrame, columns: list[str], color: str | None = None):
    data = safe_plot_df(df[columns + ([color] if color and color not in columns else [])], 1500)
    return px.scatter_matrix(data, dimensions=columns, color=color if color else None,
                             title="Pair plot — selected numerical attributes",
                             template="plotly_white")


def make_cluster_plot(df: pd.DataFrame, x: str, y: str, clusters: int = 3):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        return None
    k = max(1, min(int(clusters), len(data), 10))
    scaled = StandardScaler().fit_transform(data)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    plot_data = data.copy()
    plot_data["Cluster"] = labels.astype(str)
    plot_data["Cluster"] = "Cluster " + (plot_data["Cluster"].astype(int) + 1).astype(str)
    return px.scatter(plot_data, x=x, y=y, color="Cluster", title=f"K-Means clusters — {x} vs {y}",
                      template="plotly_white")


def make_elbow_plot(df: pd.DataFrame, columns: list[str], max_k: int = 8):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or not columns:
        return None
    data = data.sample(n=min(len(data), 3000), random_state=42) if len(data) > 3000 else data
    scaled = StandardScaler().fit_transform(data)
    upper = min(max_k, len(data) - 1)
    rows = []
    for k in range(1, upper + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(scaled)
        rows.append({"Clusters (k)": k, "Inertia": model.inertia_})
    return px.line(pd.DataFrame(rows), x="Clusters (k)", y="Inertia", markers=True,
                   title="K-Means elbow method", template="plotly_white")


def make_olap_chart(df: pd.DataFrame, dimension: str, measure: str, operation: str = "Roll-up"):
    """Illustrative OLAP-style aggregation on the uploaded dataset."""
    data = df[[dimension, measure]].copy().dropna()
    data[dimension] = data[dimension].astype(str)
    data[measure] = pd.to_numeric(data[measure], errors="coerce")
    data = data.dropna(subset=[measure])
    if data.empty:
        return None
    grouped = data.groupby(dimension, dropna=False)[measure].agg(["sum", "mean", "count"]).reset_index()
    grouped = grouped.sort_values("sum", ascending=False).head(30)
    value_col = "mean" if operation == "Roll-up (average)" else "sum"
    title = f"OLAP {operation} — {measure} by {dimension}"
    return px.bar(grouped, x=dimension, y=value_col, hover_data=["sum", "mean", "count"],
                  title=title, template="plotly_white",
                  labels={value_col: "Average" if value_col == "mean" else "Total"})
