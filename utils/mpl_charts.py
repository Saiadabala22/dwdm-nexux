"""Static (matplotlib) versions of the interactive charts, used for the PDF report.

A chart is described by a small "spec" dictionary so that it can be stored in
st.session_state and re-drawn whenever the PDF is built:

    {"type": "Box plot", "title": "Box plot - Age", "params": {"column": "Age", "group": None}}

OLAP results are stored as a snapshot table instead of parameters:

    {"type": "OLAP table", "title": "...", "table": DataFrame, "kind": "bar" | "heatmap"}
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CHART_TYPES = [
    "Box plot",
    "Scatter plot",
    "Histogram",
    "Correlation heatmap",
    "Pair plot",
    "Missing values",
    "Bar chart",
    "Line chart",
    "K-Means cluster visualization",
    "Elbow method",
]


def _sample(df, n=5000):
    return df if len(df) <= n else df.sample(n=n, random_state=42)


def _numeric(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def _box(df, p):
    column, group = p["column"], p.get("group")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    data = _sample(df)
    if group:
        top = data[group].astype("string").fillna("(Missing)").value_counts().head(12).index
        keys = data[group].astype("string").fillna("(Missing)")
        groups = [pd.to_numeric(data.loc[keys == k, column], errors="coerce").dropna() for k in top]
        ax.boxplot(groups, patch_artist=True)
        ax.set_xticklabels([str(k)[:18] for k in top])
        ax.set_xlabel(str(group))
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    else:
        ax.boxplot(pd.to_numeric(data[column], errors="coerce").dropna(), vert=False, patch_artist=True)
    ax.set_ylabel(str(column)) if group else ax.set_xlabel(str(column))
    return fig


def _scatter(df, p):
    x, y, color = p["x"], p["y"], p.get("color")
    cols = [x, y] + ([color] if color and color not in (x, y) else [])
    data = _sample(df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y]))
    fig, ax = plt.subplots(figsize=(7, 4))
    if color:
        keys = data[color]
        if pd.api.types.is_numeric_dtype(keys) and keys.nunique() > 12:
            sc = ax.scatter(data[x], data[y], c=keys, s=14, alpha=0.7, cmap="viridis")
            fig.colorbar(sc, ax=ax, label=str(color))
        else:
            labels = keys.astype("string").fillna("(Missing)")
            for key in labels.value_counts().head(12).index:
                m = labels == key
                ax.scatter(data.loc[m, x], data.loc[m, y], s=14, alpha=0.7, label=str(key)[:20])
            ax.legend(fontsize=7, title=str(color))
    else:
        ax.scatter(data[x], data[y], s=14, alpha=0.6)
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))
    return fig


def _hist(df, p):
    values = pd.to_numeric(df[p["column"]], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(values.iloc[:20000], bins=30, edgecolor="black", alpha=0.8)
    ax.set_xlabel(str(p["column"]))
    ax.set_ylabel("Frequency")
    return fig


def _heatmap(df, p):
    corr = df[_numeric(df)].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels([str(c) for c in corr.columns], rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels([str(c) for c in corr.index], fontsize=7)
    if len(corr) <= 12:
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig


def _pair(df, p):
    cols, color = p["columns"], p.get("color")
    data = _sample(df[cols + ([color] if color and color not in cols else [])].dropna(subset=cols), 1500)
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(7, 7))
    axes = np.atleast_2d(axes)
    keys = data[color].astype("string").fillna("(Missing)") if color else None
    palette = {}
    if keys is not None:
        for i, k in enumerate(keys.value_counts().head(8).index):
            palette[k] = plt.cm.tab10(i % 10)
    for i, yc in enumerate(cols):
        for j, xc in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[xc], bins=20, alpha=0.8)
            elif keys is not None:
                for k, c in palette.items():
                    m = keys == k
                    ax.scatter(data.loc[m, xc], data.loc[m, yc], s=4, alpha=0.6, color=c)
            else:
                ax.scatter(data[xc], data[yc], s=4, alpha=0.5)
            if i == n - 1:
                ax.set_xlabel(str(xc), fontsize=7)
            if j == 0:
                ax.set_ylabel(str(yc), fontsize=7)
            ax.tick_params(labelsize=5)
    return fig


def _missing(df, p):
    missing = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar([str(c) for c in missing.index], missing.values)
    ax.set_ylabel("Missing cells")
    ax.tick_params(axis="x", labelrotation=65, labelsize=7)
    return fig


def _bar(df, p):
    counts = df[p["column"]].astype("string").fillna("(Missing)").value_counts().head(20)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar([str(x)[:28] for x in counts.index], counts.values)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=55, labelsize=7)
    return fig


def _line(df, p):
    values = pd.to_numeric(_sample(df)[p["column"]], errors="coerce").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(np.arange(1, len(values) + 1), values, linewidth=1)
    ax.set_xlabel("Row index")
    ax.set_ylabel(str(p["column"]))
    return fig


def _kmeans(df, p):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    x, y, k = p["x"], p["y"], int(p.get("k", 3))
    data = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    k = max(1, min(k, len(data), 10))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(data))
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in sorted(set(labels)):
        m = labels == label
        ax.scatter(data.loc[m, x], data.loc[m, y], s=14, alpha=0.7, label=f"Cluster {label + 1}")
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))
    ax.legend(fontsize=7)
    return fig


def _elbow(df, p):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    data = df[p["columns"]].replace([np.inf, -np.inf], np.nan).dropna()
    data = _sample(data, 3000)
    scaled = StandardScaler().fit_transform(data)
    ks = list(range(1, min(8, len(data) - 1) + 1))
    inertia = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(ks, inertia, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    return fig


_RENDERERS = {
    "Box plot": _box,
    "Scatter plot": _scatter,
    "Histogram": _hist,
    "Correlation heatmap": _heatmap,
    "Pair plot": _pair,
    "Missing values": _missing,
    "Bar chart": _bar,
    "Line chart": _line,
    "K-Means cluster visualization": _kmeans,
    "Elbow method": _elbow,
}


def _olap_table_figure(table: pd.DataFrame, kind: str):
    table = table.copy()
    if kind == "heatmap":
        values = table.apply(pd.to_numeric, errors="coerce")
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(values) + 1.5)))
        image = ax.imshow(values.values.astype(float), cmap="Blues", aspect="auto")
        ax.set_xticks(range(values.shape[1]))
        ax.set_xticklabels([str(c)[:16] for c in values.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(values.shape[0]))
        ax.set_yticklabels([str(i)[:28] for i in values.index], fontsize=7)
        if values.size <= 150:
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    v = values.iloc[i, j]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:,.4g}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        return fig

    # "bar": first column(s) are labels, the last numeric column is the value
    value_col = table.columns[-1]
    labels = table.iloc[:, :-1].astype(str).agg(" › ".join, axis=1) if table.shape[1] > 1 else table.index.astype(str)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar([str(x)[:26] for x in labels[:40]], pd.to_numeric(table[value_col], errors="coerce")[:40])
    ax.set_ylabel(str(value_col))
    ax.tick_params(axis="x", labelrotation=55, labelsize=7)
    return fig


def render_chart(df, spec):
    """Return a matplotlib Figure for a chart spec. Raises on invalid data."""
    if spec["type"] == "OLAP table":
        fig = _olap_table_figure(spec["table"], spec.get("kind", "bar"))
    else:
        fig = _RENDERERS[spec["type"]](df, spec["params"])
    fig.suptitle(spec.get("title", ""), fontsize=10)
    fig.tight_layout()
    return fig
