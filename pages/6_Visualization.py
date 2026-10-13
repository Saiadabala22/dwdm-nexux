import streamlit as st
import pandas as pd

from utils.visualizations import (
    numeric_columns, categorical_columns, make_box_plot, make_scatter_plot,
    make_histogram, make_correlation_heatmap, make_missing_plot, make_bar_chart,
    make_line_chart, make_pair_plot, make_cluster_plot, make_elbow_plot,
)
from utils.mpl_charts import CHART_TYPES, render_chart
from utils.olap_ui import render_olap

st.set_page_config(page_title="Data Visualization", page_icon="📈", layout="wide")
st.title("Data Visualization & Exploration")
st.caption("Choose one visualization from the dropdown, adjust its options, and optionally add it to the PDF report.")

df = st.session_state.get("dataset")
if df is None:
    st.warning("Upload a CSV, ARFF, XLS, or XLSX dataset first.")
    st.stop()

numeric = numeric_columns(df)
categorical = categorical_columns(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Attributes", len(df.columns))
m3.metric("Numeric", len(numeric))
m4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

st.info("For very large datasets, charts may sample rows for rendering only. The dataset itself is not changed here.")

st.session_state.setdefault("report_charts", [])

sections = st.tabs(["Charts & data mining", "OLAP operations", "ETL & warehouse schemas"])

# =========================================================
# TAB 1 - one dropdown, one visualization
# =========================================================
with sections[0]:
    st.subheader("Choose a visualization")
    chart_type = st.selectbox("Visualization type", CHART_TYPES, key="viz_type")

    fig, params, description = None, None, ""

    if chart_type == "Box plot":
        if not numeric:
            st.warning("Box plots require at least one numeric attribute.")
        else:
            col = st.selectbox("Numeric attribute", numeric, key="box_col")
            group = st.selectbox("Optional grouping attribute", ["(No grouping)"] + categorical, key="box_group")
            group = None if group == "(No grouping)" else group
            fig = make_box_plot(df, col, group)
            params, description = {"column": col, "group": group}, col + (f" by {group}" if group else "")

    elif chart_type == "Scatter plot":
        if len(numeric) < 2:
            st.warning("Scatter plots require at least two numeric attributes.")
        else:
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("X attribute", numeric, index=0, key="scatter_x")
            y = c2.selectbox("Y attribute", numeric, index=1, key="scatter_y")
            color = c3.selectbox("Color by (optional)", ["(None)"] + list(df.columns), key="scatter_color")
            color = None if color == "(None)" else color
            fig = make_scatter_plot(df, x, y, color)
            params, description = {"x": x, "y": y, "color": color}, f"{x} vs {y}"

    elif chart_type == "Histogram":
        if not numeric:
            st.warning("Histograms require a numeric attribute.")
        else:
            col = st.selectbox("Numeric attribute", numeric, key="hist_col")
            fig = make_histogram(df, col)
            params, description = {"column": col}, col

    elif chart_type == "Correlation heatmap":
        fig = make_correlation_heatmap(df)
        if fig is None:
            st.warning("The heatmap requires at least two numeric attributes.")
        else:
            params, description = {}, "numeric attributes"

    elif chart_type == "Pair plot":
        if len(numeric) < 2:
            st.warning("Pair plots require at least two numeric attributes.")
        else:
            cols = st.multiselect("Select 2-5 numeric attributes", numeric,
                                  default=numeric[:min(3, len(numeric))], max_selections=5, key="pair_cols")
            color = st.selectbox("Color by (optional)", ["(None)"] + list(df.columns), key="pair_color")
            color = None if color == "(None)" else color
            if len(cols) >= 2:
                fig = make_pair_plot(df, cols, color)
                params, description = {"columns": cols, "color": color}, ", ".join(cols)
            else:
                st.caption("Select at least two numeric attributes.")

    elif chart_type == "Missing values":
        fig = make_missing_plot(df)
        st.dataframe(pd.DataFrame({"Missing values": df.isna().sum(),
                                   "Missing %": (df.isna().mean() * 100).round(2)})
                     .sort_values("Missing values", ascending=False), use_container_width=True)
        params, description = {}, "all attributes"

    elif chart_type == "Bar chart":
        col = st.selectbox("Attribute", list(df.columns), key="bar_col")
        fig = make_bar_chart(df, col)
        params, description = {"column": col}, col

    elif chart_type == "Line chart":
        if not numeric:
            st.warning("Line charts require a numeric attribute.")
        else:
            col = st.selectbox("Numeric attribute", numeric, key="line_col")
            fig = make_line_chart(df, col)
            params, description = {"column": col}, col

    elif chart_type == "K-Means cluster visualization":
        if len(numeric) < 2:
            st.warning("Select a dataset with at least two numeric attributes to use this plot.")
        else:
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("X attribute", numeric, index=0, key="cluster_x")
            y = c2.selectbox("Y attribute", numeric, index=1, key="cluster_y")
            k = c3.slider("Number of clusters", min_value=2, max_value=10, value=3, key="cluster_k")
            fig = make_cluster_plot(df, x, y, k)
            params, description = {"x": x, "y": y, "k": k}, f"{x} vs {y}, k={k}"

    elif chart_type == "Elbow method":
        if len(numeric) < 2:
            st.warning("Select a dataset with at least two numeric attributes to use this plot.")
        else:
            cols = st.multiselect("Attributes for clustering", numeric, default=numeric[:min(3, len(numeric))],
                                  max_selections=8, key="elbow_cols")
            if cols:
                with st.spinner("Calculating K-Means inertia values…"):
                    fig = make_elbow_plot(df, cols)
                params, description = {"columns": cols}, ", ".join(cols)
            else:
                st.caption("Choose at least one numeric attribute.")

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        b1, b2 = st.columns(2)
        b1.download_button("Download interactive chart (HTML)", fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                           file_name="dataset_visualization.html", mime="text/html")
        if b2.button("➕ Add to PDF report", key="viz_add"):
            spec = {"type": chart_type, "title": f"{chart_type} — {description}", "params": params}
            if any(c.get("type") == chart_type and c.get("params") == params
                   for c in st.session_state.report_charts):
                st.info("This chart is already in the PDF report.")
            else:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(render_chart(df, spec))  # make sure it can be drawn for the PDF
                    st.session_state.report_charts.append(spec)
                    st.success("Added to the PDF report. Download the PDF from the Results page.")
                except Exception as error:
                    st.error(f"This chart could not be prepared for the PDF: {error}")

    with st.expander(f"📄 Charts added to the PDF report ({len(st.session_state.report_charts)})"):
        if not st.session_state.report_charts:
            st.caption("Nothing added yet. The PDF always contains default exploratory charts as well.")
        for i, spec in enumerate(list(st.session_state.report_charts)):
            r1, r2 = st.columns([6, 1])
            r1.write(f"{i + 1}. {spec['title']}")
            if r2.button("Remove", key=f"viz_rm_{i}"):
                st.session_state.report_charts.pop(i)
                st.rerun()

    st.caption("Decision trees, ROC curves, and confusion matrices are on the Results page and are always "
               "included in the PDF.")

# =========================================================
# TAB 2 - OLAP
# =========================================================
with sections[1]:
    render_olap(df, key_prefix="olap_tab")

# =========================================================
# TAB 3 - ETL + schemas (conceptual)
# =========================================================
with sections[2]:
    st.subheader("ETL pipeline — conceptual illustration")
    st.caption("This is an illustrative diagram, not an execution engine. No ETL job or warehouse database is created by this diagram.")
    st.graphviz_chart('''digraph ETL { rankdir=LR; graph [bgcolor="transparent"]; node [shape=box, style="rounded,filled", color="#2563eb", fillcolor="#eff6ff", fontname="Arial"]; Extract -> Transform -> Validate -> Load; Extract [label="Extract\\nCSV / ARFF / Excel"]; Transform [label="Transform\\nClean / Normalize"]; Validate [label="Validate\\nTypes / Missing values"]; Load [label="Load\\nWarehouse target (concept)"]; }''')
    st.subheader("Star schema — illustrative")
    st.graphviz_chart('''graph Star { graph [bgcolor="transparent"]; node [shape=box, style="rounded,filled", fontname="Arial"]; Fact [label="Fact Table\\nMeasures + dimension keys", fillcolor="#dbeafe", color="#2563eb"]; D1 [label="Date Dimension", fillcolor="#ecfdf5", color="#059669"]; D2 [label="Product Dimension", fillcolor="#ecfdf5", color="#059669"]; D3 [label="Customer Dimension", fillcolor="#ecfdf5", color="#059669"]; D4 [label="Location Dimension", fillcolor="#ecfdf5", color="#059669"]; Fact -- D1; Fact -- D2; Fact -- D3; Fact -- D4; }''')
    st.subheader("Snowflake schema — illustrative")
    st.graphviz_chart('''graph Snowflake { graph [bgcolor="transparent"]; node [shape=box, style="rounded,filled", fontname="Arial"]; Fact [label="Fact Table", fillcolor="#dbeafe", color="#2563eb"]; Product [label="Product", fillcolor="#ecfdf5", color="#059669"]; Category [label="Category", fillcolor="#ecfdf5", color="#059669"]; Location [label="Location", fillcolor="#ecfdf5", color="#059669"]; Region [label="Region", fillcolor="#ecfdf5", color="#059669"]; Fact -- Product; Product -- Category; Fact -- Location; Location -- Region; }''')
    st.caption("These are teaching diagrams. They do not infer or represent a real schema from the uploaded file.")
