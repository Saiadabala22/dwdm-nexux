import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dataset Information", page_icon="📋", layout="wide")
st.title("Dataset Information & Data Quality")

df = st.session_state.get("dataset")
if df is None:
    st.warning("Upload a dataset first.")
    st.stop()

missing_count = int(df.isna().sum().sum())
rows, cols = df.shape
a, b, c, d = st.columns(4)
a.metric("Dataset", st.session_state.get("dataset_name") or "Uploaded dataset")
b.metric("Instances", f"{rows:,}")
c.metric("Attributes", cols)
d.metric("Missing cells", f"{missing_count:,}")

st.subheader("Attribute profile")
profile = pd.DataFrame({
    "Attribute": df.columns.astype(str),
    "Data type": df.dtypes.astype(str).values,
    "Non-null": df.notna().sum().values,
    "Missing": df.isna().sum().values,
    "Missing %": (df.isna().mean().values * 100).round(2),
    "Unique values": df.nunique(dropna=True).values,
})
st.dataframe(profile, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("Missing values by attribute")
    missing = profile.sort_values("Missing", ascending=False)
    fig = px.bar(missing, x="Attribute", y="Missing", hover_data=["Missing %"], template="plotly_white")
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Attribute data types")
    dtype_counts = profile["Data type"].value_counts().rename_axis("Data type").reset_index(name="Attributes")
    fig = px.pie(dtype_counts, names="Data type", values="Attributes", hole=0.45, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Statistical summary")
st.dataframe(df.describe(include="all").T, use_container_width=True)
st.subheader("Dataset preview")
st.dataframe(df.head(500), use_container_width=True)
st.caption("The preview is limited to 500 rows for responsiveness; the uploaded dataset remains unchanged.")
