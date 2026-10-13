import streamlit as st

from utils.olap_ui import render_olap

st.set_page_config(page_title="OLAP Operations", page_icon="🧊", layout="wide")
st.title("🧊 OLAP Operations")
st.caption("Slice, dice, drill-down, roll-up and pivot on the current dataset (after any preprocessing).")

df = st.session_state.get("dataset")
if df is None:
    st.warning("Upload a dataset first.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", len(df.columns))
m3.metric("Dataset", st.session_state.get("dataset_name") or "Uploaded dataset")

render_olap(df, key_prefix="olap_page")
