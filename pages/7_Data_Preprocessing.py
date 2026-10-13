import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import preprocessing as pp

st.set_page_config(page_title="Data Preprocessing", page_icon="🧹", layout="wide")
st.title("🧹 Data Preprocessing")
st.caption(
    "Clean and transform the dataset before analysis. Every step is applied to the dataset used by all other "
    "pages (Algorithm Selection, Results, Visualization, OLAP). Use **Undo** or **Reset** at any time."
)

df = st.session_state.get("dataset")
if df is None:
    st.warning("Upload a dataset first.")
    st.stop()

# ---------------------------------------------------------------------------
# State: original copy, undo stack, history
# ---------------------------------------------------------------------------
if st.session_state.get("original_dataset") is None:
    st.session_state.original_dataset = df.copy()
st.session_state.setdefault("preprocess_log", [])
st.session_state.setdefault("preprocess_stack", [])


def commit(new_df, description):
    """Make `new_df` the active dataset, remember the old one for Undo, and refresh the page."""
    st.session_state.preprocess_stack.append(st.session_state.dataset)
    st.session_state.preprocess_stack = st.session_state.preprocess_stack[-15:]
    st.session_state.dataset = new_df.reset_index(drop=True)
    st.session_state.preprocess_log.append(description)
    _data_changed()
    st.session_state.preprocess_flash = description
    st.rerun()


def _data_changed():
    # Results trained on the old data would no longer match the dataset.
    st.session_state.results = {}
    st.session_state.models = {}
    target = st.session_state.get("target_column")
    if target is not None and target not in st.session_state.dataset.columns:
        st.session_state.target_column = None


def undo():
    if st.session_state.preprocess_stack:
        st.session_state.dataset = st.session_state.preprocess_stack.pop()
        if st.session_state.preprocess_log:
            st.session_state.preprocess_log.pop()
        _data_changed()


def reset():
    st.session_state.dataset = st.session_state.original_dataset.copy()
    st.session_state.preprocess_stack = []
    st.session_state.preprocess_log = []
    _data_changed()


flash = st.session_state.pop("preprocess_flash", None)
if flash:
    st.success(flash)

# ---------------------------------------------------------------------------
# Summary + controls
# ---------------------------------------------------------------------------
original = st.session_state.original_dataset
a, b, c, d = st.columns(4)
a.metric("Rows", f"{len(df):,}", delta=len(df) - len(original) or None)
b.metric("Columns", len(df.columns), delta=len(df.columns) - len(original.columns) or None)
c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}",
         delta=int(df.isna().sum().sum() - original.isna().sum().sum()) or None, delta_color="inverse")
d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

u1, u2, u3 = st.columns([1, 1, 2])
u1.button("↩️ Undo last step", on_click=undo, disabled=not st.session_state.preprocess_stack,
          use_container_width=True)
u2.button("♻️ Reset to original", on_click=reset, disabled=not st.session_state.preprocess_log,
          use_container_width=True)
u3.download_button("⬇️ Download current dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                   file_name="preprocessed_dataset.csv", mime="text/csv", use_container_width=True)

tabs = st.tabs(["Overview", "Missing values", "Duplicates", "Encoding", "Scaling / Normalization",
                "Outliers", "History"])

numeric = pp.numeric_cols(df)
categorical = pp.categorical_cols(df)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    profile = pd.DataFrame({
        "Attribute": df.columns.astype(str),
        "Type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(2),
        "Unique": df.nunique(dropna=True).values,
    })
    st.dataframe(profile, use_container_width=True, hide_index=True)
    st.caption("Suggested order: placeholders → missing values → duplicates → outliers → encoding → scaling.")
    st.dataframe(df.head(100), use_container_width=True)

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
with tabs[1]:
    with st.expander("Step 0 — Turn placeholders such as '?' or 'NA' into real missing values", expanded=False):
        tokens = st.multiselect("Treat these values as missing", pp.DEFAULT_MISSING_TOKENS,
                                default=["?", "NA", "N/A", "null", "None", ""], key="pp_tokens")
        extra = st.text_input("Additional placeholders (comma separated)", key="pp_extra_tokens")
        convert = st.checkbox("Convert text columns to numbers when every remaining value is numeric",
                              value=True, key="pp_convert")
        if st.button("Apply placeholder clean-up", key="pp_apply_tokens"):
            all_tokens = tokens + [t.strip() for t in extra.split(",") if t.strip()]
            new_df, message = pp.standardize_missing(df, all_tokens, convert)
            commit(new_df, message)

    with_missing = [col for col in df.columns if df[col].isna().any()]
    if not with_missing:
        st.success("No missing values in the current dataset.")
    else:
        counts = df[with_missing].isna().sum().rename("Missing").reset_index().rename(columns={"index": "Attribute"})
        counts.columns = ["Attribute", "Missing"]
        st.dataframe(counts, use_container_width=True, hide_index=True)

        method = st.selectbox("Method", pp.MISSING_METHODS, key="pp_missing_method")
        cols = st.multiselect("Apply to columns", with_missing, default=with_missing, key="pp_missing_cols")
        constant = None
        if method == "Constant value":
            constant = st.text_input("Constant value", key="pp_missing_const")
        if method in ("Mean", "Median"):
            st.caption("Mean/Median only apply to numeric columns; other selected columns are skipped.")
        if st.button("Apply", type="primary", key="pp_missing_apply", disabled=not cols):
            try:
                new_df, message = pp.handle_missing(df, cols, method, constant)
                commit(new_df, message)
            except ValueError as error:
                st.error(str(error))

# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
with tabs[2]:
    subset = st.multiselect("Columns that define a duplicate (empty = compare whole rows)", list(df.columns),
                            key="pp_dup_subset")
    keep_label = st.selectbox("What to keep", list(pp.KEEP_OPTIONS), key="pp_dup_keep")
    keep = pp.KEEP_OPTIONS[keep_label]
    dup_all = pp.duplicate_mask(df, subset, keep=False)
    st.metric("Rows involved in duplicates", int(dup_all.sum()))
    if dup_all.any():
        with st.expander("Show duplicate rows"):
            st.dataframe(df[dup_all].sort_values(by=subset or list(df.columns)).head(300),
                         use_container_width=True)
    if st.button("Remove duplicates", type="primary", key="pp_dup_apply", disabled=not dup_all.any()):
        new_df, message = pp.remove_duplicates(df, subset, keep)
        commit(new_df, message)

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
with tabs[3]:
    if not categorical:
        st.success("There are no categorical columns to encode.")
    else:
        enc_cols = st.multiselect("Columns to encode", categorical, default=[], key="pp_enc_cols")
        method = st.radio("Method", ["Label encoding", "One-hot encoding"], horizontal=True, key="pp_enc_method")
        drop_first = False
        if method == "One-hot encoding":
            drop_first = st.checkbox("Drop first category (avoids redundant columns)", key="pp_enc_dropfirst")
            new_cols = int(sum(df[c].nunique(dropna=True) for c in enc_cols))
            if enc_cols:
                st.caption(f"This will create about {new_cols} new column(s).")
            if new_cols > 200:
                st.warning("That would create a very wide table. Consider label encoding or fewer columns.")
        st.caption("Tip: the algorithms already encode the target (class) column themselves, so you normally "
                   "leave it out here.")
        if st.button("Apply encoding", type="primary", key="pp_enc_apply", disabled=not enc_cols):
            new_df, mappings = pp.encode_columns(df, enc_cols, method, drop_first)
            st.session_state.last_label_mappings = mappings
            commit(new_df, f"{method} on {len(enc_cols)} column(s): {', '.join(map(str, enc_cols))}.")
        if st.session_state.get("last_label_mappings"):
            with st.expander("Last label-encoding mapping"):
                for col, mapping in st.session_state.last_label_mappings.items():
                    st.write(f"**{col}**", mapping)

# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
with tabs[4]:
    if not numeric:
        st.warning("There are no numeric columns to scale.")
    else:
        sc_cols = st.multiselect("Numeric columns", numeric, default=[], key="pp_scale_cols")
        method = st.radio("Method", pp.SCALING_METHODS, key="pp_scale_method")
        rng = (0.0, 1.0)
        if method == pp.SCALING_METHODS[1]:
            r1, r2 = st.columns(2)
            lo = r1.number_input("New minimum", value=0.0, key="pp_scale_lo")
            hi = r2.number_input("New maximum", value=1.0, key="pp_scale_hi")
            rng = (lo, hi)
        if sc_cols:
            st.markdown("Current statistics")
            st.dataframe(pp.describe_columns(df, sc_cols), use_container_width=True)
        st.caption("Tip: do not scale the target column for classification.")
        if st.button("Apply scaling", type="primary", key="pp_scale_apply", disabled=not sc_cols):
            if rng[0] >= rng[1]:
                st.error("The new minimum must be smaller than the new maximum.")
            else:
                new_df, message = pp.scale_columns(df, sc_cols, method, rng)
                commit(new_df, message)

# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------
with tabs[5]:
    if not numeric:
        st.warning("There are no numeric columns to check for outliers.")
    else:
        o_cols = st.multiselect("Numeric columns", numeric, default=numeric[: min(3, len(numeric))],
                                key="pp_out_cols")
        o1, o2, o3 = st.columns(3)
        method = o1.selectbox("Detection method", pp.OUTLIER_METHODS, key="pp_out_method")
        threshold = o2.number_input("Threshold (IQR multiplier or z-score)",
                                    value=1.5 if method == "IQR rule" else 3.0, min_value=0.1, step=0.1,
                                    key=f"pp_out_thr_{method}")
        action = o3.selectbox("Action", pp.OUTLIER_ACTIONS, key="pp_out_action")
        if o_cols:
            summary, mask = pp.detect_outliers(df, o_cols, method, threshold)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.metric("Rows containing at least one outlier", int(mask.sum()))
            fig = px.box(df[o_cols].melt(var_name="Column", value_name="Value"), x="Column", y="Value",
                         template="plotly_white", title="Distribution of selected columns")
            st.plotly_chart(fig, use_container_width=True)
            if st.button("Apply outlier handling", type="primary", key="pp_out_apply"):
                new_df, message = pp.handle_outliers(df, o_cols, method, threshold, action)
                commit(new_df, message)

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
with tabs[6]:
    log = st.session_state.preprocess_log
    if not log:
        st.info("No preprocessing steps applied yet.")
    else:
        for i, line in enumerate(log, 1):
            st.write(f"{i}. {line}")
    st.write(f"Original: **{original.shape[0]:,} × {original.shape[1]}** → Current: "
             f"**{df.shape[0]:,} × {df.shape[1]}**")

st.divider()
if st.button("➡️ Continue to Algorithm Selection", use_container_width=True):
    st.switch_page("pages/3_Algorithm_Selection.py")
