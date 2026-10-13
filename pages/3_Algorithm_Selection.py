import streamlit as st

from utils.ml_engine import train_models
from utils.theme import apply_saved_background


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Algorithm Selection",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# APPLY DATASET BACKGROUND
# =========================================================

apply_saved_background()


# =========================================================
# PAGE TITLE
# =========================================================

st.title("🤖 Algorithm Selection")

st.progress(
    0.66,
    text="Step 2 of 3 — Algorithm Selection"
)


# =========================================================
# CHECK DATASET
# =========================================================

if (
    "dataset" not in st.session_state
    or st.session_state.dataset is None
):

    st.warning(
        "⚠️ Please upload a dataset first."
    )

    st.stop()


# =========================================================
# GET DATASET
# =========================================================

df = st.session_state.dataset


# =========================================================
# DATASET INFORMATION
# =========================================================

st.markdown("### 📊 Dataset Ready")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "📄 Dataset",
        st.session_state.get(
            "dataset_name",
            "Uploaded Dataset"
        )
    )


with col2:

    st.metric(
        "📌 Rows",
        len(df)
    )


with col3:

    st.metric(
        "📋 Columns",
        len(df.columns)
    )


st.divider()


# =========================================================
# TARGET / CLASS ATTRIBUTE
# =========================================================

st.markdown(
    "### 🎯 Target / Class Attribute"
)


saved_target = st.session_state.get(
    "target_column",
    None
)


if saved_target in df.columns:

    target_index = list(
        df.columns
    ).index(saved_target)

else:

    target_index = len(df.columns) - 1


target = st.selectbox(
    "Select the target/class attribute",
    df.columns,
    index=target_index
)


st.session_state.target_column = target


# =========================================================
# INFORMATION BOX
# =========================================================

st.info(
    "💡 K-Means Clustering is an unsupervised learning "
    "algorithm. It does not require a target/class column. "
    "ROC/AUC is available for classification algorithms "
    "when probability scores are available."
)


# =========================================================
# ALGORITHM SELECTION
# =========================================================

st.markdown(
    "### 🤖 Select Algorithms"
)


algos = [
    "ID3",
    "J48",
    "K-NN",
    "Naive Bayes",
    "K-Means Clustering",
    "Random Forest"
]


# =========================================================
# DEFAULT SELECTION
# =========================================================

default_algorithms = [
    "ID3",
    "J48",
    "K-NN",
    "Naive Bayes",
    "K-Means Clustering",
    "Random Forest"
]


selected = st.multiselect(
    "Select algorithms to run",
    options=algos,
    default=default_algorithms
)


# =========================================================
# SELECTED ALGORITHMS
# =========================================================

if selected:

    st.success(
        "✅ Selected Algorithms: "
        + ", ".join(selected)
    )

else:

    st.warning(
        "⚠️ No algorithms selected."
    )


# =========================================================
# K-MEANS INFORMATION
# =========================================================

if "K-Means Clustering" in selected:

    st.markdown(
        "### 🎯 K-Means Clustering Settings"
    )


    numeric_columns = list(
        df.select_dtypes(
            include="number"
        ).columns
    )


    if len(numeric_columns) < 2:

        st.error(
            "❌ K-Means requires at least two numeric "
            "features in the dataset."
        )

    else:

        km_col1, km_col2 = st.columns(2)


        with km_col1:

            st.write(
                "**🔢 Numeric features available:**"
            )

            st.info(
                ", ".join(numeric_columns)
            )


        with km_col2:

            st.write(
                "**🎯 Target column:**"
            )

            st.info(
                f"{target} "
                "(not used by K-Means)"
            )


# =========================================================
# RUN ANALYSIS
# =========================================================

st.divider()


run_analysis = st.button(
    "▶ Run Analysis",
    type="primary",
    use_container_width=True
)


if run_analysis:

    # =====================================================
    # VALIDATION
    # =====================================================

    if not selected:

        st.error(
            "❌ Please select at least one algorithm."
        )

        st.stop()


    # =====================================================
    # K-MEANS VALIDATION
    # =====================================================

    if "K-Means Clustering" in selected:

        numeric_columns = list(
            df.select_dtypes(
                include="number"
            ).columns
        )


        if len(numeric_columns) < 2:

            st.error(
                "❌ K-Means cannot run because the "
                "dataset contains fewer than two "
                "numeric features."
            )

            st.stop()


    # =====================================================
    # TRAINING
    # =====================================================

    with st.spinner(
        "⚙️ Training algorithms and generating "
        "Weka-style results..."
    ):

        try:

            models, results, le = train_models(
                df,
                target,
                selected
            )

        except Exception as e:

            st.error(
                "❌ Analysis failed."
            )

            st.exception(e)

            st.stop()


    # =====================================================
    # SAVE RESULTS
    # =====================================================

    st.session_state.models = models

    st.session_state.results = results

    st.session_state.label_encoder = le

    st.session_state.selected_algorithms = selected

    st.session_state.analysis_complete = True


    # =====================================================
    # SUCCESS
    # =====================================================

    st.success(
        "✅ Analysis completed successfully!"
    )


# =========================================================
# VIEW RESULTS
# =========================================================

if st.session_state.get(
    "analysis_complete",
    False
):

    st.divider()


    st.success(
        "📊 Your analysis is ready. "
        "You can now view the complete results."
    )


    if st.button(
        "➡️ Next: View Results",
        type="primary",
        use_container_width=True
    ):

        st.switch_page(
            "pages/4_Results.py"
        )