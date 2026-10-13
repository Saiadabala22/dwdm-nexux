import streamlit as st

from utils.file_loader import load_dataset
from utils.theme import set_dataset_background, apply_saved_background


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Upload Dataset",
    page_icon="📤",
    layout="wide"
)


# =========================================================
# APPLY DEFAULT / SAVED BACKGROUND
# =========================================================

apply_saved_background()


# =========================================================
# PAGE TITLE
# =========================================================

st.title("📤 Upload Dataset")

st.progress(
    0.33,
    text="Step 1 of 3 — Upload Dataset"
)


# =========================================================
# FILE UPLOADER
# =========================================================

st.markdown(
    "### 📂 Upload your dataset"
)

f = st.file_uploader(
    "Drag and drop your dataset",
    type=["csv", "xls", "xlsx", "arff"]
)


# =========================================================
# DATASET UPLOADED
# =========================================================

if f:

    try:

        # -------------------------------------------------
        # LOAD DATASET
        # -------------------------------------------------

        df = load_dataset(f)


        # -------------------------------------------------
        # SAVE DATASET
        # -------------------------------------------------

        # Only replace the stored dataset when a NEW file is uploaded, so that
        # preprocessing done on the Data Preprocessing page is not overwritten.
        file_signature = (f.name, f.size)

        if st.session_state.get("upload_signature") != file_signature:

            st.session_state.dataset = df

            st.session_state.dataset_name = f.name

            st.session_state.original_dataset = df.copy()

            st.session_state.preprocess_log = []

            st.session_state.preprocess_stack = []

            st.session_state.report_charts = []

            st.session_state.results = {}

            st.session_state.models = {}

            st.session_state.upload_signature = file_signature

        # Everything below shows the current (possibly preprocessed) dataset.
        df = st.session_state.dataset


        # -------------------------------------------------
        # DETECT DATASET TYPE
        # AND CHANGE BACKGROUND
        # -------------------------------------------------

        set_dataset_background(df)


        # -------------------------------------------------
        # SUCCESS
        # -------------------------------------------------

        st.success(
            "✅ Dataset uploaded successfully."
        )


        # -------------------------------------------------
        # DATASET INFORMATION
        # -------------------------------------------------

        st.markdown(
            "### 📊 Dataset Ready"
        )


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
                "📌 Instances / Rows",
                len(df)
            )


        with col3:

            st.metric(
                "📋 Attributes / Columns",
                len(df.columns)
            )


        # -------------------------------------------------
        # MISSING VALUES
        # -------------------------------------------------

        missing_values = int(
            df.isna().sum().sum()
        )


        st.metric(
            "⚠️ Missing Values",
            missing_values
        )


        # -------------------------------------------------
        # DETECTED DATASET TYPE
        # -------------------------------------------------

        theme = st.session_state.get(
            "dataset_theme",
            "default"
        )


        theme_names = {

            "paddy":
                "🌾 Paddy / Agriculture Dataset",

            "medical":
                "🏥 Medical / Healthcare Dataset",

            "loan":
                "💰 Loan / Finance Dataset",

            "smartphone":
                "📱 Smartphone Dataset",

            "student":
                "🎓 Student / Education Dataset",

            "default":
                "📊 General Dataset"
        }


        st.info(
            "🔍 Detected Dataset Type: "
            f"**{theme_names.get(theme, '📊 General Dataset')}**"
        )


        # -------------------------------------------------
        # DATASET PREVIEW
        # -------------------------------------------------

        st.markdown(
            "### 📋 Dataset Preview"
        )


        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=400
        )


        # -------------------------------------------------
        # DATASET ATTRIBUTES
        # -------------------------------------------------

        st.markdown(
            "### 🏷️ Dataset Attributes"
        )


        attribute_columns = st.columns(4)


        for i, column in enumerate(df.columns):

            with attribute_columns[i % 4]:

                st.markdown(
                    f"""
                    <div style="
                        background: rgba(255,255,255,0.92);
                        padding: 10px 14px;
                        margin-bottom: 10px;
                        border-radius: 10px;
                        border: 1px solid rgba(0,0,0,0.08);
                        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
                    ">
                        <b style="color:#17351e;">
                            {i + 1}. {column}
                        </b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # -------------------------------------------------
        # NEXT BUTTON
        # -------------------------------------------------

        st.markdown("---")


        if st.button(
            "🧹 Optional: Data Preprocessing",
            use_container_width=True
        ):

            st.switch_page(
                "pages/7_Data_Preprocessing.py"
            )


        if st.button(
            "➡️ Next: Algorithm Selection",
            type="primary",
            use_container_width=True
        ):

            st.switch_page(
                "pages/3_Algorithm_Selection.py"
            )


    except Exception as e:

        st.error(
            f"❌ Error loading dataset: {str(e)}"
        )

        st.exception(e)


# =========================================================
# NO DATASET
# =========================================================

else:

    st.info(
        "📂 Upload a dataset to continue."
    )