import streamlit as st
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Dataset Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD CSS
# =========================================================

css_file = (
    Path(__file__).resolve().parent
    / "static"
    / "css"
    / "animation.css"
)

if css_file.exists():

    with open(
        css_file,
        "r",
        encoding="utf-8"
    ) as f:

        st.markdown(
            "<style>" + f.read() + "</style>",
            unsafe_allow_html=True
        )


# =========================================================
# SESSION STATE
# =========================================================

if "step" not in st.session_state:
    st.session_state.step = 0


defaults = {
    "dataset": None,
    "dataset_name": None,
    "target_column": None,
    "selected_algorithms": [],
    "results": {},
    "models": {},
    "task_type": None,
    "processed_data": None
}


for key, default in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # Simple native Streamlit sidebar
    # No custom HTML here.

    st.markdown(
        "# 🧠 Dataset Analyzer"
    )

    st.caption(
        "Weka-style Machine Learning Analysis"
    )

    st.divider()

    st.page_link(
        "app.py",
        label="Welcome",
        icon="👋"
    )

    st.page_link(
        "pages/2_Upload_Dataset.py",
        label="Upload Dataset",
        icon="📤"
    )

    st.page_link(
        "pages/7_Data_Preprocessing.py",
        label="Data Preprocessing",
        icon="🧹"
    )

    st.page_link(
        "pages/3_Algorithm_Selection.py",
        label="Algorithm Selection",
        icon="🤖"
    )

    st.page_link(
        "pages/4_Results.py",
        label="Results",
        icon="📊"
    )

    st.page_link(
        "pages/5_Dataset_Information.py",
        label="Dataset Information",
        icon="📋"
    )

    st.page_link(
        "pages/6_Visualization.py",
        label="Data Visualization",
        icon="📈"
    )

    st.page_link(
        "pages/8_OLAP_Operations.py",
        label="OLAP Operations",
        icon="🧊"
    )

    st.divider()

    if st.session_state.dataset is not None:

        st.success(
            f"Loaded: {st.session_state.dataset_name}"
        )

    else:

        st.caption(
            "No dataset loaded yet"
        )


# =========================================================
# HERO SECTION
# =========================================================

st.html(
    """
    <div class="hero-wrapper">

        <div class="welcome-card">

            <div class="hero-badge">
                ✦ MACHINE LEARNING DATASET ANALYSIS
            </div>

            <h1 class="welcome-title">
                Welcome to Dataset Analyzer
            </h1>

            <p class="welcome-description">
                A professional Weka-inspired environment for
                dataset exploration, algorithm selection,
                machine learning evaluation, decision-tree
                visualization, clustering and ROC analysis.
            </p>

            <div class="hero-line"></div>

            <div class="hero-mini-text">
                Upload &nbsp;•&nbsp;
                Analyze &nbsp;•&nbsp;
                Visualize &nbsp;•&nbsp;
                Evaluate
            </div>

        </div>

    </div>
    """
)


# =========================================================
# HOW IT WORKS
# =========================================================

st.html(
    """
    <div class="features-heading">

        <div class="section-kicker">
            HOW IT WORKS
        </div>

        <h2>
            From Dataset to Insight
        </h2>

        <p>
            Analyze your data through a simple,
            professional machine learning workflow.
        </p>

    </div>
    """
)


# =========================================================
# FEATURE CARDS
# =========================================================

c1, c2, c3 = st.columns(
    3,
    gap="large"
)


# =========================================================
# UPLOAD
# =========================================================

with c1:

    st.html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                📤
            </div>

            <div class="feature-number">
                01
            </div>

            <h3>
                Upload Dataset
            </h3>

            <p>
                Import your dataset in CSV, XLS,
                XLSX or ARFF format and begin
                exploring its structure.
            </p>

            <div class="feature-tags">

                <span>CSV</span>
                <span>XLS</span>
                <span>XLSX</span>
                <span>ARFF</span>

            </div>

        </div>
        """
    )


# =========================================================
# ANALYZE
# =========================================================

with c2:

    st.html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🤖
            </div>

            <div class="feature-number">
                02
            </div>

            <h3>
                Analyze Data
            </h3>

            <p>
                Select machine learning algorithms
                and evaluate your dataset using
                Weka-style analysis.
            </p>

            <div class="feature-tags">

                <span>ID3</span>
                <span>J48</span>
                <span>K-NN</span>
                <span>K-Means</span>

            </div>

        </div>
        """
    )


# =========================================================
# RESULTS
# =========================================================

with c3:

    st.html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                📊
            </div>

            <div class="feature-number">
                03
            </div>

            <h3>
                View Results
            </h3>

            <p>
                Explore accuracy, ROC curves,
                colored decision trees,
                clustering and detailed metrics.
            </p>

            <div class="feature-tags">

                <span>ROC</span>
                <span>Trees</span>
                <span>Metrics</span>
                <span>Clusters</span>

            </div>

        </div>
        """
    )


# =========================================================
# WORKFLOW
# =========================================================

st.html(
    """
    <div class="workflow">

        <div class="workflow-item active">
            <span>01</span>
            Dataset
        </div>

        <div class="workflow-line"></div>

        <div class="workflow-item">
            <span>02</span>
            Algorithms
        </div>

        <div class="workflow-line"></div>

        <div class="workflow-item">
            <span>03</span>
            Results
        </div>

    </div>
    """
)


# =========================================================
# READY SECTION
# =========================================================

st.html(
    """
    <div class="continue-heading">

        <div>
            Ready to begin?
        </div>

        <span>
            Upload your dataset and start the analysis.
        </span>

    </div>
    """
)


# =========================================================
# CONTINUE BUTTON
# =========================================================

if st.button(
    "🚀  Continue to Upload Dataset",
    type="primary",
    use_container_width=True
):

    st.session_state.step = 1

    st.switch_page(
        "pages/2_Upload_Dataset.py"
    )


# =========================================================
# FOOTER
# =========================================================

st.html(
    """
    <div class="premium-footer">

        <div>
            🧠 <strong>Dataset Analyzer</strong>
        </div>

        <div>
            Weka-style Machine Learning Analysis
        </div>

    </div>
    """
)