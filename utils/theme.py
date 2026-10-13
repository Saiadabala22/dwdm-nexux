import streamlit as st


# =========================================================
# DATASET THEME DETECTION
# =========================================================

def detect_dataset_theme(df):

    columns = " ".join(
        str(col).lower().replace("_", " ")
        for col in df.columns
    )

    themes = {

        "paddy": [
            "paddy",
            "rice",
            "crop",
            "yield",
            "rainfall",
            "nitrogen",
            "phosphorus",
            "potassium",
            "soil",
            "humidity",
            "temperature",
            "fertilizer",
            "irrigation",
            "agriculture",
            "farm"
        ],

        "medical": [
            "patient",
            "disease",
            "diagnosis",
            "blood",
            "glucose",
            "pressure",
            "heart",
            "kidney",
            "diabetes",
            "medical",
            "symptom"
        ],

        "loan": [
            "loan",
            "income",
            "credit",
            "interest",
            "employment",
            "borrower",
            "debt",
            "payment"
        ],

        "smartphone": [
            "smartphone",
            "mobile",
            "phone",
            "screen",
            "gaming",
            "usage",
            "addiction",
            "sleep",
            "social media"
        ],

        "student": [
            "student",
            "marks",
            "grade",
            "attendance",
            "education",
            "study",
            "exam",
            "academic"
        ]
    }

    scores = {}

    for theme, keywords in themes.items():

        score = 0

        for keyword in keywords:

            if keyword in columns:
                score += 1

        scores[theme] = score

    detected_theme = max(
        scores,
        key=scores.get
    )

    if scores[detected_theme] == 0:
        return "default"

    return detected_theme


# =========================================================
# BACKGROUND IMAGES
# =========================================================

BACKGROUND_IMAGES = {

    "paddy":
        "https://images.unsplash.com/photo-1536304993881-ff6e9eefa2a6"
        "?auto=format&fit=crop&w=2000&q=90",

    "medical":
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d"
        "?auto=format&fit=crop&w=2000&q=90",

    "loan":
        "https://images.unsplash.com/photo-1554224155-6726b3ff858f"
        "?auto=format&fit=crop&w=2000&q=90",

    "smartphone":
        "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9"
        "?auto=format&fit=crop&w=2000&q=90",

    "student":
        "https://images.unsplash.com/photo-1523240795612-9a054b0db644"
        "?auto=format&fit=crop&w=2000&q=90",

    "default":
        "https://images.unsplash.com/photo-1557683316-973673baf926"
        "?auto=format&fit=crop&w=2000&q=90"
}


# =========================================================
# COMMON THEME CSS
# =========================================================

def apply_theme_css(background):

    st.markdown(
        f"""
        <style>

        /* =================================================
           MAIN BACKGROUND
        ================================================= */

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255,255,255,0.72),
                    rgba(255,255,255,0.72)
                ),
                url("{background}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}


        /* =================================================
           MAIN CONTENT
        ================================================= */

        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}


        /* =================================================
           PAGE HEADINGS
        ================================================= */

        h1 {{
            color: #123d1c !important;
            font-weight: 800 !important;
        }}

        h2 {{
            color: #123d1c !important;
            font-weight: 800 !important;
        }}

        h3 {{
            color: #173d20 !important;
            font-weight: 800 !important;
        }}

        h4 {{
            color: #173d20 !important;
            font-weight: 700 !important;
        }}


        /* =================================================
           NORMAL STREAMLIT TEXT
           
           IMPORTANT:
           Do NOT style every div/span globally.
        ================================================= */

        .stMarkdown p {{
            color: #1b2e20 !important;
        }}

        .stText {{
            color: #1b2e20 !important;
        }}

        label {{
            color: #1b2e20 !important;
        }}


        /* =================================================
           PROGRESS BAR
        ================================================= */

        [data-testid="stProgress"] {{
            background: rgba(255,255,255,0.85);
            border-radius: 10px;
        }}


        /* =================================================
           METRIC CARDS
        ================================================= */

        [data-testid="stMetric"] {{
            background: rgba(255,255,255,0.96) !important;

            border-radius: 18px !important;

            padding: 20px !important;

            border: 1px solid rgba(0,0,0,0.10) !important;

            box-shadow:
                0 6px 20px rgba(0,0,0,0.16) !important;
        }}


        /* Metric label */

        [data-testid="stMetricLabel"] {{
            color: #31543a !important;
            font-weight: 700 !important;
        }}

        [data-testid="stMetricLabel"] * {{
            color: #31543a !important;
        }}


        /* Metric value */

        [data-testid="stMetricValue"] {{
            color: #102f18 !important;
            font-weight: 800 !important;
        }}

        [data-testid="stMetricValue"] * {{
            color: #102f18 !important;
        }}


        /* Metric delta */

        [data-testid="stMetricDelta"] {{
            color: #245d31 !important;
        }}

        [data-testid="stMetricDelta"] * {{
            color: #245d31 !important;
        }}


        /* =================================================
           FILE UPLOADER
        ================================================= */

        [data-testid="stFileUploader"] {{
            background: rgba(255,255,255,0.96) !important;

            border-radius: 18px !important;

            padding: 18px !important;

            box-shadow:
                0 6px 20px rgba(0,0,0,0.15) !important;
        }}

        [data-testid="stFileUploader"] label {{
            color: #17351e !important;
            font-weight: 700 !important;
        }}

        [data-testid="stFileUploader"] section {{
            background: rgba(255,255,255,0.96) !important;
        }}

        [data-testid="stFileUploader"] small {{
            color: #405246 !important;
        }}


        /* =================================================
           FILE UPLOADER FILE NAME
        ================================================= */

        [data-testid="stFileUploaderFileName"] {{
            color: #17351e !important;
            font-weight: 700 !important;
        }}


        /* =================================================
           SELECT BOX
        ================================================= */

        [data-baseweb="select"] > div {{
            background: rgba(255,255,255,0.97) !important;

            border-radius: 10px !important;

            border: 1px solid #b7c8ba !important;
        }}

        [data-baseweb="select"] {{
            color: #17251a !important;
        }}

        [data-baseweb="select"] span {{
            color: #17251a !important;
        }}


        /* =================================================
           SELECTBOX DROPDOWN
        ================================================= */

        [role="listbox"] {{
            background: white !important;
        }}

        [role="option"] {{
            color: #17251a !important;
            background: white !important;
        }}

        [role="option"]:hover {{
            background: #edf5ee !important;
        }}


        /* =================================================
           INFO BOX
        ================================================= */

        [data-testid="stAlert"] {{
            background: rgba(255,255,255,0.92) !important;

            border-radius: 14px !important;

            border: 1px solid rgba(30,80,40,0.15) !important;
        }}

        [data-testid="stAlert"] p {{
            color: #23452b !important;
        }}

        [data-testid="stAlert"] * {{
            color: #23452b !important;
        }}


        /* =================================================
           SUCCESS MESSAGE
        ================================================= */

        [data-testid="stAlert"][kind="success"] {{
            background: rgba(225,250,231,0.94) !important;
        }}


        /* =================================================
           DATAFRAME
        ================================================= */

        [data-testid="stDataFrame"] {{
            background: rgba(255,255,255,0.97) !important;

            border-radius: 16px !important;

            padding: 8px !important;

            box-shadow:
                0 5px 18px rgba(0,0,0,0.15) !important;
        }}


        /* =================================================
           BUTTONS
        ================================================= */

        .stButton > button {{
            border-radius: 12px !important;

            font-weight: 700 !important;

            min-height: 45px !important;
        }}

        .stButton > button p {{
            font-weight: 700 !important;
        }}


        /* =================================================
           ALGORITHM CARDS
        ================================================= */

        .algorithm-card {{
            background: rgba(255,255,255,0.96);

            border-radius: 16px;

            padding: 20px;

            border: 1px solid rgba(0,0,0,0.08);

            box-shadow:
                0 6px 18px rgba(0,0,0,0.13);
        }}


        /* =================================================
           SIDEBAR
        ================================================= */

        section[data-testid="stSidebar"] {{
            background:
                rgba(255,255,255,0.94) !important;

            backdrop-filter: blur(12px);
        }}

        section[data-testid="stSidebar"] p {{
            color: #17351e !important;
        }}


        /* =================================================
           DIVIDERS
        ================================================= */

        hr {{
            border-color:
                rgba(30,80,40,0.25) !important;
        }}


        /* =================================================
           RADIO / CHECKBOX
        ================================================= */

        [data-testid="stCheckbox"] label {{
            color: #17251a !important;
        }}

        [role="radiogroup"] label {{
            color: #17251a !important;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SET BACKGROUND AFTER UPLOAD
# =========================================================

def set_dataset_background(df):

    theme = detect_dataset_theme(df)

    background = BACKGROUND_IMAGES.get(
        theme,
        BACKGROUND_IMAGES["default"]
    )

    st.session_state["dataset_theme"] = theme

    st.session_state["background_image"] = background

    apply_theme_css(background)


# =========================================================
# APPLY SAVED BACKGROUND
# =========================================================

def apply_saved_background():

    background = st.session_state.get(
        "background_image",
        BACKGROUND_IMAGES["default"]
    )

    apply_theme_css(background)