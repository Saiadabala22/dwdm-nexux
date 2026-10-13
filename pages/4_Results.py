import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.ml_engine import (
    tree_figure,
    cluster_figure
)

from utils.report import pdf_report


# =========================================================
# PAGE TITLE
# =========================================================

st.title("📊 Weka-Style Results")


# =========================================================
# CHECK RESULTS
# =========================================================

if not st.session_state.results:

    st.warning("Run an analysis first.")

    st.stop()


# =========================================================
# BASIC DATA
# =========================================================

df = st.session_state.dataset

target = st.session_state.target_column


st.progress(
    1.0,
    text="Step 3 of 3 — Results"
)


# =========================================================
# DATASET INFORMATION
# =========================================================

st.subheader("📋 Dataset Information")


a, b, c, d = st.columns(4)


with a:

    st.metric(
        "Dataset",
        st.session_state.dataset_name
    )


with b:

    st.metric(
        "Instances",
        len(df)
    )


with c:

    st.metric(
        "Attributes",
        len(df.columns)
    )


with d:

    st.metric(
        "Target",
        target
    )


# =========================================================
# ATTRIBUTE INFORMATION
# =========================================================

st.subheader("🔎 Attribute Information")


attribute_table = pd.DataFrame(
    {
        "Attribute": df.columns,

        "Type":
            df.dtypes.astype(str).values,

        "Missing":
            df.isna().sum().values,

        "Unique":
            df.nunique().values
    }
)


st.dataframe(
    attribute_table,
    use_container_width=True
)


# =========================================================
# ALGORITHM COMPARISON
# =========================================================

st.subheader("🤖 Algorithm Comparison")


rows = []


for name, result in (
    st.session_state.results.items()
):

    rows.append(
        {
            "Algorithm":
                name,

            "Accuracy":
                result.get("Accuracy"),

            "Precision":
                result.get("Precision"),

            "Recall":
                result.get("Recall"),

            "F1 Score":
                result.get("F1 Score"),

            "ROC-AUC":
                result.get("ROC-AUC"),

            "Training Time":
                result.get("Training Time (s)"),

            "Prediction Time":
                result.get("Prediction Time (s)"),

            "Silhouette":
                result.get("Silhouette Score")
        }
    )


comparison_df = pd.DataFrame(rows)


st.dataframe(
    comparison_df,
    use_container_width=True
)


# =========================================================
# DETAILED RESULTS
# =========================================================

st.subheader("🔬 Detailed Results")


for name, result in (
    st.session_state.results.items()
):

    with st.expander(
        f"🔎 {name} — Detailed Results",
        expanded=(name == "J48")
    ):

        # =================================================
        # K-MEANS
        # =================================================

        if name == "K-Means Clustering":

            st.subheader("🎯 K-Means Clustering Analysis")

            # -------------------------------------------------
            # Basic K-Means metrics
            # -------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Number of Clusters",
                    result.get("Clusters", "N/A")
                )

            with col2:
                silhouette = result.get("Silhouette Score")

                if silhouette is not None:
                    st.metric(
                        "Silhouette Score",
                        f"{silhouette:.4f}"
                    )
                else:
                    st.metric(
                        "Silhouette Score",
                        "N/A"
                    )

            with col3:
                training_time = result.get("Training Time (s)")

                if training_time is not None:
                    st.metric(
                        "Training Time",
                        f"{training_time:.4f} sec"
                    )
                else:
                    st.metric(
                        "Training Time",
                        "N/A"
                    )

            st.info(
                result.get(
                    "Note",
                    "K-Means is an unsupervised clustering algorithm."
                )
            )

            # -------------------------------------------------
            # Get clustering information
            # -------------------------------------------------

            cluster_coordinates = result.get(
                "Cluster Coordinates"
            )

            cluster_labels = result.get(
                "Cluster Labels"
            )

            cluster_centers = result.get(
                "Cluster Centers"
            )

            explained_variance = result.get(
                "PCA Explained Variance"
            )

            cluster_sizes = result.get(
                "Cluster Sizes"
            )

            cluster_profiles = result.get(
                "Cluster Profiles",
                []
            )

            cluster_names = result.get(
                "Cluster Names"
            )

            cluster_colors = result.get(
                "Cluster Colors"
            )

            # -------------------------------------------------
            # Build fallback profiles if older results exist
            # -------------------------------------------------

            if not cluster_profiles and cluster_sizes is not None:

                cluster_profiles = []

                fallback_colors = [
                    "#2196F3",
                    "#4CAF50",
                    "#9C27B0",
                    "#FF9800",
                    "#00BCD4",
                    "#E91E63",
                    "#795548",
                    "#607D8B"
                ]

                for i, size in enumerate(cluster_sizes):

                    cluster_profiles.append(
                        {
                            "cluster": i,
                            "name": f"Cluster {i}",
                            "dominant_class": "Unknown",
                            "dominant_count": 0,
                            "dominant_percentage": 0.0,
                            "instances": int(size),
                            "color": fallback_colors[
                                i % len(fallback_colors)
                            ],
                            "distribution": {}
                        }
                    )

            if cluster_names is None and cluster_profiles:

                cluster_names = [
                    profile.get(
                        "name",
                        f"Cluster {i}"
                    )
                    for i, profile in enumerate(
                        cluster_profiles
                    )
                ]

            if cluster_colors is None and cluster_profiles:

                cluster_colors = [
                    profile.get(
                        "color",
                        "#2196F3"
                    )
                    for profile in cluster_profiles
                ]

            # =================================================
            # COLORFUL CLUSTER VISUALIZATION
            # =================================================

            st.write(
                "### 🎨 Colorful K-Means Cluster Visualization"
            )

            if (
                cluster_coordinates is not None
                and cluster_labels is not None
            ):

                try:

                    # -------------------------------------------------
                    # Generate graph using the SAME colors and names
                    # shown in the cluster information below.
                    # -------------------------------------------------

                    cluster_fig = cluster_figure(
                        cluster_coordinates,
                        cluster_labels,
                        cluster_centers,
                        explained_variance,
                        cluster_names=cluster_names,
                        cluster_colors=cluster_colors
                    )

                    st.pyplot(
                        cluster_fig,
                        use_container_width=True
                    )

                    st.success(
                        "Colorful clustering visualization generated "
                        "from the uploaded dataset."
                    )

                except TypeError:
                    # Compatibility with an older cluster_figure()
                    # that does not accept names/colors.
                    try:

                        cluster_fig = cluster_figure(
                            cluster_coordinates,
                            cluster_labels,
                            cluster_centers,
                            explained_variance
                        )

                        st.pyplot(
                            cluster_fig,
                            use_container_width=True
                        )

                    except Exception as error:

                        st.warning(
                            "Cluster visualization could not be "
                            f"displayed: {error}"
                        )

                except Exception as error:

                    st.warning(
                        "Cluster visualization could not be "
                        f"displayed: {error}"
                    )

                # =================================================
                # CLUSTER INFORMATION
                # =================================================

                st.write(
                    "### 📊 Cluster Information"
                )

                if cluster_profiles:

                    profile_rows = []

                    for profile in cluster_profiles:

                        distribution = profile.get(
                            "distribution",
                            {}
                        )

                        if distribution:

                            distribution_text = ", ".join(
                                [
                                    f"{label}: {count}"
                                    for label, count
                                    in distribution.items()
                                ]
                            )

                        else:

                            distribution_text = "N/A"

                        profile_rows.append(
                            {
                                "Cluster":
                                    f"Cluster {profile.get('cluster', 'N/A')}",

                                "Cluster Name":
                                    profile.get(
                                        "name",
                                        "Unknown"
                                    ),

                                "Color":
                                    profile.get(
                                        "color",
                                        "N/A"
                                    ),

                                "Instances":
                                    profile.get(
                                        "instances",
                                        0
                                    ),

                                "Dominant Class":
                                    profile.get(
                                        "dominant_class",
                                        "Unknown"
                                    ),

                                "Class Match (%)":
                                    round(
                                        float(
                                            profile.get(
                                                "dominant_percentage",
                                                0
                                            )
                                        ),
                                        2
                                    ),

                                "Class Distribution":
                                    distribution_text
                            }
                        )

                    cluster_info_df = pd.DataFrame(
                        profile_rows
                    )

                    st.dataframe(
                        cluster_info_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # -------------------------------------------------
                    # Visual color legend
                    # -------------------------------------------------

                    st.write(
                        "### 🎨 Cluster Color Legend"
                    )

                    legend_columns = st.columns(
                        min(
                            max(len(cluster_profiles), 1),
                            4
                        )
                    )

                    for i, profile in enumerate(
                        cluster_profiles
                    ):

                        color = profile.get(
                            "color",
                            "#2196F3"
                        )

                        name_value = profile.get(
                            "name",
                            f"Cluster {i}"
                        )

                        instances = profile.get(
                            "instances",
                            0
                        )

                        with legend_columns[
                            i % len(legend_columns)
                        ]:

                            st.markdown(
                                f"""
                                <div style="
                                    border: 1px solid {color};
                                    border-radius: 12px;
                                    padding: 14px;
                                    margin: 5px 0;
                                    background: rgba(255,255,255,0.03);
                                ">
                                    <div style="
                                        display:flex;
                                        align-items:center;
                                        gap:10px;
                                    ">
                                        <span style="
                                            width:18px;
                                            height:18px;
                                            border-radius:50%;
                                            display:inline-block;
                                            background:{color};
                                            box-shadow:0 0 8px {color};
                                        "></span>

                                        <b>{name_value}</b>
                                    </div>

                                    <div style="
                                        margin-top:8px;
                                        font-size:14px;
                                    ">
                                        Instances: <b>{instances}</b>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                elif cluster_sizes is not None:

                    cluster_info = pd.DataFrame(
                        {
                            "Cluster":
                                [
                                    f"Cluster {i}"
                                    for i in range(
                                        len(cluster_sizes)
                                    )
                                ],

                            "Number of Instances":
                                cluster_sizes
                        }
                    )

                    st.dataframe(
                        cluster_info,
                        use_container_width=True
                    )

                # =================================================
                # PCA INFORMATION
                # =================================================

                if explained_variance is not None:

                    st.write(
                        "### 📐 PCA Information"
                    )

                    pca_info = pd.DataFrame(
                        {
                            "Component": [
                                "PCA Component 1",
                                "PCA Component 2"
                            ],

                            "Variance Explained (%)": [
                                round(
                                    float(
                                        explained_variance[0]
                                    ),
                                    2
                                ),

                                round(
                                    float(
                                        explained_variance[1]
                                    ),
                                    2
                                )
                            ]
                        }
                    )

                    st.dataframe(
                        pca_info,
                        use_container_width=True,
                        hide_index=True
                    )

                    total_variance = (
                        float(explained_variance[0])
                        +
                        float(explained_variance[1])
                    )

                    st.info(
                        f"PCA reduces the processed dataset to two "
                        f"dimensions for visualization. The first two "
                        f"components explain approximately "
                        f"**{total_variance:.2f}%** of the variance."
                    )

            else:

                st.warning(
                    "Cluster coordinates were not found. "
                    "Make sure the updated K-Means code in "
                    "utils/ml_engine.py is being used."
                )

            # -------------------------------------------------
            # K-Means is unsupervised, so don't show the
            # classification-only sections below.
            # -------------------------------------------------

            continue

        # =================================================
        # CLASSIFICATION METRICS
        # =================================================

        if "Accuracy" in result:

            x, y, z, q, r = st.columns(5)


            with x:

                st.metric(
                    "Accuracy",
                    f"{result['Accuracy']:.2%}"
                )


            with y:

                st.metric(
                    "Precision",
                    f"{result['Precision']:.2%}"
                )


            with z:

                st.metric(
                    "Recall",
                    f"{result['Recall']:.2%}"
                )


            with q:

                st.metric(
                    "F1 Score",
                    f"{result['F1 Score']:.2%}"
                )


            with r:

                roc_auc = result.get(
                    "ROC-AUC"
                )


                if roc_auc is not None:

                    st.metric(
                        "ROC-AUC",
                        f"{roc_auc:.4f}"
                    )

                else:

                    st.metric(
                        "ROC-AUC",
                        "N/A"
                    )


        # =================================================
        # DECISION TREE
        # =================================================

        if (
            name in ["ID3", "J48"]
            and
            name in st.session_state.models
        ):

            st.write(
                "### 🌳 Colored Decision Tree"
            )


            try:

                class_names = list(
                    st.session_state
                    .label_encoder
                    .classes_
                )


                tree = tree_figure(
                    st.session_state.models[name],
                    class_names
                )


                st.pyplot(
                    tree,
                    use_container_width=True
                )


            except Exception as error:

                st.warning(
                    f"Decision tree could not be displayed: "
                    f"{error}"
                )


        # =================================================
        # ROC CURVE
        # =================================================

        st.write(
            "### 📈 ROC Curve"
        )


        roc_data = result.get(
            "ROC"
        )


        if roc_data is not None:

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )


            # =================================================
            # BINARY ROC
            # =================================================

            if roc_data.get("type") == "binary":

                ax.plot(
                    roc_data["fpr"],
                    roc_data["tpr"],
                    linewidth=3,
                    label=(
                        f"{name} "
                        f"(AUC = "
                        f"{roc_data['auc']:.4f})"
                    )
                )


                ax.set_title(
                    f"ROC Curve — {name} "
                    f"(AUC = "
                    f"{roc_data['auc']:.4f})"
                )


            # =================================================
            # MULTICLASS ROC
            # =================================================

            elif roc_data.get("type") == "multiclass":

                for class_name, curve in (
                    roc_data["curves"].items()
                ):

                    ax.plot(
                        curve["fpr"],
                        curve["tpr"],
                        linewidth=2,
                        label=(
                            f"{class_name} "
                            f"(AUC = "
                            f"{curve['auc']:.4f})"
                        )
                    )


                macro_auc = roc_data.get(
                    "macro_auc"
                )


                ax.set_title(
                    f"Multiclass ROC Curve — "
                    f"{name}\n"
                    f"Macro AUC = "
                    f"{macro_auc:.4f}"
                )


            # =================================================
            # RANDOM CLASSIFIER LINE
            # =================================================

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                linewidth=1.5,
                label="Random Classifier"
            )


            # =================================================
            # AXIS
            # =================================================

            ax.set_xlabel(
                "False Positive Rate"
            )


            ax.set_ylabel(
                "True Positive Rate"
            )


            ax.set_xlim(
                0,
                1
            )


            ax.set_ylim(
                0,
                1.05
            )


            ax.grid(
                alpha=0.25
            )


            ax.legend(
                loc="lower right"
            )


            fig.tight_layout()


            st.pyplot(
                fig,
                use_container_width=True
            )


            # =================================================
            # ROC INFORMATION
            # =================================================

            if roc_data.get("type") == "binary":

                st.success(
                    f"ROC-AUC Score: "
                    f"{roc_data['auc']:.4f}"
                )


            elif roc_data.get("type") == "multiclass":

                st.success(
                    f"Macro ROC-AUC Score: "
                    f"{roc_data['macro_auc']:.4f}"
                )


        else:

            st.info(
                "ROC curve is not available "
                "for this algorithm."
            )


        # =================================================
        # CONFUSION MATRIX
        # =================================================

        if "Confusion Matrix" in result:

            st.write(
                "### 🔢 Confusion Matrix"
            )


            cm = result[
                "Confusion Matrix"
            ]


            try:

                class_names = list(
                    st.session_state
                    .label_encoder
                    .classes_
                )


                cm_df = pd.DataFrame(
                    cm,

                    index=[
                        f"Actual {class_name}"
                        for class_name in class_names
                    ],

                    columns=[
                        f"Predicted {class_name}"
                        for class_name in class_names
                    ]
                )


                st.dataframe(
                    cm_df,
                    use_container_width=True
                )


            except Exception:

                st.dataframe(
                    pd.DataFrame(cm),
                    use_container_width=True
                )


        # =================================================
        # CLASSIFICATION REPORT
        # =================================================

        if "Classification Report" in result:

            st.write(
                "### 📄 Classification Report"
            )


            st.code(
                result[
                    "Classification Report"
                ]
            )


        # =================================================
        # WEKA-STYLE EVALUATION
        # =================================================

        if "Accuracy" in result:

            st.write(
                "### 📋 Weka-Style Evaluation Summary"
            )


            y_test = result.get(
                "y_test"
            )


            if y_test is not None:

                total_instances = len(
                    y_test
                )


                correct = int(
                    round(
                        result["Accuracy"]
                        *
                        total_instances
                    )
                )


                incorrect = (
                    total_instances
                    -
                    correct
                )


                st.write(
                    f"**Total Instances:** "
                    f"{total_instances}"
                )


                st.write(
                    f"**Correctly Classified Instances:** "
                    f"{correct}"
                )


                st.write(
                    f"**Incorrectly Classified Instances:** "
                    f"{incorrect}"
                )


            st.write(
                f"**Accuracy:** "
                f"{result['Accuracy']:.4f}"
            )


            st.write(
                f"**Precision:** "
                f"{result['Precision']:.4f}"
            )


            st.write(
                f"**Recall:** "
                f"{result['Recall']:.4f}"
            )


            st.write(
                f"**F1 Score:** "
                f"{result['F1 Score']:.4f}"
            )


            roc_auc = result.get(
                "ROC-AUC"
            )


            if roc_auc is not None:

                st.write(
                    f"**ROC Area:** "
                    f"{roc_auc:.4f}"
                )

            else:

                st.write(
                    "**ROC Area:** N/A"
                )


        # =================================================
        # EXECUTION TIME
        # =================================================

        if "Training Time (s)" in result:

            st.write(
                "### ⏱ Execution Time"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Training Time",
                    f"{result['Training Time (s)']:.4f} sec"
                )


            with col2:

                st.metric(
                    "Prediction Time",
                    f"{result['Prediction Time (s)']:.4f} sec"
                )


# =========================================================
# DATASET STATISTICAL SUMMARY
# =========================================================

st.subheader(
    "📊 Dataset Statistical Summary"
)


st.dataframe(
    df.describe(
        include="all"
    ).T,

    use_container_width=True
)


# =========================================================
# PDF REPORT
# =========================================================

st.subheader(
    "📥 Download Report"
)


try:

    pdf = pdf_report(
        st.session_state.dataset_name,
        df,
        target,
        st.session_state.results,
        st.session_state.models,
        st.session_state.label_encoder,
        extra_charts=st.session_state.get("report_charts", []),
        preprocessing_log=st.session_state.get("preprocess_log", [])
    )


    st.download_button(
        "📥 Download PDF Report",
        pdf,
        "dataset_analyzer_report.pdf",
        "application/pdf",
        use_container_width=True
    )


except Exception as error:

    st.error(
        f"PDF report could not be generated: {error}"
    )