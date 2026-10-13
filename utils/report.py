import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether
)

from utils.ml_engine import tree_figure


# =========================================================
# HELPER: MATPLOTLIB FIGURE → PDF IMAGE
# =========================================================

def figure_to_image(fig, width=6.5 * inch):

    image_buffer = io.BytesIO()

    fig.savefig(
        image_buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white"
    )

    image_buffer.seek(0)

    image = Image(
        image_buffer
    )

    # Keep aspect ratio
    if image.imageWidth:
        ratio = (
            width / image.imageWidth
        )

        image.drawWidth = width
        image.drawHeight = (
            image.imageHeight * ratio
        )

    return image


# =========================================================
# HEADER / FOOTER
# =========================================================

def add_page_number(canvas, document):

    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        8
    )

    canvas.drawCentredString(
        A4[0] / 2,
        0.4 * inch,
        f"Dataset Analyzer — Page {document.page}"
    )

    canvas.restoreState()


# =========================================================
# MAIN PDF FUNCTION
# =========================================================

def pdf_report(
    dataset_name,
    df,
    target,
    results,
    models=None,
    label_encoder=None,
    extra_charts=None,
    preprocessing_log=None
):

    if models is None:
        models = {}

    if extra_charts is None:
        extra_charts = []

    if preprocessing_log is None:
        preprocessing_log = []

    # -----------------------------------------------------
    # PDF BUFFER
    # -----------------------------------------------------

    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=45,
        bottomMargin=45
    )

    # -----------------------------------------------------
    # STYLES
    # -----------------------------------------------------

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading1"],
        fontSize=17,
        leading=22,
        spaceBefore=12,
        spaceAfter=10
    )

    subheading_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        spaceBefore=8,
        spaceAfter=7
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8,
        leading=11
    )

    # -----------------------------------------------------
    # STORY
    # -----------------------------------------------------

    story = []

    # =====================================================
    # COVER / TITLE
    # =====================================================

    story.append(
        Spacer(1, 25)
    )

    story.append(
        Paragraph(
            "Dataset Analyzer",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Weka-Style Machine Learning Analysis Report",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 15)
    )

    # Dataset summary table

    dataset_summary = [
        ["Dataset", str(dataset_name)],
        ["Instances / Rows", str(len(df))],
        ["Attributes / Columns", str(len(df.columns))],
        ["Target / Class", str(target)],
        [
            "Missing Values",
            str(int(df.isna().sum().sum()))
        ],
        [
            "Duplicate Rows",
            str(int(df.duplicated().sum()))
        ]
    ]

    summary_table = Table(
        dataset_summary,
        colWidths=[2.0 * inch, 4.3 * inch]
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#1769AA")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (0, -1),
                    colors.white
                ),
                (
                    "BACKGROUND",
                    (1, 0),
                    (1, -1),
                    colors.HexColor("#F3F6FA")
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    "Helvetica"
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ]
        )
    )

    story.append(
        summary_table
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Generated by Dataset Analyzer",
            normal_style
        )
    )

    story.append(
        PageBreak()
    )

    # =====================================================
    # ATTRIBUTE INFORMATION
    # =====================================================

    story.append(
        Paragraph(
            "1. Dataset Attribute Information",
            heading_style
        )
    )

    attribute_data = [
        [
            "Attribute",
            "Data Type",
            "Missing",
            "Unique"
        ]
    ]

    for column in df.columns:

        attribute_data.append(
            [
                str(column),
                str(df[column].dtype),
                str(int(df[column].isna().sum())),
                str(int(df[column].nunique()))
            ]
        )

    attribute_table = Table(
        attribute_data,
        repeatRows=1,
        colWidths=[
            2.2 * inch,
            1.5 * inch,
            1.1 * inch,
            1.1 * inch
        ]
    )

    attribute_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1769AA")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ]
        )
    )

    story.append(
        attribute_table
    )

    story.append(
        Spacer(1, 15)
    )

    # =====================================================
    # STATISTICAL SUMMARY
    # =====================================================

    story.append(
        Paragraph(
            "2. Statistical Summary",
            heading_style
        )
    )

    try:

        statistics = df.describe(
            include="all"
        ).T

        statistics = statistics.fillna("")

        stat_headers = [
            "Attribute"
        ] + list(
            statistics.columns
        )

        stat_data = [
            stat_headers
        ]

        for index, row in statistics.iterrows():

            values = [
                str(index)
            ]

            for value in row:

                if isinstance(
                    value,
                    (float, np.floating)
                ):

                    values.append(
                        f"{value:.4f}"
                    )

                else:

                    values.append(
                        str(value)
                    )

            stat_data.append(
                values
            )

        # Limit excessively wide tables
        if len(stat_data[0]) > 8:

            stat_data = [
                row[:8]
                for row in stat_data
            ]

        statistic_table = Table(
            stat_data,
            repeatRows=1,
            colWidths=None
        )

        statistic_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1769AA")
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.3,
                        colors.grey
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        4
                    )
                ]
            )
        )

        story.append(
            statistic_table
        )

    except Exception as error:

        story.append(
            Paragraph(
                f"Statistics unavailable: {error}",
                normal_style
            )
        )

    story.append(
        PageBreak()
    )

    # =====================================================
    # ALGORITHM COMPARISON
    # =====================================================

    story.append(
        Paragraph(
            "3. Algorithm Comparison",
            heading_style
        )
    )

    comparison_data = [
        [
            "Algorithm",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
    ]

    for name, result in results.items():

        accuracy = result.get(
            "Accuracy"
        )

        precision = result.get(
            "Precision"
        )

        recall = result.get(
            "Recall"
        )

        f1 = result.get(
            "F1 Score"
        )

        roc_auc = result.get(
            "ROC-AUC"
        )

        comparison_data.append(
            [
                str(name),

                f"{accuracy:.4f}"
                if isinstance(
                    accuracy,
                    (int, float)
                )
                else "-",

                f"{precision:.4f}"
                if isinstance(
                    precision,
                    (int, float)
                )
                else "-",

                f"{recall:.4f}"
                if isinstance(
                    recall,
                    (int, float)
                )
                else "-",

                f"{f1:.4f}"
                if isinstance(
                    f1,
                    (int, float)
                )
                else "-",

                f"{roc_auc:.4f}"
                if isinstance(
                    roc_auc,
                    (int, float)
                )
                else "-"
            ]
        )

    comparison_table = Table(
        comparison_data,
        repeatRows=1,
        colWidths=[
            1.55 * inch,
            0.85 * inch,
            0.85 * inch,
            0.85 * inch,
            0.75 * inch,
            0.85 * inch
        ]
    )

    comparison_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1769AA")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )

    story.append(
        comparison_table
    )

    story.append(
        PageBreak()
    )

    # =====================================================
    # EACH ALGORITHM
    # =====================================================

    for name, result in results.items():

        # -------------------------------------------------
        # ALGORITHM TITLE
        # -------------------------------------------------

        story.append(
            Paragraph(
                f"4. {name} — Detailed Evaluation",
                heading_style
            )
        )

        # =================================================
        # K-MEANS
        # =================================================

        if name == "K-Means Clustering":

            cluster_data = [
                [
                    "Metric",
                    "Value"
                ],
                [
                    "Number of Clusters",
                    str(
                        result.get(
                            "Clusters",
                            "N/A"
                        )
                    )
                ],
                [
                    "Silhouette Score",
                    (
                        f"{result['Silhouette Score']:.4f}"
                        if isinstance(
                            result.get(
                                "Silhouette Score"
                            ),
                            (int, float)
                        )
                        else "N/A"
                    )
                ]
            ]

            cluster_table = Table(
                cluster_data,
                colWidths=[
                    3 * inch,
                    3 * inch
                ]
            )

            cluster_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#1769AA")
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.4,
                            colors.grey
                        ),
                        (
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            7
                        )
                    ]
                )
            )

            story.append(
                cluster_table
            )

            story.append(
                Spacer(1, 10)
            )

            story.append(
                Paragraph(
                    result.get(
                        "Note",
                        "K-Means is an "
                        "unsupervised clustering "
                        "algorithm."
                    ),
                    normal_style
                )
            )

            story.append(
                PageBreak()
            )

            continue

        # =================================================
        # METRICS
        # =================================================

        metrics_data = [
            [
                "Metric",
                "Value"
            ],

            [
                "Accuracy",
                f"{result['Accuracy']:.4f}"
            ],

            [
                "Precision",
                f"{result['Precision']:.4f}"
            ],

            [
                "Recall",
                f"{result['Recall']:.4f}"
            ],

            [
                "F1 Score",
                f"{result['F1 Score']:.4f}"
            ],

            [
                "ROC-AUC",
                (
                    f"{result['ROC-AUC']:.4f}"
                    if result.get("ROC-AUC")
                    is not None
                    else "N/A"
                )
            ],

            [
                "Training Time",
                f"{result['Training Time (s)']:.4f} sec"
            ],

            [
                "Prediction Time",
                f"{result['Prediction Time (s)']:.4f} sec"
            ]
        ]

        metrics_table = Table(
            metrics_data,
            colWidths=[
                2.5 * inch,
                3.5 * inch
            ]
        )

        metrics_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1769AA")
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.grey
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    )
                ]
            )
        )

        story.append(
            metrics_table
        )

        story.append(
            Spacer(1, 15)
        )

        # =================================================
        # CONFUSION MATRIX
        # =================================================

        if "Confusion Matrix" in result:

            story.append(
                Paragraph(
                    "Confusion Matrix",
                    subheading_style
                )
            )

            cm = np.array(
                result["Confusion Matrix"]
            )

            cm_data = []

            class_names = (
                list(
                    label_encoder.classes_
                )
                if label_encoder is not None
                else [
                    str(i)
                    for i in range(
                        len(cm)
                    )
                ]
            )

            cm_data.append(
                ["Actual / Predicted"]
                + class_names
            )

            for i, row in enumerate(cm):

                cm_data.append(
                    [
                        class_names[i]
                    ]
                    +
                    [
                        str(value)
                        for value in row
                    ]
                )

            cm_table = Table(
                cm_data
            )

            cm_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#1769AA")
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),
                        (
                            "ALIGN",
                            (0, 0),
                            (-1, -1),
                            "CENTER"
                        ),
                        (
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        )
                    ]
                )
            )

            story.append(
                cm_table
            )

            story.append(
                Spacer(1, 15)
            )

        # =================================================
        # CLASSIFICATION REPORT
        # =================================================

        if "Classification Report" in result:

            story.append(
                Paragraph(
                    "Classification Report",
                    subheading_style
                )
            )

            report_text = (
                result[
                    "Classification Report"
                ]
                .replace(
                    "\n",
                    "<br/>"
                )
                .replace(
                    " ",
                    "&nbsp;"
                )
            )

            story.append(
                Paragraph(
                    report_text,
                    small_style
                )
            )

            story.append(
                Spacer(1, 15)
            )

        # =================================================
        # DECISION TREE
        # =================================================

        if (
            name in ["ID3", "J48"]
            and name in models
        ):

            story.append(
                PageBreak()
            )

            story.append(
                Paragraph(
                    f"🌳 {name} Colored Decision Tree",
                    heading_style
                )
            )

            try:

                class_names = (
                    list(
                        label_encoder.classes_
                    )
                    if label_encoder is not None
                    else []
                )

                tree_fig = tree_figure(
                    models[name],
                    class_names
                )

                tree_image = figure_to_image(
                    tree_fig,
                    width=7.0 * inch
                )

                story.append(
                    tree_image
                )

                story.append(
                    Spacer(1, 10)
                )

                story.append(
                    Paragraph(
                        f"The above tree represents "
                        f"the learned {name} decision "
                        f"rules from the selected dataset.",
                        normal_style
                    )
                )

                plt.close(
                    tree_fig
                )

            except Exception as error:

                story.append(
                    Paragraph(
                        f"Decision tree image could "
                        f"not be generated: {error}",
                        normal_style
                    )
                )

        # =================================================
        # ROC CURVE
        # =================================================

        roc_data = result.get(
            "ROC"
        )

        if roc_data is not None:

            story.append(
                PageBreak()
            )

            story.append(
                Paragraph(
                    f"📈 {name} ROC Curve",
                    heading_style
                )
            )

            try:

                roc_fig, roc_ax = plt.subplots(
                    figsize=(8, 6)
                )

                # -----------------------------------------
                # BINARY
                # -----------------------------------------

                if (
                    roc_data.get("type")
                    == "binary"
                ):

                    roc_ax.plot(
                        roc_data["fpr"],
                        roc_data["tpr"],
                        linewidth=3,
                        label=(
                            f"{name} "
                            f"(AUC = "
                            f"{roc_data['auc']:.4f})"
                        )
                    )

                    roc_title = (
                        f"{name} ROC Curve — "
                        f"AUC = "
                        f"{roc_data['auc']:.4f}"
                    )

                # -----------------------------------------
                # MULTICLASS
                # -----------------------------------------

                else:

                    for (
                        class_name,
                        curve
                    ) in roc_data[
                        "curves"
                    ].items():

                        roc_ax.plot(
                            curve["fpr"],
                            curve["tpr"],
                            linewidth=2,
                            label=(
                                f"{class_name} "
                                f"(AUC = "
                                f"{curve['auc']:.4f})"
                            )
                        )

                    roc_title = (
                        f"{name} Multiclass ROC Curve — "
                        f"Macro AUC = "
                        f"{roc_data['macro_auc']:.4f}"
                    )

                # -----------------------------------------
                # RANDOM CLASSIFIER
                # -----------------------------------------

                roc_ax.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    linewidth=1.5,
                    label="Random Classifier"
                )

                roc_ax.set_xlabel(
                    "False Positive Rate"
                )

                roc_ax.set_ylabel(
                    "True Positive Rate"
                )

                roc_ax.set_title(
                    roc_title
                )

                roc_ax.set_xlim(
                    0,
                    1
                )

                roc_ax.set_ylim(
                    0,
                    1.05
                )

                roc_ax.grid(
                    alpha=0.25
                )

                roc_ax.legend(
                    loc="lower right"
                )

                roc_fig.tight_layout()

                roc_image = figure_to_image(
                    roc_fig,
                    width=6.8 * inch
                )

                story.append(
                    roc_image
                )

                plt.close(
                    roc_fig
                )

            except Exception as error:

                story.append(
                    Paragraph(
                        f"ROC curve could not "
                        f"be generated: {error}",
                        normal_style
                    )
                )

        # =================================================
        # WEKA STYLE SUMMARY
        # =================================================

        story.append(
            Spacer(1, 15)
        )

        story.append(
            Paragraph(
                "Weka-Style Evaluation Summary",
                subheading_style
            )
        )

        y_test = result.get(
            "y_test"
        )

        if y_test is not None:

            total = len(
                y_test
            )

            correct = int(
                round(
                    result["Accuracy"]
                    * total
                )
            )

            incorrect = (
                total - correct
            )

            evaluation_data = [
                [
                    "Evaluation Metric",
                    "Value"
                ],

                [
                    "Total Instances",
                    str(total)
                ],

                [
                    "Correctly Classified",
                    str(correct)
                ],

                [
                    "Incorrectly Classified",
                    str(incorrect)
                ],

                [
                    "Accuracy",
                    f"{result['Accuracy']:.4f}"
                ],

                [
                    "Precision",
                    f"{result['Precision']:.4f}"
                ],

                [
                    "Recall",
                    f"{result['Recall']:.4f}"
                ],

                [
                    "F1 Score",
                    f"{result['F1 Score']:.4f}"
                ]
            ]

            if result.get(
                "ROC-AUC"
            ) is not None:

                evaluation_data.append(
                    [
                        "ROC Area",
                        f"{result['ROC-AUC']:.4f}"
                    ]
                )

            evaluation_table = Table(
                evaluation_data,
                colWidths=[
                    3.2 * inch,
                    2.8 * inch
                ]
            )

            evaluation_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor("#1769AA")
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.4,
                            colors.grey
                        ),
                        (
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        )
                    ]
                )
            )

            story.append(
                evaluation_table
            )

        story.append(
            PageBreak()
        )

    # =====================================================
    # FINAL PAGE
    # =====================================================

    story.append(
        Paragraph(
            "5. Final Analysis Summary",
            heading_style
        )
    )

    # Find best model
    classification_results = {
        name: result
        for name, result
        in results.items()
        if result.get("Accuracy") is not None
    }

    if classification_results:

        best_algorithm = max(
            classification_results,
            key=lambda x:
                classification_results[x][
                    "Accuracy"
                ]
        )

        best_accuracy = (
            classification_results[
                best_algorithm
            ]["Accuracy"]
        )

        story.append(
            Paragraph(
                f"<b>Best Performing Algorithm:</b> "
                f"{best_algorithm}",
                normal_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Best Accuracy:</b> "
                f"{best_accuracy:.4f} "
                f"({best_accuracy:.2%})",
                normal_style
            )
        )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "This report contains the dataset information, "
            "preprocessing-related statistics, algorithm "
            "evaluation results, confusion matrices, "
            "classification reports, decision-tree "
            "visualizations, ROC curves and Weka-style "
            "evaluation summaries generated by Dataset Analyzer.",
            normal_style
        )
    )

    # =====================================================
    # DATA EXPLORATION & DWDM VISUALIZATIONS
    # =====================================================

    story.append(PageBreak())
    story.append(Paragraph("6. Data Exploration & DWDM Visualizations", heading_style))
    story.append(Paragraph(
        "The following default plots are generated from the current dataset (after any preprocessing). ETL and warehouse schema figures are conceptual illustrations. Charts you added on the Visualization and OLAP pages follow in section 7.",
        normal_style
    ))
    story.append(Spacer(1, 8))

    def add_report_figure(fig, caption):
        try:
            story.append(Paragraph(caption, subheading_style))
            story.append(figure_to_image(fig, width=6.2 * inch))
            story.append(Spacer(1, 8))
        finally:
            plt.close(fig)

    try:
        numeric_df = df.select_dtypes(include=np.number)
        categorical_cols = [c for c in df.columns if c not in numeric_df.columns]

        if not numeric_df.empty:
            # Distribution / histogram for first numeric attribute
            first_numeric = numeric_df.columns[0]
            values = pd.to_numeric(numeric_df[first_numeric], errors="coerce").dropna()
            if not values.empty:
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.hist(values.iloc[:20000], bins=30, edgecolor="black", alpha=0.8)
                ax.set_title(f"Histogram — {first_numeric}")
                ax.set_xlabel(str(first_numeric))
                ax.set_ylabel("Frequency")
                fig.tight_layout()
                add_report_figure(fig, "6.1 Numerical distribution")

                fig, ax = plt.subplots(figsize=(8, 2.8))
                ax.boxplot(values.iloc[:20000], vert=False, patch_artist=True)
                ax.set_title(f"Box plot — {first_numeric}")
                ax.set_xlabel(str(first_numeric))
                fig.tight_layout()
                add_report_figure(fig, "6.2 Outlier and spread overview")

            if len(numeric_df.columns) >= 2:
                corr = numeric_df.corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(7, 5))
                image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
                ax.set_xticks(range(len(corr.columns)))
                ax.set_xticklabels([str(c) for c in corr.columns], rotation=70, ha="right", fontsize=7)
                ax.set_yticks(range(len(corr.index)))
                ax.set_yticklabels([str(c) for c in corr.index], fontsize=7)
                ax.set_title("Correlation heatmap")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                add_report_figure(fig, "6.3 Numerical attribute correlation")

                x_col, y_col = numeric_df.columns[:2]
                scatter_data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna().head(5000)
                if not scatter_data.empty:
                    fig, ax = plt.subplots(figsize=(7, 3.5))
                    ax.scatter(scatter_data[x_col], scatter_data[y_col], alpha=0.6, s=14)
                    ax.set_xlabel(str(x_col))
                    ax.set_ylabel(str(y_col))
                    ax.set_title(f"Scatter plot — {x_col} vs {y_col}")
                    fig.tight_layout()
                    add_report_figure(fig, "6.4 Relationship between numerical attributes")

                    # Lightweight K-Means illustration for the PDF.
                    try:
                        from sklearn.cluster import KMeans
                        from sklearn.preprocessing import StandardScaler
                        cluster_data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
                        if len(cluster_data) >= 3:
                            cluster_data = cluster_data.sample(n=min(len(cluster_data), 3000), random_state=42)
                            scaled = StandardScaler().fit_transform(cluster_data)
                            labels = KMeans(n_clusters=min(3, len(cluster_data)), random_state=42, n_init=10).fit_predict(scaled)
                            fig, ax = plt.subplots(figsize=(7, 3.5))
                            for label in sorted(set(labels)):
                                mask = labels == label
                                ax.scatter(cluster_data.loc[mask, x_col], cluster_data.loc[mask, y_col],
                                           s=14, alpha=0.7, label=f"Cluster {label + 1}")
                            ax.set_xlabel(str(x_col))
                            ax.set_ylabel(str(y_col))
                            ax.set_title("Illustrative K-Means clusters")
                            ax.legend(fontsize=7)
                            fig.tight_layout()
                            add_report_figure(fig, "6.5 Clustering overview (illustrative K-Means)")

                            elbow_rows = []
                            for k in range(1, min(6, len(cluster_data) - 1) + 1):
                                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                                model.fit(scaled)
                                elbow_rows.append((k, model.inertia_))
                            if elbow_rows:
                                fig, ax = plt.subplots(figsize=(7, 3.2))
                                ax.plot([r[0] for r in elbow_rows], [r[1] for r in elbow_rows], marker="o")
                                ax.set_xlabel("Number of clusters (k)")
                                ax.set_ylabel("Inertia")
                                ax.set_title("K-Means elbow method")
                                fig.tight_layout()
                                add_report_figure(fig, "6.6 Choosing a cluster count — elbow method")
                    except Exception:
                        # The core report should still be produced if optional clustering fails.
                        pass

        missing = df.isna().sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar([str(c) for c in missing.index], missing.values)
        ax.set_title("Missing values by attribute")
        ax.set_ylabel("Missing cells")
        ax.tick_params(axis="x", labelrotation=65, labelsize=7)
        fig.tight_layout()
        add_report_figure(fig, "6.7 Data quality — missing values")

        if categorical_cols:
            col = categorical_cols[0]
            counts = df[col].astype("string").fillna("(Missing)").value_counts().head(15)
            fig, ax = plt.subplots(figsize=(8, 3.4))
            ax.bar([str(x)[:28] for x in counts.index], counts.values)
            ax.set_title(f"Top categories — {col}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", labelrotation=55, labelsize=7)
            fig.tight_layout()
            add_report_figure(fig, "6.8 Categorical distribution")

        # Conceptual ETL flow diagram
        fig, ax = plt.subplots(figsize=(8, 2.2))
        ax.axis("off")
        steps = [(0.02, "Extract\nCSV / ARFF / Excel"), (0.28, "Transform\nClean / Normalize"),
                 (0.54, "Validate\nQuality checks"), (0.78, "Load\nWarehouse target")]
        for x, label in steps:
            ax.text(x + 0.09, 0.5, label, ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#eaf2ff", edgecolor="#3973ac"),
                    transform=ax.transAxes)
        for x in [0.22, 0.48, 0.74]:
            ax.annotate("", xy=(x + 0.045, 0.5), xytext=(x, 0.5), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=1.4))
        ax.set_title("ETL pipeline — conceptual illustration")
        fig.tight_layout()
        add_report_figure(fig, "6.9 Data warehousing process illustration")

        # Conceptual star/snowflake schema side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax in axes:
            ax.axis("off")
        axes[0].set_title("Star schema (concept)")
        axes[0].text(0.5, 0.5, "FACT TABLE\nMeasures + keys", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="#dbeafe", edgecolor="#3973ac"), transform=axes[0].transAxes)
        for xy, label in [((0.5, 0.9), "Date"), ((0.1, 0.5), "Product"), ((0.5, 0.1), "Customer"), ((0.9, 0.5), "Location")]:
            axes[0].text(*xy, label, ha="center", va="center", fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.35", facecolor="#e7f7ed", edgecolor="#389466"), transform=axes[0].transAxes)
            axes[0].annotate("", xy=(0.5, 0.5), xytext=xy, xycoords=axes[0].transAxes,
                             arrowprops=dict(arrowstyle="-", color="#777777"))
        axes[1].set_title("Snowflake schema (concept)")
        axes[1].text(0.5, 0.2, "FACT", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#dbeafe", edgecolor="#3973ac"), transform=axes[1].transAxes)
        axes[1].text(0.28, 0.55, "Product", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#e7f7ed", edgecolor="#389466"), transform=axes[1].transAxes)
        axes[1].text(0.72, 0.55, "Location", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#e7f7ed", edgecolor="#389466"), transform=axes[1].transAxes)
        axes[1].text(0.28, 0.85, "Category", ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#777777"), transform=axes[1].transAxes)
        axes[1].text(0.72, 0.85, "Region", ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#777777"), transform=axes[1].transAxes)
        for start, end in [((0.5, 0.27), (0.28, 0.47)), ((0.5, 0.27), (0.72, 0.47)), ((0.28, 0.63), (0.28, 0.77)), ((0.72, 0.63), (0.72, 0.77))]:
            axes[1].annotate("", xy=end, xytext=start, xycoords=axes[1].transAxes,
                             arrowprops=dict(arrowstyle="-", color="#777777"))
        fig.tight_layout()
        add_report_figure(fig, "6.10 Warehouse schema diagrams (illustrative)")

        story.append(Paragraph(
            "OLAP note: slice, dice, drill-down, roll-up and pivot are computed in memory on the current dataset (see the OLAP Operations page). They do not represent a persistent cube or a live warehouse connection.",
            small_style
        ))
    except Exception as chart_error:
        story.append(Paragraph(
            "Some exploratory charts could not be generated for this dataset: " + str(chart_error)[:300],
            small_style
        ))

    # =====================================================
    # CHARTS ADDED BY THE USER (Visualization / OLAP pages)
    # =====================================================

    if preprocessing_log:
        story.append(PageBreak())
        story.append(Paragraph("Preprocessing steps applied", heading_style))
        for index, step in enumerate(preprocessing_log, 1):
            story.append(Paragraph(f"{index}. {step}", normal_style))

    if extra_charts:
        from utils.mpl_charts import render_chart

        story.append(PageBreak())
        story.append(Paragraph("7. Charts Added from the Visualization and OLAP Pages", heading_style))
        story.append(Spacer(1, 6))

        for number, spec in enumerate(extra_charts, 1):
            try:
                fig = render_chart(df, spec)
            except Exception as chart_error:
                story.append(Paragraph(
                    f"7.{number} {spec.get('title', spec.get('type', 'Chart'))}: could not be drawn "
                    f"with the current dataset ({str(chart_error)[:150]}).",
                    small_style
                ))
                continue
            try:
                story.append(KeepTogether([
                    Paragraph(f"7.{number} {spec.get('title', spec.get('type', 'Chart'))}", subheading_style),
                    figure_to_image(fig, width=6.2 * inch),
                    Spacer(1, 8)
                ]))
            finally:
                plt.close(fig)

    # =====================================================
    # BUILD PDF
    # =====================================================

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()