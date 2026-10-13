import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)

from sklearn.decomposition import PCA

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    silhouette_score
)

from sklearn.preprocessing import label_binarize


# =========================================================
# DATA PREPARATION
# =========================================================

def prepare(df, target):

    data = df.copy()

    # -----------------------------------------------------
    # Replace common missing-value representations
    # -----------------------------------------------------

    data = data.replace(
        [
            "?",
            "NA",
            "N/A",
            "null",
            "None",
            ""
        ],
        np.nan
    )

    # -----------------------------------------------------
    # Remove rows where target is missing
    # -----------------------------------------------------

    data = data.dropna(
        subset=[target]
    )

    X = data.drop(
        columns=[target]
    )

    y = data[target]

    # -----------------------------------------------------
    # Numerical columns
    # -----------------------------------------------------

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    # -----------------------------------------------------
    # Categorical columns
    # -----------------------------------------------------

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    # -----------------------------------------------------
    # Numerical preprocessing
    # -----------------------------------------------------

    if numeric_columns:

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numerical_pipeline,
                numeric_columns
            )
        )

    # -----------------------------------------------------
    # Categorical preprocessing
    # -----------------------------------------------------

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    # -----------------------------------------------------
    # Column transformer
    # -----------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # -----------------------------------------------------
    # Encode target
    # -----------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y.astype(str)
    )

    # -----------------------------------------------------
    # Check class sizes
    # -----------------------------------------------------

    class_counts = pd.Series(
        y_encoded
    ).value_counts()

    # -----------------------------------------------------
    # Stratification
    # -----------------------------------------------------

    if (
        len(class_counts) > 1
        and class_counts.min() >= 2
    ):

        stratify = y_encoded

    else:

        stratify = None

    # -----------------------------------------------------
    # Train / test split
    # -----------------------------------------------------

    split = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=stratify
    )

    return (
        split,
        preprocessor,
        label_encoder
    )


# =========================================================
# ROC CURVE GENERATION
# =========================================================

def calculate_roc_data(
    y_test,
    probabilities,
    class_names
):

    roc_data = {}

    number_of_classes = len(
        class_names
    )

    # =====================================================
    # Binary Classification
    # =====================================================

    if number_of_classes == 2:

        fpr, tpr, thresholds = roc_curve(
            y_test,
            probabilities[:, 1]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        roc_data["type"] = "binary"

        roc_data["fpr"] = fpr

        roc_data["tpr"] = tpr

        roc_data["auc"] = roc_auc

        roc_data["classes"] = class_names

        return roc_data

    # =====================================================
    # Multiclass Classification
    # =====================================================

    y_test_binary = label_binarize(
        y_test,
        classes=np.arange(
            number_of_classes
        )
    )

    roc_data["type"] = "multiclass"

    roc_data["classes"] = class_names

    roc_data["curves"] = {}

    auc_values = []

    for i, class_name in enumerate(
        class_names
    ):

        fpr, tpr, _ = roc_curve(
            y_test_binary[:, i],
            probabilities[:, i]
        )

        class_auc = auc(
            fpr,
            tpr
        )

        roc_data["curves"][
            class_name
        ] = {

            "fpr": fpr,

            "tpr": tpr,

            "auc": class_auc
        }

        auc_values.append(
            class_auc
        )

    # -----------------------------------------------------
    # Macro average AUC
    # -----------------------------------------------------

    roc_data["macro_auc"] = np.mean(
        auc_values
    )

    return roc_data


# =========================================================
# K-MEANS CLUSTER COLORS
# =========================================================

CLUSTER_COLORS = [
    "#2196F3",  # Blue
    "#4CAF50",  # Green
    "#9C27B0",  # Purple
    "#FF9800",  # Orange
    "#00BCD4",  # Cyan
    "#E91E63",  # Pink
    "#795548",  # Brown
    "#607D8B",  # Blue Grey
]


def build_cluster_profiles(target_values, cluster_labels, number_of_clusters):
    """Build human-readable cluster names, colors and class distributions."""

    target_values = np.asarray(target_values).astype(str)
    cluster_labels = np.asarray(cluster_labels)
    profiles = []

    for cluster_id in range(number_of_clusters):
        mask = cluster_labels == cluster_id
        values = target_values[mask]
        counts = pd.Series(values).value_counts() if len(values) else pd.Series(dtype=int)

        if len(counts):
            dominant_class = str(counts.index[0])
            dominant_count = int(counts.iloc[0])
            dominant_percentage = (dominant_count / len(values)) * 100
            distribution = {
                str(label): int(count)
                for label, count in counts.items()
            }
        else:
            dominant_class = "Unknown"
            dominant_count = 0
            dominant_percentage = 0.0
            distribution = {}

        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        profiles.append({
            "cluster": int(cluster_id),
            "name": f"Cluster {cluster_id} — {dominant_class}",
            "dominant_class": dominant_class,
            "dominant_count": dominant_count,
            "dominant_percentage": dominant_percentage,
            "instances": int(len(values)),
            "color": color,
            "distribution": distribution,
        })

    return profiles


# =========================================================
# TRAIN MODELS
# =========================================================

def train_models(
    df,
    target,
    selected_algorithms
):

    # -----------------------------------------------------
    # Prepare classification data
    # -----------------------------------------------------

    split, preprocessor, label_encoder = prepare(
        df,
        target
    )

    X_train, X_test, y_train, y_test = split

    models = {}

    results = {}

    # =====================================================
    # CLASSIFICATION ESTIMATORS
    # =====================================================

    estimators = {

        "ID3":
            DecisionTreeClassifier(
                criterion="entropy",
                random_state=42
            ),

        "J48":
            DecisionTreeClassifier(
                criterion="entropy",
                min_samples_leaf=2,
                random_state=42
            ),

        "K-NN":
            KNeighborsClassifier(
                n_neighbors=5
            ),

        "Naive Bayes":
            GaussianNB(),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
    }

    # =====================================================
    # CLASSIFICATION ALGORITHMS
    # =====================================================

    for algorithm in selected_algorithms:

        if algorithm == "K-Means Clustering":

            continue

        # -------------------------------------------------
        # Safety check
        # -------------------------------------------------

        if algorithm not in estimators:

            continue

        estimator = estimators[
            algorithm
        ]

        # -------------------------------------------------
        # Pipeline
        # -------------------------------------------------

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    estimator
                )
            ]
        )

        # =================================================
        # TRAINING TIME
        # =================================================

        training_start = time.perf_counter()

        pipeline.fit(
            X_train,
            y_train
        )

        training_time = (
            time.perf_counter()
            - training_start
        )

        # =================================================
        # PREDICTION
        # =================================================

        prediction_start = time.perf_counter()

        predictions = pipeline.predict(
            X_test
        )

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        # =================================================
        # METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        confusion = confusion_matrix(
            y_test,
            predictions
        )

        report = classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0
        )

        # =================================================
        # RESULT OBJECT
        # =================================================

        result = {

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1 Score":
                f1,

            "Training Time (s)":
                training_time,

            "Prediction Time (s)":
                prediction_time,

            "Confusion Matrix":
                confusion,

            "Classification Report":
                report,

            "y_test":
                y_test,

            "prediction":
                predictions,

            "classes":
                label_encoder.classes_
        }

        # =================================================
        # ROC
        # =================================================

        if hasattr(
            pipeline,
            "predict_proba"
        ):

            try:

                probabilities = (
                    pipeline.predict_proba(
                        X_test
                    )
                )

                roc_data = calculate_roc_data(
                    y_test,
                    probabilities,
                    label_encoder.classes_
                )

                result[
                    "ROC"
                ] = roc_data

                # -------------------------------------------------
                # Binary ROC
                # -------------------------------------------------

                if (
                    roc_data["type"]
                    == "binary"
                ):

                    result[
                        "ROC-AUC"
                    ] = roc_data[
                        "auc"
                    ]

                # -------------------------------------------------
                # Multiclass ROC
                # -------------------------------------------------

                else:

                    result[
                        "ROC-AUC"
                    ] = roc_data[
                        "macro_auc"
                    ]

            except Exception as error:

                result[
                    "ROC"
                ] = None

                result[
                    "ROC-AUC"
                ] = None

                result[
                    "ROC Error"
                ] = str(error)

        else:

            result[
                "ROC"
            ] = None

            result[
                "ROC-AUC"
            ] = None

        # -------------------------------------------------
        # Save model and result
        # -------------------------------------------------

        models[
            algorithm
        ] = pipeline

        results[
            algorithm
        ] = result

    # =====================================================
    # K-MEANS CLUSTERING
    # =====================================================

    if (
        "K-Means Clustering"
        in selected_algorithms
    ):

        try:

            # -------------------------------------------------
            # Remove target
            # -------------------------------------------------

            clustering_data = df.drop(
                columns=[target]
            ).copy()

            # -------------------------------------------------
            # Use the existing preprocessing pipeline
            #
            # This allows numerical and categorical
            # attributes to participate in clustering.
            # -------------------------------------------------

            clustering_preprocessor = preprocessor

            processed_data = (
                clustering_preprocessor.fit_transform(
                    clustering_data
                )
            )

            processed_data = np.asarray(
                processed_data,
                dtype=float
            )

            # -------------------------------------------------
            # Remove invalid rows
            # -------------------------------------------------

            valid_rows = np.isfinite(
                processed_data
            ).all(axis=1)

            processed_data = (
                processed_data[
                    valid_rows
                ]
            )

            # -------------------------------------------------
            # Validate data
            # -------------------------------------------------

            if len(
                processed_data
            ) < 2:

                raise ValueError(
                    "K-Means requires at least "
                    "2 valid records."
                )

            # -------------------------------------------------
            # Number of clusters
            #
            # Default = 3
            # -------------------------------------------------

            number_of_clusters = min(
                3,
                len(processed_data)
            )

            number_of_clusters = max(
                2,
                number_of_clusters
            )

            # -------------------------------------------------
            # Train K-Means
            # -------------------------------------------------

            kmeans_start = (
                time.perf_counter()
            )

            kmeans = KMeans(
                n_clusters=number_of_clusters,
                random_state=42,
                n_init=10
            )

            cluster_labels = (
                kmeans.fit_predict(
                    processed_data
                )
            )

            kmeans_training_time = (
                time.perf_counter()
                - kmeans_start
            )

            # -------------------------------------------------
            # Silhouette Score
            # -------------------------------------------------

            unique_labels = np.unique(
                cluster_labels
            )

            if (
                len(unique_labels) >= 2
                and len(unique_labels)
                < len(processed_data)
            ):

                silhouette = (
                    silhouette_score(
                        processed_data,
                        cluster_labels
                    )
                )

            else:

                silhouette = None

            # =================================================
            # PCA 2D VISUALIZATION
            # =================================================

            if processed_data.shape[1] >= 2:

                pca = PCA(
                    n_components=2,
                    random_state=42
                )

                cluster_coordinates = (
                    pca.fit_transform(
                        processed_data
                    )
                )

                # -------------------------------------------------
                # Convert K-Means centroids to PCA space
                # -------------------------------------------------

                cluster_centers = (
                    pca.transform(
                        kmeans.cluster_centers_
                    )
                )

                explained_variance = (
                    pca.explained_variance_ratio_
                    * 100
                )

            else:

                # -------------------------------------------------
                # One feature only
                # -------------------------------------------------

                cluster_coordinates = (
                    np.column_stack(
                        (
                            processed_data[
                                :, 0
                            ],

                            np.zeros(
                                len(
                                    processed_data
                                )
                            )
                        )
                    )
                )

                cluster_centers = (
                    np.column_stack(
                        (
                            kmeans.cluster_centers_[
                                :, 0
                            ],

                            np.zeros(
                                number_of_clusters
                            )
                        )
                    )
                )

                explained_variance = [
                    100.0,
                    0.0
                ]

            # =================================================
            # CLUSTER SIZES
            # =================================================

            cluster_sizes = np.bincount(
                cluster_labels,
                minlength=number_of_clusters
            )

            # =================================================
            # CLUSTER NAMES / COLORS / DATASET CLASS MAPPING
            # =================================================

            # Keep the target values aligned with the rows that
            # survived preprocessing so we can explain what each
            # unsupervised cluster represents in the uploaded dataset.
            target_values = (
                df.loc[clustering_data.index, target]
                .astype(str)
                .to_numpy()[valid_rows]
            )

            cluster_profiles = build_cluster_profiles(
                target_values,
                cluster_labels,
                number_of_clusters
            )

            cluster_names = [
                profile["name"]
                for profile in cluster_profiles
            ]

            cluster_colors = [
                profile["color"]
                for profile in cluster_profiles
            ]

            # =================================================
            # SAVE K-MEANS MODEL
            # =================================================

            models[
                "K-Means Clustering"
            ] = kmeans

            # =================================================
            # SAVE K-MEANS RESULT
            # =================================================

            results[
                "K-Means Clustering"
            ] = {

                "Silhouette Score":
                    silhouette,

                "Clusters":
                    number_of_clusters,

                "Cluster Labels":
                    cluster_labels,

                "Cluster Coordinates":
                    cluster_coordinates,

                "Cluster Centers":
                    cluster_centers,

                "PCA Explained Variance":
                    explained_variance,

                "Cluster Sizes":
                    cluster_sizes,

                "Cluster Profiles":
                    cluster_profiles,

                "Cluster Names":
                    cluster_names,

                "Cluster Colors":
                    cluster_colors,

                "Cluster Target Values":
                    target_values,

                "Training Time (s)":
                    kmeans_training_time,

                "Note":
                    "K-Means is an unsupervised "
                    "clustering algorithm. "
                    "ROC-AUC is not applicable."
            }

        except Exception as error:

            results[
                "K-Means Clustering"
            ] = {

                "Silhouette Score":
                    None,

                "Clusters":
                    None,

                "Cluster Labels":
                    None,

                "Cluster Coordinates":
                    None,

                "Cluster Centers":
                    None,

                "PCA Explained Variance":
                    None,

                "Cluster Sizes":
                    None,

                "Training Time (s)":
                    None,

                "Note":
                    f"K-Means clustering failed: {error}"
            }

    return (
        models,
        results,
        label_encoder
    )


# =========================================================
# COLORED DECISION TREE
# =========================================================

def tree_figure(
    pipeline,
    class_names
):

    figure, axis = plt.subplots(
        figsize=(20, 12)
    )

    model = pipeline.named_steps[
        "model"
    ]

    plot_tree(
        model,
        filled=True,
        rounded=True,
        class_names=class_names,
        fontsize=9,
        ax=axis
    )

    axis.set_title(
        "Colored Decision Tree",
        fontsize=18,
        fontweight="bold"
    )

    figure.tight_layout()

    return figure


# =========================================================
# COLORED K-MEANS CLUSTER FIGURE
# =========================================================

def cluster_figure(
    cluster_coordinates,
    cluster_labels,
    cluster_centers=None,
    explained_variance=None,
    cluster_names=None,
    cluster_colors=None
):

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    cluster_coordinates = np.asarray(
        cluster_coordinates
    )

    cluster_labels = np.asarray(
        cluster_labels
    )

    unique_clusters = np.unique(
        cluster_labels
    )

    # -----------------------------------------------------
    # Colors
    # -----------------------------------------------------

    colors = (
        cluster_colors
        if cluster_colors is not None
        else CLUSTER_COLORS
    )

    # -----------------------------------------------------
    # Plot clusters
    # -----------------------------------------------------

    for index, cluster_id in enumerate(
        unique_clusters
    ):

        points = (
            cluster_labels
            == cluster_id
        )

        axis.scatter(
            cluster_coordinates[
                points,
                0
            ],

            cluster_coordinates[
                points,
                1
            ],

            s=85,

            color=colors[
                index % len(colors)
            ],

            alpha=0.85,

            edgecolors="white",

            linewidths=0.7,

            label=(
                cluster_names[index]
                if cluster_names is not None
                and index < len(cluster_names)
                else f"Cluster {cluster_id}"
            )
        )

    # -----------------------------------------------------
    # Add cluster labels near the cluster center
    # -----------------------------------------------------

    for index, cluster_id in enumerate(unique_clusters):
        points = cluster_labels == cluster_id
        if not np.any(points):
            continue

        center_x = float(np.mean(cluster_coordinates[points, 0]))
        center_y = float(np.mean(cluster_coordinates[points, 1]))

        display_name = (
            cluster_names[index]
            if cluster_names is not None
            and index < len(cluster_names)
            else f"Cluster {cluster_id}"
        )

        axis.annotate(
            display_name,
            (center_x, center_y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=colors[index % len(colors)],
                alpha=0.88
            )
        )

    # -----------------------------------------------------
    # Plot centroids
    # -----------------------------------------------------

    if cluster_centers is not None:

        cluster_centers = np.asarray(
            cluster_centers
        )

        axis.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],

            s=320,

            marker="X",

            color="white",

            edgecolors="black",

            linewidths=1.5,

            label="Centroids",

            zorder=10
        )

    # -----------------------------------------------------
    # Axis labels
    # -----------------------------------------------------

    x_label = (
        "PCA Component 1"
    )

    y_label = (
        "PCA Component 2"
    )

    if explained_variance is not None:

        if len(
            explained_variance
        ) >= 2:

            x_label = (
                f"PCA Component 1 "
                f"({explained_variance[0]:.1f}%)"
            )

            y_label = (
                f"PCA Component 2 "
                f"({explained_variance[1]:.1f}%)"
            )

    axis.set_xlabel(
        x_label,
        fontsize=12,
        fontweight="bold"
    )

    axis.set_ylabel(
        y_label,
        fontsize=12,
        fontweight="bold"
    )

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    axis.set_title(
        "K-Means Clustering Visualization",
        fontsize=18,
        fontweight="bold",
        pad=18
    )

    # -----------------------------------------------------
    # Grid
    # -----------------------------------------------------

    axis.grid(
        True,
        linestyle="--",
        alpha=0.25
    )

    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------

    axis.legend(
        loc="best",
        frameon=True
    )

    figure.tight_layout()

    return figure