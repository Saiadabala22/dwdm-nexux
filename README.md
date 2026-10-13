# Dataset Analyzer — Weka Style

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Workflow:
Welcome → Upload Dataset → Data Preprocessing (optional) → Algorithm Selection → Results → Data Visualization / OLAP Operations

Algorithms:
- ID3
- J48
- K-NN
- Naive Bayes
- K-Means Clustering
- Random Forest

Results include:
- Weka-style dataset information
- algorithm comparison
- accuracy, precision, recall, F1, ROC-AUC
- confusion matrix
- classification report
- colored decision tree for ID3/J48
- ROC curve for binary classification
- K-Means silhouette score
- PDF report


Data visualization page:
- Interactive box plots, scatter plots, histograms, correlation heatmaps, pair plots, missing-value charts, bar charts and line charts.
- K-Means cluster visualization and elbow-method analysis.
- Illustrative ETL pipeline, OLAP roll-up-style aggregation, and star/snowflake schema diagrams.
- Download interactive charts as HTML. The PDF report includes exploratory charts, conceptual warehouse diagrams, and the existing model evaluation outputs.

Note: ETL pipeline and warehouse schema diagrams are conceptual illustrations. This release does not implement a persistent warehouse, ETL execution engine, or full OLAP cube.


## New in this version

**Data Preprocessing page** (`pages/7_Data_Preprocessing.py`)
- Placeholder clean-up (`?`, `NA`, ... → real missing values, text → numeric when safe)
- Missing values: mean, median, mode, constant, drop rows, drop columns
- Remove duplicates (whole row or chosen columns; keep first / last / none)
- Encoding: label or one-hot
- Scaling / normalization: z-score, min-max (custom range), robust
- Outliers: IQR or z-score; remove rows or cap values
- Undo, reset to original, step history, download cleaned CSV
- The cleaned dataset is used by every other page; trained results are cleared when the data changes.

**OLAP Operations** (`pages/8_OLAP_Operations.py`, also a tab on the Visualization page)
- Slice, dice, drill-down, roll-up (hierarchy levels incl. grand total), pivot (with totals)
- Derived dimensions: numeric binning and Year/Quarter/Month/Day from a date column
- Computed in memory on the current dataset; not a persistent cube.

**Visualization → PDF**
- One dropdown for all chart types (box, scatter, histogram, heatmap, pair, missing, bar, line, K-Means, elbow).
- "Add to PDF report" stores a chart (and OLAP results) for section 7 of the PDF; default exploratory charts stay in section 6.
- The PDF also lists the preprocessing steps applied.
