# 🍷 PCA & LDA Explorer: Dimensionality Reduction on the Wine Dataset

An interactive **R Shiny** dashboard for exploring and comparing **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)** using the chemical attributes of Italian wine cultivars.

---

## 📌 Project Overview

When dealing with high-dimensional datasets, visualizing patterns and building efficient machine learning models can be challenging. Dimensionality reduction methods solve this by transforming high-dimensional spaces into lower-dimensional representations:

* **Principal Component Analysis (PCA)** is an **unsupervised** method that discovers orthogonal axes of maximum overall variance across all features, regardless of class labels.
* **Linear Discriminant Analysis (LDA)** is a **supervised** method that identifies projection vectors that maximize the separation between known classes relative to the scatter within each class (**Fisher's Criterion**).

This application allows students, data scientists, and researchers to interactively explore these principles on the well-known **Wine Dataset** (178 samples, 13 chemical features, 3 cultivars).

---

## 🚀 Key Features

### 1. 📊 Dataset Overview
* **Summary KPI Cards**: Instant visibility into total samples (178), continuous chemical features (13), classes (3 cultivars), and missing value audit (0 missing / 100% complete).
* **Class Distribution Bar Chart**: Sample count and percentage breakdown across Cultivar 1 (59), Cultivar 2 (71), and Cultivar 3 (48).
* **Correlation Heatmap**: Pairwise Pearson correlation coefficients between all 13 chemical features with diverging color gradient and numerical cell overlays.
* **Selectable Feature Box Plot**: Interactive distribution comparison (median, interquartile range, outliers, and jitter points) across the three cultivars for any selected chemical attribute.
* **Interactive Data Table**: Paginated, sortable, and searchable preview of the entire standardized dataset.
* **Export**: One-click download of the complete dataset in CSV format.

### 2. 🔍 Principal Component Analysis (PCA)
* **Unsupervised Modeling**: Fitted using R's `stats::prcomp` with strict standardization (`center = TRUE`, `scale. = TRUE`) with class labels excluded from the transformation matrix.
* **Variance Metrics**: Real-time display of PC1 variance (36.20%), PC2 variance (19.21%), and cumulative 2D explained variance (55.41%).
* **2D Projection Scatter Plot**: Samples projected on PC1 vs PC2 with 95% confidence ellipses and color-coded cultivar identities.
* **Scree & Cumulative Variance Plot**: Dual-axis visualization showing individual component variance alongside cumulative variance with 80% and 90% benchmark thresholds.
* **Feature Loadings Table**: Eigenvector weights showing which chemical attributes drive PC1 and PC2.
* **Scores Preview & Downloads**: Download PCA scores as CSV and export high-resolution (300 DPI) publication-ready scatter plots as PNG.

### 3. 🎯 Linear Discriminant Analysis (LDA)
* **Supervised Modeling**: Fitted using `MASS::lda` on standardized features using cultivar labels to compute optimal discriminant functions.
* **Trace Proportions**: Displays LD1 trace (68.75%) and LD2 trace (31.25%), accounting for 100% of between-class separability in a 2D plane (since $\text{rank} = \min(p, C-1) = 2$).
* **Discriminant Scatter Plot**: Samples projected onto LD1 vs LD2 showcasing clean, linearly separable cultivar boundaries.
* **Educational Callout**: Beginner-friendly explanations of Fisher's Discriminant Criterion, scatter matrices ($S_B$ and $S_W$), and mathematical bounds on discriminant dimensions.
* **LDA Coefficients Table**: Linear combination multipliers highlighting key distinguishing chemical attributes.
* **Scores Preview & Downloads**: Download LDA scores as CSV and export 300 DPI PNG scatter plots.

### 4. ⚖️ Model Comparison & Empirical Results
* **Side-by-Side Projections**: Synchronized 2D visualizations comparing PCA (PC1 vs PC2) and LDA (LD1 vs LD2).
* **Quantitative Separation Metric**: Dynamically computed **Fisher Separation Ratio** ($\text{trace}(S_B) / \text{trace}(S_W)$):
  * **PCA (PC1 + PC2)**: $3.41$
  * **LDA (LD1 + LD2)**: $6.61$ (**+93.8% higher class separation**)
* **Structured Comparison Matrix**: Side-by-side breakdown of learning paradigms, objectives, coordinate constraints, scaling sensitivity, and real-world use cases.
* **Evidence-Based Conclusion**: Clear summary explaining why LDA achieves superior class distinction while PCA excels at unbiased variance retention.

### 5. ⚡ Real-Time Class Filtering (No Model Refit)
* Toggle any cultivar on or off in the scatter plots without recalculating eigenvectors or discriminant coefficients.
* Preserves mathematical integrity by keeping the coordinate projection space fixed and stable during interactive exploration.

---

## 📁 Repository Structure

```text
PCA-LDA-Dimensionality-Reduction/
├── app.R              # Complete R Shiny web application (UI and Server logic)
├── run_app.R          # Quick one-command launcher with automatic dependency installation
├── .gitignore         # Standard git ignore rules for R, RStudio, and Windows
└── README.md          # Project documentation, mathematical guide, and setup instructions
```

---

## 🛠️ Windows Setup & Launch Guide

### Prerequisites
* **R (version 4.0 or later)**: Download and install from [CRAN](https://cran.r-project.org/bin/windows/base/).
* *(Optional)* **RStudio Desktop**: Download from [Posit](https://posit.co/download/rstudio-desktop/).

### Option 1: Quick Launch via Terminal / PowerShell (Recommended)
1. Open **Command Prompt** or **PowerShell**.
2. Navigate to the project directory:
   ```powershell
   cd C:\Users\HaniS\OneDrive\Desktop\PCA-LDA-Dimensionality-Reduction
   ```
3. Run the launcher script with `Rscript`:
   ```powershell
   Rscript run_app.R
   ```
   *(If `Rscript` is not in your system PATH, use the full path, e.g., `"C:\Program Files\R\R-4.6.1\bin\Rscript.exe" run_app.R`)*.
4. The script will automatically verify dependencies, install any missing packages (`shiny`, `ggplot2`, `MASS`, `rattle.data`, `DT`), and open the app in your default web browser.

### Option 2: Launch from RStudio or R Console
1. Open RStudio and set the working directory to this folder:
   ```r
   setwd("C:/Users/HaniS/OneDrive/Desktop/PCA-LDA-Dimensionality-Reduction")
   ```
2. Open `app.R` and click the **Run App** button in the top right corner of the editor, or run:
   ```r
   shiny::runApp()
   ```

---

## 📦 Required Packages

The application relies on minimal, robust R packages:
* `shiny` - Interactive web framework
* `ggplot2` - Data visualization
* `MASS` - Linear Discriminant Analysis (`lda`)
* `rattle.data` - Wine dataset source
* `DT` - Interactive data tables

To install all required packages manually in R:
```r
install.packages(c("shiny", "ggplot2", "MASS", "rattle.data", "DT"), repos = "https://cloud.r-project.org")
```

---

## 📊 Summary of Mathematical Findings

| Metric / Property | PCA (PC1 & PC2) | LDA (LD1 & LD2) | Takeaway |
| :--- | :--- | :--- | :--- |
| **Supervision** | Unsupervised | Supervised | LDA uses cultivar labels; PCA does not |
| **Criterion** | $\max \text{Var}(X w)$ | $\max \frac{w^T S_B w}{w^T S_W w}$ | PCA maximizes total scatter; LDA maximizes class separation |
| **Explained Metric** | 55.41% Total Variance | 100% Between-Class Trace | LDA exhausts all between-class separation across 2 dimensions |
| **Fisher Separation Ratio** | **3.41** | **6.61** | LDA delivers **+93.8% higher cluster distinction** |
| **Class Overlap in 2D** | Slight overlap (Cultivar 1 vs 2) | None (Clean linear boundaries) | LDA is optimal for classification tasks |

---

## 📜 Dataset Citation
The Wine dataset was acquired through chemical analysis of wines grown in the same region in Italy but derived from three different cultivars:
* **Source**: Forina, M. et al., PARVUS - An Extendible Package for Data Exploration, Classification and Correlation. Institute of Pharmaceutical and Food Analysis and Technologies, Genoa, Italy.
* Available through `rattle.data::wine` and the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine).

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
