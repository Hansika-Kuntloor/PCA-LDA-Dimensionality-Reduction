# 🍷 PCA & LDA Explorer: Dimensionality Reduction on the Wine Dataset

An interactive **R Shiny** dashboard for exploring and evaluating **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)** using a strict, leak-free **70:30 stratified train-test split** on the chemical attributes of Italian wine cultivars.

---

## 📌 Project Overview

When evaluating dimensionality reduction and classification techniques, maintaining rigorous data hygiene is critical. This application implements a strict **70:30 stratified train-test split**:

* **Strict 70:30 Partitioning**: Exactly 70% of the dataset (125 samples) is used for training and 30% (53 samples) is held out for out-of-sample testing, using a reproducible random seed (`set.seed(42)`). No other split ratio (e.g. 80:20) is permitted.
* **Stratified Proportions**: Class proportions are strictly preserved across both partitions:
  * **Cultivar 1**: 41 Training / 18 Testing (59 total)
  * **Cultivar 2**: 50 Training / 21 Testing (71 total)
  * **Cultivar 3**: 34 Training / 14 Testing (48 total)
* **Zero Data Leakage**: Standardization parameters ($\mu$ and $\sigma$) are estimated **exclusively on the training partition**. Test features are standardized using the training-set means and standard deviations.
* **No Test Refitting**: PCA rotation eigenvectors and LDA discriminant functions are fitted **solely on the training set**. Test observations are transformed into the learned coordinate systems via projection without refitting.
* **Separation of Exploration & Evaluation**: The full dataset is preserved for Exploratory Data Analysis (EDA) in the Dataset section, while all modeling and evaluation in PCA, LDA, and Comparison strictly adhere to the 70:30 protocol.

---

## 🚀 Key Features & Workflow

### 1. 📊 Dataset Overview (Full EDA + Stratified Breakdown)
* **Summary KPI Cards**: Instant visibility into total samples (178), continuous chemical features (13), training partition (125), and testing partition (53).
* **Partitioned Class Distribution**: Grouped bar chart displaying sample counts for both Training (70%) and Testing (30%) subsets across each cultivar.
* **Correlation Heatmap**: Pairwise Pearson correlation matrix across all 13 chemical features with color gradients and coefficient overlays.
* **Selectable Feature Box Plot**: Displays feature distributions across cultivars, visually distinguishing Training samples (circles) and Testing samples (triangles).
* **Interactive Data Table**: Sortable and searchable preview of the wine dataset with an explicit `Split` partition column (`DT::dataTableOutput`).
* **Export**: Download full dataset with train/test partition tags as CSV.

### 2. 🔍 Principal Component Analysis (PCA - Unsupervised)
* **Training-Only Fit**: Fitted via `stats::prcomp` on the 125 standardized training observations without class labels.
* **Variance Metrics**: PC1 explains **37.00%**, PC2 explains **20.12%**, and the first two components explain **57.12%** of cumulative training variance.
* **2D Projection Scatter Plot**:
  * **Circles (●)** depict the 125 Training samples.
  * **Triangles (▲)** depict the 53 Held-out Test samples.
  * Consistent color palette across cultivars (Cultivar 1 = Blue, Cultivar 2 = Green, Cultivar 3 = Red).
  * 95% confidence ellipses.
* **Scree Plot**: Visualizes individual and cumulative variance explained by all 13 principal components with 80% and 90% benchmark lines.
* **Loadings & Scores**: Eigenvector feature weights and sample projection scores with CSV and 300 DPI PNG export.

### 3. 🎯 Linear Discriminant Analysis (LDA - Supervised)
* **Training-Only Fit**: Fitted via `MASS::lda` on the 125 training observations using cultivar labels.
* **Trace Proportions**: Displays LD1 trace (**70.29%**) and LD2 trace (**29.71%**), exhausting 100% of between-class separability across 2 dimensions ($\min(13, 3-1) = 2$).
* **Out-of-Sample Test Set Evaluation**:
  * Evaluated strictly on the 53 unseen test samples.
  * **Held-Out Test Accuracy**: **98.11%** (52 of 53 test observations correctly classified).
  * **Test Confusion Matrix**:
    * Cultivar 1: 18 / 18 correct (100.0%)
    * Cultivar 2: 20 / 21 correct (95.2% — only 1 bottle misclassified as Cultivar 1)
    * Cultivar 3: 14 / 14 correct (100.0%)
* **Discriminant Scatter Plot**: Clear visual separation of training (●) and testing (▲) points with zero cluster overlap.
* **Coefficients & Scores**: Linear discriminant scaling coefficients and projection scores with CSV and 300 DPI PNG export.

### 4. ⚖️ Model Comparison & Empirical Findings
* **Side-by-Side Projections**: Synchronized 2D visualizations comparing unsupervised PCA vs supervised LDA with points distinguished by partition shape.
* **Quantitative Separation & Classification Metrics**:
  * **Fisher Separation Ratio ($S_B / S_W$)**: PCA = $3.65$ vs LDA = $7.15$ (**+95.9% separation gain**).
  * **Held-Out Test Classification Accuracy**:
    * **2D LDA Classifier**: **98.11%** (52/53 correct).
    * **2D PCA Nearest-Centroid Classifier**: **92.45%** (49/53 correct).
* **Comprehensive Comparison Table**: Structured side-by-side assessment covering learning paradigm, mathematical objective, output dimensions, test accuracy, and practical use cases.
* **Empirical Takeaway**: Explains why LDA's supervised projection preserves critical class-distinguishing chemical traits that unsupervised PCA discards as lower variance.

### 5. ⚡ Real-Time Class & Partition Filtering (No Refit)
* Interactive controls to filter by cultivar or view specific partitions ("Both", "Training Only", or "Testing Only") without refitting the models or altering projection coordinates.

---

## 📊 Summary of Verified Empirical Results (70:30 Split)

| Evaluation Metric | PCA (Unsupervised) | LDA (Supervised) | Practical Implication |
| :--- | :--- | :--- | :--- |
| **Training Partition (70%)** | 125 samples | 125 samples | Proportions strictly preserved (41 / 50 / 34) |
| **Testing Partition (30%)** | 53 samples | 53 samples | Unseen held-out test set (18 / 21 / 14) |
| **Standardization Baseline** | Training set only ($\mu, \sigma$) | Training set only ($\mu, \sigma$) | Prevents data leakage between partitions |
| **2D Representation** | PC1 + PC2 (57.12% variance) | LD1 + LD2 (100% trace) | LDA captures complete discriminant space |
| **Fisher Separation Ratio** | **3.65** | **7.15** | LDA achieves **+95.9% higher class separation** |
| **Held-Out Test Accuracy** | **92.45%** (49/53 correct) | **98.11%** (52/53 correct) | LDA classifier achieves superior generalization |
| **Boundary Distinction** | Partial overlap (Cultivar 1 vs 2) | Clean linear separation | LDA is optimal for cultivar classification |

---

## 📁 Repository Structure

```text
PCA-LDA-Dimensionality-Reduction/
├── app.R              # Complete R Shiny web application (70:30 split, UI, Server logic)
├── run_app.R          # One-command launcher script with dependency check
├── validate_app.R     # Automated test suite asserting split counts, leak-free scaling, & accuracy
├── .gitignore         # Git ignore rules for R, RStudio, and Windows
└── README.md          # Comprehensive documentation and empirical report
```

---

## 🛠️ Windows Setup & Launch Guide

### Prerequisites
* **R (version 4.0 or later)**: Download and install from [CRAN](https://cran.r-project.org/bin/windows/base/).
* *(Optional)* **RStudio Desktop**: Download from [Posit](https://posit.co/download/rstudio-desktop/).

### Option 1: Quick Launch via Terminal / PowerShell (Recommended)
1. Open PowerShell or Command Prompt.
2. Navigate to the project root:
   ```powershell
   cd C:\Users\HaniS\OneDrive\Desktop\PCA-LDA-Dimensionality-Reduction
   ```
3. Execute the launcher script:
   ```powershell
   & "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" run_app.R
   ```
   *(Or simply `Rscript run_app.R` if R is on your system PATH)*.
4. The application will automatically verify dependencies, start the server, and open in your default browser.

### Option 2: Launch from RStudio or R Console
1. Open RStudio and set working directory:
   ```r
   setwd("C:/Users/HaniS/OneDrive/Desktop/PCA-LDA-Dimensionality-Reduction")
   ```
2. Run the application:
   ```r
   shiny::runApp()
   ```

### Option 3: Run Automated Verification Tests
To run the automated assertion suite that verifies the 70:30 split, sample counts, and out-of-sample accuracy:
```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" validate_app.R
```

---

## 📦 Required Packages

* `shiny` - Interactive web framework
* `ggplot2` - Data visualization
* `MASS` - Linear Discriminant Analysis (`lda`)
* `rattle.data` - Wine dataset source
* `DT` - Interactive data tables

---

## 📜 Dataset Citation
* **Source**: Forina, M. et al., PARVUS - An Extendible Package for Data Exploration, Classification and Correlation. Institute of Pharmaceutical and Food Analysis and Technologies, Genoa, Italy.
* Available via `rattle.data::wine` and the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine).

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
