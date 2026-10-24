# 🍷 Feature Extraction Studio — PCA and LDA on the Wine Dataset

An academic **R Shiny** application for feature extraction and dimensionality reduction on the Wine dataset using **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)** with a strict, leak-free **70:30 stratified train-test split** (reproducible seed `42`).

Built with a minimal, modern white-and-sky-blue design system, the application operates as a focused single-page studio dashboard with real-time feature selection, standardization settings, optional CSV uploads, circular variance gauges, feature loading contributions, PCA-exclusive scree plots, and full CSV exports.

---

## 📌 Architecture & Design Overview

### 1. Minimal White & Sky Blue Design
* **Header**: Title **“Feature Extraction Studio”** with subtitle **“PCA and LDA on the Wine Dataset”** and a compact dataset badge.
* **Focused Dashboard**: Completely removes separate Dataset and Compare navigation tabs, presenting all analysis controls, visualizations, metrics, and transformed data in one cohesive view.
* **Palette**: Minimal white cards (`#ffffff`), very light sky-blue surfaces (`#f4f8fd`), dark navy text (`#0f2744`), and dedicated cultivar accents:
  * **Cultivar 1**: Royal Blue (`#2563eb`)
  * **Cultivar 2**: Warm Orange (`#f97316`)
  * **Cultivar 3**: Vibrant Purple (`#9333ea`)
* **Partition Shapes**:
  * **● Circles** (`shape = 21`): 70% Training partition (125 observations)
  * **▲ Triangles** (`shape = 24`): 30% Testing partition (53 observations)

### 2. Left Control Panel
* **Method Selector**: Segmented toggle buttons (`PCA` vs `LDA`) that update all plots, metrics, gauges, and data previews.
* **Dataset Information**: Compact label showing dataset name, sample count (178), numerical feature count (13), and class count (3).
* **Optional CSV Upload**: Compact `Upload CSV` button. Loads the Wine Recognition dataset automatically by default (uploading is never required). Validates uploaded files, ensures a class-label column exists for LDA, and provides a quick reset link back to the Wine dataset.
* **Feature Selection**: Scrollable list of numeric feature checkboxes with working **Select All** and **Deselect All** controls.
* **Standardization Setting**: Checkbox toggle to enable/disable Z-score standardization ($\mu=0, \sigma=1$) using training-only statistics.
* **Fixed 70:30 Train-Test Split**: Visual two-tone progress bar (70% deep blue, 30% light blue) showing **125 Training** / **53 Testing** samples with seed `42`.
* **Run Analysis Button**: Primary blue action button (`▶ Run Analysis`) recalculating models and updating views.

### 3. Center Analytics Panel
* **Main Scatter Plot**:
  * **PCA**: PC1 vs PC2 scatter plot with variance explained labels on both axes and 95% shaded confidence ellipses.
  * **LDA**: LD1 vs LD2 scatter plot with between-class trace proportion labels on both axes.
* **Secondary Visualization (Directly Below Scatter Plot)**:
  * **PCA Mode**: Displays the **PCA Scree Plot** showing explained variance ratios for every principal component with numerical values labeled above points.
  * **LDA Mode**: Displays the **LDA Discriminant Variance Bar Chart** showing separate bars for LD1 (70.29%) and LD2 (29.71%) reflecting the actual between-class discriminant variance proportions.

### 4. Right Metrics & Feature Contributions Panel
* **Explained Variance (PCA) / Discriminant Variance (LDA)**:
  * High-resolution SVG circular donut gauges for Component 1, Component 2, and Combined (Component 1 + Component 2).
  * Shows total variance explained for PCA (PC1: 37.00%, PC2: 20.12%, Combined: 57.12%) and between-class trace for LDA (LD1: 70.29%, LD2: 29.71%, Combined: 100%).
* **Feature Contributions**:
  * Single dedicated panel with a dynamic dropdown to select **PC1/PC2** (for PCA) or **LD1/LD2** (for LDA).
  * Horizontal gradient bars (`#38bdf8` to `#2563eb`) displaying relative absolute loadings $|w_j|$ or scaling coefficients $|c_j|$ with exact numerical values.

### 5. Bottom Transformed Data Preview
* **Preview Table**: First 5 rows of projected scores with colored badges for Class and Train/Test partition.
* **CSV Export**: `Download CSV` button exporting all 178 transformed observations (or all rows if custom data is loaded).

---

## 🔬 Statistical Methodology (Zero Data Leakage)

1. **Stratified Split**: 
   The dataset is split once using `set.seed(42)` into 70% training (125 samples) and 30% testing (53 samples) in a stratified manner across the three cultivars:
   * **Cultivar 1**: 41 Train / 18 Test (59 total)
   * **Cultivar 2**: 50 Train / 21 Test (71 total)
   * **Cultivar 3**: 34 Train / 14 Test (48 total)
2. **Training-Set-Only Preprocessing**:
   Feature means ($\mu_{\text{train}}$) and standard deviations ($\sigma_{\text{train}}$) are computed **strictly on the 125 training observations**. The held-out 53 test observations are standardized using training parameters.
3. **Unsupervised PCA Fitting**:
   Fitted strictly on the training partition without class labels. Test observations are projected via matrix multiplication with the training rotation eigenvectors ($V_{\text{train}}$):
   $$Z_{\text{test}} = X_{\text{test, scaled}} \, V_{\text{train}}$$
4. **Supervised LDA Fitting**:
   Fitted strictly on the training partition using cultivar labels. Test observations are projected onto the learned linear discriminants without refitting.

---

## 📊 Summary of Baseline Model Findings (All 13 Features)

| Property / Metric | PCA (Unsupervised) | LDA (Supervised) |
| :--- | :--- | :--- |
| **Learning Paradigm** | Unsupervised (no class labels) | Supervised (uses cultivar labels) |
| **Optimization Target** | Maximize total feature variance | Maximize between-to-within variance ratio ($S_B / S_W$) |
| **Training Partition (70%)** | 125 observations | 125 observations |
| **Testing Partition (30%)** | 53 observations | 53 observations |
| **Component 1** | PC1 explains **37.00%** of variance | LD1 accounts for **70.29%** of between-class trace |
| **Component 2** | PC2 explains **20.12%** of variance | LD2 accounts for **29.71%** of between-class trace |
| **Cumulative 2D Representation** | **57.12%** total variance retained | **100.0%** between-class space captured ($\min(13, 3-1)=2$) |
| **Out-of-Sample LDA Test Accuracy** | N/A (Descriptive feature extraction) | **98.11%** (52 of 53 held-out test samples correctly classified) |

---

## 📁 Repository Structure

```text
PCA-LDA-Dimensionality-Reduction/
├── app.R              # Complete R Shiny Feature Extraction Studio
├── run_app.R          # One-click startup script with automated dependency installer
├── validate_app.R     # Validation suite testing 70:30 split, standardization, and dynamic models
├── .gitignore         # Git ignore rules for R, RStudio, and Windows
└── README.md          # Complete project documentation and methodology
```

---

## 🛠️ Run Instructions

### Option 1: Quick Launch via PowerShell / Terminal
```powershell
cd C:\Users\HaniS\OneDrive\Desktop\PCA-LDA-Dimensionality-Reduction
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" run_app.R
```
*(Or simply `Rscript run_app.R` if `Rscript` is on your system PATH).*

### Option 2: Launch within R / RStudio
```r
setwd("C:/Users/HaniS/OneDrive/Desktop/PCA-LDA-Dimensionality-Reduction")
shiny::runApp(launch.browser = TRUE)
```

### Option 3: Run the Verification Suite
```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" validate_app.R
```

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
