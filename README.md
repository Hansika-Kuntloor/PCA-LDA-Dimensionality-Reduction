# 🍷 PCA & LDA Feature Extraction Studio

An academic **R Shiny** application for dimensionality reduction and feature extraction on the Wine dataset using **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)** with a strict, leak-free **70:30 stratified train-test split** (reproducible seed `42`).

The application is structured into **Three Dedicated Main Navigation Tabs**:
1. **Analysis**: Primary inputs (method, dataset, feature checkboxes, standardization, split info) and the main projection scatter plot.
2. **Insights**: Supporting visualisations tailored dynamically to the selected method (PCA scree plot & loadings vs. LDA discriminant variance & coefficients, with restored horizontal feature-contribution bars).
3. **Overview**: Summary cards and transformed dataset preview with CSV download, clearly distinguishing the latest completed analysis from pending input modifications.

---

## 📌 Architecture & Tab Suite

### 1. Persistent Top Navigation Bar
* **Positioning**: Sticky header (`position: sticky; top: 0; z-index: 1000;`) remaining visible during scrolling.
* **Left**: Project title **“PCA & LDA Feature Extraction Studio”** with brand icon.
* **Center**: Exactly three horizontal navigation tabs:
  * **Analysis**
  * **Insights**
  * **Overview**
* **Right**: Compact Wine Recognition dataset badge (`178 samples • 13 features • 3 classes`).
* **Styling**: White background (`#FFFFFF`), dark navy typography (`#0F172A`), rounded subtle hover states, thin light-grey bottom border (`#E5E7EB`), and active tab highlight with primary blue (`#315DEB`).

---

### 2. Tab 1 — Analysis
* **Left Control Panel**:
  * **Method Selector**: Segmented toggle buttons (`PCA` vs `LDA`).
  * **Dataset Information**: Summary identifying the Wine Recognition dataset (178 observations, 13 features, 3 classes).
  * **CSV Upload & Reset**: `Upload CSV` button supporting custom datasets, with a quick `Reset to Wine Dataset` link.
  * **Feature Selection**: Scrollable list of feature checkboxes with **Select All** and **Deselect All** quick links.
  * **Standardization Setting**: Checkbox toggle to enable/disable Z-score standardization ($\mu=0, \sigma=1$) using training-only statistics.
  * **70:30 Train-Test Split**: Visual progress bar showing **125 Training (70.2%)** and **53 Testing (29.8%)** samples (seed `42`), with an optional toggle to distinguish train/test marker shapes.
  * **Run Analysis Button**: Primary blue action button (`▶ Run Analysis`).
* **Right Scatter Plot Panel**:
  * Large, high-resolution scatter plot on a white background with subtle light-grey gridlines.
  * **PCA Mode**: Displays PC1 against PC2 with actual explained variance percentages in the axis titles and 95% shaded confidence ellipses.
  * **LDA Mode**: Displays LD1 against LD2 with actual between-class discriminant variance percentages.
  * **Class Palette**:
    * **Cultivar 1**: Blue (`#315DEB`)
    * **Cultivar 2**: Rose Pink (`#E64980`)
    * **Cultivar 3**: Muted Purple (`#8055C7`)
  * `Save PNG` export button.

---

### 3. Tab 2 — Insights
Dynamically displays supporting visualisations according to the active method:

#### A. When PCA is Selected:
* **Explained Variance Summary**: Metric chips for PC1 (37.00%), PC2 (20.12%), and Combined 2D (57.12%), plus a compact table with Component, Explained Variance (%), and Cumulative Variance (%).
* **PCA Scree Plot**: Clean `#315DEB` blue bar chart showing explained variance percentages across principal components without any connecting line or curve.
* **Restored PCA Feature Contributions**:
  * Dropdown selector for principal components (`PC1` through `PC13`).
  * Unclipped feature names aligned neatly on the left (`155px` width).
  * Horizontal bar tracks: **Blue (`#315DEB`)** for positive eigenvector loadings, **Rose Pink (`#E64980`)** for negative eigenvector loadings.
  * Right-aligned signed numerical values (e.g. `+0.428`, `-0.312`) in monospace font.
  * Sorted automatically by absolute loading magnitude, largest first.
  * Caption: *"Blue = positive · Pink = negative · Sorted by magnitude"*.

#### B. When LDA is Selected (Clean, Balanced 6/6 Layout):
* **LDA Discriminant Variance Chart & Fisher Context**: Summary chips for LD1 (70.29%), LD2 (29.71%), and Total Between-Class Trace (100.00%), plus clean blue bars and an explanatory box detailing Fisher's criterion ($S_B / S_W$) and 98.11% out-of-sample test accuracy (the duplicate scatter plot has been removed from Insights since the primary projection is always available in Analysis).
* **Restored LDA Feature Contributions**:
  * Dropdown selector for discriminant functions (`LD1` / `LD2`).
  * Horizontal bar layout matching PCA, with blue positive and pink negative coefficients.
  * Labeled explicitly as linear discriminant scaling coefficients (not percentages).

---

### 4. Tab 3 — Overview (Dynamic & Method-Aware)
* **Latest Completed Run Synchronization**:
  * Displays metrics, transformation summaries, and coordinates strictly corresponding to the **latest successfully completed analysis run**.
  * If the user selects a different method or alters features in the Analysis tab without clicking *Run Analysis*, Overview preserves the completed run's data and displays a warning notification indicating a run is pending.
  * In the event of an invalid or failed run, previous successful metrics and labels are preserved.
* **Dynamic Summary Cards**:
  * **Active Method**: Method name (PCA / LDA), coordinate transformation description, and Z-score standardization badge.
  * **Dataset Information**: Dataset name, total observation count, feature count, and cultivar class count.
  * **Method-Specific Metric Card**:
    * **PCA Completed**: Displays **PCA Variance Explained (2D)** (e.g. `57.12% Total Variance`, `PC1: 37.00% • PC2: 20.12%`, with `Unsupervised Eigen-Projection` badge).
    * **LDA Completed**: Displays **LDA Discriminant Trace (2D)** (e.g. `100.00% Between-Class Trace`, `LD1: 70.29% • LD2: 29.71%`, with `Supervised Fisher Criterion` badge).
  * **Train / Test Split**: 70:30 Train-Test split breakdown (125 Train / 53 Test) with seed 42 zero-leakage indicator.
* **Transformed Dataset Preview & Export (Method-Specific)**:
  * Dynamic preview table displaying the first 10 observations with method-specific coordinate columns:
    * `PC1` & `PC2` for completed PCA runs.
    * `LD1` & `LD2` for completed LDA runs.
  * Cultivar class tags and train/test partition badges.
  * `Download CSV` button exporting all observations with method-specific naming (`pca_transformed_data_all_*.csv` or `lda_transformed_data_all_*.csv`).

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
├── app.R              # Complete R Shiny Feature Extraction Studio (3-Tab Navigation)
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
