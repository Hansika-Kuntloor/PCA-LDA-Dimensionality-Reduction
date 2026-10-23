# 🍷 Feature Extraction (Dimensionality Reduction) Using PCA and LDA

An academic **R Shiny** application for feature extraction and dimensionality reduction on the Wine dataset using **Principal Component Analysis (PCA)** and **Linear Discriminant Analysis (LDA)** with a strict, leak-free **70:30 stratified train-test split**.

---

## 📌 Project Overview

This application provides a clean, academic interface focused on the core steps of feature extraction:

* **Three Core Sections**:
  1. **Dataset**: Introduction to the Wine dataset, chemical features description, and full interactive dataset table with split tags.
  2. **PCA**: Unsupervised linear feature extraction with beginner-friendly explanations and a PC1 vs PC2 scatter plot.
  3. **LDA**: Supervised linear feature extraction with beginner-friendly explanations and an LD1 vs LD2 scatter plot.
* **Strict 70:30 Stratified Split**: Exactly 70% of the dataset (125 samples) is used for training and 30% (53 samples) is held out for testing (`set.seed(42)`):
  * **Cultivar 1**: 41 Training / 18 Testing (59 total)
  * **Cultivar 2**: 50 Training / 21 Testing (71 total)
  * **Cultivar 3**: 34 Training / 14 Testing (48 total)
* **Zero Data Leakage**: Standardization parameters ($\mu$ and $\sigma$) are estimated **exclusively on the training partition**. Test features are transformed using the training-set parameters.
* **No Test Refitting**: PCA rotation eigenvectors and LDA discriminant functions are fitted **solely on the training partition**. Test observations are projected onto the learned axes without refitting.

---

## 🚀 Application Sections

### 1. 📊 Dataset
* Academic introduction to the Wine dataset (178 samples from 3 Italian wine cultivars).
* Description table of the 13 continuous chemical constituents.
* Concise sample count breakdown across training (125) and testing (53) partitions.
* Interactive, paginated data table showing cultivar labels, partition assignment, and rounded chemical values.
* Download full dataset as CSV.

### 2. 🔍 PCA (Unsupervised)
* Beginner-friendly academic explanation of variance maximization and orthogonality.
* Fitted strictly on the 125 standardized training observations without class labels.
* Variance explained: PC1 (**37.00%**), PC2 (**20.12%**), Cumulative (**57.12%**).
* Clean **PC1 vs PC2 scatter plot** with:
  * Observations colored by cultivar class.
  * Observations shaped by partition: **● Circles** = Training (70%), **▲ Triangles** = Testing (30%).
  * 95% confidence ellipses.
* Export scatter plot as high-resolution PNG (300 DPI) and PCA scores as CSV.

### 3. 🎯 LDA (Supervised)
* Beginner-friendly academic explanation of Fisher's discriminant criterion ($S_B / S_W$).
* Fitted strictly on the 125 training observations with cultivar labels.
* Trace explained: LD1 (**70.29%**), LD2 (**29.71%**), exhausting 100% of the 2D discriminant space ($\min(13, 3-1) = 2$).
* Clean **LD1 vs LD2 scatter plot** showing distinct class separation between cultivars.
* Export scatter plot as high-resolution PNG (300 DPI) and LDA scores as CSV.

---

## 📊 Summary of Model Findings (70:30 Split)

| Property / Metric | PCA (Unsupervised) | LDA (Supervised) |
| :--- | :--- | :--- |
| **Learning Paradigm** | Unsupervised (no class labels) | Supervised (uses cultivar labels) |
| **Optimization Goal** | Maximize total feature variance | Maximize between-class separation |
| **Training Partition (70%)** | 125 samples | 125 samples |
| **Testing Partition (30%)** | 53 samples | 53 samples |
| **2D Representation** | PC1 + PC2 (57.12% variance) | LD1 + LD2 (100% trace) |
| **Scatter Separation** | Partial cluster overlap (Cultivars 1 & 2) | Distinct, linearly separated clusters |

---

## 📁 Repository Structure

```text
PCA-LDA-Dimensionality-Reduction/
├── app.R              # Clean R Shiny application (Dataset, PCA, LDA sections)
├── run_app.R          # One-click launch script with automated dependency check
├── validate_app.R     # Assertion script verifying 70:30 split and mathematical integrity
├── .gitignore         # Git ignore rules for R, RStudio, and Windows
└── README.md          # Academic documentation and setup guide
```

---

## 🛠️ Windows Setup & Launch Instructions

### Prerequisites
* **R (version 4.0 or later)**: Installed on Windows ([CRAN](https://cran.r-project.org/bin/windows/base/)).

### Option 1: Quick Launch via Terminal / PowerShell
1. Open PowerShell or Command Prompt.
2. Navigate to the project root:
   ```powershell
   cd C:\Users\HaniS\OneDrive\Desktop\PCA-LDA-Dimensionality-Reduction
   ```
3. Run the launcher script:
   ```powershell
   & "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" run_app.R
   ```
   *(Or simply `Rscript run_app.R` if R is on your system PATH)*.

### Option 2: Launch from R or RStudio
```r
setwd("C:/Users/HaniS/OneDrive/Desktop/PCA-LDA-Dimensionality-Reduction")
shiny::runApp()
```

### Option 3: Run Automated Verification Tests
To verify the 70:30 split, zero leakage assertions, and model integrity:
```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" validate_app.R
```

---

## 📦 Required Packages

* `shiny`
* `ggplot2`
* `MASS`
* `rattle.data`
* `DT`

To install missing packages in R:
```r
install.packages(c("shiny", "ggplot2", "MASS", "rattle.data", "DT"), repos = "https://cloud.r-project.org")
```

---

## 📜 Dataset Citation
* **Source**: Forina, M. et al., PARVUS - An Extendible Package for Data Exploration, Classification and Correlation. Institute of Pharmaceutical and Food Analysis and Technologies, Genoa, Italy.
* Available via `rattle.data::wine` and the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine).

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
