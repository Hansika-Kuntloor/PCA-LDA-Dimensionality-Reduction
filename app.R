# ==============================================================================
# Feature Extraction Studio — PCA and LDA on the Wine Dataset
# Design: Minimal White & Sky Blue Academic Interface (Single-Page Studio)
# Methodology: Strict 70:30 Stratified Train-Test Split (Seed 42, Zero Leakage)
# ==============================================================================

# --- 1. Load Required Libraries ---
library(shiny)
library(ggplot2)
library(MASS)

# Optional interactive table & scales libraries
has_dt <- requireNamespace("DT", quietly = TRUE)
if (has_dt) {
  library(DT)
}
has_scales <- requireNamespace("scales", quietly = TRUE)

# --- 2. Load Default Dataset (Wine Recognition Dataset) ---
if (requireNamespace("rattle.data", quietly = TRUE)) {
  data(wine, package = "rattle.data")
} else {
  wine_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
  wine <- read.csv(wine_url, header = FALSE)
  colnames(wine) <- c("Type", "Alcohol", "Malic", "Ash", "Alcalinity", 
                      "Magnesium", "Phenols", "Flavanoids", "Nonflavanoids", 
                      "Proanthocyanins", "Color", "Hue", "Dilution", "Proline")
}

# Standardize column naming and clean factor labels for 3 cultivars
wine$Type <- as.factor(wine$Type)
wine$Cultivar <- factor(wine$Type, 
                        levels = c("1", "2", "3"),
                        labels = c("Cultivar 1", "Cultivar 2", "Cultivar 3"))

# Feature Metadata dictionary
features_meta <- data.frame(
  key = c("alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", 
          "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", 
          "color_intensity", "hue", "od280_od315", "proline"),
  col_name = c("Alcohol", "Malic", "Ash", "Alcalinity", "Magnesium", 
               "Phenols", "Flavanoids", "Nonflavanoids", "Proanthocyanins", 
               "Color", "Hue", "Dilution", "Proline"),
  display_name = c("alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", 
                   "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", 
                   "color_intensity", "hue", "od280_od315", "proline"),
  unit = c("% vol", "g/L", "g/L", "meq/L", "mg/L", 
           "mg/L GAE", "mg/L", "mg/L", "mg/L", 
           "Abs units", "Ratio 420/520 nm", "OD ratio", "mg/L"),
  stringsAsFactors = FALSE
)

feature_cols <- features_meta$col_name
feature_keys <- features_meta$key
names(feature_cols) <- feature_keys

# --- 3. Fixed, Stratified 70:30 Train-Test Split (Seed 42) ---
set.seed(42)

cultivar_levels <- levels(wine$Cultivar)
train_indices <- integer(0)

# Stratified sampling: 70% of each cultivar into training, 30% into testing
for (cult in cultivar_levels) {
  cult_idx <- which(wine$Cultivar == cult)
  n_cult <- length(cult_idx)
  n_train <- round(n_cult * 0.70)
  train_indices <- c(train_indices, sample(cult_idx, size = n_train))
}

train_indices <- sort(train_indices)
test_indices <- setdiff(seq_len(nrow(wine)), train_indices)

# Assign Split identifier
wine$Split <- "Testing (30%)"
wine$Split[train_indices] <- "Training (70%)"
wine$Split <- factor(wine$Split, levels = c("Training (70%)", "Testing (30%)"))

wine_train <- wine[train_indices, ]
wine_test  <- wine[test_indices, ]

# Curated palette for cultivars matching reference design
cultivar_colors_default <- c(
  "Cultivar 1" = "#2563eb",  # Royal Blue
  "Cultivar 2" = "#f97316",  # Warm Orange
  "Cultivar 3" = "#9333ea"   # Vibrant Purple
)

split_shapes <- c(
  "Training (70%)" = 21,  # Circle with border
  "Testing (30%)"  = 24   # Triangle with border
)

# --- 4. Modeling Engine (Zero Leakage, Training Partition Only) ---
compute_feature_models <- function(selected_keys, 
                                   standardize = TRUE,
                                   tr_data = wine_train, 
                                   te_data = wine_test,
                                   class_col = "Cultivar",
                                   cols_dict = feature_cols) {
  if (length(selected_keys) < 2) {
    return(NULL)
  }
  
  sel_cols <- cols_dict[selected_keys]
  
  # Standardize strictly using training partition statistics
  mu <- colMeans(tr_data[, sel_cols, drop = FALSE], na.rm = TRUE)
  sigma <- apply(tr_data[, sel_cols, drop = FALSE], 2, sd, na.rm = TRUE)
  sigma[is.na(sigma) | sigma == 0] <- 1
  
  if (standardize) {
    tr_scaled <- as.data.frame(scale(tr_data[, sel_cols, drop = FALSE], center = mu, scale = sigma))
    te_scaled <- as.data.frame(scale(te_data[, sel_cols, drop = FALSE], center = mu, scale = sigma))
  } else {
    tr_scaled <- as.data.frame(scale(tr_data[, sel_cols, drop = FALSE], center = mu, scale = FALSE))
    te_scaled <- as.data.frame(scale(te_data[, sel_cols, drop = FALSE], center = mu, scale = FALSE))
  }
  
  # 1. PCA (Unsupervised: fitted on training features only, without class labels)
  pca_fit <- prcomp(tr_scaled, center = FALSE, scale. = FALSE)
  pca_var <- pca_fit$sdev^2
  pca_var_pct <- (pca_var / sum(pca_var)) * 100
  
  pca_tr_scores <- as.data.frame(pca_fit$x)
  pca_tr_scores$Cultivar <- tr_data[[class_col]]
  pca_tr_scores$Split    <- tr_data$Split
  
  pca_te_scores <- as.data.frame(predict(pca_fit, newdata = te_scaled))
  pca_te_scores$Cultivar <- te_data[[class_col]]
  pca_te_scores$Split    <- te_data$Split
  
  pca_all_scores <- rbind(pca_tr_scores, pca_te_scores)
  
  # 2. LDA (Supervised: fitted on training features with training cultivar labels)
  tr_lda_input <- tr_scaled
  tr_lda_input$Cultivar <- tr_data[[class_col]]
  
  lda_fit <- MASS::lda(Cultivar ~ ., data = tr_lda_input)
  lda_trace_pct <- (lda_fit$svd^2 / sum(lda_fit$svd^2)) * 100
  
  lda_tr_pred <- predict(lda_fit, newdata = tr_lda_input)
  lda_tr_scores <- as.data.frame(lda_tr_pred$x)
  lda_tr_scores$Cultivar <- tr_data[[class_col]]
  lda_tr_scores$Split    <- tr_data$Split
  
  lda_te_pred <- predict(lda_fit, newdata = te_scaled)
  lda_te_scores <- as.data.frame(lda_te_pred$x)
  lda_te_scores$Cultivar <- te_data[[class_col]]
  lda_te_scores$Split    <- te_data$Split
  
  lda_all_scores <- rbind(lda_tr_scores, lda_te_scores)
  
  # Out-of-sample LDA test classification accuracy
  lda_test_conf <- table(Predicted = lda_te_pred$class, Actual = te_data[[class_col]])
  lda_test_acc  <- (sum(diag(lda_test_conf)) / sum(lda_test_conf)) * 100
  
  list(
    selected_keys = selected_keys,
    selected_cols = sel_cols,
    standardize = standardize,
    train_means = mu,
    train_sds = sigma,
    pca_fit = pca_fit,
    pca_var_pct = pca_var_pct,
    pca_train_scores = pca_tr_scores,
    pca_test_scores = pca_te_scores,
    pca_scores_df = pca_all_scores,
    lda_fit = lda_fit,
    lda_trace_pct = lda_trace_pct,
    lda_train_scores = lda_tr_scores,
    lda_test_scores = lda_te_scores,
    lda_scores_df = lda_all_scores,
    lda_test_acc = lda_test_acc
  )
}

# --- 5. Baseline Global Model (All 13 Features) for Script & Assertion Integrity ---
baseline_res <- compute_feature_models(feature_keys)

train_means      <- baseline_res$train_means
train_sds        <- baseline_res$train_sds
pca_fit          <- baseline_res$pca_fit
pca_var_pct      <- baseline_res$pca_var_pct
pca_train_scores <- baseline_res$pca_train_scores
pca_test_scores  <- baseline_res$pca_test_scores
pca_scores_df    <- baseline_res$pca_scores_df
lda_fit          <- baseline_res$lda_fit
lda_trace_pct    <- baseline_res$lda_trace_pct
lda_train_scores <- baseline_res$lda_train_scores
lda_test_scores  <- baseline_res$lda_test_scores
lda_scores_df    <- baseline_res$lda_scores_df
lda_test_acc     <- baseline_res$lda_test_acc


# --- 6. User Interface (UI) Definition ---
ui <- fluidPage(
  title = "Feature Extraction Studio — PCA and LDA on the Wine Dataset",
  
  tags$head(
    tags$style(HTML("
      /* Google Font & Layout Reset */
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
      
      *, *::before, *::after {
        box-sizing: border-box;
      }
      
      body {
        background-color: #f4f8fd;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        margin: 0;
        padding: 0;
        line-height: 1.5;
        overflow-x: hidden;
      }
      
      /* Subtle Leaf Background Flourishes */
      .bg-leaf-tl {
        position: fixed;
        top: -20px;
        left: -20px;
        pointer-events: none;
        z-index: 0;
        opacity: 0.6;
      }
      .bg-leaf-tr {
        position: fixed;
        top: -20px;
        right: -20px;
        pointer-events: none;
        z-index: 0;
        opacity: 0.45;
      }
      .bg-leaf-bl {
        position: fixed;
        bottom: -20px;
        left: -20px;
        pointer-events: none;
        z-index: 0;
        opacity: 0.5;
      }
      
      /* Studio Header (Single-Page, No separate tabs) */
      .studio-header {
        position: relative;
        z-index: 10;
        background: rgba(255, 255, 255, 0.94);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid #e2e8f0;
        padding: 16px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
      }
      .brand-box {
        display: flex;
        align-items: center;
        gap: 14px;
      }
      .brand-icon-svg {
        width: 38px;
        height: 38px;
        flex-shrink: 0;
      }
      .brand-title {
        font-size: 20px;
        font-weight: 700;
        color: #0f2744;
        margin: 0;
        letter-spacing: -0.4px;
      }
      .brand-subtitle {
        font-size: 13px;
        font-weight: 500;
        color: #64748b;
        margin: 2px 0 0 0;
      }
      .header-dataset-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
      }
      
      /* Main Content Container */
      .studio-container {
        position: relative;
        z-index: 5;
        padding: 22px 28px 40px 28px;
        max-width: 1620px;
        margin: 0 auto;
      }
      
      /* Cards */
      .studio-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px -2px rgba(15, 35, 75, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
      }
      
      /* Card Headers */
      .card-header-flex {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
      }
      .card-title-group {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .card-icon {
        color: #2563eb;
        font-size: 16px;
      }
      .card-main-title {
        font-size: 15px;
        font-weight: 700;
        color: #0f2744;
        margin: 0;
      }
      .card-subtitle {
        font-size: 12px;
        color: #64748b;
        margin: 2px 0 0 0;
      }
      
      /* Segmented Method Selector */
      .method-segmented .shiny-options-group {
        display: flex;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 4px;
        margin: 0 0 16px 0;
      }
      .method-segmented .radio-inline {
        flex: 1;
        margin: 0;
        padding: 0;
        text-align: center;
      }
      .method-segmented .radio-inline input[type='radio'] {
        display: none;
      }
      .method-segmented .radio-inline span {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 9px 12px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        color: #475569;
        cursor: pointer;
        transition: all 0.2s ease;
      }
      .method-segmented .radio-inline input[type='radio']:checked + span {
        background: #2563eb;
        color: #ffffff;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.28);
      }
      
      /* Compact Dataset Info & Upload Box */
      .dataset-compact-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 16px;
      }
      .dataset-compact-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
      }
      .dataset-icon-circle {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: #eff6ff;
        color: #2563eb;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
      }
      .dataset-compact-title {
        font-size: 13px;
        font-weight: 700;
        color: #0f2744;
        margin: 0;
      }
      .dataset-compact-meta {
        font-size: 11px;
        color: #64748b;
        margin: 0;
      }
      .dataset-upload-row {
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #e2e8f0;
      }
      .dataset-upload-row .shiny-input-container {
        margin-bottom: 0;
      }
      .dataset-upload-row .btn-file {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        color: #2563eb;
        font-weight: 600;
        font-size: 12px;
        padding: 5px 10px;
        border-radius: 6px;
      }
      .dataset-upload-row .btn-file:hover {
        background: #eff6ff;
        border-color: #3b82f6;
      }
      .dataset-upload-row .form-control {
        font-size: 11px;
        height: 30px;
      }
      .reset-link {
        font-size: 11px;
        font-weight: 600;
        color: #dc2626;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        margin-top: 6px;
      }
      .reset-link:hover {
        text-decoration: underline;
      }
      
      /* Feature Selection Section & Alignment Fix */
      .feature-select-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
      }
      .feature-select-title {
        font-size: 13px;
        font-weight: 700;
        color: #0f2744;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .feature-quick-links {
        font-size: 12px;
      }
      .feature-quick-links a {
        color: #2563eb;
        font-weight: 600;
        text-decoration: none;
        cursor: pointer;
        margin-left: 6px;
      }
      .feature-quick-links a:hover {
        text-decoration: underline;
      }
      .features-scroll-box {
        max-height: 200px;
        overflow-y: auto;
        padding: 10px 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin-bottom: 14px;
      }
      .features-scroll-box::-webkit-scrollbar {
        width: 6px;
      }
      .features-scroll-box::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
      }
      .features-scroll-box .shiny-options-group {
        display: flex;
        flex-direction: column;
        gap: 4px;
      }
      .features-scroll-box .checkbox {
        position: relative !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        min-height: auto !important;
      }
      .features-scroll-box .checkbox label {
        display: flex !important;
        align-items: center !important;
        padding: 4px 6px !important;
        padding-left: 0 !important;
        margin: 0 !important;
        min-height: auto !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #1e293b !important;
        cursor: pointer !important;
        border-radius: 6px !important;
        transition: background 0.15s ease !important;
      }
      .features-scroll-box .checkbox label:hover {
        background: #f1f5f9 !important;
      }
      /* Separate, fixed-width area for checkbox input with NO negative margins */
      .features-scroll-box .checkbox input[type='checkbox'] {
        position: static !important;
        float: none !important;
        margin: 0 10px 0 0 !important;
        width: 16px !important;
        height: 16px !important;
        min-width: 16px !important;
        max-width: 16px !important;
        flex-shrink: 0 !important;
        accent-color: #2563eb !important;
        cursor: pointer !important;
      }
      /* Feature label text beside the checkbox with clean separation */
      .features-scroll-box .checkbox label span {
        display: inline-block !important;
        padding-left: 0 !important;
        margin-left: 0 !important;
        color: #334155 !important;
        font-size: 13px !important;
        line-height: 1.3 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
      }
      
      /* Standardization Setting Box */
      .standardize-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 14px;
      }
      .standardize-box .checkbox {
        margin: 0;
      }
      .standardize-box .checkbox label {
        font-size: 12.5px;
        font-weight: 600;
        color: #0f2744;
        display: flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
      }
      .standardize-box input[type='checkbox'] {
        accent-color: #2563eb;
        width: 15px;
        height: 15px;
        cursor: pointer;
      }
      .standardize-help {
        font-size: 11px;
        color: #64748b;
        margin: 4px 0 0 23px;
        line-height: 1.4;
      }
      
      /* Train-Test Split Block */
      .split-block {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 16px;
      }
      .split-block-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
      }
      .split-title {
        font-size: 13px;
        font-weight: 700;
        color: #0f2744;
      }
      .split-pct-text {
        font-size: 12px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 6px;
      }
      .split-bar-container {
        height: 8px;
        border-radius: 6px;
        background: #f1f5f9;
        overflow: hidden;
        display: flex;
        margin-bottom: 6px;
      }
      .split-bar-train {
        height: 100%;
        background: #2563eb;
        width: 70.2%;
      }
      .split-bar-test {
        height: 100%;
        background: #bae6fd;
        width: 29.8%;
      }
      .split-detail-text {
        font-size: 11px;
        color: #64748b;
      }
      
      /* Run Analysis Button */
      .btn-run-studio {
        background: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 14px;
        font-weight: 700;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.28);
        transition: all 0.2s ease;
      }
      .btn-run-studio:hover {
        background: #1d4ed8;
        color: #ffffff;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
      }
      
      /* Circular SVG Gauges */
      .gauges-container {
        display: flex;
        align-items: center;
        justify-content: space-around;
        gap: 10px;
        padding: 8px 0;
      }
      .gauge-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
      }
      .gauge-header-label {
        font-size: 12px;
        font-weight: 700;
        color: #475569;
        margin-bottom: 4px;
      }
      .gauge-svg {
        transform: rotate(-90deg);
        display: block;
      }
      .gauge-track {
        fill: none;
        stroke: #f1f5f9;
        stroke-width: 8;
      }
      .gauge-fill {
        fill: none;
        stroke-width: 8;
        stroke-linecap: round;
        transition: stroke-dashoffset 0.6s ease;
      }
      .gauge-fill-pc1 { stroke: #2563eb; }
      .gauge-fill-pc2 { stroke: #8b5cf6; }
      .gauge-fill-total { stroke: #1e3a8a; }
      .gauge-text {
        font-size: 14px;
        font-weight: 700;
        fill: #0f2744;
        text-anchor: middle;
        dominant-baseline: central;
        transform: rotate(90deg);
        transform-origin: 50% 50%;
      }
      .gauge-sub-label {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
      }
      
      /* Feature Contributions */
      .contrib-select-wrapper {
        width: 90px;
      }
      .contrib-select-wrapper .form-group {
        margin-bottom: 0;
      }
      .contrib-select-wrapper select {
        padding: 4px 8px;
        height: 32px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 6px;
        border: 1px solid #cbd5e1;
      }
      .feature-contrib-list {
        max-height: 250px;
        overflow-y: auto;
        padding-right: 4px;
      }
      .feature-contrib-list::-webkit-scrollbar {
        width: 5px;
      }
      .feature-contrib-list::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
      }
      .feature-contrib-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
      }
      .contrib-name {
        width: 120px;
        font-size: 12px;
        font-weight: 500;
        color: #1e293b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .contrib-bar-track {
        flex-grow: 1;
        height: 8px;
        background: #f1f5f9;
        border-radius: 6px;
        overflow: hidden;
      }
      .contrib-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #38bdf8, #2563eb);
        border-radius: 6px;
        transition: width 0.4s ease;
      }
      .contrib-val {
        width: 38px;
        text-align: right;
        font-size: 12px;
        font-weight: 600;
        color: #0f2744;
        font-variant-numeric: tabular-nums;
      }
      
      /* Clean Academic Table Preview */
      .clean-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }
      .clean-table th {
        background: #f8fafc;
        color: #475569;
        font-weight: 700;
        padding: 10px 14px;
        border-bottom: 1px solid #e2e8f0;
        text-align: left;
      }
      .clean-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #f1f5f9;
        color: #334155;
      }
      .clean-table tr:hover {
        background: #f8fafc;
      }
      
      /* Badges */
      .badge-cultivar-1 {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
      }
      .badge-cultivar-2 {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #fff7ed;
        color: #c2410c;
        border: 1px solid #fed7aa;
      }
      .badge-cultivar-3 {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #faf5ff;
        color: #7e22ce;
        border: 1px solid #e9d5ff;
      }
      .badge-train {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
      }
      .badge-test {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #f8fafc;
        color: #475569;
        border: 1px solid #cbd5e1;
      }
      
      /* Action / Export Buttons */
      .btn-export {
        background: #ffffff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        cursor: pointer;
        transition: all 0.15s ease;
        text-decoration: none !important;
      }
      .btn-export:hover {
        background: #eff6ff;
        border-color: #3b82f6;
        color: #1d4ed8;
      }
      
      /* Info Tag */
      .info-tag {
        font-size: 11px;
        color: #64748b;
        background: #f1f5f9;
        border-radius: 6px;
        padding: 5px 8px;
        margin-top: 6px;
        line-height: 1.4;
      }
      
      /* Validation Alert */
      .studio-alert-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 8px;
        color: #92400e;
        font-size: 13px;
        margin-bottom: 16px;
      }
    "))
  ),
  
  # Decorative Leaf SVG Background Elements
  tags$div(class = "bg-leaf-tl",
    HTML('<svg width="220" height="220" viewBox="0 0 200 200" fill="none"><path d="M10 190C30 110 110 30 190 10C160 80 130 130 80 160C40 180 20 190 10 190Z" fill="#3b82f6" fill-opacity="0.10"/><path d="M10 190C50 140 100 90 190 10" stroke="#3b82f6" stroke-opacity="0.15" stroke-width="2"/></svg>')
  ),
  tags$div(class = "bg-leaf-tr",
    HTML('<svg width="240" height="240" viewBox="0 0 200 200" fill="none"><path d="M190 10C110 30 30 110 10 190C80 160 130 130 160 80C180 40 190 20 190 10Z" fill="#60a5fa" fill-opacity="0.08"/></svg>')
  ),
  tags$div(class = "bg-leaf-bl",
    HTML('<svg width="200" height="200" viewBox="0 0 200 200" fill="none"><path d="M10 190C60 130 120 70 190 10C130 60 70 120 10 190Z" fill="#93c5fd" fill-opacity="0.09"/></svg>')
  ),
  
  # Studio Header (Clean, focused single-page header)
  div(class = "studio-header",
    div(class = "brand-box",
      HTML('<svg class="brand-icon-svg" viewBox="0 0 40 40" fill="none"><path d="M4 36C8 20 22 6 36 4C30 16 24 24 14 30C8 34 5 36 4 36Z" fill="#2563eb"/><path d="M4 36C12 26 20 18 36 4" stroke="#ffffff" stroke-width="1.8" stroke-linecap="round"/><path d="M14 24C16 18 22 14 28 12" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" opacity="0.8"/></svg>'),
      div(
        h1(class = "brand-title", "Feature Extraction Studio"),
        p(class = "brand-subtitle", "PCA and LDA on the Wine Dataset")
      )
    ),
    div(class = "header-dataset-badge",
      HTML("🍇"), textOutput("header_badge_text", inline = TRUE)
    )
  ),
  
  # Main Studio Dashboard Container
  div(class = "studio-container",
    
    # UI Warning if fewer than 2 features selected
    uiOutput("validation_alert"),
    
    fluidRow(
      # ------------------------------------------------------------------------
      # COLUMN 1: LEFT CONTROLS & FEATURE SELECTION (~25% width)
      # ------------------------------------------------------------------------
      column(width = 3,
        div(class = "studio-card",
          
          # 1. Method Selector
          div(class = "card-header-flex", style = "margin-bottom: 8px;",
            div(class = "card-title-group",
              span(class = "card-icon", HTML("⚙")),
              h4(class = "card-main-title", "Method")
            )
          ),
          div(class = "method-segmented",
            radioButtons("method", label = NULL,
                         choices = c("PCA" = "pca", "LDA" = "lda"),
                         selected = "pca", inline = TRUE)
          ),
          
          # 2. Compact Dataset Information & Optional Upload
          div(class = "card-header-flex", style = "margin-bottom: 8px;",
            div(class = "card-title-group",
              span(class = "card-icon", HTML("🗄")),
              h4(class = "card-main-title", "Dataset Information")
            )
          ),
          div(class = "dataset-compact-box",
            div(class = "dataset-compact-header",
              div(class = "dataset-icon-circle", HTML("🍷")),
              div(
                p(class = "dataset-compact-title", textOutput("ds_name", inline = TRUE)),
                p(class = "dataset-compact-meta", textOutput("ds_meta", inline = TRUE))
              )
            ),
            div(class = "dataset-upload-row",
              fileInput("upload_csv", label = NULL, buttonLabel = "Upload CSV", 
                        placeholder = "Wine dataset default", accept = c(".csv", "text/csv")),
              uiOutput("ui_reset_dataset")
            )
          ),
          
          # 3. Feature Selection
          div(class = "feature-select-header",
            div(class = "feature-select-title",
              span(class = "card-icon", HTML("▼")),
              span("Feature Selection")
            ),
            div(class = "feature-quick-links",
              actionLink("select_all", "Select All"),
              span("|", style = "color: #cbd5e1; margin: 0 2px;"),
              actionLink("deselect_all", "Deselect All")
            )
          ),
          div(class = "features-scroll-box",
            checkboxGroupInput("selected_features", label = NULL,
                               choiceNames = features_meta$display_name,
                               choiceValues = features_meta$key,
                               selected = features_meta$key)
          ),
          
          # 4. Standardization Setting
          div(class = "standardize-box",
            checkboxInput("standardize", label = "Standardize Features (Z-Score)", value = TRUE),
            p(class = "standardize-help", 
              "Scales features to zero mean and unit variance using training statistics (recommended for continuous features with different units).")
          ),
          
          # 5. Fixed Train-Test Split (Strict 70:30, Seed 42)
          div(class = "split-block",
            div(class = "split-block-header",
              span(class = "card-icon", HTML("🕒")),
              span(class = "split-title", "Train-Test Split")
            ),
            div(class = "split-pct-text", "70% Training | 30% Testing"),
            div(class = "split-bar-container",
              div(class = "split-bar-train"),
              div(class = "split-bar-test")
            ),
            div(class = "split-detail-text", textOutput("split_detail_text", inline = TRUE))
          ),
          
          # 6. Run Analysis Button
          actionButton("btn_run", label = tagList(HTML("▶"), "Run Analysis"),
                       class = "btn-run-studio")
        )
      ),
      
      # ------------------------------------------------------------------------
      # COLUMN 2: CENTER PLOTS (SCATTER & CONDITIONAL SCREE) (~50% width)
      # ------------------------------------------------------------------------
      column(width = 6,
        # Main Scatter Plot Card (Large)
        div(class = "studio-card",
          div(class = "card-header-flex",
            div(class = "card-title-group",
              span(class = "card-icon", HTML("📊")),
              div(
                h4(class = "card-main-title", textOutput("scatter_card_title", inline = TRUE)),
                p(class = "card-subtitle", textOutput("scatter_card_subtitle", inline = TRUE))
              )
            ),
            downloadButton("download_scatter_png", "Save PNG", class = "btn-export")
          ),
          plotOutput("plot_scatter", height = "440px")
        ),
        
        # Secondary Chart: PCA Scree Plot (when PCA is selected)
        conditionalPanel(
          condition = "input.method == 'pca'",
          div(class = "studio-card",
            div(class = "card-header-flex",
              div(class = "card-title-group",
                span(class = "card-icon", HTML("☊")),
                div(
                  h4(class = "card-main-title", "PCA Scree Plot"),
                  p(class = "card-subtitle", "Explained variance ratio per principal component")
                )
              ),
              downloadButton("download_scree_png", "Save PNG", class = "btn-export")
            ),
            plotOutput("plot_scree", height = "240px")
          )
        ),
        
        # Secondary Chart: LDA Discriminant Variance Bar Chart (when LDA is selected)
        conditionalPanel(
          condition = "input.method == 'lda'",
          div(class = "studio-card",
            div(class = "card-header-flex",
              div(class = "card-title-group",
                span(class = "card-icon", HTML("📊")),
                div(
                  h4(class = "card-main-title", "LDA Discriminant Variance"),
                  p(class = "card-subtitle", "Relative discriminant information captured by each linear discriminant")
                )
              ),
              downloadButton("download_lda_variance_png", "Save PNG", class = "btn-export")
            ),
            plotOutput("plot_lda_variance", height = "240px")
          )
        )
      ),
      
      # ------------------------------------------------------------------------
      # COLUMN 3: METRICS & FEATURE CONTRIBUTIONS (~25% width)
      # ------------------------------------------------------------------------
      column(width = 3,
        # Explained Variance / Discriminant Variance Card
        div(class = "studio-card",
          div(class = "card-header-flex",
            div(class = "card-title-group",
              span(class = "card-icon", HTML("❖")),
              h4(class = "card-main-title", textOutput("variance_card_title", inline = TRUE))
            ),
            span(title = "Variance or between-class separation explained by the components",
                 style = "color: #94a3b8; cursor: help; font-size: 14px;", "ⓘ")
          ),
          uiOutput("ui_circular_gauges")
        ),
        
        # Feature Contributions Card (Single Panel with Dynamic Dropdown)
        div(class = "studio-card",
          div(class = "card-header-flex",
            div(class = "card-title-group",
              span(class = "card-icon", HTML("❄")),
              h4(class = "card-main-title", textOutput("contrib_card_title", inline = TRUE))
            ),
            div(class = "contrib-select-wrapper",
              selectInput("contrib_comp", label = NULL, choices = c("PC1" = "PC1"))
            )
          ),
          div(class = "info-tag", textOutput("contrib_info_desc")),
          tags$hr(style = "margin: 10px 0 12px 0; border: none; border-top: 1px solid #f1f5f9;"),
          uiOutput("ui_feature_contributions")
        )
      )
    ),
    
    # --------------------------------------------------------------------------
    # BOTTOM ROW: TRANSFORMED DATA PREVIEW (FIRST 5 ROWS)
    # --------------------------------------------------------------------------
    fluidRow(
      column(width = 12,
        div(class = "studio-card",
          div(class = "card-header-flex",
            div(class = "card-title-group",
              span(class = "card-icon", HTML("▦")),
              div(
                h4(class = "card-main-title", "Transformed Data (First 5 Rows)"),
                p(class = "card-subtitle", "Projected coordinates alongside original class labels and partition tags")
              )
            ),
            downloadButton("download_transformed_csv", "Download CSV", class = "btn-export")
          ),
          tableOutput("table_transformed_preview")
        )
      )
    )
  )
)


# --- 7. Server Logic Definition ---
server <- function(input, output, session) {
  
  # Reactive state storing current dataset (Wine default or custom uploaded)
  dataset_state <- reactiveValues(
    name = "Wine Dataset",
    is_custom = FALSE,
    raw = wine,
    train = wine_train,
    test = wine_test,
    class_col = "Cultivar",
    keys = feature_keys,
    cols = feature_cols,
    display_names = setNames(features_meta$display_name, features_meta$key)
  )
  
  # Reactive container holding current fitted model results
  active_model <- reactiveVal(baseline_res)
  
  # Header & Dataset Info Text
  output$header_badge_text <- renderText({
    sprintf("%s • %d Samples • %d Features",
            dataset_state$name, nrow(dataset_state$raw), length(dataset_state$keys))
  })
  
  output$ds_name <- renderText({
    dataset_state$name
  })
  
  output$ds_meta <- renderText({
    n_samp <- nrow(dataset_state$raw)
    n_feat <- length(dataset_state$keys)
    n_cls  <- length(unique(dataset_state$raw[[dataset_state$class_col]]))
    sprintf("%d samples • %d features • %d classes", n_samp, n_feat, n_cls)
  })
  
  output$split_detail_text <- renderText({
    n_tr <- nrow(dataset_state$train)
    n_te <- nrow(dataset_state$test)
    sprintf("%d Training • %d Testing (Seed 42)", n_tr, n_te)
  })
  
  output$ui_reset_dataset <- renderUI({
    if (dataset_state$is_custom) {
      actionLink("reset_wine", "↺ Reset to Wine Dataset", class = "reset-link")
    } else {
      NULL
    }
  })
  
  # Handle Optional CSV Upload
  observeEvent(input$upload_csv, {
    file <- input$upload_csv
    req(file)
    
    df_uploaded <- tryCatch({
      read.csv(file$datapath, header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
    }, error = function(e) {
      showNotification(paste("Error reading CSV:", e$message), type = "error")
      return(NULL)
    })
    
    if (is.null(df_uploaded)) return()
    
    if (nrow(df_uploaded) < 10) {
      showNotification("Uploaded CSV must contain at least 10 observations.", type = "error")
      return()
    }
    
    # Detect numeric and potential class columns
    num_mask <- sapply(df_uploaded, is.numeric)
    num_names <- names(df_uploaded)[num_mask]
    non_num_names <- names(df_uploaded)[!num_mask]
    class_cand <- NULL
    
    if (length(non_num_names) > 0) {
      for (cn in non_num_names) {
        if (length(unique(df_uploaded[[cn]])) >= 2) {
          class_cand <- cn
          break
        }
      }
    }
    
    # If no non-numeric column, check integer columns with small cardinality (2 to 10 unique values)
    if (is.null(class_cand)) {
      for (cn in num_names) {
        u_cnt <- length(unique(df_uploaded[[cn]]))
        if (u_cnt >= 2 && u_cnt <= 10) {
          class_cand <- cn
          num_names <- setdiff(num_names, cn)
          break
        }
      }
    }
    
    if (is.null(class_cand)) {
      showNotification("Validation error: Uploaded CSV must contain a class-label column with at least 2 distinct classes for LDA.", 
                       type = "error", duration = 6)
      return()
    }
    
    if (length(num_names) < 2) {
      showNotification("Validation error: Uploaded CSV must contain at least 2 numeric features for dimensionality reduction.", 
                       type = "error", duration = 6)
      return()
    }
    
    # Format class column
    df_uploaded[[class_cand]] <- as.factor(df_uploaded[[class_cand]])
    
    # Fixed stratified 70:30 split with seed 42
    set.seed(42)
    tr_idx <- integer(0)
    for (cls in levels(df_uploaded[[class_cand]])) {
      idx_cls <- which(df_uploaded[[class_cand]] == cls)
      n_tr <- max(1, round(length(idx_cls) * 0.70))
      tr_idx <- c(tr_idx, sample(idx_cls, size = n_tr))
    }
    tr_idx <- sort(tr_idx)
    te_idx <- setdiff(seq_len(nrow(df_uploaded)), tr_idx)
    
    df_uploaded$Split <- "Testing (30%)"
    df_uploaded$Split[tr_idx] <- "Training (70%)"
    df_uploaded$Split <- factor(df_uploaded$Split, levels = c("Training (70%)", "Testing (30%)"))
    
    up_train <- df_uploaded[tr_idx, ]
    up_test  <- df_uploaded[te_idx, ]
    
    # Generate clean keys and dictionaries
    up_keys <- tolower(gsub("[^A-Za-z0-9_]", "_", num_names))
    up_keys <- make.unique(up_keys)
    up_cols <- num_names
    names(up_cols) <- up_keys
    
    # Update dataset state
    dataset_state$name <- file$name
    dataset_state$is_custom <- TRUE
    dataset_state$raw <- df_uploaded
    dataset_state$train <- up_train
    dataset_state$test <- up_test
    dataset_state$class_col <- class_cand
    dataset_state$keys <- up_keys
    dataset_state$cols <- up_cols
    dataset_state$display_names <- setNames(up_cols, up_keys)
    
    # Update checkboxes
    updateCheckboxGroupInput(session, "selected_features",
                             choices = setNames(up_keys, up_cols),
                             selected = up_keys)
    
    # Recompute models
    res_up <- compute_feature_models(
      selected_keys = up_keys,
      standardize = input$standardize,
      tr_data = up_train,
      te_data = up_test,
      class_col = class_cand,
      cols_dict = up_cols
    )
    if (!is.null(res_up)) {
      active_model(res_up)
    }
    
    showNotification(sprintf("Uploaded '%s' loaded successfully (%d samples, %d features).",
                             file$name, nrow(df_uploaded), length(up_keys)),
                     type = "message")
  })
  
  # Reset back to default Wine dataset
  observeEvent(input$reset_wine, {
    dataset_state$name <- "Wine Dataset"
    dataset_state$is_custom <- FALSE
    dataset_state$raw <- wine
    dataset_state$train <- wine_train
    dataset_state$test <- wine_test
    dataset_state$class_col <- "Cultivar"
    dataset_state$keys <- feature_keys
    dataset_state$cols <- feature_cols
    dataset_state$display_names <- setNames(features_meta$display_name, features_meta$key)
    
    updateCheckboxGroupInput(session, "selected_features",
                             choices = setNames(features_meta$key, features_meta$display_name),
                             selected = features_meta$key)
    
    res_wine <- compute_feature_models(
      selected_keys = feature_keys,
      standardize = input$standardize,
      tr_data = wine_train,
      te_data = wine_test,
      class_col = "Cultivar",
      cols_dict = feature_cols
    )
    if (!is.null(res_wine)) {
      active_model(res_wine)
    }
    showNotification("Reverted to default Wine Recognition dataset.", type = "message")
  })
  
  # Quick Link Observers: Select All / Deselect All
  observeEvent(input$select_all, {
    updateCheckboxGroupInput(session, "selected_features", selected = dataset_state$keys)
  })
  observeEvent(input$deselect_all, {
    updateCheckboxGroupInput(session, "selected_features", selected = character(0))
  })
  
  # Run Analysis Trigger
  observeEvent(input$btn_run, {
    req(input$selected_features)
    
    if (length(input$selected_features) < 2) {
      showNotification("Please select at least 2 features to perform 2D dimensionality reduction.",
                       type = "warning", duration = 4)
      return()
    }
    
    res <- compute_feature_models(
      selected_keys = input$selected_features,
      standardize = input$standardize,
      tr_data = dataset_state$train,
      te_data = dataset_state$test,
      class_col = dataset_state$class_col,
      cols_dict = dataset_state$cols
    )
    if (!is.null(res)) {
      active_model(res)
    }
  })
  
  # Dynamic updates to Feature Contributions Dropdown choices
  observe({
    res <- active_model()
    req(res)
    m <- input$method
    
    if (m == "pca") {
      n_comps <- length(res$selected_keys)
      comp_choices <- paste0("PC", seq_len(n_comps))
      curr_sel <- if (input$contrib_comp %in% comp_choices) input$contrib_comp else "PC1"
      updateSelectInput(session, "contrib_comp", choices = comp_choices, selected = curr_sel)
    } else {
      n_disc <- length(res$lda_trace_pct)
      comp_choices <- paste0("LD", seq_len(n_disc))
      curr_sel <- if (input$contrib_comp %in% comp_choices) input$contrib_comp else "LD1"
      updateSelectInput(session, "contrib_comp", choices = comp_choices, selected = curr_sel)
    }
  })
  
  # Validation Alert banner
  output$validation_alert <- renderUI({
    if (length(input$selected_features) < 2) {
      div(class = "studio-alert-warning",
        strong("Input Warning: "),
        "At least 2 features are required to construct a 2D projection. Please select 2 or more features and click 'Run Analysis'."
      )
    } else {
      NULL
    }
  })
  
  # Dynamic Card Titles & Subtitles
  output$scatter_card_title <- renderText({
    if (input$method == "pca") "PCA Analysis" else "LDA Analysis"
  })
  output$scatter_card_subtitle <- renderText({
    if (input$method == "pca") "PC1 vs PC2 Scatter Plot" else "LD1 vs LD2 Scatter Plot"
  })
  
  output$variance_card_title <- renderText({
    if (input$method == "pca") "Explained Variance (PCA)" else "Discriminant Variance (LDA)"
  })
  
  output$contrib_card_title <- renderText({
    if (input$method == "pca") "Feature Contributions (PCA)" else "Feature Contributions (LDA)"
  })
  
  output$contrib_info_desc <- renderText({
    if (input$method == "pca") {
      "Relative absolute eigenvector loading |w_j| on the selected principal component."
    } else {
      "Relative absolute scaling coefficient |c_j| on the selected linear discriminant."
    }
  })
  
  # ----------------------------------------------------------------------------
  # PLOT 1: MAIN SCATTER PLOT (PCA or LDA)
  # ----------------------------------------------------------------------------
  build_scatter_plot <- function() {
    res <- active_model()
    req(res)
    m <- input$method
    
    # Determine color palette for classes
    classes <- levels(dataset_state$raw[[dataset_state$class_col]])
    if (length(classes) == 3 && all(classes %in% names(cultivar_colors_default))) {
      pal <- cultivar_colors_default
    } else {
      curated_colors <- c("#2563eb", "#f97316", "#9333ea", "#10b981", "#ef4444", "#06b6d4")
      pal <- setNames(curated_colors[seq_along(classes)], classes)
    }
    
    if (m == "pca") {
      df <- res$pca_scores_df
      pct1 <- res$pca_var_pct[1]
      pct2 <- res$pca_var_pct[2]
      
      ggplot(df, aes(x = PC1, y = PC2, color = Cultivar, fill = Cultivar)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "#cbd5e1", linewidth = 0.5) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "#cbd5e1", linewidth = 0.5) +
        stat_ellipse(geom = "polygon", alpha = 0.12, level = 0.95, show.legend = FALSE, color = NA) +
        geom_point(aes(shape = Split), size = 3.6, alpha = 0.88, stroke = 0.8) +
        scale_color_manual(values = pal) +
        scale_fill_manual(values = pal) +
        scale_shape_manual(values = split_shapes) +
        theme_minimal(base_size = 13) +
        theme(
          legend.position = "top",
          legend.box = "horizontal",
          legend.title = element_text(face = "bold", size = 11, color = "#0f2744"),
          legend.text = element_text(size = 10, color = "#334155"),
          panel.grid.minor = element_blank(),
          panel.grid.major = element_line(color = "#f1f5f9", linetype = "dashed"),
          axis.title = element_text(face = "bold", size = 11, color = "#0f2744"),
          axis.text = element_text(color = "#64748b", size = 10),
          plot.margin = margin(8, 12, 8, 12)
        ) +
        labs(
          x = sprintf("PC1 (%.2f%%)", pct1),
          y = sprintf("PC2 (%.2f%%)", pct2),
          color = "Class:",
          fill  = "Class:",
          shape = "Partition:"
        )
    } else {
      df <- res$lda_scores_df
      pct1 <- res$lda_trace_pct[1]
      pct2 <- if (length(res$lda_trace_pct) >= 2) res$lda_trace_pct[2] else 0
      
      ggplot(df, aes(x = LD1, y = LD2, color = Cultivar, fill = Cultivar)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "#cbd5e1", linewidth = 0.5) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "#cbd5e1", linewidth = 0.5) +
        stat_ellipse(geom = "polygon", alpha = 0.12, level = 0.95, show.legend = FALSE, color = NA) +
        geom_point(aes(shape = Split), size = 3.6, alpha = 0.88, stroke = 0.8) +
        scale_color_manual(values = pal) +
        scale_fill_manual(values = pal) +
        scale_shape_manual(values = split_shapes) +
        theme_minimal(base_size = 13) +
        theme(
          legend.position = "top",
          legend.box = "horizontal",
          legend.title = element_text(face = "bold", size = 11, color = "#0f2744"),
          legend.text = element_text(size = 10, color = "#334155"),
          panel.grid.minor = element_blank(),
          panel.grid.major = element_line(color = "#f1f5f9", linetype = "dashed"),
          axis.title = element_text(face = "bold", size = 11, color = "#0f2744"),
          axis.text = element_text(color = "#64748b", size = 10),
          plot.margin = margin(8, 12, 8, 12)
        ) +
        labs(
          x = sprintf("LD1 (%.2f%% Between-Class Trace)", pct1),
          y = sprintf("LD2 (%.2f%% Between-Class Trace)", pct2),
          color = "Class:",
          fill  = "Class:",
          shape = "Partition:"
        )
    }
  }
  
  output$plot_scatter <- renderPlot({
    build_scatter_plot()
  }, res = 100)
  
  # ----------------------------------------------------------------------------
  # PLOT 2: SCREE PLOT (Displayed only for PCA)
  # ----------------------------------------------------------------------------
  build_scree_plot <- function() {
    res <- active_model()
    req(res)
    
    var_ratios <- res$pca_var_pct / 100
    n_comps <- length(var_ratios)
    comp_names <- factor(paste0("PC", seq_len(n_comps)), levels = paste0("PC", seq_len(n_comps)))
    
    scree_df <- data.frame(
      Component = comp_names,
      Ratio = var_ratios,
      Index = seq_len(n_comps),
      Label = ifelse(seq_len(n_comps) <= min(5, n_comps), sprintf("%.3f", var_ratios), "")
    )
    
    y_top <- max(0.4, max(var_ratios) * 1.25)
    
    ggplot(scree_df, aes(x = Component, y = Ratio, group = 1)) +
      geom_area(fill = "#dbeafe", alpha = 0.5) +
      geom_line(color = "#2563eb", linewidth = 1.1) +
      geom_point(color = "#2563eb", fill = "#ffffff", shape = 21, size = 3.2, stroke = 1.5) +
      geom_text(aes(label = Label), vjust = -0.9, size = 3.4, fontface = "bold", color = "#1e3a8a") +
      scale_y_continuous(limits = c(0, y_top), breaks = seq(0, 1, 0.2)) +
      theme_minimal(base_size = 12) +
      theme(
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = "#e2e8f0", linetype = "dashed"),
        axis.title = element_text(face = "bold", color = "#0f2744", size = 11),
        axis.text = element_text(color = "#64748b", size = 10),
        plot.margin = margin(6, 12, 6, 12)
      ) +
      labs(x = "Principal Component", y = "Explained Variance Ratio")
  }
  
  output$plot_scree <- renderPlot({
    build_scree_plot()
  }, res = 100)
  
  # ----------------------------------------------------------------------------
  # PLOT 3: LDA DISCRIMINANT VARIANCE BAR CHART (Displayed for LDA below scatter)
  # ----------------------------------------------------------------------------
  build_lda_variance_plot <- function() {
    res <- active_model()
    req(res)
    
    trace_pcts <- res$lda_trace_pct
    
    # If the fitted model cannot provide valid proportions, display a helpful message
    if (is.null(trace_pcts) || length(trace_pcts) == 0 || any(is.na(trace_pcts))) {
      return(
        ggplot() +
          annotate("text", x = 0.5, y = 0.5, 
                   label = "Discriminant variance proportions could not be computed for this configuration.",
                   color = "#64748b", size = 4) +
          theme_void()
      )
    }
    
    n_comps <- length(trace_pcts)
    comp_names <- factor(paste0("LD", seq_len(n_comps)), levels = paste0("LD", seq_len(n_comps)))
    
    lda_df <- data.frame(
      Discriminant = comp_names,
      TracePct = trace_pcts,
      Label = sprintf("%.2f%%", trace_pcts)
    )
    
    y_max <- max(100, max(trace_pcts) * 1.25)
    
    ggplot(lda_df, aes(x = Discriminant, y = TracePct, fill = Discriminant)) +
      geom_col(width = 0.45, fill = "#9333ea", alpha = 0.88, color = "#7e22ce", linewidth = 0.8) +
      geom_text(aes(label = Label), vjust = -0.7, size = 4.2, fontface = "bold", color = "#581c87") +
      scale_y_continuous(limits = c(0, y_max), breaks = seq(0, 100, 20),
                         labels = function(x) paste0(x, "%")) +
      theme_minimal(base_size = 12) +
      theme(
        legend.position = "none",
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(color = "#e2e8f0", linetype = "dashed"),
        axis.title = element_text(face = "bold", color = "#0f2744", size = 11),
        axis.text = element_text(color = "#475569", size = 10.5),
        plot.margin = margin(6, 16, 6, 16)
      ) +
      labs(
        x = "Linear Discriminant Function",
        y = "Between-Class Variance Proportion (%)"
      )
  }
  
  output$plot_lda_variance <- renderPlot({
    build_lda_variance_plot()
  }, res = 100)
  
  # ----------------------------------------------------------------------------
  # CARD: CIRCULAR GAUGES (PCA Explained Variance / LDA Discriminant Variance)
  # ----------------------------------------------------------------------------
  output$ui_circular_gauges <- renderUI({
    res <- active_model()
    req(res)
    m <- input$method
    
    circ <- 251.33 # Circumference for r=40
    
    if (m == "pca") {
      v1 <- res$pca_var_pct[1]
      v2 <- if (length(res$pca_var_pct) >= 2) res$pca_var_pct[2] else 0
      v_tot <- v1 + v2
      
      off1 <- circ * (1 - v1 / 100)
      off2 <- circ * (1 - v2 / 100)
      off_tot <- circ * (1 - v_tot / 100)
      
      div(class = "gauges-container",
        # Gauge PC1
        div(class = "gauge-item",
          div(class = "gauge-header-label", "PC1"),
          HTML(sprintf('
            <svg class="gauge-svg" width="82" height="82" viewBox="0 0 100 100">
              <circle class="gauge-track" cx="50" cy="50" r="40" />
              <circle class="gauge-fill gauge-fill-pc1" cx="50" cy="50" r="40"
                      style="stroke-dasharray: 251.33; stroke-dashoffset: %.2f;" />
              <text class="gauge-text" x="50" y="50">%.2f%%</text>
            </svg>
          ', off1, v1))
        ),
        # Gauge PC2
        div(class = "gauge-item",
          div(class = "gauge-header-label", "PC2"),
          HTML(sprintf('
            <svg class="gauge-svg" width="82" height="82" viewBox="0 0 100 100">
              <circle class="gauge-track" cx="50" cy="50" r="40" />
              <circle class="gauge-fill gauge-fill-pc2" cx="50" cy="50" r="40"
                      style="stroke-dasharray: 251.33; stroke-dashoffset: %.2f;" />
              <text class="gauge-text" x="50" y="50">%.2f%%</text>
            </svg>
          ', off2, v2))
        ),
        # Gauge Total
        div(class = "gauge-item",
          div(class = "gauge-header-label", "Combined"),
          HTML(sprintf('
            <svg class="gauge-svg" width="82" height="82" viewBox="0 0 100 100">
              <circle class="gauge-track" cx="50" cy="50" r="40" />
              <circle class="gauge-fill gauge-fill-total" cx="50" cy="50" r="40"
                      style="stroke-dasharray: 251.33; stroke-dashoffset: %.2f;" />
              <text class="gauge-text" x="50" y="50">%.2f%%</text>
            </svg>
          ', off_tot, v_tot)),
          div(class = "gauge-sub-label", "(PC1 + PC2)")
        )
      )
    } else {
      v1 <- res$lda_trace_pct[1]
      v2 <- if (length(res$lda_trace_pct) >= 2) res$lda_trace_pct[2] else 0
      v_tot <- v1 + v2
      
      off1 <- circ * (1 - v1 / 100)
      off2 <- circ * (1 - v2 / 100)
      off_tot <- circ * (1 - v_tot / 100)
      
      div(
        div(class = "gauges-container",
          # Gauge LD1
          div(class = "gauge-item",
            div(class = "gauge-header-label", "LD1"),
            HTML(sprintf('
              <svg class="gauge-svg" width="82" height="82" viewBox="0 0 100 100">
                <circle class="gauge-track" cx="50" cy="50" r="40" />
                <circle class="gauge-fill gauge-fill-pc1" cx="50" cy="50" r="40"
                        style="stroke-dasharray: 251.33; stroke-dashoffset: %.2f;" />
                <text class="gauge-text" x="50" y="50">%.2f%%</text>
              </svg>
            ', off1, v1))
          ),
          # Gauge LD2
          div(class = "gauge-item",
            div(class = "gauge-header-label", "LD2"),
            HTML(sprintf('
              <svg class="gauge-svg" width="82" height="82" viewBox="0 0 100 100">
                <circle class="gauge-track" cx="50" cy="50" r="40" />
                <circle class="gauge-fill gauge-fill-pc2" cx="50" cy="50" r="40"
                        style="stroke-dasharray: 251.33; stroke-dashoffset: %.2f;" />
                <text class="gauge-text" x="50" y="50">%.2f%%</text>
              </svg>
            ', off2, v2))
          ),
          # Gauge Total
          div(class = "gauge-item",
            div(class = "gauge-header-label", "Combined"),
            HTML(sprintf('
              <svg class="gauge-svg" width="82" height="82" viewBox="0 0 100 100">
                <circle class="gauge-track" cx="50" cy="50" r="40" />
                <circle class="gauge-fill gauge-fill-total" cx="50" cy="50" r="40"
                        style="stroke-dasharray: 251.33; stroke-dashoffset: %.2f;" />
                <text class="gauge-text" x="50" y="50">%.1f%%</text>
              </svg>
            ', off_tot, v_tot)),
            div(class = "gauge-sub-label", "(LD1 + LD2)")
          )
        ),
        div(style = "font-size: 11px; color: #64748b; text-align: center; margin-top: 6px; padding: 0 4px;",
            "Between-class discriminant trace proportion (Fisher's criterion)")
      )
    }
  })
  
  # ----------------------------------------------------------------------------
  # CARD: FEATURE CONTRIBUTIONS (Single Dynamic Panel)
  # ----------------------------------------------------------------------------
  output$ui_feature_contributions <- renderUI({
    res <- active_model()
    req(res)
    m <- input$method
    comp <- input$contrib_comp
    
    if (m == "pca") {
      req(comp %in% colnames(res$pca_fit$rotation))
      loadings <- res$pca_fit$rotation[, comp]
      weights <- abs(loadings)
    } else {
      req(comp %in% colnames(res$lda_fit$scaling))
      scalings <- res$lda_fit$scaling[, comp]
      weights <- abs(scalings)
    }
    
    f_keys <- res$selected_keys
    f_names <- dataset_state$display_names[f_keys]
    
    # Sort features by contribution magnitude
    order_idx <- order(weights, decreasing = TRUE)
    sorted_weights <- weights[order_idx]
    sorted_names <- f_names[order_idx]
    
    max_w <- max(sorted_weights)
    if (max_w == 0) max_w <- 1
    
    rows <- lapply(seq_along(sorted_weights), function(i) {
      pct_width <- (sorted_weights[i] / max_w) * 100
      val_fmt <- sprintf("%.2f", sorted_weights[i])
      
      div(class = "feature-contrib-row",
        div(class = "contrib-name", title = sorted_names[i], sorted_names[i]),
        div(class = "contrib-bar-track",
          div(class = "contrib-bar-fill", style = sprintf("width: %.1f%%;", pct_width))
        ),
        div(class = "contrib-val", val_fmt)
      )
    })
    
    div(class = "feature-contrib-list", rows)
  })
  
  # ----------------------------------------------------------------------------
  # BOTTOM: TRANSFORMED DATA PREVIEW (FIRST 5 ROWS)
  # ----------------------------------------------------------------------------
  output$table_transformed_preview <- renderUI({
    res <- active_model()
    req(res)
    m <- input$method
    
    if (m == "pca") {
      df <- res$pca_scores_df
      sub_df <- head(df, 5)
      
      table_rows <- lapply(seq_len(nrow(sub_df)), function(i) {
        c_badge <- "badge-cultivar-1"
        if (as.character(sub_df$Cultivar[i]) == "Cultivar 2") c_badge <- "badge-cultivar-2"
        if (as.character(sub_df$Cultivar[i]) == "Cultivar 3") c_badge <- "badge-cultivar-3"
        s_badge <- if (grepl("Training", sub_df$Split[i])) "badge-train" else "badge-test"
        
        tags$tr(
          tags$td(i),
          tags$td(sprintf("%.4f", sub_df$PC1[i])),
          tags$td(sprintf("%.4f", sub_df$PC2[i])),
          tags$td(span(class = c_badge, as.character(sub_df$Cultivar[i]))),
          tags$td(span(class = s_badge, as.character(sub_df$Split[i])))
        )
      })
      
      tags$table(class = "clean-table",
        tags$thead(
          tags$tr(
            tags$th("#"),
            tags$th("PC1"),
            tags$th("PC2"),
            tags$th("Class"),
            tags$th("Partition")
          )
        ),
        tags$tbody(table_rows)
      )
    } else {
      df <- res$lda_scores_df
      sub_df <- head(df, 5)
      
      table_rows <- lapply(seq_len(nrow(sub_df)), function(i) {
        c_badge <- "badge-cultivar-1"
        if (as.character(sub_df$Cultivar[i]) == "Cultivar 2") c_badge <- "badge-cultivar-2"
        if (as.character(sub_df$Cultivar[i]) == "Cultivar 3") c_badge <- "badge-cultivar-3"
        s_badge <- if (grepl("Training", sub_df$Split[i])) "badge-train" else "badge-test"
        
        tags$tr(
          tags$td(i),
          tags$td(sprintf("%.4f", sub_df$LD1[i])),
          tags$td(sprintf("%.4f", sub_df$LD2[i])),
          tags$td(span(class = c_badge, as.character(sub_df$Cultivar[i]))),
          tags$td(span(class = s_badge, as.character(sub_df$Split[i])))
        )
      })
      
      tags$table(class = "clean-table",
        tags$thead(
          tags$tr(
            tags$th("#"),
            tags$th("LD1"),
            tags$th("LD2"),
            tags$th("Class"),
            tags$th("Partition")
          )
        ),
        tags$tbody(table_rows)
      )
    }
  })
  
  # ----------------------------------------------------------------------------
  # DOWNLOAD HANDLERS
  # ----------------------------------------------------------------------------
  
  # 1. Download Current Scatter Plot PNG
  output$download_scatter_png <- downloadHandler(
    filename = function() {
      paste0(input$method, "_scatter_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- build_scatter_plot()
      ggsave(file, plot = p, width = 8.5, height = 5.5, dpi = 300)
    }
  )
  
  # 2. Download Scree Plot PNG
  output$download_scree_png <- downloadHandler(
    filename = function() {
      paste0("pca_scree_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- build_scree_plot()
      ggsave(file, plot = p, width = 8.5, height = 4.8, dpi = 300)
    }
  )
  
  # 3. Download LDA Variance PNG
  output$download_lda_variance_png <- downloadHandler(
    filename = function() {
      paste0("lda_discriminant_variance_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- build_lda_variance_plot()
      ggsave(file, plot = p, width = 8.5, height = 4.8, dpi = 300)
    }
  )
  
  # 3. Download Transformed CSV (All transformed observations)
  output$download_transformed_csv <- downloadHandler(
    filename = function() {
      paste0(input$method, "_transformed_data_all_", Sys.Date(), ".csv")
    },
    content = function(file) {
      res <- active_model()
      df <- if (input$method == "pca") res$pca_scores_df else res$lda_scores_df
      write.csv(df, file, row.names = FALSE)
    }
  )
}

# --- 8. Launch Application ---
shinyApp(ui = ui, server = server)
