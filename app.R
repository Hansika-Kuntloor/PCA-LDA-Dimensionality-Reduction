# ==============================================================================
# PCA & LDA Feature Extraction Studio
# Methodology: Strict 70:30 Stratified Train-Test Split (Seed 42, Zero Leakage)
# UI: Exactly Three Main Navigation Tabs (Analysis, Insights, Overview)
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
  "Cultivar 1" = "#315DEB",  # Cultivar 1: Blue
  "Cultivar 2" = "#E64980",  # Cultivar 2: Rose pink
  "Cultivar 3" = "#8055C7"   # Cultivar 3: Muted purple
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
  lda_test_correct <- sum(diag(lda_test_conf))
  lda_test_total <- sum(lda_test_conf)
  
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
    lda_test_acc = lda_test_acc,
    lda_test_correct = lda_test_correct,
    lda_test_total = lda_test_total
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
  title = "PCA & LDA Feature Extraction Studio",
  
  tags$head(
    tags$style(HTML("
      /* Clean Modern Academic Typography & Reset */
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
      
      *, *::before, *::after {
        box-sizing: border-box;
      }
      
      body {
        background-color: #F7F8FC;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #0F172A;
        margin: 0;
        padding: 0;
        line-height: 1.45;
        overflow-x: hidden;
      }
      
      /* STICKY TOP NAVIGATION BAR */
      .studio-navbar {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: #FFFFFF;
        border-bottom: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        width: 100%;
      }
      .navbar-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 32px;
        height: 76px;
        max-width: 1720px;
        margin: 0 auto;
      }
      .navbar-left {
        display: flex;
        align-items: center;
        gap: 14px;
        flex-shrink: 0;
      }
      .brand-icon-svg {
        width: 36px;
        height: 36px;
        flex-shrink: 0;
      }
      .brand-title {
        font-size: 21px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.3px;
        white-space: nowrap;
      }
      .navbar-center {
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .studio-nav-pills {
        display: flex;
        gap: 8px;
        margin: 0;
        padding: 0;
        list-style: none;
        border-bottom: none;
      }
      .studio-nav-pills > li > a {
        display: inline-flex;
        align-items: center;
        padding: 10px 24px;
        font-size: 16.5px;
        font-weight: 600;
        color: #475569;
        text-decoration: none !important;
        border-radius: 8px;
        transition: all 0.15s ease;
        position: relative;
        cursor: pointer;
        border-bottom: 2px solid transparent;
      }
      .studio-nav-pills > li > a:hover {
        background: #F1F5F9;
        color: #0F172A;
      }
      .studio-nav-pills > li.active > a,
      .studio-nav-pills > li.active > a:focus,
      .studio-nav-pills > li.active > a:hover {
        color: #315DEB !important;
        background: #EFF4FF !important;
        border-bottom: 2px solid #315DEB !important;
      }
      .navbar-right {
        display: flex;
        align-items: center;
        flex-shrink: 0;
      }
      .header-dataset-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #EFF4FF;
        border: 1px solid #C7D7FE;
        color: #315DEB;
        padding: 7px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        white-space: nowrap;
      }
      
      /* Main Content Container */
      .studio-container {
        position: relative;
        z-index: 5;
        padding: 20px 32px 36px 32px;
        max-width: 1720px;
        margin: 0 auto;
      }
      
      /* Cards */
      .studio-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
      }
      
      /* Card Header */
      .card-header-flex {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        min-height: 40px;
      }
      .card-title-group {
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1;
        min-width: 0;
      }
      .card-icon {
        color: #315DEB;
        font-size: 18px;
      }
      .card-main-title {
        font-size: 16.5px;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        line-height: 1.3;
      }
      .card-subtitle {
        font-size: 12.5px;
        color: #64748B;
        margin: 2px 0 0 0;
        line-height: 1.35;
      }
      .card-header-select {
        margin-left: 12px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
      }
      .card-header-select .form-group,
      .card-header-select .shiny-input-container,
      .card-header-select .selectize-control {
        margin-bottom: 0 !important;
      }
      .card-header-select select.form-control,
      .card-header-select .selectize-input {
        height: 36px !important;
        min-height: 36px !important;
        padding: 6px 12px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
        line-height: 1.4 !important;
        display: flex !important;
        align-items: center !important;
      }
      
      /* Segmented Method Selector */
      .method-segmented .shiny-options-group {
        display: flex;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 3px;
        margin: 0 0 10px 0;
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
        gap: 6px;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 12.5px;
        font-weight: 600;
        color: #475569;
        cursor: pointer;
        transition: all 0.2s ease;
      }
      .method-segmented .radio-inline input[type='radio']:checked + span {
        background: #315DEB;
        color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(49, 93, 235, 0.25);
      }
      
      /* Compact Dataset Info & Upload Box */
      .dataset-compact-box {
        background: #FAFAFC;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 10px;
      }
      .dataset-compact-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
      }
      .dataset-icon-circle {
        width: 28px;
        height: 28px;
        border-radius: 6px;
        background: #EFF4FF;
        color: #315DEB;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        flex-shrink: 0;
      }
      .dataset-compact-title {
        font-size: 12.5px;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
      }
      .dataset-compact-meta {
        font-size: 11px;
        color: #64748B;
        margin: 0;
      }
      .dataset-upload-row {
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid #E5E7EB;
      }
      .dataset-upload-row .shiny-input-container {
        margin-bottom: 0;
      }
      .dataset-upload-row .btn-file {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        color: #315DEB;
        font-weight: 600;
        font-size: 11px;
        padding: 4px 8px;
        border-radius: 5px;
        transition: all 0.15s ease;
      }
      .dataset-upload-row .btn-file:hover {
        background: #EFF4FF;
        border-color: #C7D7FE;
      }
      .dataset-upload-row .form-control {
        font-size: 10.5px;
        height: 26px;
      }
      .reset-link {
        font-size: 10.5px;
        font-weight: 600;
        color: #DC2626;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        margin-top: 4px;
      }
      .reset-link:hover {
        text-decoration: underline;
      }
      
      /* Feature Selection Section (Compact Scrollable Box) */
      .feature-select-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
      }
      .feature-select-title {
        font-size: 12.5px;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .feature-quick-links {
        font-size: 11.5px;
      }
      .feature-quick-links a {
        color: #315DEB;
        font-weight: 600;
        text-decoration: none;
        cursor: pointer;
        margin-left: 4px;
      }
      .feature-quick-links a:hover {
        text-decoration: underline;
      }
      .features-scroll-box {
        max-height: 145px;
        overflow-y: auto;
        padding: 6px 10px;
        background: #FAFAFC;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        margin-bottom: 10px;
      }
      .features-scroll-box::-webkit-scrollbar {
        width: 5px;
      }
      .features-scroll-box::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
      }
      .features-scroll-box .shiny-options-group {
        display: flex;
        flex-direction: column;
        gap: 1px;
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
        padding: 2.5px 4px !important;
        padding-left: 0 !important;
        margin: 0 !important;
        min-height: auto !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #1E293B !important;
        cursor: pointer !important;
        border-radius: 4px !important;
        transition: background 0.15s ease !important;
      }
      .features-scroll-box .checkbox label:hover {
        background: #F1F5F9 !important;
      }
      .features-scroll-box .checkbox input[type='checkbox'] {
        position: static !important;
        float: none !important;
        margin: 0 8px 0 0 !important;
        width: 14px !important;
        height: 14px !important;
        min-width: 14px !important;
        max-width: 14px !important;
        flex-shrink: 0 !important;
        accent-color: #315DEB !important;
        cursor: pointer !important;
      }
      .features-scroll-box .checkbox label span {
        display: inline-block !important;
        color: #334155 !important;
        font-size: 12px !important;
        line-height: 1.25 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
      }
      
      /* Standardization Setting Box */
      .standardize-box {
        background: #FAFAFC;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 12px;
      }
      .standardize-box .form-group {
        margin: 0 !important;
        margin-bottom: 0 !important;
      }
      .standardize-box .checkbox {
        position: relative !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        min-height: auto !important;
      }
      .standardize-box .checkbox label {
        display: flex !important;
        align-items: center !important;
        padding: 0 !important;
        padding-left: 0 !important;
        margin: 0 !important;
        min-height: auto !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
        color: #0F172A !important;
        cursor: pointer !important;
        line-height: 1.35 !important;
        user-select: none;
      }
      .standardize-box .checkbox input[type='checkbox'] {
        position: static !important;
        float: none !important;
        margin: 0 8px 0 0 !important;
        width: 15px !important;
        height: 15px !important;
        min-width: 15px !important;
        max-width: 15px !important;
        flex-shrink: 0 !important;
        accent-color: #315DEB !important;
        cursor: pointer !important;
        vertical-align: middle;
      }
      .standardize-help {
        font-size: 11px;
        color: #64748B;
        margin: 4px 0 0 23px;
        line-height: 1.35;
      }
      
      /* Split Display Box */
      .split-block {
        background: #FAFAFC;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 12px;
      }
      .split-block-header {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
      }
      .split-title {
        font-size: 12.5px;
        font-weight: 700;
        color: #0F172A;
      }
      .split-pct-text {
        font-size: 12px;
        font-weight: 600;
        color: #315DEB;
        margin-bottom: 5px;
      }
      .split-bar-container {
        height: 6px;
        display: flex;
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 6px;
        background: #E2E8F0;
      }
      .split-bar-train {
        width: 70.2%;
        background: #315DEB;
      }
      .split-bar-test {
        width: 29.8%;
        background: #94A3B8;
      }
      .split-detail-text {
        font-size: 11px;
        color: #64748B;
        margin-bottom: 6px;
      }
      .split-toggle-box {
        margin-top: 6px;
        padding-top: 7px;
        border-top: 1px dashed #E2E8F0;
      }
      .split-toggle-box .form-group {
        margin: 0 !important;
        margin-bottom: 0 !important;
      }
      .split-toggle-box .checkbox {
        position: relative !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        min-height: auto !important;
      }
      .split-toggle-box .checkbox label {
        display: flex !important;
        align-items: center !important;
        padding: 0 !important;
        padding-left: 0 !important;
        margin: 0 !important;
        min-height: auto !important;
        font-size: 11.5px !important;
        font-weight: 600 !important;
        color: #334155 !important;
        cursor: pointer !important;
        line-height: 1.35 !important;
        user-select: none;
      }
      .split-toggle-box .checkbox input[type='checkbox'] {
        position: static !important;
        float: none !important;
        margin: 0 8px 0 0 !important;
        width: 14px !important;
        height: 14px !important;
        min-width: 14px !important;
        max-width: 14px !important;
        flex-shrink: 0 !important;
        accent-color: #315DEB !important;
        cursor: pointer !important;
        vertical-align: middle;
      }
      
      /* Primary Run Button */
      .btn-run-studio {
        width: 100%;
        background: #315DEB;
        color: #FFFFFF;
        font-size: 13px;
        font-weight: 600;
        padding: 9px;
        border-radius: 8px;
        border: none;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(49, 93, 235, 0.2);
      }
      .btn-run-studio:hover {
        background: #254EC9;
        color: #FFFFFF;
        box-shadow: 0 3px 8px rgba(49, 93, 235, 0.3);
      }
      
      /* RESTORED CLEAN FEATURE CONTRIBUTION CHART STYLING */
      .feature-contrib-list {
        padding: 4px 6px 4px 0;
        margin-top: 6px;
      }
      .feature-contrib-row {
        display: flex;
        align-items: center;
        height: 26px;
        min-height: 26px;
        gap: 12px;
        margin-bottom: 6px;
        padding-right: 6px;
      }
      .contrib-name {
        width: 155px;
        min-width: 155px;
        max-width: 155px;
        flex-shrink: 0;
        font-size: 12.5px;
        font-weight: 600;
        color: #1E293B;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 26px;
      }
      .contrib-bar-track {
        flex: 1 1 auto;
        min-width: 60px;
        height: 10px;
        background: #F1F5F9;
        border-radius: 5px;
        overflow: hidden;
      }
      .contrib-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.25s ease;
      }
      .contrib-val {
        width: 64px;
        min-width: 64px;
        max-width: 64px;
        flex-shrink: 0;
        font-size: 12.5px;
        font-weight: 700;
        text-align: right;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        line-height: 26px;
        font-variant-numeric: tabular-nums;
        margin-right: 4px;
      }
      
      /* Table Scroll Wrap & Clean Styling */
      .table-scroll-wrap {
        max-height: 420px;
        overflow-y: auto;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background: #FFFFFF;
      }
      .table-scroll-wrap::-webkit-scrollbar {
        width: 5px;
      }
      .table-scroll-wrap::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
      }
      .table-scroll-wrap .clean-table th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #F8FAFC;
        box-shadow: 0 1px 0 #E2E8F0;
      }
      
      /* Balanced 3-Column Layout for Desktop & Responsive Stacking */
      @media (min-width: 992px) {
        .insights-three-col {
          display: grid !important;
          grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.14fr) minmax(0, 1.05fr) !important;
          gap: 18px !important;
          margin-left: 0 !important;
          margin-right: 0 !important;
          align-items: start !important;
        }
        .insights-three-col::before,
        .insights-three-col::after {
          display: none !important;
        }
        .insights-three-col > [class*='col-'] {
          width: 100% !important;
          max-width: 100% !important;
          padding-left: 0 !important;
          padding-right: 0 !important;
          display: flex !important;
          flex-direction: column !important;
        }
        .insights-three-col .studio-card {
          width: 100% !important;
          margin-bottom: 0 !important;
          min-height: auto !important;
          height: auto !important;
        }
      }
      @media (max-width: 991px) {
        .insights-three-col {
          display: flex !important;
          flex-direction: column !important;
          gap: 16px !important;
          margin-left: 0 !important;
          margin-right: 0 !important;
        }
        .insights-three-col::before,
        .insights-three-col::after {
          display: none !important;
        }
        .insights-three-col > [class*='col-'] {
          width: 100% !important;
          max-width: 100% !important;
          padding-left: 0 !important;
          padding-right: 0 !important;
          margin-bottom: 0 !important;
        }
        .insights-three-col .studio-card {
          min-height: auto !important;
          margin-bottom: 0 !important;
        }
      }
      
      /* Metric Chips Grid (2-on-top, 1-spanning-bottom) */
      .metric-chips-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 12px;
      }
      .metric-chip {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 12px;
        display: flex;
        flex-direction: column;
        gap: 3px;
      }
      .metric-chip-full {
        grid-column: span 2;
        flex-direction: row !important;
        justify-content: space-between !important;
        align-items: center !important;
      }
      .metric-chip-primary {
        background: #EFF4FF;
        border-color: #C7D7FE;
      }
      .metric-chip-label {
        font-size: 11.5px;
        font-weight: 600;
        color: #64748B;
      }
      .metric-chip-primary .metric-chip-label {
        color: #315DEB;
      }
      .metric-chip-val {
        font-size: 16.5px;
        font-weight: 700;
        color: #0F172A;
      }
      .metric-chip-primary .metric-chip-val {
        color: #315DEB;
      }
      
      /* Context Info Box */
      .insights-context-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 14px;
        margin-top: 10px;
        font-size: 12.5px;
        color: #334155;
        line-height: 1.45;
      }
      .insights-context-box strong {
        color: #0F172A;
      }
      
      /* Overview Cards */
      .overview-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }
      .overview-card-label {
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        margin-bottom: 6px;
      }
      .overview-card-val {
        font-size: 19px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
      }
      .overview-card-desc {
        font-size: 12px;
        color: #64748B;
        line-height: 1.45;
        margin-bottom: 8px;
      }
      .overview-card-badge {
        display: inline-block;
        padding: 3.5px 9px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #EFF4FF;
        color: #315DEB;
        border: 1px solid #C7D7FE;
        align-self: flex-start;
      }
      .overview-card-badge-stale {
        background: #FFFBEB;
        color: #B45309;
        border-color: #FDE68A;
      }
      
      /* Alerts & Notices */
      .studio-alert-notice {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid #315DEB;
        padding: 10px 14px;
        border-radius: 6px;
        color: #1E40AF;
        font-size: 12px;
        margin-bottom: 14px;
      }
      .studio-alert-warning {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #F59E0B;
        padding: 8px 12px;
        border-radius: 6px;
        color: #92400E;
        font-size: 12px;
        margin-bottom: 12px;
      }
      
      /* Clean Table */
      .clean-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12.5px;
      }
      .clean-table th {
        background: #F8FAFC;
        color: #475569;
        font-weight: 700;
        padding: 8px 12px;
        border-bottom: 1px solid #E5E7EB;
        text-align: left;
      }
      .clean-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #F1F5F9;
        color: #334155;
      }
      .clean-table tr:hover {
        background: #F8FAFC;
      }
      
      /* Badges */
      .badge-cultivar-1 {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 10.5px;
        font-weight: 600;
        background: #EFF4FF;
        color: #315DEB;
        border: 1px solid #C7D7FE;
      }
      .badge-cultivar-2 {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 10.5px;
        font-weight: 600;
        background: #FDF2F8;
        color: #E64980;
        border: 1px solid #FBCFE8;
      }
      .badge-cultivar-3 {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 10.5px;
        font-weight: 600;
        background: #F5F3FF;
        color: #8055C7;
        border: 1px solid #DDD6FE;
      }
      .badge-train {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 10.5px;
        font-weight: 600;
        background: #F0FDF4;
        color: #16A34A;
        border: 1px solid #BBF7D0;
      }
      .badge-test {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 10.5px;
        font-weight: 600;
        background: #F8FAFC;
        color: #64748B;
        border: 1px solid #E2E8F0;
      }
      
      /* Action Buttons */
      .btn-export {
        background: #FFFFFF;
        color: #315DEB;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 5px 11px;
        font-size: 11.5px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        cursor: pointer;
        transition: all 0.15s ease;
        text-decoration: none !important;
      }
      .btn-export:hover {
        background: #F8FAFC;
        border-color: #315DEB;
        color: #254EC9;
      }
      
      /* Info Tag */
      .info-tag {
        font-size: 11px;
        color: #475569;
        background: #F1F5F9;
        border-radius: 5px;
        padding: 4px 8px;
        margin-bottom: 8px;
        display: inline-block;
        line-height: 1.35;
      }
    ")),
    
    tags$script(HTML("
      // Window resize trigger on tab change for responsive ggplot sizing
      $(document).on('shown.bs.tab', '#main_nav_tabs a[data-toggle=\"tab\"]', function (e) {
        var targetVal = $(e.target).data('value');
        Shiny.setInputValue('active_tab', targetVal);
        $(window).trigger('resize');
      });
    "))
  ),
  
  # ----------------------------------------------------------------------------
  # 1. PERSISTENT TOP NAVIGATION BAR (EXACTLY THREE TABS)
  # ----------------------------------------------------------------------------
  tags$header(class = "studio-navbar",
    div(class = "navbar-container",
      # Left: Title
      div(class = "navbar-left",
        HTML('<svg class="brand-icon-svg" viewBox="0 0 40 40" fill="none"><path d="M4 36C8 20 22 6 36 4C30 16 24 24 14 30C8 34 5 36 4 36Z" fill="#315DEB"/><path d="M4 36C12 26 20 18 36 4" stroke="#ffffff" stroke-width="1.8" stroke-linecap="round"/><path d="M14 24C16 18 22 14 28 12" stroke="#ffffff" stroke-width="1.2" stroke-linecap="round" opacity="0.8"/></svg>'),
        span(class = "brand-title", "PCA & LDA Feature Extraction Studio")
      ),
      
      # Center: Exactly Three Navigation Tabs
      div(class = "navbar-center",
        tags$ul(class = "nav nav-pills studio-nav-pills", id = "main_nav_tabs",
          tags$li(class = "active", tags$a(href = "#tab-analysis", `data-toggle` = "tab", `data-value` = "analysis", "Analysis")),
          tags$li(tags$a(href = "#tab-insights", `data-toggle` = "tab", `data-value` = "insights", "Insights")),
          tags$li(tags$a(href = "#tab-overview", `data-toggle` = "tab", `data-value` = "overview", "Overview"))
        )
      ),
      
      # Right: Wine Dataset Badge
      div(class = "navbar-right",
        div(class = "header-dataset-badge",
          HTML("🍇"), textOutput("header_badge_text", inline = TRUE)
        )
      )
    )
  ),
  
  # ----------------------------------------------------------------------------
  # 2. MAIN CONTAINER WITH THREE TAB PANES
  # ----------------------------------------------------------------------------
  div(class = "studio-container",
    
    div(class = "tab-content",
      
      # ========================================================================
      # TAB 1: ANALYSIS
      # ========================================================================
      div(class = "tab-pane active", id = "tab-analysis",
        # UI Warning if fewer than 2 features selected
        uiOutput("validation_alert"),
        
        fluidRow(
          # Left: Compact Control Panel (width = 3)
          column(width = 3,
            div(class = "studio-card",
              
              # 1. Method Selector
              div(class = "card-header-flex", style = "margin-bottom: 6px;",
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
              
              # 2. Dataset Information & CSV Upload
              div(class = "card-header-flex", style = "margin-bottom: 6px;",
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
              
              # 4. Standardization Option
              div(class = "standardize-box",
                checkboxInput("standardize", label = "Standardize Features (Z-Score)", value = TRUE),
                p(class = "standardize-help", 
                  "Scales features to zero mean and unit variance using training statistics.")
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
                div(class = "split-detail-text", textOutput("split_detail_text", inline = TRUE)),
                div(class = "split-toggle-box",
                  checkboxInput("distinguish_split", "Distinguish training & testing markers", value = FALSE)
                )
              ),
              
              # 6. Run Analysis Button
              actionButton("btn_run", label = tagList(HTML("▶"), "Run Analysis"),
                           class = "btn-run-studio")
            )
          ),
          
          # Right: Main Scatter Plot (width = 9)
          column(width = 9,
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
              plotOutput("plot_scatter", height = "510px")
            )
          )
        )
      ),
      
      # ========================================================================
      # TAB 2: INSIGHTS (DYNAMICALLY UPDATES BY METHOD; NO DUPLICATE SCATTER PLOT)
      # ========================================================================
      div(class = "tab-pane", id = "tab-insights",
        
        # --- A. When PCA is selected (Balanced 3-Column Layout) ---
        conditionalPanel(
          condition = "input.method == 'pca'",
          fluidRow(class = "insights-three-col",
            # Column 1: Explained Variance Summary & Table (width = 4)
            column(width = 4,
              div(class = "studio-card",
                div(class = "card-header-flex",
                  div(class = "card-title-group",
                    span(class = "card-icon", HTML("❖")),
                    div(
                      h4(class = "card-main-title", "Explained Variance Summary (PCA)"),
                      p(class = "card-subtitle", "Variance explained per principal component (training fit)")
                    )
                  )
                ),
                uiOutput("ui_pca_variance_stats"),
                uiOutput("table_pca_variance")
              )
            ),
            
            # Column 2: PCA Feature Contributions (width = 4)
            column(width = 4,
              div(class = "studio-card",
                div(class = "card-header-flex",
                  div(class = "card-title-group",
                    span(class = "card-icon", HTML("❄")),
                    div(
                      h4(class = "card-main-title", "PCA Feature Contributions"),
                      p(class = "card-subtitle", "Eigenvector loadings for selected component")
                    )
                  ),
                  div(class = "card-header-select", style = "width: 105px;",
                    selectInput("pca_contrib_comp", label = NULL, choices = paste0("PC", 1:13), selected = "PC1")
                  )
                ),
                div(class = "info-tag", "Blue = positive · Pink = negative · Sorted by magnitude"),
                p(style = "font-size: 11.5px; color: #64748B; margin: 0 0 8px 0; line-height: 1.35;",
                  "Loadings represent feature coefficients in the principal component eigenvector. Sorted by absolute magnitude."),
                uiOutput("ui_pca_feature_contributions")
              )
            ),
            
            # Column 3: PCA Scree Plot (width = 4)
            column(width = 4,
              div(class = "studio-card",
                div(class = "card-header-flex",
                  div(class = "card-title-group",
                    span(class = "card-icon", HTML("☊")),
                    div(
                      h4(class = "card-main-title", "PCA Scree Plot"),
                      p(class = "card-subtitle", "Explained variance percentage per principal component")
                    )
                  ),
                  downloadButton("download_scree_png", "Save PNG", class = "btn-export")
                ),
                plotOutput("plot_scree", height = "380px")
              )
            )
          )
        ),
        
        # --- B. When LDA is selected (Balanced 3-Column Layout) ---
        conditionalPanel(
          condition = "input.method == 'lda'",
          fluidRow(class = "insights-three-col",
            # Column 1: Discriminant Variance Summary & Trace Table (width = 4)
            column(width = 4,
              div(class = "studio-card",
                div(class = "card-header-flex",
                  div(class = "card-title-group",
                    span(class = "card-icon", HTML("❖")),
                    div(
                      h4(class = "card-main-title", "Discriminant Variance Summary (LDA)"),
                      p(class = "card-subtitle", "Between-class separation captured by linear discriminants")
                    )
                  )
                ),
                uiOutput("ui_lda_variance_stats"),
                uiOutput("table_lda_variance")
              )
            ),
            
            # Column 2: LDA Feature Contributions (width = 4)
            column(width = 4,
              div(class = "studio-card",
                div(class = "card-header-flex",
                  div(class = "card-title-group",
                    span(class = "card-icon", HTML("❄")),
                    div(
                      h4(class = "card-main-title", "LDA Feature Contributions"),
                      p(class = "card-subtitle", "Linear discriminant scaling coefficients")
                    )
                  ),
                  div(class = "card-header-select", style = "width: 105px;",
                    selectInput("lda_contrib_comp", label = NULL, choices = c("LD1" = "LD1", "LD2" = "LD2"), selected = "LD1")
                  )
                ),
                div(class = "info-tag", "Blue = positive · Pink = negative · Sorted by magnitude"),
                p(style = "font-size: 11.5px; color: #64748B; margin: 0 0 8px 0; line-height: 1.35;",
                  "Coefficients indicate each feature's contribution to cultivar separation on this axis. Sorted by absolute magnitude."),
                uiOutput("ui_lda_feature_contributions")
              )
            ),
            
            # Column 3: LDA Discriminant Variance Chart & Fisher Context (width = 4)
            column(width = 4,
              div(class = "studio-card",
                div(class = "card-header-flex",
                  div(class = "card-title-group",
                    span(class = "card-icon", HTML("☊")),
                    div(
                      h4(class = "card-main-title", "LDA Discriminant Variance"),
                      p(class = "card-subtitle", "Between-class variance proportion per discriminant function")
                    )
                  ),
                  downloadButton("download_lda_variance_png", "Save PNG", class = "btn-export")
                ),
                plotOutput("plot_lda_variance", height = "280px"),
                div(class = "insights-context-box", style = "margin-top: 12px;",
                  strong("Fisher's Criterion: "),
                  "LDA maximizes between-to-within class variance (SB / SW). With 3 distinct cultivars, LDA generates min(p, 3 - 1) = 2 discriminant functions (LD1 and LD2), capturing 100% of between-class separation."
                )
              )
            )
          )
        )
      ),
      
      # ========================================================================
      # TAB 3: OVERVIEW (DYNAMIC, METHOD-AWARE, LATEST COMPLETED ANALYSIS)
      # ========================================================================
      div(class = "tab-pane", id = "tab-overview",
        
        # Sync notice if inputs in Analysis tab have changed since last completed calculation
        uiOutput("overview_sync_notice"),
        
        # Top Row: Dynamic Summary Cards
        fluidRow(
          # Card 1: Active Method
          column(width = 3,
            div(class = "overview-card",
              div(
                div(class = "overview-card-label", "Active Method"),
                div(class = "overview-card-val", textOutput("overview_method_val")),
                div(class = "overview-card-desc", textOutput("overview_method_desc"))
              ),
              uiOutput("overview_method_badge_ui")
            )
          ),
          
          # Card 2: Dataset Information
          column(width = 3,
            div(class = "overview-card",
              div(
                div(class = "overview-card-label", "Dataset Information"),
                div(class = "overview-card-val", textOutput("overview_dataset_val")),
                div(class = "overview-card-desc", textOutput("overview_dataset_desc"))
              ),
              div(class = "overview-card-badge", textOutput("overview_dataset_badge"))
            )
          ),
          
          # Card 3: Dynamic Method-Specific Metric (PCA Variance vs LDA Trace)
          column(width = 3,
            div(class = "overview-card",
              div(
                div(class = "overview-card-label", textOutput("overview_metric_card_label")),
                div(class = "overview-card-val", textOutput("overview_metric_card_val")),
                div(class = "overview-card-desc", textOutput("overview_metric_card_desc"))
              ),
              uiOutput("overview_metric_card_badge_ui")
            )
          ),
          
          # Card 4: Partition
          column(width = 3,
            div(class = "overview-card",
              div(
                div(class = "overview-card-label", "Train / Test Partition"),
                div(class = "overview-card-val", "70:30 Split"),
                div(class = "overview-card-desc", textOutput("overview_split_desc"))
              ),
              div(class = "overview-card-badge", "Fixed Seed 42 • Zero Leakage")
            )
          )
        ),
        
        tags$div(style = "height: 16px;"),
        
        # Bottom Row: Transformed Data Preview Table & CSV Download (Method-Aware)
        fluidRow(
          column(width = 12,
            div(class = "studio-card",
              div(class = "card-header-flex",
                div(class = "card-title-group",
                  span(class = "card-icon", HTML("▦")),
                  div(
                    h4(class = "card-main-title", textOutput("overview_table_title", inline = TRUE)),
                    p(class = "card-subtitle", textOutput("overview_table_subtitle", inline = TRUE))
                  )
                ),
                downloadButton("download_transformed_csv", "Download CSV", class = "btn-export")
              ),
              uiOutput("table_transformed_preview")
            )
          )
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
  
  # State tracking the exact settings & model of the latest successfully completed analysis run
  latest_analysis <- reactiveValues(
    method = "pca",
    keys = feature_keys,
    standardize = TRUE,
    dataset_name = "Wine Dataset",
    num_samples = nrow(wine),
    res = baseline_res,
    timestamp = Sys.time()
  )
  
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
    
    # Recompute models safely with tryCatch
    res_up <- tryCatch({
      compute_feature_models(
        selected_keys = up_keys,
        standardize = input$standardize,
        tr_data = up_train,
        te_data = up_test,
        class_col = class_cand,
        cols_dict = up_cols
      )
    }, error = function(e) {
      showNotification(paste("Error running model:", conditionMessage(e)), type = "error")
      NULL
    })
    
    if (!is.null(res_up)) {
      active_model(res_up)
      latest_analysis$method <- input$method
      latest_analysis$keys <- up_keys
      latest_analysis$standardize <- input$standardize
      latest_analysis$dataset_name <- file$name
      latest_analysis$num_samples <- nrow(df_uploaded)
      latest_analysis$res <- res_up
      latest_analysis$timestamp <- Sys.time()
      showNotification(sprintf("Uploaded '%s' loaded successfully (%d samples, %d features).",
                               file$name, nrow(df_uploaded), length(up_keys)),
                       type = "message")
    }
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
    
    res_wine <- tryCatch({
      compute_feature_models(
        selected_keys = feature_keys,
        standardize = input$standardize,
        tr_data = wine_train,
        te_data = wine_test,
        class_col = "Cultivar",
        cols_dict = feature_cols
      )
    }, error = function(e) {
      NULL
    })
    
    if (!is.null(res_wine)) {
      active_model(res_wine)
      latest_analysis$method <- input$method
      latest_analysis$keys <- feature_keys
      latest_analysis$standardize <- input$standardize
      latest_analysis$dataset_name <- "Wine Dataset"
      latest_analysis$num_samples <- nrow(wine)
      latest_analysis$res <- res_wine
      latest_analysis$timestamp <- Sys.time()
      showNotification("Reverted to default Wine Recognition dataset.", type = "message")
    }
  })
  
  # Quick Link Observers: Select All / Deselect All
  observeEvent(input$select_all, {
    updateCheckboxGroupInput(session, "selected_features", selected = dataset_state$keys)
  })
  observeEvent(input$deselect_all, {
    updateCheckboxGroupInput(session, "selected_features", selected = character(0))
  })
  
  # Run Analysis Trigger (Strictly Updates latest_analysis only on success)
  observeEvent(input$btn_run, {
    feats <- input$selected_features
    if (is.null(feats) || length(feats) < 2) {
      showNotification("Please select at least 2 features to perform 2D dimensionality reduction.",
                       type = "warning", duration = 4)
      return()
    }
    
    res <- tryCatch({
      compute_feature_models(
        selected_keys = feats,
        standardize = input$standardize,
        tr_data = dataset_state$train,
        te_data = dataset_state$test,
        class_col = dataset_state$class_col,
        cols_dict = dataset_state$cols
      )
    }, error = function(e) {
      showNotification(paste("Analysis calculation error:", conditionMessage(e)), type = "error")
      NULL
    })
    
    if (!is.null(res)) {
      active_model(res)
      latest_analysis$method <- input$method
      latest_analysis$keys <- feats
      latest_analysis$standardize <- input$standardize
      latest_analysis$dataset_name <- dataset_state$name
      latest_analysis$num_samples <- nrow(dataset_state$raw)
      latest_analysis$res <- res
      latest_analysis$timestamp <- Sys.time()
      showNotification(sprintf("%s analysis completed successfully.", toupper(input$method)), type = "message")
    }
  })
  
  # Dynamic updates to PCA Component Dropdown choices
  observe({
    res <- active_model()
    req(res)
    n_comps <- length(res$selected_keys)
    comp_choices <- paste0("PC", seq_len(n_comps))
    curr_sel <- if (!is.null(input$pca_contrib_comp) && input$pca_contrib_comp %in% comp_choices) input$pca_contrib_comp else "PC1"
    updateSelectInput(session, "pca_contrib_comp", choices = comp_choices, selected = curr_sel)
  })
  
  # Dynamic updates to LDA Component Dropdown choices
  observe({
    res <- active_model()
    req(res)
    n_disc <- length(res$lda_trace_pct)
    comp_choices <- paste0("LD", seq_len(n_disc))
    curr_sel <- if (!is.null(input$lda_contrib_comp) && input$lda_contrib_comp %in% comp_choices) input$lda_contrib_comp else "LD1"
    updateSelectInput(session, "lda_contrib_comp", choices = comp_choices, selected = curr_sel)
  })
  
  # Validation Alert banner (Null-Safe)
  output$validation_alert <- renderUI({
    if (!is.null(input$selected_features) && length(input$selected_features) < 2) {
      div(class = "studio-alert-warning",
        strong("Input Warning: "),
        "At least 2 features are required to construct a 2D projection. Please select 2 or more features and click 'Run Analysis'."
      )
    } else {
      NULL
    }
  })
  
  # Dynamic Card Titles & Subtitles for Main Scatter Plot in Tab 1
  output$scatter_card_title <- renderText({
    if (input$method == "pca") "PCA Projection" else "LDA Projection"
  })
  output$scatter_card_subtitle <- renderText({
    if (input$method == "pca") "PC1 vs PC2 Scatter Plot (Unsupervised)" else "LD1 vs LD2 Scatter Plot (Supervised Separation)"
  })
  
  # ----------------------------------------------------------------------------
  # SHARED ELEGANT ACADEMIC GGPLOT THEME
  # ----------------------------------------------------------------------------
  theme_studio <- function(base_size = 12.5) {
    theme_minimal(base_size = base_size) +
      theme(
        plot.background       = element_rect(fill = "#FFFFFF", color = NA),
        panel.background      = element_rect(fill = "#FFFFFF", color = NA),
        panel.grid.minor      = element_blank(),
        panel.grid.major      = element_line(color = "#F1F5F9", linewidth = 0.5),
        axis.title.x          = element_text(face = "bold", color = "#0F172A", size = base_size, margin = margin(t = 8)),
        axis.title.y          = element_text(face = "bold", color = "#0F172A", size = base_size, margin = margin(r = 8)),
        axis.text             = element_text(color = "#475569", size = base_size - 1.5),
        legend.position       = "top",
        legend.box            = "horizontal",
        legend.box.just       = "center",
        legend.justification  = "center",
        legend.title          = element_text(face = "bold", color = "#0F172A", size = base_size - 1),
        legend.text           = element_text(color = "#334155", size = base_size - 1),
        legend.key            = element_blank(),
        legend.background     = element_blank(),
        legend.box.background = element_blank(),
        legend.margin         = margin(0, 0, 4, 0),
        legend.box.margin     = margin(0, 0, 4, 0),
        legend.spacing.x      = unit(10, "pt"),
        plot.margin           = margin(8, 14, 8, 14)
      )
  }
  
  # Helper: Class Palette Builder
  get_class_palette <- function() {
    classes <- levels(dataset_state$raw[[dataset_state$class_col]])
    if (all(classes %in% names(cultivar_colors_default))) {
      cultivar_colors_default[classes]
    } else {
      curated_colors <- c("#315DEB", "#E64980", "#8055C7", "#10B981", "#F59E0B", "#06B6D4")
      setNames(curated_colors[seq_along(classes)], classes)
    }
  }
  
  # ----------------------------------------------------------------------------
  # TAB 1: MAIN SCATTER PLOT (PCA or LDA)
  # ----------------------------------------------------------------------------
  build_scatter_plot <- function() {
    res <- active_model()
    req(res)
    m <- input$method
    distinguish <- isTRUE(input$distinguish_split)
    pal <- get_class_palette()
    
    if (m == "pca") {
      df <- res$pca_scores_df
      pct1 <- res$pca_var_pct[1]
      pct2 <- if (length(res$pca_var_pct) >= 2) res$pca_var_pct[2] else 0
      
      p <- ggplot(df, aes(x = PC1, y = PC2)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "#E2E8F0", linewidth = 0.5) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "#E2E8F0", linewidth = 0.5) +
        stat_ellipse(geom = "polygon", aes(fill = Cultivar), alpha = 0.08, level = 0.95, show.legend = FALSE, color = NA)
      
      if (distinguish) {
        p <- p +
          geom_point(aes(fill = Cultivar, shape = Split), size = 3.4, stroke = 0.6, color = "#FFFFFF", alpha = 0.88) +
          scale_shape_manual(name = "Partition (Shape):", values = c("Training (70%)" = 21, "Testing (30%)" = 24)) +
          scale_fill_manual(name = "Wine Class (Color):", values = pal) +
          guides(
            fill = guide_legend(
              order = 1,
              nrow = 1,
              override.aes = list(shape = 21, size = 3.8, stroke = 0.6, color = "#FFFFFF", alpha = 1)
            ),
            shape = guide_legend(
              order = 2,
              nrow = 1,
              override.aes = list(fill = "#475569", size = 3.8, stroke = 0.6, color = "#FFFFFF", alpha = 1)
            )
          )
      } else {
        p <- p +
          geom_point(aes(fill = Cultivar), shape = 21, size = 3.4, stroke = 0.6, color = "#FFFFFF", alpha = 0.88) +
          scale_fill_manual(name = "Wine Class:", values = pal) +
          guides(
            fill = guide_legend(
              nrow = 1,
              override.aes = list(shape = 21, size = 3.8, stroke = 0.6, color = "#FFFFFF", alpha = 1)
            )
          )
      }
      
      p <- p +
        theme_studio(base_size = 12.5) +
        labs(
          x = sprintf("PC1 (%.2f%% Explained Variance)", pct1),
          y = sprintf("PC2 (%.2f%% Explained Variance)", pct2)
        )
      return(p)
    } else {
      df <- res$lda_scores_df
      
      # If LD2 is not available (e.g. 1 discriminant axis only)
      if (!("LD2" %in% colnames(df)) || length(res$lda_trace_pct) < 2) {
        pct1 <- if (length(res$lda_trace_pct) >= 1) res$lda_trace_pct[1] else 100
        p <- ggplot(df, aes(x = LD1, y = 0)) +
          geom_hline(yintercept = 0, linetype = "dashed", color = "#E2E8F0", linewidth = 0.5) +
          geom_vline(xintercept = 0, linetype = "dashed", color = "#E2E8F0", linewidth = 0.5)
        
        if (distinguish) {
          p <- p +
            geom_point(aes(fill = Cultivar, shape = Split), size = 3.4, stroke = 0.6, color = "#FFFFFF", alpha = 0.88,
                       position = position_jitter(height = 0.05, seed = 42)) +
            scale_shape_manual(name = "Partition (Shape):", values = c("Training (70%)" = 21, "Testing (30%)" = 24)) +
            scale_fill_manual(name = "Wine Class (Color):", values = pal) +
            guides(
              fill = guide_legend(
                order = 1,
                nrow = 1,
                override.aes = list(shape = 21, size = 3.8, stroke = 0.6, color = "#FFFFFF", alpha = 1)
              ),
              shape = guide_legend(
                order = 2,
                nrow = 1,
                override.aes = list(fill = "#475569", size = 3.8, stroke = 0.6, color = "#FFFFFF", alpha = 1)
              )
            )
        } else {
          p <- p +
            geom_point(aes(fill = Cultivar), shape = 21, size = 3.4, stroke = 0.6, color = "#FFFFFF", alpha = 0.88,
                       position = position_jitter(height = 0.05, seed = 42)) +
            scale_fill_manual(name = "Wine Class:", values = pal) +
            guides(
              fill = guide_legend(
                nrow = 1,
                override.aes = list(shape = 21, size = 3.8, stroke = 0.6, color = "#FFFFFF", alpha = 1)
              )
            )
        }
        
        p <- p +
          theme_studio(base_size = 12.5) +
          theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
          labs(
            x = sprintf("LD1 (%.2f%% Between-Class Trace)", pct1),
            y = "1D Projection (LD2 not available)"
          )
        return(p)
      }
      
      pct1 <- res$lda_trace_pct[1]
      pct2 <- res$lda_trace_pct[2]
      
      p <- ggplot(df, aes(x = LD1, y = LD2)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "#E2E8F0", linewidth = 0.5) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "#E2E8F0", linewidth = 0.5) +
        stat_ellipse(geom = "polygon", aes(fill = Cultivar), alpha = 0.08, level = 0.95, show.legend = FALSE, color = NA)
      
      if (distinguish) {
        p <- p +
          geom_point(aes(fill = Cultivar, shape = Split), size = 3.4, stroke = 0.6, color = "#FFFFFF", alpha = 0.88) +
          scale_shape_manual(name = "Partition (Shape):", values = c("Training (70%)" = 21, "Testing (30%)" = 24)) +
          scale_fill_manual(name = "Wine Class (Color):", values = pal) +
          guides(
            fill = guide_legend(
              order = 1,
              nrow = 1,
              override.aes = list(shape = 21, size = 3.8, stroke = 0.6, color = "#FFFFFF", alpha = 1)
            ),
            shape = guide_legend(
              order = 2,
              nrow = 1,
              override.aes = list(fill = "#475569", size = 3.8, stroke = 0.6, color = "#FFFFFF", alpha = 1)
            )
          )
      } else {
        p <- p +
          geom_point(aes(fill = Cultivar), shape = 21, size = 3.4, stroke = 0.6, color = "#FFFFFF", alpha = 0.88) +
          scale_fill_manual(name = "Wine Class:", values = pal) +
          guides(
            fill = guide_legend(
              nrow = 1,
              override.aes = list(shape = 21, size = 3.8, stroke = 0.6, color = "#FFFFFF", alpha = 1)
            )
          )
      }
      
      p <- p +
        theme_studio(base_size = 12.5) +
        labs(
          x = sprintf("LD1 (%.2f%% Between-Class Trace)", pct1),
          y = sprintf("LD2 (%.2f%% Between-Class Trace)", pct2)
        )
      return(p)
    }
  }
  
  output$plot_scatter <- renderPlot({
    build_scatter_plot()
  }, res = 100)
  
  # ----------------------------------------------------------------------------
  # TAB 2: INSIGHTS (PCA) - VARIANCE SUMMARY & SCREE PLOT
  # ----------------------------------------------------------------------------
  output$ui_pca_variance_stats <- renderUI({
    res <- active_model()
    req(res)
    v1 <- res$pca_var_pct[1]
    v2 <- if (length(res$pca_var_pct) >= 2) res$pca_var_pct[2] else 0
    v_tot <- v1 + v2
    
    div(class = "metric-chips-grid",
      div(class = "metric-chip",
        span(class = "metric-chip-label", "PC1 Variance:"),
        span(class = "metric-chip-val", sprintf("%.2f%%", v1))
      ),
      div(class = "metric-chip",
        span(class = "metric-chip-label", "PC2 Variance:"),
        span(class = "metric-chip-val", sprintf("%.2f%%", v2))
      ),
      div(class = "metric-chip metric-chip-primary metric-chip-full",
        span(class = "metric-chip-label", "Combined 2D (PC1 + PC2):"),
        span(class = "metric-chip-val", sprintf("%.2f%% Total Variance", v_tot))
      )
    )
  })
  
  output$table_pca_variance <- renderUI({
    res <- active_model()
    req(res)
    
    var_pcts <- res$pca_var_pct
    cum_pcts <- cumsum(var_pcts)
    n_comps <- length(var_pcts)
    
    table_rows <- lapply(seq_len(n_comps), function(i) {
      tags$tr(
        tags$td(strong(paste0("PC", i))),
        tags$td(sprintf("%.2f%%", var_pcts[i])),
        tags$td(sprintf("%.2f%%", cum_pcts[i]))
      )
    })
    
    tags$div(class = "table-scroll-wrap",
      tags$table(class = "clean-table",
        tags$thead(
          tags$tr(
            tags$th("Component"),
            tags$th("Explained Variance"),
            tags$th("Cumulative Variance")
          )
        ),
        tags$tbody(table_rows)
      )
    )
  })
  
  # Scree Plot (Simple Blue Bars #315DEB, No Curve/Connecting Line, Uncluttered Labels)
  build_scree_plot <- function() {
    res <- active_model()
    req(res)
    
    var_pcts <- res$pca_var_pct
    n_comps <- length(var_pcts)
    comp_names <- factor(paste0("PC", seq_len(n_comps)), levels = paste0("PC", seq_len(n_comps)))
    
    scree_df <- data.frame(
      Component = comp_names,
      Pct = var_pcts,
      Label = ifelse(seq_len(n_comps) <= min(4, n_comps), sprintf("%.1f%%", var_pcts), "")
    )
    
    y_max <- min(100, max(var_pcts) * 1.30)
    if (y_max < 25) y_max <- 25
    
    ggplot(scree_df, aes(x = Component, y = Pct)) +
      geom_col(fill = "#315DEB", width = 0.58, alpha = 0.95) +
      geom_text(aes(label = Label), vjust = -0.7, size = 3.6, fontface = "bold", color = "#0F172A") +
      scale_y_continuous(
        limits = c(0, y_max),
        labels = function(x) paste0(x, "%"),
        expand = expansion(mult = c(0, 0.08))
      ) +
      coord_cartesian(clip = "off") +
      theme_studio(base_size = 12.5) +
      theme(
        panel.grid.major.x = element_blank(),
        axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 1, size = 10, face = "bold", color = "#475569"),
        axis.text.y = element_text(size = 11, color = "#475569"),
        axis.title.x = element_text(face = "bold", color = "#0F172A", size = 12.5, margin = margin(t = 10)),
        axis.title.y = element_text(face = "bold", color = "#0F172A", size = 12.5, margin = margin(r = 10)),
        plot.margin = margin(16, 16, 10, 16)
      ) +
      labs(x = "Principal Component", y = "Explained Variance (%)")
  }
  
  output$plot_scree <- renderPlot({
    build_scree_plot()
  }, res = 100)
  
  # ----------------------------------------------------------------------------
  # TAB 2: INSIGHTS (PCA) - RESTORED CLEAN FEATURE CONTRIBUTIONS
  # ----------------------------------------------------------------------------
  output$ui_pca_feature_contributions <- renderUI({
    res <- active_model()
    req(res)
    comp <- input$pca_contrib_comp
    req(comp %in% colnames(res$pca_fit$rotation))
    
    raw_vals <- res$pca_fit$rotation[, comp]
    f_keys <- res$selected_keys
    f_cols <- res$selected_cols
    f_names <- dataset_state$display_names[f_keys]
    
    val_vec <- as.numeric(raw_vals[f_cols])
    
    df_contrib <- data.frame(
      Name = unname(f_names),
      Value = val_vec,
      AbsValue = abs(val_vec),
      stringsAsFactors = FALSE
    )
    
    # Sort descending by absolute loading magnitude
    df_contrib <- df_contrib[order(df_contrib$AbsValue, decreasing = TRUE), ]
    
    max_abs <- max(df_contrib$AbsValue, na.rm = TRUE)
    if (is.na(max_abs) || max_abs == 0) max_abs <- 1
    
    rows <- lapply(seq_len(nrow(df_contrib)), function(i) {
      v <- df_contrib$Value[i]
      pct_width <- max(2, (abs(v) / max_abs) * 100)
      val_fmt <- sprintf("%+.3f", v)
      bar_color <- if (v >= 0) "#315DEB" else "#E64980"
      
      div(class = "feature-contrib-row",
        div(class = "contrib-name", title = df_contrib$Name[i], df_contrib$Name[i]),
        div(class = "contrib-bar-track",
          div(class = "contrib-bar-fill", 
              style = sprintf("width: %.1f%%; background-color: %s;", pct_width, bar_color))
        ),
        div(class = "contrib-val", 
            style = if (v >= 0) "color: #315DEB;" else "color: #E64980;",
            val_fmt)
      )
    })
    
    div(class = "feature-contrib-list", rows)
  })
  
  # ----------------------------------------------------------------------------
  # TAB 2: INSIGHTS (LDA) - DISCRIMINANT VARIANCE BAR CHART
  # ----------------------------------------------------------------------------
  output$ui_lda_variance_stats <- renderUI({
    res <- active_model()
    req(res)
    v1 <- if (length(res$lda_trace_pct) >= 1) res$lda_trace_pct[1] else 100
    v2 <- if (length(res$lda_trace_pct) >= 2) res$lda_trace_pct[2] else 0
    v_tot <- v1 + v2
    
    div(class = "metric-chips-grid",
      div(class = "metric-chip",
        span(class = "metric-chip-label", "LD1 Trace:"),
        span(class = "metric-chip-val", sprintf("%.2f%%", v1))
      ),
      div(class = "metric-chip",
        span(class = "metric-chip-label", "LD2 Trace:"),
        span(class = "metric-chip-val", sprintf("%.2f%%", v2))
      ),
      div(class = "metric-chip metric-chip-primary metric-chip-full",
        span(class = "metric-chip-label", "Total Between-Class Trace:"),
        span(class = "metric-chip-val", sprintf("%.2f%% (LD1 + LD2)", v_tot))
      )
    )
  })
  
  output$table_lda_variance <- renderUI({
    res <- active_model()
    req(res)
    
    trace_pcts <- res$lda_trace_pct
    cum_trace <- cumsum(trace_pcts)
    n_disc <- length(trace_pcts)
    
    table_rows <- lapply(seq_len(n_disc), function(i) {
      tags$tr(
        tags$td(strong(paste0("LD", i))),
        tags$td(sprintf("%.2f%%", trace_pcts[i])),
        tags$td(sprintf("%.2f%%", cum_trace[i]))
      )
    })
    
    tagList(
      tags$div(class = "table-scroll-wrap",
        tags$table(class = "clean-table",
          tags$thead(
            tags$tr(
              tags$th("Discriminant"),
              tags$th("Between-Class Trace"),
              tags$th("Cumulative Trace")
            )
          ),
          tags$tbody(table_rows)
        )
      ),
      div(class = "insights-context-box", style = "margin-top: 10px;",
        strong("Held-Out Test Accuracy: "),
        sprintf("%.2f%% (%d of %d test observations correctly classified with zero data leakage).", 
                res$lda_test_acc, res$lda_test_correct, res$lda_test_total)
      )
    )
  })
  
  build_lda_variance_plot <- function() {
    res <- active_model()
    req(res)
    
    trace_pcts <- res$lda_trace_pct
    if (is.null(trace_pcts) || length(trace_pcts) == 0 || any(is.na(trace_pcts))) {
      return(
        ggplot() +
          annotate("text", x = 0.5, y = 0.5, 
                   label = "Discriminant variance proportions could not be computed for this configuration.",
                   color = "#64748B", size = 4) +
          theme_void() +
          theme(plot.background = element_rect(fill = "#FFFFFF", color = NA))
      )
    }
    
    n_comps <- length(trace_pcts)
    comp_names <- factor(paste0("LD", seq_len(n_comps)), levels = paste0("LD", seq_len(n_comps)))
    
    lda_df <- data.frame(
      Discriminant = comp_names,
      TracePct = trace_pcts,
      Label = sprintf("%.2f%%", trace_pcts)
    )
    
    y_max <- min(100, max(trace_pcts) * 1.25)
    if (y_max < 30) y_max <- 30
    
    ggplot(lda_df, aes(x = Discriminant, y = TracePct)) +
      geom_col(fill = "#315DEB", width = 0.45, alpha = 0.95) +
      geom_text(aes(label = Label), vjust = -0.7, size = 4.0, fontface = "bold", color = "#0F172A") +
      scale_y_continuous(
        limits = c(0, y_max),
        breaks = seq(0, 100, 20),
        labels = function(x) paste0(x, "%"),
        expand = expansion(mult = c(0, 0.08))
      ) +
      coord_cartesian(clip = "off") +
      theme_studio(base_size = 12.5) +
      theme(
        panel.grid.major.x = element_blank(),
        axis.text.x = element_text(size = 12, face = "bold", color = "#0F172A"),
        axis.text.y = element_text(size = 11, color = "#475569"),
        axis.title.x = element_text(face = "bold", color = "#0F172A", size = 12.5, margin = margin(t = 10)),
        axis.title.y = element_text(face = "bold", color = "#0F172A", size = 12.5, margin = margin(r = 12)),
        plot.margin = margin(16, 16, 10, 24)
      ) +
      labs(
        x = "Linear Discriminant Function",
        y = "Between-Class Variance (%)"
      )
  }
  
  output$plot_lda_variance <- renderPlot({
    build_lda_variance_plot()
  }, res = 100)
  
  # ----------------------------------------------------------------------------
  # TAB 2: INSIGHTS (LDA) - RESTORED CLEAN FEATURE CONTRIBUTIONS
  # ----------------------------------------------------------------------------
  output$ui_lda_feature_contributions <- renderUI({
    res <- active_model()
    req(res)
    comp <- input$lda_contrib_comp
    req(comp %in% colnames(res$lda_fit$scaling))
    
    raw_vals <- res$lda_fit$scaling[, comp]
    f_keys <- res$selected_keys
    f_cols <- res$selected_cols
    f_names <- dataset_state$display_names[f_keys]
    
    val_vec <- as.numeric(raw_vals[f_cols])
    
    df_contrib <- data.frame(
      Name = unname(f_names),
      Value = val_vec,
      AbsValue = abs(val_vec),
      stringsAsFactors = FALSE
    )
    
    # Sort descending by absolute coefficient magnitude
    df_contrib <- df_contrib[order(df_contrib$AbsValue, decreasing = TRUE), ]
    
    max_abs <- max(df_contrib$AbsValue, na.rm = TRUE)
    if (is.na(max_abs) || max_abs == 0) max_abs <- 1
    
    rows <- lapply(seq_len(nrow(df_contrib)), function(i) {
      v <- df_contrib$Value[i]
      pct_width <- max(2, (abs(v) / max_abs) * 100)
      val_fmt <- sprintf("%+.3f", v)
      bar_color <- if (v >= 0) "#315DEB" else "#E64980"
      
      div(class = "feature-contrib-row",
        div(class = "contrib-name", title = df_contrib$Name[i], df_contrib$Name[i]),
        div(class = "contrib-bar-track",
          div(class = "contrib-bar-fill", 
              style = sprintf("width: %.1f%%; background-color: %s;", pct_width, bar_color))
        ),
        div(class = "contrib-val", 
            style = if (v >= 0) "color: #315DEB;" else "color: #E64980;",
            val_fmt)
      )
    })
    
    div(class = "feature-contrib-list", rows)
  })
  
  # ----------------------------------------------------------------------------
  # TAB 3: OVERVIEW - METHOD-AWARE STATE & METRIC CARDS
  # ----------------------------------------------------------------------------
  settings_changed <- reactive({
    method_diff <- !is.null(input$method) && input$method != latest_analysis$method
    feat_diff <- !is.null(input$selected_features) && !setequal(input$selected_features, latest_analysis$keys)
    std_diff <- !is.null(input$standardize) && input$standardize != latest_analysis$standardize
    method_diff || feat_diff || std_diff
  })
  
  output$overview_sync_notice <- renderUI({
    if (settings_changed()) {
      curr_m_desc <- if (input$method == "pca") "PCA" else "LDA"
      last_m_desc <- if (latest_analysis$method == "pca") "PCA" else "LDA"
      div(class = "studio-alert-notice",
        strong("Input Modification Notice: "),
        sprintf("Settings in the Analysis tab have changed (Method: %s selected, Run pending). The summary and data below reflect the latest completed %s analysis run. Click 'Run Analysis' in the Analysis tab to recalculate and display %s results.",
                curr_m_desc, last_m_desc, curr_m_desc)
      )
    } else {
      NULL
    }
  })
  
  # Card 1: Active Method
  output$overview_method_val <- renderText({
    if (latest_analysis$method == "pca") "PCA (Unsupervised)" else "LDA (Supervised)"
  })
  
  output$overview_method_desc <- renderText({
    if (settings_changed() && input$method != latest_analysis$method) {
      sprintf("Active Run: %s | Current Selected: %s (Click 'Run Analysis' to sync)",
              toupper(latest_analysis$method), toupper(input$method))
    } else {
      if (latest_analysis$method == "pca") {
        "Unsupervised coordinate transformation maximizing total feature variance"
      } else {
        "Supervised coordinate transformation maximizing cultivar class separation"
      }
    }
  })
  
  output$overview_method_badge_ui <- renderUI({
    badge_cls <- if (settings_changed() && input$method != latest_analysis$method) "overview-card-badge overview-card-badge-stale" else "overview-card-badge"
    badge_text <- if (isTRUE(latest_analysis$standardize)) "Z-Score Standardized" else "Mean-Centered Only"
    span(class = badge_cls, badge_text)
  })
  
  # Card 2: Dataset Information
  output$overview_dataset_val <- renderText({
    latest_analysis$dataset_name
  })
  
  output$overview_dataset_desc <- renderText({
    sprintf("%d total observations • %d features selected",
            latest_analysis$num_samples, length(latest_analysis$keys))
  })
  
  output$overview_dataset_badge <- renderText({
    n_cls <- length(unique(dataset_state$raw[[dataset_state$class_col]]))
    sprintf("%d Distinct Cultivar Classes", n_cls)
  })
  
  # Card 3: Dynamic Method-Specific Metric Card
  output$overview_metric_card_label <- renderText({
    if (latest_analysis$method == "pca") {
      "PCA Variance Explained (2D)"
    } else {
      "LDA Discriminant Trace (2D)"
    }
  })
  
  output$overview_metric_card_val <- renderText({
    res <- latest_analysis$res
    req(res)
    if (latest_analysis$method == "pca") {
      v1 <- res$pca_var_pct[1]
      v2 <- if (length(res$pca_var_pct) >= 2) res$pca_var_pct[2] else 0
      sprintf("%.2f%% Total Variance", v1 + v2)
    } else {
      t1 <- if (length(res$lda_trace_pct) >= 1) res$lda_trace_pct[1] else 100
      t2 <- if (length(res$lda_trace_pct) >= 2) res$lda_trace_pct[2] else 0
      sprintf("%.2f%% Between-Class Trace", t1 + t2)
    }
  })
  
  output$overview_metric_card_desc <- renderText({
    res <- latest_analysis$res
    req(res)
    if (latest_analysis$method == "pca") {
      v1 <- res$pca_var_pct[1]
      v2 <- if (length(res$pca_var_pct) >= 2) res$pca_var_pct[2] else 0
      sprintf("PC1: %.2f%% • PC2: %.2f%% of total feature variance", v1, v2)
    } else {
      t1 <- if (length(res$lda_trace_pct) >= 1) res$lda_trace_pct[1] else 100
      t2 <- if (length(res$lda_trace_pct) >= 2) res$lda_trace_pct[2] else 0
      sprintf("LD1: %.2f%% • LD2: %.2f%% between-class discriminant trace", t1, t2)
    }
  })
  
  output$overview_metric_card_badge_ui <- renderUI({
    if (latest_analysis$method == "pca") {
      span(class = "overview-card-badge", "Unsupervised Eigen-Projection")
    } else {
      span(class = "overview-card-badge", "Supervised Fisher Criterion")
    }
  })
  
  # Card 4: Partition
  output$overview_split_desc <- renderText({
    n_tr <- nrow(dataset_state$train)
    n_te <- nrow(dataset_state$test)
    n_tot <- n_tr + n_te
    pct_tr <- if (n_tot > 0) (n_tr / n_tot) * 100 else 70.2
    pct_te <- if (n_tot > 0) (n_te / n_tot) * 100 else 29.8
    sprintf("%d Training (%.1f%%) • %d Testing (%.1f%%)",
            n_tr, pct_tr, n_te, pct_te)
  })
  
  # Bottom Table: Transformed Data Preview (Method-Aware)
  output$overview_table_title <- renderText({
    m_name <- if (latest_analysis$method == "pca") "PCA" else "LDA"
    sprintf("Transformed Data Preview (%s Coordinates)", m_name)
  })
  
  output$overview_table_subtitle <- renderText({
    res <- latest_analysis$res
    req(res)
    m_name <- if (latest_analysis$method == "pca") "PCA" else "LDA"
    df <- if (latest_analysis$method == "pca") res$pca_scores_df else res$lda_scores_df
    sprintf("Displaying first 10 observations of %d total records projected onto %s and %s (latest completed %s run):",
            nrow(df), 
            if (latest_analysis$method == "pca") "PC1" else "LD1",
            if (latest_analysis$method == "pca") "PC2" else "LD2",
            m_name)
  })
  
  output$table_transformed_preview <- renderUI({
    res <- latest_analysis$res
    req(res)
    m <- latest_analysis$method
    
    if (m == "pca") {
      df <- res$pca_scores_df
      sub_df <- head(df, 10)
      
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
            tags$th("Wine Cultivar"),
            tags$th("Partition")
          )
        ),
        tags$tbody(table_rows)
      )
    } else {
      df <- res$lda_scores_df
      sub_df <- head(df, 10)
      
      table_rows <- lapply(seq_len(nrow(sub_df)), function(i) {
        c_badge <- "badge-cultivar-1"
        if (as.character(sub_df$Cultivar[i]) == "Cultivar 2") c_badge <- "badge-cultivar-2"
        if (as.character(sub_df$Cultivar[i]) == "Cultivar 3") c_badge <- "badge-cultivar-3"
        s_badge <- if (grepl("Training", sub_df$Split[i])) "badge-train" else "badge-test"
        
        tags$tr(
          tags$td(i),
          tags$td(sprintf("%.4f", sub_df$LD1[i])),
          tags$td(if ("LD2" %in% colnames(sub_df)) sprintf("%.4f", sub_df$LD2[i]) else "N/A"),
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
            tags$th("Wine Cultivar"),
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
  
  # 1. Download Current Scatter Plot PNG (From Tab 1 Analysis)
  output$download_scatter_png <- downloadHandler(
    filename = function() {
      paste0(input$method, "_scatter_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- build_scatter_plot()
      ggsave(file, plot = p, width = 8.5, height = 5.2, dpi = 300)
    }
  )
  
  # 2. Download Scree Plot PNG (From Tab 2 Insights PCA)
  output$download_scree_png <- downloadHandler(
    filename = function() {
      paste0("pca_scree_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- build_scree_plot()
      ggsave(file, plot = p, width = 8.5, height = 4.2, dpi = 300)
    }
  )
  
  # 3. Download LDA Variance PNG (From Tab 2 Insights LDA)
  output$download_lda_variance_png <- downloadHandler(
    filename = function() {
      paste0("lda_discriminant_variance_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- build_lda_variance_plot()
      ggsave(file, plot = p, width = 8.5, height = 4.2, dpi = 300)
    }
  )
  
  # 4. Download Transformed CSV (All transformed observations from latest completed run)
  output$download_transformed_csv <- downloadHandler(
    filename = function() {
      paste0(latest_analysis$method, "_transformed_data_all_", Sys.Date(), ".csv")
    },
    content = function(file) {
      res <- latest_analysis$res
      df <- if (latest_analysis$method == "pca") res$pca_scores_df else res$lda_scores_df
      write.csv(df, file, row.names = FALSE)
    }
  )
}

# --- 8. Launch Application ---
shinyApp(ui = ui, server = server)
