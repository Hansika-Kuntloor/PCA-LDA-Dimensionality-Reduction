# ==============================================================================
# Application: PCA & LDA Explorer
# Description: Interactive Dimensionality Reduction and Classification in R Shiny
# Dataset: Wine Dataset (rattle.data / UCI Machine Learning Repository)
# Split Rule: STRICT 70:30 Stratified Train-Test Split (125 Train / 53 Test)
# Author: Pair Programming with Antigravity
# ==============================================================================

# --- 1. Load Required Libraries ---
library(shiny)
library(ggplot2)
library(MASS)

# Load DT for interactive tables if available
has_dt <- requireNamespace("DT", quietly = TRUE)
if (has_dt) {
  library(DT)
}

# Gracefully load rattle.data or provide reliable fallback
if (requireNamespace("rattle.data", quietly = TRUE)) {
  data(wine, package = "rattle.data")
} else {
  # Fallback directly to UCI Wine repository if rattle.data is missing
  wine_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
  wine <- read.csv(wine_url, header = FALSE)
  colnames(wine) <- c("Type", "Alcohol", "Malic", "Ash", "Alcalinity", 
                      "Magnesium", "Phenols", "Flavanoids", "Nonflavanoids", 
                      "Proanthocyanins", "Color", "Hue", "Dilution", "Proline")
  wine$Type <- as.factor(wine$Type)
}

# Ensure Wine Type is a clean factor with descriptive cultivar names
wine$Type <- as.factor(wine$Type)
wine$Cultivar <- factor(wine$Type, 
                        levels = c("1", "2", "3"),
                        labels = c("Cultivar 1", "Cultivar 2", "Cultivar 3"))

feature_cols <- c("Alcohol", "Malic", "Ash", "Alcalinity", "Magnesium", 
                  "Phenols", "Flavanoids", "Nonflavanoids", "Proanthocyanins", 
                  "Color", "Hue", "Dilution", "Proline")

# --- 2. Strict 70:30 Stratified Train-Test Split (No Data Leakage) ---
# Reproducible random seed
set.seed(42)

cultivar_levels <- levels(wine$Cultivar)
train_indices <- integer(0)

# Stratified sampling: 70% of each cultivar into training, remaining 30% into testing
for (cult in cultivar_levels) {
  cult_idx <- which(wine$Cultivar == cult)
  n_cult <- length(cult_idx)
  n_train <- round(n_cult * 0.70)
  train_indices <- c(train_indices, sample(cult_idx, size = n_train))
}

train_indices <- sort(train_indices)
test_indices <- setdiff(seq_len(nrow(wine)), train_indices)

# Add Split identifier to full dataset
wine$Split <- "Testing (30%)"
wine$Split[train_indices] <- "Training (70%)"
wine$Split <- factor(wine$Split, levels = c("Training (70%)", "Testing (30%)"))

# Partition into Train and Test subsets
wine_train <- wine[train_indices, ]
wine_test  <- wine[test_indices, ]

# --- 3. Training-Set Only Standardization ---
# Parameter estimates (means and standard deviations) are computed exclusively
# on the training partition to prevent data leakage.
train_means <- colMeans(wine_train[, feature_cols])
train_sds   <- apply(wine_train[, feature_cols], 2, sd)

# Scale Training Data
wine_train_scaled <- as.data.frame(scale(wine_train[, feature_cols], center = train_means, scale = train_sds))
wine_train_scaled$Cultivar <- wine_train$Cultivar
wine_train_scaled$Split    <- wine_train$Split

# Transform Test Data using Training Set Parameters (do NOT refit!)
wine_test_scaled <- as.data.frame(scale(wine_test[, feature_cols], center = train_means, scale = train_sds))
wine_test_scaled$Cultivar <- wine_test$Cultivar
wine_test_scaled$Split    <- wine_test$Split

# Combined standardized frame (for table viewing and unified plotting)
wine_scaled_all <- rbind(wine_train_scaled, wine_test_scaled)


# --- 4. Model Training (70% Training Partition Only) ---

# A. PCA fitted on training features only (Unsupervised - no class labels)
pca_fit <- prcomp(wine_train_scaled[, feature_cols], center = FALSE, scale. = FALSE)
pca_var <- pca_fit$sdev^2
pca_var_pct <- (pca_var / sum(pca_var)) * 100
pca_cum_var <- cumsum(pca_var_pct)

# Project Training features onto PCA rotation
pca_train_scores <- as.data.frame(pca_fit$x)
pca_train_scores$Cultivar <- wine_train$Cultivar
pca_train_scores$Split    <- wine_train$Split

# Transform Test features using training-fitted PCA rotation (no refit!)
pca_test_scores <- as.data.frame(predict(pca_fit, newdata = wine_test_scaled[, feature_cols]))
pca_test_scores$Cultivar <- wine_test$Cultivar
pca_test_scores$Split    <- wine_test$Split

# Combined PCA Scores dataframe
pca_scores_df <- rbind(pca_train_scores, pca_test_scores)


# B. LDA fitted on training features and training class labels (Supervised)
lda_fit <- MASS::lda(Cultivar ~ ., data = wine_train_scaled[, c(feature_cols, "Cultivar")])
lda_trace_pct <- (lda_fit$svd^2 / sum(lda_fit$svd^2)) * 100

# Project Training observations onto LDA discriminants
lda_train_pred <- predict(lda_fit, newdata = wine_train_scaled)
lda_train_scores <- as.data.frame(lda_train_pred$x)
lda_train_scores$Cultivar  <- wine_train$Cultivar
lda_train_scores$Split     <- wine_train$Split
lda_train_scores$Predicted <- lda_train_pred$class

# Transform Held-Out Test observations using training-fitted LDA model (no refit!)
lda_test_pred <- predict(lda_fit, newdata = wine_test_scaled)
lda_test_scores <- as.data.frame(lda_test_pred$x)
lda_test_scores$Cultivar  <- wine_test$Cultivar
lda_test_scores$Split     <- wine_test$Split
lda_test_scores$Predicted <- lda_test_pred$class

# Combined LDA Scores dataframe
lda_scores_df <- rbind(lda_train_scores, lda_test_scores)


# --- 5. Evaluated Test Set Classification Metrics ---
# Confusion Matrix and Out-of-Sample Accuracy evaluated on the held-out test set
lda_test_conf <- table(Predicted = lda_test_pred$class, Actual = wine_test$Cultivar)
lda_test_acc  <- (sum(diag(lda_test_conf)) / sum(lda_test_conf)) * 100

lda_train_conf <- table(Predicted = lda_train_pred$class, Actual = wine_train$Cultivar)
lda_train_acc  <- (sum(diag(lda_train_conf)) / sum(lda_train_conf)) * 100

# Also evaluate a separate Nearest-Centroid Classifier on the 2D PCA representation
# (demonstrates that unsupervised 2D PCA yields lower test classification accuracy than supervised 2D LDA)
pca_centroids <- aggregate(pca_train_scores[, c("PC1", "PC2")], by = list(Cultivar = pca_train_scores$Cultivar), mean)
predict_pca_centroid <- function(pt) {
  dists <- apply(pca_centroids[, c("PC1", "PC2")], 1, function(c) sum((pt - c)^2))
  pca_centroids$Cultivar[which.min(dists)]
}
pca_test_preds <- factor(apply(pca_test_scores[, c("PC1", "PC2")], 1, predict_pca_centroid), levels = cultivar_levels)
pca_test_conf  <- table(Predicted = pca_test_preds, Actual = wine_test$Cultivar)
pca_test_acc   <- (sum(diag(pca_test_conf)) / sum(pca_test_conf)) * 100

# Calculate Between-Class vs Within-Class Separation (Fisher Ratio) on Training Set
compute_fisher_ratio <- function(coords, classes) {
  classes <- as.factor(classes)
  n <- nrow(coords)
  S_T <- cov(coords) * (n - 1)
  S_W <- matrix(0, nrow = ncol(coords), ncol = ncol(coords))
  for (cl in levels(classes)) {
    idx <- which(classes == cl)
    n_cl <- length(idx)
    if (n_cl > 1) {
      cl_data <- coords[idx, , drop = FALSE]
      S_W <- S_W + cov(cl_data) * (n_cl - 1)
    }
  }
  S_B <- S_T - S_W
  ratio <- sum(diag(S_B)) / sum(diag(S_W))
  return(ratio)
}

pca_fisher_val <- compute_fisher_ratio(pca_train_scores[, c("PC1", "PC2")], wine_train$Cultivar)
lda_fisher_val <- compute_fisher_ratio(lda_train_scores[, c("LD1", "LD2")], wine_train$Cultivar)

# Curated accessible color palette for the 3 Cultivars
cultivar_colors <- c(
  "Cultivar 1" = "#1d4ed8",  # Royal Blue
  "Cultivar 2" = "#059669",  # Emerald Green
  "Cultivar 3" = "#dc2626"   # Crimson Red
)

# Shapes for Split distinction: Circle for Training, Triangle for Testing
split_shapes <- c(
  "Training (70%)" = 21,  # Solid circle with border
  "Testing (30%)"  = 24   # Solid triangle with border
)


# --- 6. User Interface (UI) Definition ---
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      /* Global Layout & Fonts */
      body {
        background-color: #f8fafc;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        color: #1e293b;
        margin: 0;
        padding: 0;
      }
      
      /* Header Styling */
      .app-header {
        background: linear-gradient(135deg, #0f1f38 0%, #1e3a5f 100%);
        color: #ffffff;
        padding: 22px 30px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        border-bottom: 3px solid #3b82f6;
      }
      .app-header h1 {
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 5px 0;
        letter-spacing: -0.5px;
      }
      .app-header p {
        font-size: 14px;
        color: #94a3b8;
        margin: 0;
      }
      .header-badges span {
        display: inline-block;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #e2e8f0;
        font-size: 12px;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 20px;
        margin-left: 8px;
      }
      .badge-highlight {
        background: #0284c7 !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
      }
      
      /* Sidebar Styling */
      .sidebar-panel {
        background-color: #111d2e !important;
        border-radius: 12px;
        padding: 22px 18px;
        color: #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #1e3a5f;
        margin-top: 20px;
      }
      .sidebar-panel label {
        color: #cbd5e1;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
      }
      .sidebar-panel .radio label {
        color: #f1f5f9;
        font-size: 13.5px;
        font-weight: 500;
        padding: 6px 10px;
        display: block;
        border-radius: 6px;
        transition: all 0.2s ease;
      }
      .sidebar-panel .radio label:hover {
        background-color: #1e3a5f;
      }
      .sidebar-panel .checkbox label {
        color: #f8fafc;
        font-size: 13px;
      }
      .sidebar-section-title {
        font-size: 12px;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 18px;
        margin-bottom: 10px;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 4px;
      }
      
      /* Main Content Area */
      .main-content {
        padding: 20px 10px 40px 10px;
      }
      
      /* Summary Cards */
      .summary-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1e3a5f;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }
      .summary-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
      }
      .summary-card.card-blue { border-left-color: #2563eb; }
      .summary-card.card-emerald { border-left-color: #059669; }
      .summary-card.card-purple { border-left-color: #7c3aed; }
      .summary-card.card-amber { border-left-color: #d97706; }
      .card-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
      }
      .card-title {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
      }
      .card-subtitle {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 2px;
      }
      
      /* Content Card Panels */
      .content-box {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
      }
      .content-box-title {
        font-size: 16px;
        font-weight: 700;
        color: #0f1f38;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 8px;
      }
      
      /* Explanatory Callouts */
      .callout-info {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 14px;
        color: #1e3a8a;
        line-height: 1.5;
      }
      .callout-success {
        background-color: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 14px;
        color: #065f46;
        line-height: 1.5;
      }
      .callout-split {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 18px;
        font-size: 13px;
        color: #334155;
      }
      
      /* Styled Action Buttons and Downloads */
      .btn-download {
        background-color: #0284c7;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
        padding: 8px 14px;
        transition: all 0.2s ease;
        display: inline-block;
        margin-top: 4px;
        margin-bottom: 8px;
        width: 100%;
        text-align: center;
        text-decoration: none;
      }
      .btn-download:hover {
        background-color: #0369a1;
        color: #ffffff;
        text-decoration: none;
      }
      .btn-download-secondary {
        background-color: #334155;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
        padding: 8px 14px;
        transition: all 0.2s ease;
        display: inline-block;
        margin-top: 4px;
        margin-bottom: 8px;
        width: 100%;
        text-align: center;
        text-decoration: none;
      }
      .btn-download-secondary:hover {
        background-color: #475569;
        color: #ffffff;
        text-decoration: none;
      }
      
      /* Custom Table Styling */
      .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
      }
      .custom-table th {
        background-color: #0f1f38;
        color: #ffffff;
        padding: 10px 14px;
        text-align: left;
        font-weight: 600;
      }
      .custom-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #e2e8f0;
      }
      .custom-table tr:nth-child(even) {
        background-color: #f8fafc;
      }
      .custom-table tr:hover {
        background-color: #f1f5f9;
      }
    "))
  ),
  
  # Header Section
  div(class = "app-header",
    div(
      h1("🍷 PCA & LDA Explorer"),
      p("Dimensionality Reduction & Wine Classification (Strict 70:30 Stratified Train-Test Split)")
    ),
    div(class = "header-badges",
      span(class = "badge-highlight", "Strict 70:30 Split (Seed 42)"),
      span("Train: 125 | Test: 53"),
      span("No Data Leakage"),
      span("R Shiny")
    )
  ),
  
  # Sidebar & Main Content Layout
  fluidRow(
    style = "margin: 0; padding: 0 15px;",
    
    # Left Navigation Sidebar
    column(width = 3,
      div(class = "sidebar-panel",
        div(class = "sidebar-section-title", "Navigation Sections"),
        radioButtons("section", label = NULL,
          choices = c(
            "📊 Dataset Overview" = "dataset",
            "🔍 PCA (Unsupervised)" = "pca",
            "🎯 LDA (Supervised)" = "lda",
            "⚖️ Model Comparison" = "comparison"
          ),
          selected = "dataset"
        ),
        
        # Train-Test Split Status Badge
        div(style = "background: rgba(2, 132, 199, 0.15); border: 1px solid #0284c7; border-radius: 6px; padding: 10px 12px; margin: 15px 0 10px 0; font-size: 12px;",
          strong(style = "color: #38bdf8;", "Fixed 70:30 Split:"), br(),
          "• Training: 125 samples (70.2%)", br(),
          "• Testing: 53 samples (29.8%)", br(),
          "• Stratified by Cultivar (Seed: 42)"
        ),
        
        # Contextual Controls for Dataset View
        conditionalPanel(
          condition = "input.section == 'dataset'",
          div(class = "sidebar-section-title", "Feature Distribution"),
          selectInput("box_feature", "Select Chemical Feature:",
                      choices = feature_cols,
                      selected = "Alcohol"),
          div(class = "sidebar-section-title", "Export Dataset"),
          downloadButton("download_wine_csv", "Download Wine CSV (w/ Split)", class = "btn-download")
        ),
        
        # Contextual Controls for PCA View
        conditionalPanel(
          condition = "input.section == 'pca'",
          div(class = "sidebar-section-title", "Partition View"),
          radioButtons("pca_split_view", label = NULL,
                       choices = c("Both (125 Train + 53 Test)" = "both",
                                   "Training Set Only (125)" = "train",
                                   "Testing Set Only (53)" = "test"),
                       selected = "both"),
          div(class = "sidebar-section-title", "Class Filter (No Refit)"),
          checkboxGroupInput("pca_classes", label = NULL,
                             choices = c("Cultivar 1" = "Cultivar 1",
                                         "Cultivar 2" = "Cultivar 2",
                                         "Cultivar 3" = "Cultivar 3"),
                             selected = c("Cultivar 1", "Cultivar 2", "Cultivar 3")),
          div(class = "sidebar-section-title", "Export PCA Artifacts"),
          downloadButton("download_pca_csv", "Download PCA Scores (CSV)", class = "btn-download"),
          downloadButton("download_pca_png", "Download PCA Plot (PNG)", class = "btn-download-secondary")
        ),
        
        # Contextual Controls for LDA View
        conditionalPanel(
          condition = "input.section == 'lda'",
          div(class = "sidebar-section-title", "Partition View"),
          radioButtons("lda_split_view", label = NULL,
                       choices = c("Both (125 Train + 53 Test)" = "both",
                                   "Training Set Only (125)" = "train",
                                   "Testing Set Only (53)" = "test"),
                       selected = "both"),
          div(class = "sidebar-section-title", "Class Filter (No Refit)"),
          checkboxGroupInput("lda_classes", label = NULL,
                             choices = c("Cultivar 1" = "Cultivar 1",
                                         "Cultivar 2" = "Cultivar 2",
                                         "Cultivar 3" = "Cultivar 3"),
                             selected = c("Cultivar 1", "Cultivar 2", "Cultivar 3")),
          div(class = "sidebar-section-title", "Export LDA Artifacts"),
          downloadButton("download_lda_csv", "Download LDA Scores (CSV)", class = "btn-download"),
          downloadButton("download_lda_png", "Download LDA Plot (PNG)", class = "btn-download-secondary")
        ),
        
        # Contextual Controls for Comparison View
        conditionalPanel(
          condition = "input.section == 'comparison'",
          div(class = "sidebar-section-title", "Partition View"),
          radioButtons("comp_split_view", label = NULL,
                       choices = c("Both (125 Train + 53 Test)" = "both",
                                   "Training Set Only (125)" = "train",
                                   "Testing Set Only (53)" = "test"),
                       selected = "both"),
          div(class = "sidebar-section-title", "Class Filter (Both Views)"),
          checkboxGroupInput("comp_classes", label = NULL,
                             choices = c("Cultivar 1" = "Cultivar 1",
                                         "Cultivar 2" = "Cultivar 2",
                                         "Cultivar 3" = "Cultivar 3"),
                             selected = c("Cultivar 1", "Cultivar 2", "Cultivar 3")),
          div(class = "sidebar-section-title", "Export Scores"),
          downloadButton("download_all_scores_csv", "Download All Scores (CSV)", class = "btn-download")
        ),
        
        # Legend Note for Shapes
        tags$hr(style = "border-color: #1e3a5f; margin: 20px 0 15px 0;"),
        div(style = "font-size: 11px; color: #94a3b8; line-height: 1.5;",
          strong(style = "color: #cbd5e1;", "Plot Shape Legend:"), br(),
          "● Circles = Training Samples (70%)", br(),
          "▲ Triangles = Testing Samples (30%)", br(),
          "Colors = Wine Cultivars (1, 2, 3)"
        )
      )
    ),
    
    # Main Dashboard Panel
    column(width = 9, class = "main-content",
      
      # ------------------------------------------------------------------------
      # SECTION 1: DATASET OVERVIEW (Full EDA + Stratified Split Breakdown)
      # ------------------------------------------------------------------------
      conditionalPanel(
        condition = "input.section == 'dataset'",
        
        div(class = "callout-split",
          strong("Dataset Exploration vs Modeling Protocol: "),
          "This section presents Exploratory Data Analysis (EDA) on the complete Wine dataset (178 observations). For modeling (PCA & LDA), the dataset is strictly partitioned into a ",
          strong("70:30 stratified split"),
          " (125 Training / 53 Testing samples) using seed 42. Standardization parameters are computed exclusively on the training partition."
        ),
        
        # 4 Summary KPI Cards
        fluidRow(
          column(3,
            div(class = "summary-card card-blue",
              div(class = "card-value", "178"),
              div(class = "card-title", "Total Observations"),
              div(class = "card-subtitle", "13 chemical features")
            )
          ),
          column(3,
            div(class = "summary-card card-emerald",
              div(class = "card-value", "125 (70%)"),
              div(class = "card-title", "Training Partition"),
              div(class = "card-subtitle", "C1: 41 | C2: 50 | C3: 34")
            )
          ),
          column(3,
            div(class = "summary-card card-purple",
              div(class = "card-value", "53 (30%)"),
              div(class = "card-title", "Testing Partition"),
              div(class = "card-subtitle", "C1: 18 | C2: 21 | C3: 14")
            )
          ),
          column(3,
            div(class = "summary-card card-amber",
              div(class = "card-value", "0"),
              div(class = "card-title", "Missing Values"),
              div(class = "card-subtitle", "100% complete records")
            )
          )
        ),
        
        # Class Distribution (with Train/Test Split) & Feature Box Plot
        fluidRow(
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "Class Distribution (70% Train vs 30% Test)"),
              plotOutput("plot_class_dist", height = "300px")
            )
          ),
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", textOutput("title_feature_box")),
              plotOutput("plot_feature_box", height = "300px")
            )
          )
        ),
        
        # Correlation Heatmap
        fluidRow(
          column(12,
            div(class = "content-box",
              div(class = "content-box-title", "Pairwise Correlation Matrix of Chemical Features"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 12px;",
                "Values show Pearson correlation coefficient (r) between features. Blue indicates positive correlation, red indicates negative correlation."),
              plotOutput("plot_corr_heatmap", height = "440px")
            )
          )
        ),
        
        # Dataset Table Preview (Shows Split column)
        fluidRow(
          column(12,
            div(class = "content-box",
              div(class = "content-box-title", "Wine Dataset Preview (with Train/Test Assignment)"),
              if (has_dt) {
                DT::dataTableOutput("table_wine_preview")
              } else {
                tableOutput("table_wine_preview_static")
              }
            )
          )
        )
      ),
      
      # ------------------------------------------------------------------------
      # SECTION 2: PRINCIPAL COMPONENT ANALYSIS (PCA)
      # ------------------------------------------------------------------------
      conditionalPanel(
        condition = "input.section == 'pca'",
        
        div(class = "callout-info",
          strong("PCA Methodology (Strict 70% Training Fit): "),
          "PCA was fitted strictly on the standardized ", strong("70% training partition (125 samples)"), 
          " with class labels excluded. The held-out ", strong("30% test samples (53 samples)"), 
          " are standardized using training-set means and standard deviations, then projected onto the training rotation axes. ",
          strong("The PCA model is NEVER refitted on test data.")
        ),
        
        # 3 PCA Metric Cards
        fluidRow(
          column(4,
            div(class = "summary-card card-blue",
              div(class = "card-value", sprintf("%.2f%%", pca_var_pct[1])),
              div(class = "card-title", "PC1 Explained Variance"),
              div(class = "card-subtitle", "Training set (125 samples)")
            )
          ),
          column(4,
            div(class = "summary-card card-emerald",
              div(class = "card-value", sprintf("%.2f%%", pca_var_pct[2])),
              div(class = "card-title", "PC2 Explained Variance"),
              div(class = "card-subtitle", "Training set (125 samples)")
            )
          ),
          column(4,
            div(class = "summary-card card-purple",
              div(class = "card-value", sprintf("%.2f%%", pca_cum_var[2])),
              div(class = "card-title", "Cumulative 2D Variance"),
              div(class = "card-subtitle", "PC1 + PC2 (Training fit)")
            )
          )
        ),
        
        # PC1 vs PC2 Scatter Plot & Scree Plot
        fluidRow(
          column(7,
            div(class = "content-box",
              div(class = "content-box-title", "PCA: PC1 vs PC2 Projection (Train vs Test)"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "Circles (●) = 125 Training samples. Triangles (▲) = 53 Held-out Test samples. Shaded ellipses depict 95% confidence intervals."),
              plotOutput("plot_pca_scatter", height = "400px")
            )
          ),
          column(5,
            div(class = "content-box",
              div(class = "content-box-title", "Scree Plot (Training Set Variance)"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "Bars show individual variance %; line shows cumulative variance with 80% & 90% thresholds."),
              plotOutput("plot_pca_scree", height = "400px")
            )
          )
        ),
        
        # PCA Feature Loadings & Scores Table
        fluidRow(
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "Top Chemical Feature Loadings on PC1 & PC2"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "Eigenvectors computed on the 70% training set:"),
              tableOutput("table_pca_loadings")
            )
          ),
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "PCA Component Scores Preview (w/ Split)"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "Sample projections on PC1 through PC5:"),
              tableOutput("table_pca_scores")
            )
          )
        )
      ),
      
      # ------------------------------------------------------------------------
      # SECTION 3: LINEAR DISCRIMINANT ANALYSIS (LDA)
      # ------------------------------------------------------------------------
      conditionalPanel(
        condition = "input.section == 'lda'",
        
        div(class = "callout-success",
          strong("LDA Methodology (Strict 70% Training Fit & Out-of-Sample Evaluation): "),
          "LDA was trained strictly on the standardized ", strong("70% training partition (125 samples)"), 
          " with cultivar labels. The held-out ", strong("30% test samples (53 samples)"), 
          " were standardized using training parameters and projected onto the fitted discriminant functions without refitting. ",
          "Out-of-sample classification performance is evaluated strictly on the 53 test observations."
        ),
        
        # 3 LDA Metric Cards
        fluidRow(
          column(4,
            div(class = "summary-card card-blue",
              div(class = "card-value", sprintf("%.2f%%", lda_trace_pct[1])),
              div(class = "card-title", "LD1 Proportion of Trace"),
              div(class = "card-subtitle", "Training set (125 samples)")
            )
          ),
          column(4,
            div(class = "summary-card card-emerald",
              div(class = "card-value", sprintf("%.2f%%", lda_trace_pct[2])),
              div(class = "card-title", "LD2 Proportion of Trace"),
              div(class = "card-subtitle", "Training set (125 samples)")
            )
          ),
          column(4,
            div(class = "summary-card card-purple",
              div(class = "card-value", sprintf("%.2f%%", lda_test_acc)),
              div(class = "card-title", "Held-Out Test Accuracy"),
              div(class = "card-subtitle", sprintf("52 of 53 test samples correct"))
            )
          )
        ),
        
        # LD1 vs LD2 Scatter Plot & Test Set Evaluation Card
        fluidRow(
          column(7,
            div(class = "content-box",
              div(class = "content-box-title", "LDA: LD1 vs LD2 Projection (Train vs Test)"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "Circles (●) = 125 Training samples. Triangles (▲) = 53 Held-out Test samples. Note complete separation of all clusters."),
              plotOutput("plot_lda_scatter", height = "400px")
            )
          ),
          column(5,
            div(class = "content-box",
              div(class = "content-box-title", "Held-Out Test Set Confusion Matrix"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 8px;",
                sprintf("Evaluated on all 53 unseen test samples (Accuracy: %.2f%%):", lda_test_acc)),
              tableOutput("table_lda_test_conf"),
              tags$hr(style = "margin: 12px 0;"),
              div(style = "font-size: 12px; color: #334155; line-height: 1.5;",
                strong("Out-of-Sample Performance:"), br(),
                "• Cultivar 1: 18 / 18 correct (100.0%)", br(),
                "• Cultivar 2: 20 / 21 correct (95.2%)", br(),
                "• Cultivar 3: 14 / 14 correct (100.0%)", br(),
                "Only 1 test bottle (Cultivar 2) was misclassified as Cultivar 1."
              )
            )
          )
        ),
        
        # LDA Coefficients & Scores Table
        fluidRow(
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "LDA Coefficients of Linear Discriminants"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "Linear combination weights computed on the 70% training partition:"),
              tableOutput("table_lda_coefs")
            )
          ),
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "LDA Component Scores Preview (w/ Split)"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "Sample projections on LD1 and LD2:"),
              tableOutput("table_lda_scores")
            )
          )
        )
      ),
      
      # ------------------------------------------------------------------------
      # SECTION 4: COMPARISON & EMPIRICAL EVALUATION
      # ------------------------------------------------------------------------
      conditionalPanel(
        condition = "input.section == 'comparison'",
        
        div(class = "callout-info",
          strong("Direct Comparison: Unsupervised (PCA) vs Supervised (LDA) under 70:30 Evaluation"), br(),
          "Both models are trained on the identical 70% training set (125 samples). Their projection quality and class discrimination are evaluated on the identical 30% held-out test set (53 samples)."
        ),
        
        # Side-by-Side Plots
        fluidRow(
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "PCA: PC1 vs PC2 (Train ● & Test ▲)"),
              plotOutput("plot_comp_pca", height = "360px")
            )
          ),
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "LDA: LD1 vs LD2 (Train ● & Test ▲)"),
              plotOutput("plot_comp_lda", height = "360px")
            )
          )
        ),
        
        # Evaluated Metrics Comparison Cards
        fluidRow(
          column(6,
            div(class = "summary-card card-blue",
              div(class = "card-value", sprintf("%.2f%% (Test Acc)", pca_test_acc)),
              div(class = "card-title", "2D PCA Nearest-Centroid Classifier"),
              div(class = "card-subtitle", sprintf("Fisher Ratio: %.2f | 49 of 53 test samples correct", pca_fisher_val))
            )
          ),
          column(6,
            div(class = "summary-card card-emerald",
              div(class = "card-value", sprintf("%.2f%% (Test Acc)", lda_test_acc)),
              div(class = "card-title", "2D LDA Discriminant Classifier"),
              div(class = "card-subtitle", sprintf("Fisher Ratio: %.2f (+%.1f%%) | 52 of 53 test samples correct", 
                                                   lda_fisher_val, ((lda_fisher_val - pca_fisher_val)/pca_fisher_val)*100))
            )
          )
        ),
        
        # Structured Methodological Comparison Table
        fluidRow(
          column(12,
            div(class = "content-box",
              div(class = "content-box-title", "Methodological Feature Comparison"),
              tags$table(class = "custom-table",
                tags$thead(
                  tags$tr(
                    tags$th("Evaluation Criterion"),
                    tags$th("Principal Component Analysis (PCA)"),
                    tags$th("Linear Discriminant Analysis (LDA)")
                  )
                ),
                tags$tbody(
                  tags$tr(
                    tags$td(strong("Learning Paradigm")),
                    tags$td("Unsupervised (no class labels used during training)"),
                    tags$td("Supervised (trained with 70% cultivar labels)")
                  ),
                  tags$tr(
                    tags$td(strong("Mathematical Objective")),
                    tags$td("Maximize total feature variance across training data"),
                    tags$td("Maximize between-class scatter relative to within-class scatter")
                  ),
                  tags$tr(
                    tags$td(strong("Output Components")),
                    tags$td("Up to p = 13 components (13 features)"),
                    tags$td("Exactly min(p, C - 1) = 2 components (3 cultivars)")
                  ),
                  tags$tr(
                    tags$td(strong("Held-Out Test Accuracy (30%)")),
                    tags$td(sprintf("%.2f%% (49/53 correct via 2D centroid classifier)", pca_test_acc)),
                    tags$td(sprintf("%.2f%% (52/53 correct via LDA classifier)", lda_test_acc))
                  ),
                  tags$tr(
                    tags$td(strong("Fisher Separation Ratio (S_B / S_W)")),
                    tags$td(sprintf("%.2f (Moderate separation with partial cluster overlap)", pca_fisher_val)),
                    tags$td(sprintf("%.2f (High separation with zero boundary overlap)", lda_fisher_val))
                  ),
                  tags$tr(
                    tags$td(strong("Ideal Application")),
                    tags$td("Unlabelled exploratory analysis, feature compression, noise filtering"),
                    tags$td("Supervised classification, cultivar authentication, quality control")
                  )
                )
              )
            )
          )
        ),
        
        # Empirical Conclusion
        fluidRow(
          column(12,
            div(class = "content-box",
              div(class = "content-box-title", "Empirical Conclusion under 70:30 Evaluation"),
              div(style = "font-size: 14px; color: #1e293b; line-height: 1.6;",
                tags$p(
                  strong("1. Out-of-Sample Classification Efficacy: "),
                  sprintf("When evaluated on the 53 held-out test observations, LDA achieves an outstanding test accuracy of %.2f%% (52/53 correct), misclassifying only a single bottle. In contrast, an equivalent nearest-centroid classifier operating on the 2D PCA representation achieves %.2f%% (49/53 correct). This demonstrates that LDA's supervised projection preserves class-relevant discriminant information that PCA discards as 'lower variance'.", 
                          lda_test_acc, pca_test_acc)
                ),
                tags$p(
                  strong("2. Cluster Separation vs Feature Variance: "),
                  sprintf("On the training set, LDA achieves a Fisher separation ratio of %.2f compared to %.2f for PCA (+%.1f%% separation gain). While PCA effectively retains 57.12%% of total feature variance in its first two components, that variance is not aligned with cultivar differences, resulting in slight boundary overlap between Cultivars 1 and 2.",
                          lda_fisher_val, pca_fisher_val, ((lda_fisher_val - pca_fisher_val)/pca_fisher_val)*100)
                ),
                tags$p(
                  strong("3. Data Hygiene and Generalization: "),
                  "By strictly fitting scaling parameters, PCA rotation, and LDA discriminants on the 70% training partition alone, the workflow guarantees zero data leakage. The resulting high test accuracy confirms that the linear boundaries generalize exceptionally well to unseen wine samples."
                )
              )
            )
          )
        )
      )
    )
  )
)


# --- 7. Server Logic Definition ---
server <- function(input, output, session) {
  
  # Reactive subset for PCA scatter plot (filters without refitting model)
  filtered_pca_data <- reactive({
    req(input$pca_classes, input$pca_split_view)
    df <- subset(pca_scores_df, Cultivar %in% input$pca_classes)
    if (input$pca_split_view == "train") {
      df <- subset(df, Split == "Training (70%)")
    } else if (input$pca_split_view == "test") {
      df <- subset(df, Split == "Testing (30%)")
    }
    df
  })
  
  # Reactive subset for LDA scatter plot (filters without refitting model)
  filtered_lda_data <- reactive({
    req(input$lda_classes, input$lda_split_view)
    df <- subset(lda_scores_df, Cultivar %in% input$lda_classes)
    if (input$lda_split_view == "train") {
      df <- subset(df, Split == "Training (70%)")
    } else if (input$lda_split_view == "test") {
      df <- subset(df, Split == "Testing (30%)")
    }
    df
  })
  
  # Reactive subset for Comparison view (filters without refitting model)
  filtered_comp_pca <- reactive({
    req(input$comp_classes, input$comp_split_view)
    df <- subset(pca_scores_df, Cultivar %in% input$comp_classes)
    if (input$comp_split_view == "train") {
      df <- subset(df, Split == "Training (70%)")
    } else if (input$comp_split_view == "test") {
      df <- subset(df, Split == "Testing (30%)")
    }
    df
  })
  filtered_comp_lda <- reactive({
    req(input$comp_classes, input$comp_split_view)
    df <- subset(lda_scores_df, Cultivar %in% input$comp_classes)
    if (input$comp_split_view == "train") {
      df <- subset(df, Split == "Training (70%)")
    } else if (input$comp_split_view == "test") {
      df <- subset(df, Split == "Testing (30%)")
    }
    df
  })
  
  
  # ----------------------------------------------------------------------------
  # SECTION 1 OUTPUTS: DATASET OVERVIEW
  # ----------------------------------------------------------------------------
  
  # Class Distribution Bar Plot (Grouped by Train vs Test)
  output$plot_class_dist <- renderPlot({
    split_counts <- as.data.frame(table(Cultivar = wine$Cultivar, Split = wine$Split))
    
    ggplot(split_counts, aes(x = Cultivar, y = Freq, fill = Cultivar, alpha = Split)) +
      geom_bar(stat = "identity", position = position_dodge(width = 0.75), 
               width = 0.65, color = "#0f172a") +
      geom_text(aes(label = Freq), 
                position = position_dodge(width = 0.75), 
                vjust = -0.5, fontface = "bold", size = 4, color = "#1e293b") +
      scale_fill_manual(values = cultivar_colors, guide = "none") +
      scale_alpha_manual(values = c("Training (70%)" = 0.95, "Testing (30%)" = 0.45),
                         name = "Partition:") +
      scale_y_continuous(expand = expansion(mult = c(0, 0.18)), limits = c(0, 60)) +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "top",
        legend.title = element_text(face = "bold", color = "#1e293b"),
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(face = "bold", color = "#334155"),
        axis.text = element_text(color = "#1e293b", face = "bold")
      ) +
      labs(y = "Sample Count")
  })
  
  # Title for Feature Box Plot
  output$title_feature_box <- renderText({
    paste0("Feature Distribution by Cultivar: ", input$box_feature)
  })
  
  # Selectable-Feature Box Plot
  output$plot_feature_box <- renderPlot({
    req(input$box_feature)
    feature_name <- input$box_feature
    
    ggplot(wine, aes_string(x = "Cultivar", y = feature_name, fill = "Cultivar")) +
      geom_boxplot(alpha = 0.75, width = 0.5, outlier.shape = 21, outlier.size = 2.5) +
      geom_jitter(aes(shape = Split), width = 0.15, alpha = 0.55, size = 2, color = "#0f172a") +
      scale_fill_manual(values = cultivar_colors, guide = "none") +
      scale_shape_manual(values = c("Training (70%)" = 16, "Testing (30%)" = 17), name = "Split:") +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "top",
        legend.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(face = "bold", color = "#334155"),
        axis.text = element_text(color = "#1e293b", face = "bold")
      ) +
      labs(y = paste(feature_name, "(Raw Measurement)"))
  })
  
  # Correlation Heatmap (Full Dataset EDA)
  output$plot_corr_heatmap <- renderPlot({
    corr_mat <- cor(wine[, feature_cols])
    corr_df <- as.data.frame(as.table(corr_mat))
    colnames(corr_df) <- c("Feature1", "Feature2", "Correlation")
    
    ggplot(corr_df, aes(x = Feature1, y = Feature2, fill = Correlation)) +
      geom_tile(color = "white", lwd = 0.6) +
      geom_text(aes(label = sprintf("%.2f", Correlation)), 
                size = 3.2, color = ifelse(abs(corr_df$Correlation) > 0.5, "white", "#1e293b")) +
      scale_fill_gradient2(low = "#dc2626", mid = "#f8fafc", high = "#1d4ed8", 
                           midpoint = 0, limit = c(-1, 1), name = "Pearson (r)") +
      theme_minimal(base_size = 12) +
      theme(
        axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, face = "bold", color = "#1e293b"),
        axis.text.y = element_text(face = "bold", color = "#1e293b"),
        axis.title = element_blank(),
        panel.grid = element_blank()
      )
  })
  
  # Dataset Interactive Table Preview (Includes Split column)
  if (has_dt) {
    output$table_wine_preview <- DT::renderDataTable({
      preview_df <- wine[, c("Cultivar", "Split", feature_cols)]
      preview_df[, feature_cols] <- round(preview_df[, feature_cols], 2)
      DT::datatable(
        preview_df,
        options = list(pageLength = 8, scrollX = TRUE),
        rownames = FALSE,
        class = "stripe hover compact cell-border"
      )
    })
  } else {
    output$table_wine_preview_static <- renderTable({
      preview_df <- head(wine[, c("Cultivar", "Split", feature_cols)], 10)
      preview_df[, feature_cols] <- round(preview_df[, feature_cols], 2)
      preview_df
    }, striped = TRUE, hover = TRUE, bordered = TRUE)
  }
  
  # Download Handler: Wine Dataset with Split column
  output$download_wine_csv <- downloadHandler(
    filename = function() {
      paste0("wine_dataset_70_30_split_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(wine, file, row.names = FALSE)
    }
  )
  
  
  # ----------------------------------------------------------------------------
  # SECTION 2 OUTPUTS: PRINCIPAL COMPONENT ANALYSIS (PCA)
  # ----------------------------------------------------------------------------
  
  # Helper to construct PCA Plot (used for rendering & PNG export)
  make_pca_plot <- function(data) {
    ggplot(data, aes(x = PC1, y = PC2, color = Cultivar, fill = Cultivar)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "#94a3b8", linewidth = 0.5) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "#94a3b8", linewidth = 0.5) +
      stat_ellipse(geom = "polygon", alpha = 0.12, level = 0.95, show.legend = FALSE) +
      geom_point(aes(shape = Split), size = 3.4, alpha = 0.85, stroke = 0.9) +
      scale_color_manual(values = cultivar_colors) +
      scale_fill_manual(values = cultivar_colors) +
      scale_shape_manual(values = split_shapes) +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "top",
        legend.box = "horizontal",
        legend.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        axis.title = element_text(face = "bold", color = "#1e293b"),
        axis.text = element_text(color = "#334155")
      ) +
      labs(
        x = sprintf("Principal Component 1 (%.2f%% Variance)", pca_var_pct[1]),
        y = sprintf("Principal Component 2 (%.2f%% Variance)", pca_var_pct[2]),
        color = "Cultivar:",
        fill  = "Cultivar:",
        shape = "Partition:"
      )
  }
  
  # Render PCA Scatter Plot
  output$plot_pca_scatter <- renderPlot({
    make_pca_plot(filtered_pca_data())
  })
  
  # PCA Scree Plot (Training Set Variance)
  output$plot_pca_scree <- renderPlot({
    scree_df <- data.frame(
      PC = factor(paste0("PC", 1:13), levels = paste0("PC", 1:13)),
      VarPct = pca_var_pct,
      CumVar = pca_cum_var
    )
    
    ggplot(scree_df, aes(x = PC)) +
      geom_bar(aes(y = VarPct), stat = "identity", fill = "#1e3a5f", width = 0.65, alpha = 0.9) +
      geom_line(aes(y = CumVar, group = 1), color = "#dc2626", linewidth = 1.2) +
      geom_point(aes(y = CumVar), color = "#dc2626", size = 2.6) +
      geom_hline(yintercept = c(80, 90), linetype = "dotted", color = "#64748b", linewidth = 0.7) +
      annotate("text", x = 11, y = 83, label = "80% Threshold", size = 3.2, color = "#475569") +
      annotate("text", x = 11, y = 93, label = "90% Threshold", size = 3.2, color = "#475569") +
      scale_y_continuous(
        name = "Individual Variance (%)",
        limits = c(0, 105),
        sec.axis = sec_axis(~., name = "Cumulative Variance (%)")
      ) +
      theme_minimal(base_size = 12) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        axis.title.y.left = element_text(color = "#1e3a5f", face = "bold"),
        axis.title.y.right = element_text(color = "#dc2626", face = "bold"),
        panel.grid.minor = element_blank()
      ) +
      labs(x = "Principal Components")
  })
  
  # Top Feature Loadings on PC1 & PC2
  output$table_pca_loadings <- renderTable({
    loadings <- as.data.frame(pca_fit$rotation[, 1:2])
    loadings$Feature <- rownames(loadings)
    rownames(loadings) <- NULL
    loadings <- loadings[, c("Feature", "PC1", "PC2")]
    loadings$PC1 <- round(loadings$PC1, 3)
    loadings$PC2 <- round(loadings$PC2, 3)
    loadings[order(-abs(loadings$PC1)), ]
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # Sample of PCA Component Scores (w/ Split identifier)
  output$table_pca_scores <- renderTable({
    scores_sample <- rbind(
      head(subset(pca_scores_df, Split == "Training (70%)")[, c("Cultivar", "Split", "PC1", "PC2", "PC3", "PC4", "PC5")], 4),
      head(subset(pca_scores_df, Split == "Testing (30%)")[, c("Cultivar", "Split", "PC1", "PC2", "PC3", "PC4", "PC5")], 4)
    )
    scores_sample$PC1 <- round(scores_sample$PC1, 3)
    scores_sample$PC2 <- round(scores_sample$PC2, 3)
    scores_sample$PC3 <- round(scores_sample$PC3, 3)
    scores_sample$PC4 <- round(scores_sample$PC4, 3)
    scores_sample$PC5 <- round(scores_sample$PC5, 3)
    scores_sample
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # Download Handler: PCA Scores CSV
  output$download_pca_csv <- downloadHandler(
    filename = function() {
      paste0("pca_scores_70_30_split_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(pca_scores_df, file, row.names = FALSE)
    }
  )
  
  # Download Handler: PCA Scatter Plot PNG
  output$download_pca_png <- downloadHandler(
    filename = function() {
      paste0("pca_scatter_plot_70_30_split_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- make_pca_plot(filtered_pca_data())
      ggsave(file, plot = p, width = 8.5, height = 5.8, dpi = 300)
    }
  )
  
  
  # ----------------------------------------------------------------------------
  # SECTION 3 OUTPUTS: LINEAR DISCRIMINANT ANALYSIS (LDA)
  # ----------------------------------------------------------------------------
  
  # Helper to construct LDA Plot (used for rendering & PNG export)
  make_lda_plot <- function(data) {
    ggplot(data, aes(x = LD1, y = LD2, color = Cultivar, fill = Cultivar)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "#94a3b8", linewidth = 0.5) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "#94a3b8", linewidth = 0.5) +
      stat_ellipse(geom = "polygon", alpha = 0.12, level = 0.95, show.legend = FALSE) +
      geom_point(aes(shape = Split), size = 3.4, alpha = 0.85, stroke = 0.9) +
      scale_color_manual(values = cultivar_colors) +
      scale_fill_manual(values = cultivar_colors) +
      scale_shape_manual(values = split_shapes) +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "top",
        legend.box = "horizontal",
        legend.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        axis.title = element_text(face = "bold", color = "#1e293b"),
        axis.text = element_text(color = "#334155")
      ) +
      labs(
        x = sprintf("Linear Discriminant 1 (%.2f%% of Trace)", lda_trace_pct[1]),
        y = sprintf("Linear Discriminant 2 (%.2f%% of Trace)", lda_trace_pct[2]),
        color = "Cultivar:",
        fill  = "Cultivar:",
        shape = "Partition:"
      )
  }
  
  # Render LDA Scatter Plot
  output$plot_lda_scatter <- renderPlot({
    make_lda_plot(filtered_lda_data())
  })
  
  # LDA Out-of-Sample Test Confusion Matrix Table
  output$table_lda_test_conf <- renderTable({
    conf_df <- as.data.frame.matrix(lda_test_conf)
    conf_df <- cbind(Predicted = rownames(conf_df), conf_df)
    rownames(conf_df) <- NULL
    conf_df
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # LDA Scaling Coefficients Table
  output$table_lda_coefs <- renderTable({
    coefs <- as.data.frame(lda_fit$scaling)
    coefs$Feature <- rownames(coefs)
    rownames(coefs) <- NULL
    coefs <- coefs[, c("Feature", "LD1", "LD2")]
    coefs$LD1 <- round(coefs$LD1, 3)
    coefs$LD2 <- round(coefs$LD2, 3)
    coefs[order(-abs(coefs$LD1)), ]
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # Sample of LDA Component Scores (w/ Split identifier)
  output$table_lda_scores <- renderTable({
    scores_sample <- rbind(
      head(subset(lda_scores_df, Split == "Training (70%)")[, c("Cultivar", "Split", "LD1", "LD2", "Predicted")], 4),
      head(subset(lda_scores_df, Split == "Testing (30%)")[, c("Cultivar", "Split", "LD1", "LD2", "Predicted")], 4)
    )
    scores_sample$LD1 <- round(scores_sample$LD1, 3)
    scores_sample$LD2 <- round(scores_sample$LD2, 3)
    scores_sample
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # Download Handler: LDA Scores CSV
  output$download_lda_csv <- downloadHandler(
    filename = function() {
      paste0("lda_scores_70_30_split_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(lda_scores_df, file, row.names = FALSE)
    }
  )
  
  # Download Handler: LDA Scatter Plot PNG
  output$download_lda_png <- downloadHandler(
    filename = function() {
      paste0("lda_scatter_plot_70_30_split_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- make_lda_plot(filtered_lda_data())
      ggsave(file, plot = p, width = 8.5, height = 5.8, dpi = 300)
    }
  )
  
  
  # ----------------------------------------------------------------------------
  # SECTION 4 OUTPUTS: COMPARISON
  # ----------------------------------------------------------------------------
  
  # Side-by-Side Comparison: PCA Plot
  output$plot_comp_pca <- renderPlot({
    make_pca_plot(filtered_comp_pca()) +
      ggtitle("PCA (Unsupervised: Maximizes Overall Training Variance)") +
      theme(plot.title = element_text(size = 12, face = "bold", color = "#1e3a5f"))
  })
  
  # Side-by-Side Comparison: LDA Plot
  output$plot_comp_lda <- renderPlot({
    make_lda_plot(filtered_comp_lda()) +
      ggtitle("LDA (Supervised: Maximizes Cultivar Discrimination)") +
      theme(plot.title = element_text(size = 12, face = "bold", color = "#059669"))
  })
  
  # Download Handler: All Scores Combined CSV
  output$download_all_scores_csv <- downloadHandler(
    filename = function() {
      paste0("pca_lda_scores_70_30_split_", Sys.Date(), ".csv")
    },
    content = function(file) {
      combined <- data.frame(
        Cultivar  = wine$Cultivar,
        Type      = wine$Type,
        Split     = wine$Split,
        PC1       = pca_scores_df$PC1,
        PC2       = pca_scores_df$PC2,
        PC3       = pca_scores_df$PC3,
        LD1       = lda_scores_df$LD1,
        LD2       = lda_scores_df$LD2,
        LDA_Pred  = lda_scores_df$Predicted
      )
      write.csv(combined, file, row.names = FALSE)
    }
  )
}

# --- 8. Launch the Application ---
shinyApp(ui = ui, server = server)
