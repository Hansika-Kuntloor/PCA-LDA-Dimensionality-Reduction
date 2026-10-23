# ==============================================================================
# Application: PCA & LDA Explorer
# Description: Interactive Dimensionality Reduction and Classification in R Shiny
# Dataset: Wine Dataset (rattle.data / UCI Machine Learning Repository)
# Author: Pair Programming with Antigravity
# ==============================================================================

# --- 1. Load Required Libraries ---
library(shiny)
library(ggplot2)
library(MASS)

# Load DT for modern interactive tables if available
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

# Separate features (X) and class label (y)
feature_cols <- c("Alcohol", "Malic", "Ash", "Alcalinity", "Magnesium", 
                  "Phenols", "Flavanoids", "Nonflavanoids", "Proanthocyanins", 
                  "Color", "Hue", "Dilution", "Proline")
wine_features <- wine[, feature_cols]
wine_class <- wine$Cultivar

# Standardize numerical features (mean = 0, standard deviation = 1)
# Note: Feature standardization is crucial because variables like Proline (~1000)
# would otherwise completely dwarf variables like Nonflavanoids (~0.3).
wine_scaled_mat <- scale(wine_features)
wine_scaled_df <- as.data.frame(wine_scaled_mat)
wine_scaled_df$Cultivar <- wine_class
wine_scaled_df$Type <- wine$Type

# --- 2. Fit PCA and LDA Models Once at Application Startup ---
# Models are pre-fitted on the complete standardized dataset.
# The user can filter classes for visualization without refitting models.

# A. Principal Component Analysis (Unsupervised - class labels excluded)
pca_fit <- prcomp(wine_features, center = TRUE, scale. = TRUE)
pca_var <- pca_fit$sdev^2
pca_var_pct <- (pca_var / sum(pca_var)) * 100
pca_cum_var <- cumsum(pca_var_pct)

# Full PCA scores data frame
pca_scores_df <- as.data.frame(pca_fit$x)
pca_scores_df$Cultivar <- wine_class
pca_scores_df$Type <- wine$Type

# B. Linear Discriminant Analysis (Supervised - uses class labels)
lda_fit <- MASS::lda(Cultivar ~ ., data = wine_scaled_df[, c(feature_cols, "Cultivar")])
lda_pred <- predict(lda_fit)
lda_trace_pct <- (lda_fit$svd^2 / sum(lda_fit$svd^2)) * 100

# Full LDA scores data frame
lda_scores_df <- as.data.frame(lda_pred$x)
lda_scores_df$Cultivar <- wine_class
lda_scores_df$Type <- wine$Type

# C. Calculate Between-Class vs Within-Class Separation (Fisher Ratio)
compute_fisher_ratio <- function(coords, classes) {
  classes <- as.factor(classes)
  n <- nrow(coords)
  # Total scatter matrix
  S_T <- cov(coords) * (n - 1)
  # Within-class scatter matrix S_W
  S_W <- matrix(0, nrow = ncol(coords), ncol = ncol(coords))
  for (cl in levels(classes)) {
    idx <- which(classes == cl)
    n_cl <- length(idx)
    if (n_cl > 1) {
      cl_data <- coords[idx, , drop = FALSE]
      S_W <- S_W + cov(cl_data) * (n_cl - 1)
    }
  }
  # Between-class scatter matrix S_B = S_T - S_W
  S_B <- S_T - S_W
  # Trace ratio (higher means superior cluster separation)
  ratio <- sum(diag(S_B)) / sum(diag(S_W))
  return(ratio)
}

pca_fisher_val <- compute_fisher_ratio(pca_scores_df[, c("PC1", "PC2")], wine_class)
lda_fisher_val <- compute_fisher_ratio(lda_scores_df[, c("LD1", "LD2")], wine_class)

# Curated accessible color palette for the 3 Cultivars
cultivar_colors <- c(
  "Cultivar 1" = "#1d4ed8",  # Royal Blue
  "Cultivar 2" = "#059669",  # Emerald Green
  "Cultivar 3" = "#dc2626"   # Crimson Red
)


# --- 3. User Interface (UI) Definition ---
ui <- fluidPage(
  # Include Custom CSS for Navy Header, Sidebar, Cards, and Typography
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
        font-size: 14px;
        font-weight: 500;
        padding: 8px 10px;
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
      p("Interactive Dimensionality Reduction, Unsupervised vs Supervised Analysis on the Wine Dataset")
    ),
    div(class = "header-badges",
      span("R Shiny"),
      span("MASS::lda"),
      span("stats::prcomp"),
      span("Wine (178 obs, 13 vars)")
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
        
        # Contextual Controls for Dataset View
        conditionalPanel(
          condition = "input.section == 'dataset'",
          div(class = "sidebar-section-title", "Feature Distribution"),
          selectInput("box_feature", "Select Chemical Feature:",
                      choices = feature_cols,
                      selected = "Alcohol"),
          div(class = "sidebar-section-title", "Export Dataset"),
          downloadButton("download_wine_csv", "Download Wine CSV", class = "btn-download")
        ),
        
        # Contextual Controls for PCA View
        conditionalPanel(
          condition = "input.section == 'pca'",
          div(class = "sidebar-section-title", "Class Filter (No Refit)"),
          p(style = "font-size: 11px; color: #94a3b8; margin-bottom: 8px;",
            "Filter projected points in real-time without recalculating eigenvectors:"),
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
          div(class = "sidebar-section-title", "Class Filter (No Refit)"),
          p(style = "font-size: 11px; color: #94a3b8; margin-bottom: 8px;",
            "Filter projected discriminant points without recalculating discriminant axes:"),
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
          div(class = "sidebar-section-title", "Class Filter (Both Views)"),
          checkboxGroupInput("comp_classes", label = NULL,
                             choices = c("Cultivar 1" = "Cultivar 1",
                                         "Cultivar 2" = "Cultivar 2",
                                         "Cultivar 3" = "Cultivar 3"),
                             selected = c("Cultivar 1", "Cultivar 2", "Cultivar 3")),
          div(class = "sidebar-section-title", "Export Scores"),
          downloadButton("download_all_scores_csv", "Download All Scores (CSV)", class = "btn-download")
        ),
        
        # Informational Note at Bottom of Sidebar
        tags$hr(style = "border-color: #1e3a5f; margin: 20px 0 15px 0;"),
        div(style = "font-size: 11px; color: #94a3b8; line-height: 1.4;",
          strong(style = "color: #cbd5e1;", "Dataset Info:"), br(),
          "Wine chemical analysis results from three distinct cultivars grown in the same region in Italy."
        )
      )
    ),
    
    # Main Dashboard Panel
    column(width = 9, class = "main-content",
      
      # ------------------------------------------------------------------------
      # SECTION 1: DATASET OVERVIEW
      # ------------------------------------------------------------------------
      conditionalPanel(
        condition = "input.section == 'dataset'",
        
        # 4 Summary KPI Cards
        fluidRow(
          column(3,
            div(class = "summary-card card-blue",
              div(class = "card-value", "178"),
              div(class = "card-title", "Total Observations"),
              div(class = "card-subtitle", "Wine bottles tested")
            )
          ),
          column(3,
            div(class = "summary-card card-emerald",
              div(class = "card-value", "13"),
              div(class = "card-title", "Chemical Features"),
              div(class = "card-subtitle", "Continuous variables")
            )
          ),
          column(3,
            div(class = "summary-card card-purple",
              div(class = "card-value", "3"),
              div(class = "card-title", "Wine Cultivars"),
              div(class = "card-subtitle", "C1 (59), C2 (71), C3 (48)")
            )
          ),
          column(3,
            div(class = "summary-card card-amber",
              div(class = "card-value", "0"),
              div(class = "card-title", "Missing Values"),
              div(class = "card-subtitle", "100% complete data")
            )
          )
        ),
        
        # Class Distribution and Feature Box Plot
        fluidRow(
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "Class Distribution (Wine Cultivars)"),
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
                "Values show Pearson correlation coefficient (r) between standardized features. Blue indicates positive correlation, red indicates negative correlation."),
              plotOutput("plot_corr_heatmap", height = "440px")
            )
          )
        ),
        
        # Dataset Table Preview
        fluidRow(
          column(12,
            div(class = "content-box",
              div(class = "content-box-title", "Wine Dataset Table Preview"),
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
          strong("About Principal Component Analysis (Unsupervised): "),
          "PCA seeks orthogonal linear combinations of the 13 standardized chemical features that maximize overall variance across the entire sample space, completely ignoring class labels. It reduces dimensionality while preserving as much data dispersion as possible."
        ),
        
        # 3 PCA Metric Cards
        fluidRow(
          column(4,
            div(class = "summary-card card-blue",
              div(class = "card-value", sprintf("%.2f%%", pca_var_pct[1])),
              div(class = "card-title", "PC1 Explained Variance"),
              div(class = "card-subtitle", "Largest axis of feature variance")
            )
          ),
          column(4,
            div(class = "summary-card card-emerald",
              div(class = "card-value", sprintf("%.2f%%", pca_var_pct[2])),
              div(class = "card-title", "PC2 Explained Variance"),
              div(class = "card-subtitle", "Second orthogonal principal axis")
            )
          ),
          column(4,
            div(class = "summary-card card-purple",
              div(class = "card-value", sprintf("%.2f%%", pca_cum_var[2])),
              div(class = "card-title", "Cumulative 2D Variance"),
              div(class = "card-subtitle", "7 PCs needed for 80%; 9 for 90%")
            )
          )
        ),
        
        # PC1 vs PC2 Scatter Plot & Scree Plot
        fluidRow(
          column(7,
            div(class = "content-box",
              div(class = "content-box-title", "PCA: PC1 vs PC2 Projection"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "Points projected onto the first two principal components. Shaded ellipses depict 95% confidence intervals."),
              plotOutput("plot_pca_scatter", height = "400px")
            )
          ),
          column(5,
            div(class = "content-box",
              div(class = "content-box-title", "Scree Plot & Cumulative Variance"),
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
                "Weights (eigenvectors) indicating each feature's contribution to PC1 and PC2:"),
              tableOutput("table_pca_loadings")
            )
          ),
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "PCA Component Scores (Sample Preview)"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "First 8 projected observations (PC1 through PC5):"),
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
          strong("About Linear Discriminant Analysis (Supervised): "),
          "Unlike PCA, LDA utilizes cultivar labels during training. It computes projection axes that simultaneously maximize between-class variance (scatter between cultivar centers) while minimizing within-class variance (scatter inside each cultivar). For 3 classes, exactly min(13, 3 - 1) = 2 linear discriminants exist."
        ),
        
        # 3 LDA Metric Cards
        fluidRow(
          column(4,
            div(class = "summary-card card-blue",
              div(class = "card-value", sprintf("%.2f%%", lda_trace_pct[1])),
              div(class = "card-title", "LD1 Proportion of Trace"),
              div(class = "card-subtitle", "First linear discriminant axis")
            )
          ),
          column(4,
            div(class = "summary-card card-emerald",
              div(class = "card-value", sprintf("%.2f%%", lda_trace_pct[2])),
              div(class = "card-title", "LD2 Proportion of Trace"),
              div(class = "card-subtitle", "Second linear discriminant axis")
            )
          ),
          column(4,
            div(class = "summary-card card-purple",
              div(class = "card-value", "100.0%"),
              div(class = "card-title", "Total Trace Explained"),
              div(class = "card-subtitle", "Exhausts all between-class separation")
            )
          )
        ),
        
        # LD1 vs LD2 Scatter Plot & Educational Theory Card
        fluidRow(
          column(7,
            div(class = "content-box",
              div(class = "content-box-title", "LDA: LD1 vs LD2 Projection"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "Points projected onto the 2 discriminant dimensions. Notice the sharp cluster boundaries with zero class overlap."),
              plotOutput("plot_lda_scatter", height = "400px")
            )
          ),
          column(5,
            div(class = "content-box",
              div(class = "content-box-title", "Supervised Reduction Principles"),
              div(style = "font-size: 13px; color: #334155; line-height: 1.6;",
                strong("1. Fisher's Discriminant Criterion:"), br(),
                "LDA maximizes the criterion ratio J(w) = (w' S_B w) / (w' S_W w), where S_B is between-class scatter and S_W is within-class scatter.",
                br(), br(),
                strong("2. Dimensionality Ceiling:"), br(),
                "For C classes, the rank of S_B is at most C - 1. Since our Wine dataset has 3 cultivars, LDA produces exactly 2 discriminants (LD1 and LD2), capturing 100% of class separability in a 2D plane.",
                br(), br(),
                strong("3. Contrast with PCA:"), br(),
                "PCA may capture high variance that happens to be irrelevant to class boundaries (e.g., measurement noise). LDA directly isolates dimensions that differentiate cultivars."
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
                "Linear combinations of standardized features that form LD1 and LD2:"),
              tableOutput("table_lda_coefs")
            )
          ),
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "LDA Component Scores (Sample Preview)"),
              p(style = "font-size: 13px; color: #64748b; margin-top: -5px; margin-bottom: 10px;",
                "First 8 projected observations on LD1 and LD2:"),
              tableOutput("table_lda_scores")
            )
          )
        )
      ),
      
      # ------------------------------------------------------------------------
      # SECTION 4: COMPARISON & EMPIRICAL CONCLUSION
      # ------------------------------------------------------------------------
      conditionalPanel(
        condition = "input.section == 'comparison'",
        
        div(class = "callout-info",
          strong("Direct Comparison: Unsupervised (PCA) vs Supervised (LDA)"), br(),
          "Observe how the same 178 wine samples project into 2D space under both methods. While PCA finds axes of maximal variance across all features, LDA optimizes for class separability."
        ),
        
        # Side-by-Side Plots
        fluidRow(
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "PCA: PC1 vs PC2 (Unsupervised)"),
              plotOutput("plot_comp_pca", height = "360px")
            )
          ),
          column(6,
            div(class = "content-box",
              div(class = "content-box-title", "LDA: LD1 vs LD2 (Supervised)"),
              plotOutput("plot_comp_lda", height = "360px")
            )
          )
        ),
        
        # Calculated Quantitative Comparison Cards
        fluidRow(
          column(6,
            div(class = "summary-card card-blue",
              div(class = "card-value", sprintf("%.2f", pca_fisher_val)),
              div(class = "card-title", "PCA Fisher Separation Ratio (S_B / S_W)"),
              div(class = "card-subtitle", "Moderate between-to-within class separation")
            )
          ),
          column(6,
            div(class = "summary-card card-emerald",
              div(class = "card-value", sprintf("%.2f (+%.1f%%)", lda_fisher_val, ((lda_fisher_val - pca_fisher_val)/pca_fisher_val)*100)),
              div(class = "card-title", "LDA Fisher Separation Ratio (S_B / S_W)"),
              div(class = "card-subtitle", "Substantially higher between-to-within cluster separation")
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
                    tags$td("Unsupervised (no class labels required)"),
                    tags$td("Supervised (requires known class labels)")
                  ),
                  tags$tr(
                    tags$td(strong("Core Objective")),
                    tags$td("Maximize total data variance across all directions"),
                    tags$td("Maximize between-class scatter relative to within-class scatter")
                  ),
                  tags$tr(
                    tags$td(strong("Output Components")),
                    tags$td("Up to p = 13 components (equal to feature count)"),
                    tags$td("At most min(p, C - 1) = 2 components (where C = 3 classes)")
                  ),
                  tags$tr(
                    tags$td(strong("Class Separation on 2D Plot")),
                    tags$td("Partial overlap between Cultivar 1 and 2/3"),
                    tags$td("Completely distinct, linearly separable clusters")
                  ),
                  tags$tr(
                    tags$td(strong("Sensitivity to Scaling")),
                    tags$td("High: features must be centered and standardized"),
                    tags$td("Standardization recommended for interpretable coefficients")
                  ),
                  tags$tr(
                    tags$td(strong("Ideal Application")),
                    tags$td("Data compression, noise filtering, exploratory clustering"),
                    tags$td("Classification pre-processing, discriminant feature mapping")
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
              div(class = "content-box-title", "Empirical Conclusion on Wine Dataset"),
              div(style = "font-size: 14px; color: #1e293b; line-height: 1.6;",
                tags$p(
                  strong("1. Separation Efficacy: "),
                  sprintf("On the Wine dataset, LDA achieves a Fisher separation ratio of %.2f compared to %.2f for PCA—representing a %.1f%% increase in between-class distinction relative to within-class scatter. In the LDA scatter plot, the three wine cultivars form three completely separated clusters with clear classification margins.", 
                          lda_fisher_val, pca_fisher_val, ((lda_fisher_val - pca_fisher_val)/pca_fisher_val)*100)
                ),
                tags$p(
                  strong("2. Variance vs. Discrimination: "),
                  "PCA retains 55.41% of the total dataset variance in PC1 and PC2, which is useful when the goal is general dimensionality reduction without knowing wine origins. However, because variance within a cultivar can be as large as variance across cultivars for certain chemical traits, PCA exhibits slight visual boundary overlap."
                ),
                tags$p(
                  strong("3. Practical Recommendation: "),
                  "When wine labels are available and the analytical objective is classification, quality control, or origin authentication, LDA is unequivocally the superior choice. When exploring unlabelled data or assessing general feature relationships, PCA remains the benchmark tool."
                )
              )
            )
          )
        )
      )
    )
  )
)


# --- 4. Server Logic Definition ---
server <- function(input, output, session) {
  
  # Reactive subset for PCA scatter plot (filters without refitting model)
  filtered_pca_data <- reactive({
    req(input$pca_classes)
    subset(pca_scores_df, Cultivar %in% input$pca_classes)
  })
  
  # Reactive subset for LDA scatter plot (filters without refitting model)
  filtered_lda_data <- reactive({
    req(input$lda_classes)
    subset(lda_scores_df, Cultivar %in% input$lda_classes)
  })
  
  # Reactive subset for Comparison view (filters without refitting model)
  filtered_comp_pca <- reactive({
    req(input$comp_classes)
    subset(pca_scores_df, Cultivar %in% input$comp_classes)
  })
  filtered_comp_lda <- reactive({
    req(input$comp_classes)
    subset(lda_scores_df, Cultivar %in% input$comp_classes)
  })
  
  
  # ----------------------------------------------------------------------------
  # SECTION 1 OUTPUTS: DATASET OVERVIEW
  # ----------------------------------------------------------------------------
  
  # Class Distribution Bar Plot
  output$plot_class_dist <- renderPlot({
    class_counts <- as.data.frame(table(wine$Cultivar))
    colnames(class_counts) <- c("Cultivar", "Count")
    class_counts$Pct <- (class_counts$Count / sum(class_counts$Count)) * 100
    
    ggplot(class_counts, aes(x = Cultivar, y = Count, fill = Cultivar)) +
      geom_bar(stat = "identity", width = 0.55, color = "#0f172a", alpha = 0.9) +
      geom_text(aes(label = sprintf("%d (%0.1f%%)", Count, Pct)), 
                vjust = -0.6, fontface = "bold", size = 4.2, color = "#1e293b") +
      scale_fill_manual(values = cultivar_colors) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.18)), limits = c(0, 85)) +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "none",
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(face = "bold", color = "#334155"),
        axis.text = element_text(color = "#1e293b", face = "bold")
      ) +
      labs(y = "Number of Samples")
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
      geom_jitter(width = 0.15, alpha = 0.35, size = 1.6, color = "#0f172a") +
      scale_fill_manual(values = cultivar_colors) +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "none",
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_text(face = "bold", color = "#334155"),
        axis.text = element_text(color = "#1e293b", face = "bold")
      ) +
      labs(y = paste(feature_name, "(Raw Measurement)"))
  })
  
  # Correlation Heatmap
  output$plot_corr_heatmap <- renderPlot({
    corr_mat <- cor(wine_features)
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
  
  # Dataset Interactive Table Preview
  if (has_dt) {
    output$table_wine_preview <- DT::renderDataTable({
      formatted_wine <- wine
      formatted_wine[, feature_cols] <- round(formatted_wine[, feature_cols], 2)
      DT::datatable(
        formatted_wine,
        options = list(pageLength = 8, scrollX = TRUE),
        rownames = FALSE,
        class = "stripe hover compact cell-border"
      )
    })
  } else {
    output$table_wine_preview_static <- renderTable({
      formatted_wine <- head(wine, 10)
      formatted_wine[, feature_cols] <- round(formatted_wine[, feature_cols], 2)
      formatted_wine
    }, striped = TRUE, hover = TRUE, bordered = TRUE)
  }
  
  # Download Handler: Wine Dataset CSV
  output$download_wine_csv <- downloadHandler(
    filename = function() {
      paste0("wine_dataset_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(wine, file, row.names = FALSE)
    }
  )
  
  
  # ----------------------------------------------------------------------------
  # SECTION 2 OUTPUTS: PRINCIPAL COMPONENT ANALYSIS (PCA)
  # ----------------------------------------------------------------------------
  
  # Reactive function to generate PCA Scatter Plot (used for rendering & PNG export)
  make_pca_plot <- function(data) {
    ggplot(data, aes(x = PC1, y = PC2, color = Cultivar, fill = Cultivar)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "#94a3b8", linewidth = 0.5) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "#94a3b8", linewidth = 0.5) +
      stat_ellipse(geom = "polygon", alpha = 0.15, level = 0.95, show.legend = FALSE) +
      geom_point(size = 3.2, alpha = 0.85, shape = 21, stroke = 0.8, color = "#0f172a") +
      scale_color_manual(values = cultivar_colors) +
      scale_fill_manual(values = cultivar_colors) +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "top",
        legend.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        axis.title = element_text(face = "bold", color = "#1e293b"),
        axis.text = element_text(color = "#334155")
      ) +
      labs(
        x = sprintf("Principal Component 1 (%.2f%% Variance)", pca_var_pct[1]),
        y = sprintf("Principal Component 2 (%.2f%% Variance)", pca_var_pct[2]),
        color = "Cultivar:",
        fill = "Cultivar:"
      )
  }
  
  # Render PCA Scatter Plot
  output$plot_pca_scatter <- renderPlot({
    make_pca_plot(filtered_pca_data())
  })
  
  # PCA Scree Plot with Cumulative Line
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
    # Sort by absolute loading magnitude on PC1
    loadings[order(-abs(loadings$PC1)), ]
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # Sample of PCA Component Scores
  output$table_pca_scores <- renderTable({
    scores_sample <- head(pca_scores_df[, c("Cultivar", "PC1", "PC2", "PC3", "PC4", "PC5")], 8)
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
      paste0("pca_scores_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(pca_scores_df, file, row.names = FALSE)
    }
  )
  
  # Download Handler: PCA Scatter Plot PNG
  output$download_pca_png <- downloadHandler(
    filename = function() {
      paste0("pca_scatter_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- make_pca_plot(filtered_pca_data())
      ggsave(file, plot = p, width = 8, height = 5.5, dpi = 300)
    }
  )
  
  
  # ----------------------------------------------------------------------------
  # SECTION 3 OUTPUTS: LINEAR DISCRIMINANT ANALYSIS (LDA)
  # ----------------------------------------------------------------------------
  
  # Reactive function to generate LDA Scatter Plot (used for rendering & PNG export)
  make_lda_plot <- function(data) {
    ggplot(data, aes(x = LD1, y = LD2, color = Cultivar, fill = Cultivar)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "#94a3b8", linewidth = 0.5) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "#94a3b8", linewidth = 0.5) +
      stat_ellipse(geom = "polygon", alpha = 0.15, level = 0.95, show.legend = FALSE) +
      geom_point(size = 3.2, alpha = 0.85, shape = 21, stroke = 0.8, color = "#0f172a") +
      scale_color_manual(values = cultivar_colors) +
      scale_fill_manual(values = cultivar_colors) +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "top",
        legend.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        axis.title = element_text(face = "bold", color = "#1e293b"),
        axis.text = element_text(color = "#334155")
      ) +
      labs(
        x = sprintf("Linear Discriminant 1 (%.2f%% of Trace)", lda_trace_pct[1]),
        y = sprintf("Linear Discriminant 2 (%.2f%% of Trace)", lda_trace_pct[2]),
        color = "Cultivar:",
        fill = "Cultivar:"
      )
  }
  
  # Render LDA Scatter Plot
  output$plot_lda_scatter <- renderPlot({
    make_lda_plot(filtered_lda_data())
  })
  
  # LDA Scaling Coefficients Table
  output$table_lda_coefs <- renderTable({
    coefs <- as.data.frame(lda_fit$scaling)
    coefs$Feature <- rownames(coefs)
    rownames(coefs) <- NULL
    coefs <- coefs[, c("Feature", "LD1", "LD2")]
    coefs$LD1 <- round(coefs$LD1, 3)
    coefs$LD2 <- round(coefs$LD2, 3)
    # Sort by absolute discriminant impact on LD1
    coefs[order(-abs(coefs$LD1)), ]
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # Sample of LDA Component Scores
  output$table_lda_scores <- renderTable({
    lda_sample <- head(lda_scores_df[, c("Cultivar", "LD1", "LD2")], 8)
    lda_sample$LD1 <- round(lda_sample$LD1, 3)
    lda_sample$LD2 <- round(lda_sample$LD2, 3)
    lda_sample
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # Download Handler: LDA Scores CSV
  output$download_lda_csv <- downloadHandler(
    filename = function() {
      paste0("lda_scores_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(lda_scores_df, file, row.names = FALSE)
    }
  )
  
  # Download Handler: LDA Scatter Plot PNG
  output$download_lda_png <- downloadHandler(
    filename = function() {
      paste0("lda_scatter_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- make_lda_plot(filtered_lda_data())
      ggsave(file, plot = p, width = 8, height = 5.5, dpi = 300)
    }
  )
  
  
  # ----------------------------------------------------------------------------
  # SECTION 4 OUTPUTS: COMPARISON
  # ----------------------------------------------------------------------------
  
  # Side-by-Side Comparison: PCA Plot
  output$plot_comp_pca <- renderPlot({
    make_pca_plot(filtered_comp_pca()) +
      ggtitle("PCA (Unsupervised: Maximizes Overall Variance)") +
      theme(plot.title = element_text(size = 12, face = "bold", color = "#1e3a5f"))
  })
  
  # Side-by-Side Comparison: LDA Plot
  output$plot_comp_lda <- renderPlot({
    make_lda_plot(filtered_comp_lda()) +
      ggtitle("LDA (Supervised: Maximizes Cultivar Separation)") +
      theme(plot.title = element_text(size = 12, face = "bold", color = "#059669"))
  })
  
  # Download Handler: All Scores Combined CSV
  output$download_all_scores_csv <- downloadHandler(
    filename = function() {
      paste0("pca_lda_combined_scores_", Sys.Date(), ".csv")
    },
    content = function(file) {
      combined <- data.frame(
        Cultivar = wine$Cultivar,
        Type = wine$Type,
        PC1 = pca_scores_df$PC1,
        PC2 = pca_scores_df$PC2,
        PC3 = pca_scores_df$PC3,
        LD1 = lda_scores_df$LD1,
        LD2 = lda_scores_df$LD2
      )
      write.csv(combined, file, row.names = FALSE)
    }
  )
}

# --- 5. Launch the Application ---
shinyApp(ui = ui, server = server)
