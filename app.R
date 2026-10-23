# ==============================================================================
# Application: Perform Feature Extraction (Dimensionality Reduction) Using PCA and LDA
# Dataset: Wine Dataset (rattle.data / UCI Machine Learning Repository)
# Design: Simple, Clean Academic Web Interface
# Split Rule: Fixed, Stratified 70:30 Train-Test Split (Seed 42)
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

# Load Wine dataset
if (requireNamespace("rattle.data", quietly = TRUE)) {
  data(wine, package = "rattle.data")
} else {
  wine_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
  wine <- read.csv(wine_url, header = FALSE)
  colnames(wine) <- c("Type", "Alcohol", "Malic", "Ash", "Alcalinity", 
                      "Magnesium", "Phenols", "Flavanoids", "Nonflavanoids", 
                      "Proanthocyanins", "Color", "Hue", "Dilution", "Proline")
  wine$Type <- as.factor(wine$Type)
}

# Ensure clean factor labels for the 3 wine cultivars
wine$Type <- as.factor(wine$Type)
wine$Cultivar <- factor(wine$Type, 
                        levels = c("1", "2", "3"),
                        labels = c("Cultivar 1", "Cultivar 2", "Cultivar 3"))

feature_cols <- c("Alcohol", "Malic", "Ash", "Alcalinity", "Magnesium", 
                  "Phenols", "Flavanoids", "Nonflavanoids", "Proanthocyanins", 
                  "Color", "Hue", "Dilution", "Proline")

# --- 2. Fixed, Stratified 70:30 Train-Test Split (Seed 42) ---
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

# Assign Split identifier
wine$Split <- "Testing (30%)"
wine$Split[train_indices] <- "Training (70%)"
wine$Split <- factor(wine$Split, levels = c("Training (70%)", "Testing (30%)"))

wine_train <- wine[train_indices, ]
wine_test  <- wine[test_indices, ]

# --- 3. Training-Set Only Standardization (Zero Data Leakage) ---
train_means <- colMeans(wine_train[, feature_cols])
train_sds   <- apply(wine_train[, feature_cols], 2, sd)

# Scale Training Data
wine_train_scaled <- as.data.frame(scale(wine_train[, feature_cols], center = train_means, scale = train_sds))
wine_train_scaled$Cultivar <- wine_train$Cultivar
wine_train_scaled$Split    <- wine_train$Split

# Transform Test Data using Training Parameters
wine_test_scaled <- as.data.frame(scale(wine_test[, feature_cols], center = train_means, scale = train_sds))
wine_test_scaled$Cultivar <- wine_test$Cultivar
wine_test_scaled$Split    <- wine_test$Split

# --- 4. PCA Modeling (70% Training Partition Only) ---
# Unsupervised: class labels are excluded
pca_fit <- prcomp(wine_train_scaled[, feature_cols], center = FALSE, scale. = FALSE)
pca_var <- pca_fit$sdev^2
pca_var_pct <- (pca_var / sum(pca_var)) * 100

# Training projection scores
pca_train_scores <- as.data.frame(pca_fit$x)
pca_train_scores$Cultivar <- wine_train$Cultivar
pca_train_scores$Split    <- wine_train$Split

# Held-out test projection scores (projected using training rotation)
pca_test_scores <- as.data.frame(predict(pca_fit, newdata = wine_test_scaled[, feature_cols]))
pca_test_scores$Cultivar <- wine_test$Cultivar
pca_test_scores$Split    <- wine_test$Split

pca_scores_df <- rbind(pca_train_scores, pca_test_scores)

# --- 5. LDA Modeling (70% Training Partition Only) ---
# Supervised: trained with training cultivar labels
lda_fit <- MASS::lda(Cultivar ~ ., data = wine_train_scaled[, c(feature_cols, "Cultivar")])
lda_trace_pct <- (lda_fit$svd^2 / sum(lda_fit$svd^2)) * 100

# Training projection scores
lda_train_pred <- predict(lda_fit, newdata = wine_train_scaled)
lda_train_scores <- as.data.frame(lda_train_pred$x)
lda_train_scores$Cultivar <- wine_train$Cultivar
lda_train_scores$Split    <- wine_train$Split

# Held-out test projection scores
lda_test_pred <- predict(lda_fit, newdata = wine_test_scaled)
lda_test_scores <- as.data.frame(lda_test_pred$x)
lda_test_scores$Cultivar <- wine_test$Cultivar
lda_test_scores$Split    <- wine_test$Split

lda_scores_df <- rbind(lda_train_scores, lda_test_scores)

# Out-of-sample test evaluation (evaluated on 53 held-out test observations)
lda_test_conf <- table(Predicted = lda_test_pred$class, Actual = wine_test$Cultivar)
lda_test_acc  <- (sum(diag(lda_test_conf)) / sum(lda_test_conf)) * 100

# Color and shape aesthetic configurations
cultivar_colors <- c(
  "Cultivar 1" = "#1d4ed8",  # Blue
  "Cultivar 2" = "#059669",  # Green
  "Cultivar 3" = "#dc2626"   # Red
)

split_shapes <- c(
  "Training (70%)" = 21,  # Circle with border
  "Testing (30%)"  = 24   # Triangle with border
)

# Feature description data
features_info <- data.frame(
  Feature = c("Alcohol", "Malic", "Ash", "Alcalinity", "Magnesium", 
              "Phenols", "Flavanoids", "Nonflavanoids", "Proanthocyanins", 
              "Color", "Hue", "Dilution", "Proline"),
  Description = c("Alcohol content by volume (%)",
                  "Malic acid concentration (g/L)",
                  "Ash content from mineral residue",
                  "Alcalinity of ash",
                  "Magnesium concentration (mg/L)",
                  "Total phenolic compounds",
                  "Flavanoid polyphenol content",
                  "Nonflavanoid phenolic compounds",
                  "Proanthocyanin content",
                  "Color intensity of the wine",
                  "Hue (absorbance ratio at 420/520 nm)",
                  "OD280/OD315 ratio of diluted wines",
                  "Proline amino acid concentration (mg/L)"),
  stringsAsFactors = FALSE
)


# --- 6. User Interface (UI) Definition ---
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      /* Clean Academic Styling */
      body {
        background-color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        color: #1f2937;
        margin: 0;
        padding: 0;
        line-height: 1.6;
      }
      
      /* Minimalist Academic Header */
      .academic-header {
        background-color: #ffffff;
        padding: 22px 30px 18px 30px;
        border-bottom: 2px solid #2563eb;
        margin-bottom: 20px;
      }
      .academic-header h1 {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        margin: 0 0 4px 0;
        letter-spacing: -0.3px;
      }
      .academic-header p {
        font-size: 14px;
        color: #4b5563;
        margin: 0;
      }
      
      /* Sidebar Panel */
      .sidebar-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 16px;
        margin-bottom: 20px;
      }
      .sidebar-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        color: #1e3a8a;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
      }
      .sidebar-box .radio label {
        font-size: 14px;
        font-weight: 500;
        color: #374151;
        padding: 4px 6px;
      }
      .sidebar-box .checkbox label {
        font-size: 13px;
        color: #374151;
      }
      
      /* Subtle Partition Box */
      .split-info-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 3px solid #16a34a;
        padding: 10px 12px;
        border-radius: 6px;
        font-size: 12px;
        color: #166534;
        margin: 14px 0;
        line-height: 1.5;
      }
      
      /* Content Containers */
      .academic-content {
        padding: 0 10px 40px 10px;
      }
      .academic-section-title {
        font-size: 19px;
        font-weight: 700;
        color: #111827;
        margin: 0 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #e5e7eb;
      }
      
      /* Academic Explanatory Box */
      .academic-callout {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #2563eb;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 20px;
        font-size: 14px;
        color: #334155;
        line-height: 1.6;
      }
      
      /* Stat Line */
      .academic-stat-line {
        background-color: #f1f5f9;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 13px;
        color: #1e293b;
        margin-bottom: 18px;
      }
      
      /* Clean Download Buttons */
      .btn-academic {
        background-color: #ffffff;
        color: #1d4ed8;
        border: 1px solid #93c5fd;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
        padding: 7px 12px;
        width: 100%;
        margin-top: 6px;
        margin-bottom: 6px;
        text-align: center;
        display: inline-block;
        transition: all 0.15s ease;
      }
      .btn-academic:hover {
        background-color: #eff6ff;
        border-color: #3b82f6;
        color: #1e40af;
        text-decoration: none;
      }
      
      /* Simple Academic Tables */
      .academic-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-top: 8px;
        margin-bottom: 20px;
      }
      .academic-table th {
        background-color: #f1f5f9;
        color: #1e293b;
        font-weight: 600;
        padding: 8px 12px;
        border: 1px solid #e2e8f0;
        text-align: left;
      }
      .academic-table td {
        padding: 8px 12px;
        border: 1px solid #e2e8f0;
        color: #334155;
      }
      .academic-table tr:nth-child(even) {
        background-color: #f8fafc;
      }
    "))
  ),
  
  # Clean Header
  div(class = "academic-header",
    h1("Perform Feature Extraction (Dimensionality Reduction) Using PCA and LDA"),
    p("Comparative pattern analysis on the Wine dataset using a fixed 70:30 stratified train-test split")
  ),
  
  # Sidebar & Main Content
  fluidRow(
    style = "margin: 0; padding: 0 15px;",
    
    # Left Control Sidebar
    column(width = 3,
      div(class = "sidebar-box",
        div(class = "sidebar-title", "Sections"),
        radioButtons("section", label = NULL,
          choices = c(
            "Dataset" = "dataset",
            "PCA"     = "pca",
            "LDA"     = "lda"
          ),
          selected = "dataset"
        ),
        
        # Concise Split Specification
        div(class = "split-info-box",
          strong("Stratified 70:30 Split (Seed 42)"), br(),
          "• Training: 125 samples (70.2%)", br(),
          "• Testing: 53 samples (29.8%)", br(),
          "• Standardization: Training only"
        ),
        
        # Section Controls: Dataset
        conditionalPanel(
          condition = "input.section == 'dataset'",
          div(class = "sidebar-title", "Data Export"),
          downloadButton("download_wine_csv", "Download Dataset (CSV)", class = "btn-academic")
        ),
        
        # Section Controls: PCA
        conditionalPanel(
          condition = "input.section == 'pca'",
          div(class = "sidebar-title", "Partition View"),
          radioButtons("pca_split_view", label = NULL,
                       choices = c("Both (125 Train + 53 Test)" = "both",
                                   "Training Set Only (125)" = "train",
                                   "Testing Set Only (53)" = "test"),
                       selected = "both"),
          div(class = "sidebar-title", "Cultivar Filter"),
          checkboxGroupInput("pca_classes", label = NULL,
                             choices = c("Cultivar 1" = "Cultivar 1",
                                         "Cultivar 2" = "Cultivar 2",
                                         "Cultivar 3" = "Cultivar 3"),
                             selected = c("Cultivar 1", "Cultivar 2", "Cultivar 3")),
          div(class = "sidebar-title", "Export"),
          downloadButton("download_pca_png", "Download Scatter Plot (PNG)", class = "btn-academic"),
          downloadButton("download_pca_csv", "Download PCA Scores (CSV)", class = "btn-academic")
        ),
        
        # Section Controls: LDA
        conditionalPanel(
          condition = "input.section == 'lda'",
          div(class = "sidebar-title", "Partition View"),
          radioButtons("lda_split_view", label = NULL,
                       choices = c("Both (125 Train + 53 Test)" = "both",
                                   "Training Set Only (125)" = "train",
                                   "Testing Set Only (53)" = "test"),
                       selected = "both"),
          div(class = "sidebar-title", "Cultivar Filter"),
          checkboxGroupInput("lda_classes", label = NULL,
                             choices = c("Cultivar 1" = "Cultivar 1",
                                         "Cultivar 2" = "Cultivar 2",
                                         "Cultivar 3" = "Cultivar 3"),
                             selected = c("Cultivar 1", "Cultivar 2", "Cultivar 3")),
          div(class = "sidebar-title", "Export"),
          downloadButton("download_lda_png", "Download Scatter Plot (PNG)", class = "btn-academic"),
          downloadButton("download_lda_csv", "Download LDA Scores (CSV)", class = "btn-academic")
        ),
        
        # Plot Legend Note
        conditionalPanel(
          condition = "input.section == 'pca' || input.section == 'lda'",
          tags$hr(style = "margin: 14px 0 10px 0; border-color: #e2e8f0;"),
          div(style = "font-size: 11px; color: #64748b; line-height: 1.5;",
            strong("Shape Notation:"), br(),
            "● Circle = Training Sample (70%)", br(),
            "▲ Triangle = Testing Sample (30%)"
          )
        )
      )
    ),
    
    # Main Content Area
    column(width = 9, class = "academic-content",
      
      # ------------------------------------------------------------------------
      # SECTION 1: DATASET (Default Section)
      # ------------------------------------------------------------------------
      conditionalPanel(
        condition = "input.section == 'dataset'",
        
        div(class = "academic-section-title", "Wine Dataset Overview & Features"),
        
        # Short Academic Introduction
        div(class = "academic-callout",
          strong("Introduction to the Wine Dataset: "),
          "The Wine dataset comprises chemical analysis results of wines derived from three different cultivars grown in the same region in Italy. It contains 178 observations across 13 continuous chemical constituents (e.g., alcohol, phenols, flavanoids, color intensity, and proline).",
          br(), br(),
          "To evaluate feature extraction under strict data hygiene, the dataset is partitioned using a ",
          strong("fixed, stratified 70:30 train-test split"),
          " (random seed 42). Feature standardization parameters (means and standard deviations) are computed strictly on the training partition to prevent data leakage."
        ),
        
        # Concise Sample Counts Line
        div(class = "academic-stat-line",
          strong("Dataset Specification: "),
          "178 Total Observations | 13 Continuous Features | 3 Classes | 0 Missing Values", br(),
          strong("Sample Counts by Partition: "),
          "Training (70%): 125 samples (C1: 41, C2: 50, C3: 34) | Testing (30%): 53 samples (C1: 18, C2: 21, C3: 14)"
        ),
        
        # Chemical Features Table
        h4(style = "font-size: 15px; font-weight: 700; color: #1f2937; margin: 16px 0 8px 0;", "Measured Chemical Features (13 Attributes)"),
        tableOutput("table_features_info"),
        
        # Interactive Dataset Table Preview
        h4(style = "font-size: 15px; font-weight: 700; color: #1f2937; margin: 20px 0 8px 0;", "Dataset Table (with Train/Test Split Column)"),
        if (has_dt) {
          DT::dataTableOutput("table_wine_preview")
        } else {
          tableOutput("table_wine_preview_static")
        }
      ),
      
      # ------------------------------------------------------------------------
      # SECTION 2: PCA (Principal Component Analysis)
      # ------------------------------------------------------------------------
      conditionalPanel(
        condition = "input.section == 'pca'",
        
        div(class = "academic-section-title", "Principal Component Analysis (PCA)"),
        
        # Beginner-Friendly Explanation
        div(class = "academic-callout",
          strong("What is Principal Component Analysis (PCA)?"), br(),
          "PCA is an ", strong("unsupervised linear feature extraction"), " technique. It transforms high-dimensional data into orthogonal axes (principal components) that maximize total feature variance, without using class labels.",
          br(), br(),
          strong("Methodology: "),
          "The PCA model was fitted strictly on the standardized 70% training set (125 observations). The held-out 30% test observations (53 samples) were transformed using the training-set rotation vectors without refitting.",
          br(), br(),
          sprintf("• PC1 explains %.2f%% of training variance (direction of greatest data dispersion).", pca_var_pct[1]), br(),
          sprintf("• PC2 explains %.2f%% of training variance (orthogonal to PC1).", pca_var_pct[2]), br(),
          sprintf("• Together, PC1 and PC2 retain %.2f%% of the total cumulative variance in a 2D plane.", pca_var_pct[1] + pca_var_pct[2])
        ),
        
        # PC1 vs PC2 Scatter Plot (No scree plot, no biplot)
        div(style = "background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 20px;",
          h4(style = "font-size: 15px; font-weight: 700; color: #1f2937; margin: 0 0 12px 0;", "PC1 vs PC2 Scatter Plot"),
          plotOutput("plot_pca_scatter", height = "520px")
        )
      ),
      
      # ------------------------------------------------------------------------
      # SECTION 3: LDA (Linear Discriminant Analysis)
      # ------------------------------------------------------------------------
      conditionalPanel(
        condition = "input.section == 'lda'",
        
        div(class = "academic-section-title", "Linear Discriminant Analysis (LDA)"),
        
        # Beginner-Friendly Explanation
        div(class = "academic-callout",
          strong("What is Linear Discriminant Analysis (LDA)?"), br(),
          "LDA is a ", strong("supervised dimensionality reduction"), " technique. Unlike PCA, LDA utilizes known class labels during training to find linear combinations of features that maximize between-class separation while minimizing within-class spread (Fisher's criterion).",
          br(), br(),
          strong("Methodology: "),
          "The LDA model was trained strictly on the standardized 70% training partition (125 observations) using cultivar labels. The held-out 30% test observations (53 samples) were projected onto the fitted discriminant functions without refitting.",
          br(), br(),
          sprintf("• Linear Discriminant 1 (LD1) accounts for %.2f%% of between-class separation.", lda_trace_pct[1]), br(),
          sprintf("• Linear Discriminant 2 (LD2) accounts for %.2f%% of between-class separation.", lda_trace_pct[2]), br(),
          "• For 3 classes, exactly min(13, 3 - 1) = 2 linear discriminants exist, capturing 100% of the between-class discriminant space in 2D."
        ),
        
        # LD1 vs LD2 Scatter Plot
        div(style = "background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 20px;",
          h4(style = "font-size: 15px; font-weight: 700; color: #1f2937; margin: 0 0 12px 0;", "LD1 vs LD2 Scatter Plot"),
          plotOutput("plot_lda_scatter", height = "520px")
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
  
  
  # ----------------------------------------------------------------------------
  # SECTION 1: DATASET OUTPUTS
  # ----------------------------------------------------------------------------
  
  # Features description table
  output$table_features_info <- renderTable({
    features_info
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  # Interactive Dataset Table
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
  
  # Download Handler: Wine Dataset CSV
  output$download_wine_csv <- downloadHandler(
    filename = function() {
      paste0("wine_dataset_70_30_split_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(wine, file, row.names = FALSE)
    }
  )
  
  
  # ----------------------------------------------------------------------------
  # SECTION 2: PCA OUTPUTS
  # ----------------------------------------------------------------------------
  
  # Construct PCA Scatter Plot
  make_pca_plot <- function(data) {
    ggplot(data, aes(x = PC1, y = PC2, color = Cultivar, fill = Cultivar)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "#cbd5e1", linewidth = 0.5) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "#cbd5e1", linewidth = 0.5) +
      stat_ellipse(geom = "polygon", alpha = 0.10, level = 0.95, show.legend = FALSE) +
      geom_point(aes(shape = Split), size = 3.6, alpha = 0.85, stroke = 0.8) +
      scale_color_manual(values = cultivar_colors) +
      scale_fill_manual(values = cultivar_colors) +
      scale_shape_manual(values = split_shapes) +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "top",
        legend.box = "horizontal",
        legend.title = element_text(face = "bold", color = "#1f2937"),
        panel.grid.minor = element_blank(),
        axis.title = element_text(face = "bold", color = "#1f2937"),
        axis.text = element_text(color = "#4b5563")
      ) +
      labs(
        x = sprintf("Principal Component 1 (%.2f%% Variance)", pca_var_pct[1]),
        y = sprintf("Principal Component 2 (%.2f%% Variance)", pca_var_pct[2]),
        color = "Cultivar:",
        fill  = "Cultivar:",
        shape = "Partition:"
      )
  }
  
  output$plot_pca_scatter <- renderPlot({
    make_pca_plot(filtered_pca_data())
  })
  
  # Download Handler: PCA Plot PNG
  output$download_pca_png <- downloadHandler(
    filename = function() {
      paste0("pca_scatter_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- make_pca_plot(filtered_pca_data())
      ggsave(file, plot = p, width = 8.5, height = 5.8, dpi = 300)
    }
  )
  
  # Download Handler: PCA Scores CSV
  output$download_pca_csv <- downloadHandler(
    filename = function() {
      paste0("pca_scores_70_30_split_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(pca_scores_df, file, row.names = FALSE)
    }
  )
  
  
  # ----------------------------------------------------------------------------
  # SECTION 3: LDA OUTPUTS
  # ----------------------------------------------------------------------------
  
  # Construct LDA Scatter Plot
  make_lda_plot <- function(data) {
    ggplot(data, aes(x = LD1, y = LD2, color = Cultivar, fill = Cultivar)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "#cbd5e1", linewidth = 0.5) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "#cbd5e1", linewidth = 0.5) +
      stat_ellipse(geom = "polygon", alpha = 0.10, level = 0.95, show.legend = FALSE) +
      geom_point(aes(shape = Split), size = 3.6, alpha = 0.85, stroke = 0.8) +
      scale_color_manual(values = cultivar_colors) +
      scale_fill_manual(values = cultivar_colors) +
      scale_shape_manual(values = split_shapes) +
      theme_minimal(base_size = 13) +
      theme(
        legend.position = "top",
        legend.box = "horizontal",
        legend.title = element_text(face = "bold", color = "#1f2937"),
        panel.grid.minor = element_blank(),
        axis.title = element_text(face = "bold", color = "#1f2937"),
        axis.text = element_text(color = "#4b5563")
      ) +
      labs(
        x = sprintf("Linear Discriminant 1 (%.2f%% of Trace)", lda_trace_pct[1]),
        y = sprintf("Linear Discriminant 2 (%.2f%% of Trace)", lda_trace_pct[2]),
        color = "Cultivar:",
        fill  = "Cultivar:",
        shape = "Partition:"
      )
  }
  
  output$plot_lda_scatter <- renderPlot({
    make_lda_plot(filtered_lda_data())
  })
  
  # Download Handler: LDA Plot PNG
  output$download_lda_png <- downloadHandler(
    filename = function() {
      paste0("lda_scatter_plot_", Sys.Date(), ".png")
    },
    content = function(file) {
      p <- make_lda_plot(filtered_lda_data())
      ggsave(file, plot = p, width = 8.5, height = 5.8, dpi = 300)
    }
  )
  
  # Download Handler: LDA Scores CSV
  output$download_lda_csv <- downloadHandler(
    filename = function() {
      paste0("lda_scores_70_30_split_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(lda_scores_df, file, row.names = FALSE)
    }
  )
}

# --- 8. Launch the Application ---
shinyApp(ui = ui, server = server)
