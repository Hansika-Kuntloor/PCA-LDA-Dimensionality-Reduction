# ==============================================================================
# Quick Launcher for PCA & LDA Explorer
# Run in terminal or R: Rscript run_app.R
# ==============================================================================

required_packages <- c("shiny", "ggplot2", "MASS", "rattle.data", "DT")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

if (length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

cat("Launching PCA & LDA Explorer in your default web browser...\n")
shiny::runApp(".", launch.browser = TRUE)
