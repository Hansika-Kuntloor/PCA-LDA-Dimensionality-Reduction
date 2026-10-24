# ==============================================================================
# Quick Launcher for Feature Extraction Studio
# Run in terminal or PowerShell: Rscript run_app.R
# Or in Windows: & "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" run_app.R
# ==============================================================================

required_packages <- c("shiny", "ggplot2", "MASS", "rattle.data", "DT")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]

if (length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

cat("Launching Feature Extraction Studio in your default web browser...\n")
shiny::runApp(".", launch.browser = TRUE)
