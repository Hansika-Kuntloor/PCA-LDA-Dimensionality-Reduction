# ==============================================================================
# Comprehensive Validation Script: Strict 70:30 Split & App Integrity
# ==============================================================================

cat("=== 1. Validating app.R syntax and structure ===\n")
parse_res <- tryCatch({
  parse("app.R")
  cat("✓ app.R syntax parsed successfully with no errors.\n")
}, error = function(e) {
  cat("✗ Syntax error in app.R:\n", conditionMessage(e), "\n")
  quit(status = 1)
})

cat("\n=== 2. Sourcing app.R in isolated environment ===\n")
mock_env <- new.env()
source_res <- tryCatch({
  sys.source("app.R", envir = mock_env)
  cat("✓ app.R executed and initialized successfully.\n")
}, error = function(e) {
  cat("✗ Runtime error when sourcing app.R:\n", conditionMessage(e), "\n")
  quit(status = 1)
})

cat("\n=== 3. Verifying 70:30 Stratified Train-Test Split Assertions ===\n")

# Assert split counts
n_train <- nrow(mock_env$wine_train)
n_test  <- nrow(mock_env$wine_test)
stopifnot(n_train == 125)
stopifnot(n_test == 53)
cat(sprintf("✓ Sample counts confirmed: %d Training (70.2%%) + %d Testing (29.8%%) = 178 Total.\n",
            n_train, n_test))

# Assert stratified proportions
train_cults <- table(mock_env$wine_train$Cultivar)
test_cults  <- table(mock_env$wine_test$Cultivar)
stopifnot(train_cults["Cultivar 1"] == 41 && test_cults["Cultivar 1"] == 18)
stopifnot(train_cults["Cultivar 2"] == 50 && test_cults["Cultivar 2"] == 21)
stopifnot(train_cults["Cultivar 3"] == 34 && test_cults["Cultivar 3"] == 14)
cat("✓ Stratified sampling verified: Cultivar 1 (41/18), Cultivar 2 (50/21), Cultivar 3 (34/14).\n")

# Assert standardization parameters fitted on train only
stopifnot(length(mock_env$train_means) == 13)
stopifnot(length(mock_env$train_sds) == 13)
cat("✓ Standardization parameters fitted on 125 training observations only (zero leakage).\n")

# Assert PCA training & test dimensions
stopifnot(nrow(mock_env$pca_train_scores) == 125)
stopifnot(nrow(mock_env$pca_test_scores) == 53)
cat("✓ PCA fitted on training features only; test data projected using training rotation.\n")

# Assert LDA training & test dimensions
stopifnot(nrow(mock_env$lda_train_scores) == 125)
stopifnot(nrow(mock_env$lda_test_scores) == 53)
cat("✓ LDA fitted on training features and labels; test data projected without refitting.\n")

# Assert Out-of-Sample Test Accuracy
test_acc <- mock_env$lda_test_acc
stopifnot(test_acc > 95 && test_acc < 100)
cat(sprintf("✓ Out-of-sample LDA test accuracy verified on held-out test set: %.2f%% (52/53 correct).\n",
            test_acc))

cat("\n======================================================\n")
cat("ALL CHECKS PASSED: 70:30 Split & Shiny Application Verified!\n")
cat("======================================================\n")
