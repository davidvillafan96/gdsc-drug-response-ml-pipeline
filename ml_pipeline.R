# ============================================================
# GDSC Pharmacogenomics - Machine Learning Pipeline
# Author: David Villafañe
# ============================================================

# -----------------------------
# 1. Libraries
# -----------------------------
if (!require("pacman")) install.packages("pacman")
library(pacman)

p_load(
  tidyverse, dplyr, tidyr,
  caret,
  randomForest,
  xgboost,
  e1071,
  class,
  Metrics
)

# -----------------------------
# 2. Load Data
# -----------------------------
df_ml <- readr::read_csv("data/GDSC_ML_Ready.csv")

cat("Dataset loaded. Rows:", nrow(df_ml), "\n")

# -----------------------------
# 3. Memory Optimization
# -----------------------------
set.seed(123)
df_ml <- df_ml %>% sample_n(min(30000, nrow(df_ml)))

gc()

# -----------------------------
# 4. Feature Cleaning
# -----------------------------
cols_to_discard <- c(
  "DATASET", "NLME_RESULT_ID", "NLME_CURVE_ID", "COSMIC_ID",
  "CELL_LINE_NAME", "SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME",
  "SYNONYMS", "WEBRELEASE", "COMPANY_ID",
  "PUTATIVE_TARGET", "PATHWAY_NAME"
)

df_clean <- df_ml %>%
  select(-any_of(cols_to_discard)) %>%
  mutate(across(where(is.character), as.factor)) %>%
  drop_na() %>%
  select(where(~ n_distinct(.) > 1))

rm(df_ml); gc()

# -----------------------------
# 5. One-Hot Encoding
# -----------------------------
df_dummy <- model.matrix(~ . - 1, data = df_clean) %>%
  as.data.frame()

rm(df_clean); gc()

# -----------------------------
# 6. Preprocessing
# -----------------------------
preproc <- preProcess(df_dummy, method = c("nzv", "center", "scale"))
df_proc <- predict(preproc, df_dummy)

rm(df_dummy, preproc); gc()

# -----------------------------
# 7. Define Targets & Predictors
# -----------------------------
target_metrics <- c("LN_IC50_scaled", "AUC_scaled", "Z_SCORE_scaled")

X_data <- df_proc %>%
  select(-any_of(target_metrics)) %>%
  select(-starts_with("Sensitivity_Class"))

# -----------------------------
# 8. Regression Setup
# -----------------------------
y_reg <- df_proc$LN_IC50_scaled

set.seed(123)
idx_reg <- createDataPartition(y_reg, p = 0.8, list = FALSE)

X_train_reg <- X_data[idx_reg, ]
X_test_reg  <- X_data[-idx_reg, ]

y_train_reg <- y_reg[idx_reg]
y_test_reg  <- y_reg[-idx_reg]

# -----------------------------
# 9. Classification Setup
# -----------------------------
q25_threshold <- quantile(df_proc$LN_IC50_scaled, 0.25, na.rm = TRUE)

y_clf <- as.factor(ifelse(df_proc$LN_IC50_scaled <= q25_threshold, 1, 0))

set.seed(123)
idx_clf <- createDataPartition(y_clf, p = 0.8, list = FALSE)

X_train_clf <- X_data[idx_clf, ]
X_test_clf  <- X_data[-idx_clf, ]

y_train_clf <- y_clf[idx_clf]
y_test_clf  <- y_clf[-idx_clf]

# ============================================================
# REGRESSION MODELS
# ============================================================

cat("\n--- Training Regression Models ---\n")

# Random Forest
rf_reg <- randomForest(
  x = X_train_reg,
  y = y_train_reg,
  ntree = 100
)

pred_rf <- predict(rf_reg, X_test_reg)
cat("RF RMSE:", rmse(y_test_reg, pred_rf), "\n")

# XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(X_train_reg), label = y_train_reg)
dtest  <- xgb.DMatrix(data = as.matrix(X_test_reg), label = y_test_reg)

params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8
)

xgb_reg <- xgb.train(params = params, data = dtrain, nrounds = 150)

pred_xgb <- predict(xgb_reg, dtest)
cat("XGB RMSE:", rmse(y_test_reg, pred_xgb), "\n")

# ============================================================
# CLASSIFICATION PIPELINE
# ============================================================

cat("\n--- Classification Pipeline ---\n")

# -----------------------------
# PCA
# -----------------------------
pca_res <- prcomp(X_train_clf, center = TRUE, scale. = TRUE)

var_explained <- cumsum(pca_res$sdev^2) / sum(pca_res$sdev^2)
n_comp <- which(var_explained >= 0.95)[1]

cat("PCA components:", n_comp, "\n")

X_train_pca <- predict(pca_res, X_train_clf)[, 1:n_comp]
X_test_pca  <- predict(pca_res, X_test_clf)[, 1:n_comp]

# -----------------------------
# Class Balancing
# -----------------------------
set.seed(123)

idx_0 <- which(y_train_clf == 0)
idx_1 <- which(y_train_clf == 1)

sample_0 <- sample(idx_0, 7500)
sample_1 <- sample(idx_1, 7500)

idx_bal <- c(sample_0, sample_1)

X_train_bal <- X_train_pca[idx_bal, ]
y_train_bal <- y_train_clf[idx_bal]

cat("Balanced classes:\n")
print(table(y_train_bal))

# -----------------------------
# Logistic Regression
# -----------------------------
logit_model <- glm(
  y_train_bal ~ .,
  data = data.frame(X_train_bal, y_train_bal),
  family = binomial
)

pred_prob <- predict(logit_model, as.data.frame(X_test_pca), type = "response")
pred_class <- as.factor(ifelse(pred_prob > 0.5, 1, 0))

cat("\nLogistic Regression:\n")
print(confusionMatrix(pred_class, y_test_clf))

# -----------------------------
# SVM
# -----------------------------
svm_model <- svm(
  x = X_train_bal,
  y = y_train_bal,
  kernel = "linear",
  cost = 1
)

pred_svm <- predict(svm_model, X_test_pca)

cat("\nSVM:\n")
print(confusionMatrix(pred_svm, y_test_clf))

# -----------------------------
# k-NN
# -----------------------------
pred_knn <- knn(
  train = X_train_bal,
  test = X_test_pca,
  cl = y_train_bal,
  k = 7
)

cat("\nk-NN:\n")
print(confusionMatrix(pred_knn, y_test_clf))

# -----------------------------
# Random Forest Classifier
# -----------------------------
rf_clf <- randomForest(X_train_bal, y_train_bal, ntree = 200)

pred_rf_clf <- predict(rf_clf, X_test_pca)

cat("\nRandom Forest Classifier:\n")
print(confusionMatrix(pred_rf_clf, y_test_clf))

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

cat("\n--- Hyperparameter Tuning ---\n")

control <- trainControl(method = "cv", number = 5)

tunegrid <- expand.grid(.mtry = c(3, 5, 10))

tuned_rf <- train(
  x = X_train_bal,
  y = y_train_bal,
  method = "rf",
  trControl = control,
  tuneGrid = tunegrid,
  ntree = 100
)

print(tuned_rf)

# --- Random Search ---
cat("\n---Random Search --- \n")
control_rand <- trainControl(method = "cv", number = 5, search = "random")

random_rf <- train(
  x = X_train_bal,
  y = y_train_bal,
  method = "rf",
  trControl = control_rand,
  tuneLength = 5, 
  ntree = 100
)
print(random_rf)

# ============================================================
# END
# ============================================================

cat("\n--- Pipeline completed successfully ---\n")