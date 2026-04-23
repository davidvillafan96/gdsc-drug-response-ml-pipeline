#🧬 GDSC Pharmacogenomics ML Pipeline
Drug Response Prediction using Machine Learning

📌 Overview
This project is part of a structured pharmacogenomics pipeline based on the Genomics of Drug Sensitivity in Cancer (GDSC) dataset.
It represents Project 3 (Machine Learning) in a multi-step workflow:
Project 1: Data integration & harmonization
Project 2: Exploratory Data Analysis (EDA) & statistical inference
Project 3 (this repo): Machine Learning modeling
The goal is to predict drug response using molecular and pharmacological features.

🎯 Objectives
Two complementary prediction tasks were defined:
🔹 Regression
Predict continuous drug response:
LN_IC50_scaled
🔹 Classification
Predict binary drug sensitivity:
Sensitive (1): bottom 25% (Q1) of LN_IC50
Resistant/Intermediate (0): remaining samples

⚙️ Tech Stack
Language: R
Environment: Google Colab
Libraries:
tidyverse, dplyr, tidyr
caret
randomForest
xgboost
e1071 (SVM)
class (k-NN)
Metrics
pROC

🧠 Machine Learning Pipeline
1. Data Loading
The preprocessed dataset (GDSC_ML_Ready.csv) was loaded from Google Drive.
2. Memory Optimization
To prevent RAM overflow in Colab:
Random sampling of 30,000 rows
Removal of original full dataset from memory (rm() + gc())
3. Feature Cleaning
Removed metadata and identifiers:
Cell line IDs
Drug names
Dataset references
Converted categorical variables → factors
Removed:
Missing values
Constant features
Near-zero-variance predictors
4. Feature Engineering
One-hot encoding using model.matrix
Standardization:
Centering
Scaling
5. Data Leakage Prevention 🚨
To ensure valid modeling:
Removed:
LN_IC50_scaled
AUC_scaled
Z_SCORE_scaled
Any Sensitivity_Class variables
from predictor matrix before training.
6. Train/Test Split
80% training
20% testing
Stratified sampling using caret::createDataPartition

📈 Regression Models
Models trained:
🌲 Random Forest Regressor
⚡ XGBoost Regressor
Evaluation metric:
RMSE (Root Mean Squared Error)

🧬 Classification Pipeline
1. Dimensionality Reduction (PCA)
Applied only on training data
Retained components explaining 95% variance
Test set projected into same PCA space
2. Class Imbalance Handling
To avoid bias:
Downsampling to balanced dataset (50/50)
7,500 sensitive
7,500 resistant/intermediate
3. Models trained
Logistic Regression
Linear SVM
k-NN (k = 7)
Random Forest Classifier
4. Evaluation
Confusion Matrix (caret)
Predictions performed on original (unbalanced) test set

🔧 Model Tuning
Random Forest Optimization
5-fold cross-validation
Grid Search:
mtry = {3, 5, 10}
Random Search:
Reduced tuning space for computational efficiency

🚀 Key Insights
High-dimensional pharmacogenomic data requires:
Strong preprocessing
Dimensionality reduction (PCA)
Class imbalance significantly affects classification performance
Tree-based models (RF, XGBoost) showed strong baseline performance
Memory constraints are a real-world limitation → required sampling strategy

📂 Repository Structure
├── data/
│   └── GDSC_ML_Ready.csv
├── notebooks/
│   └── gdsc_ml_pipeline.ipynb
├── src/
│   └── ml_pipeline.R
├── results/
│   └── model_outputs/
└── README.md

🧪 Reproducibility
To reproduce the analysis:
# Install required packages
install.packages(c("tidyverse", "caret", "randomForest", "xgboost", "e1071", "class", "Metrics"))

# Run pipeline
source("src/ml_pipeline.R")

📊 Future Improvements
Hyperparameter tuning for XGBoost
Feature importance analysis
SHAP values / interpretability
Deep learning approaches
External validation dataset

👨‍🔬 Author
David Villafañe
PhD in Biological Sciences | Biotechnologist
Focused on:
Bioinformatics
Genomics
Machine Learning in biomedical data

🌐 Related Projects
GDSC Project 1 → Data Integration
GDSC Project 2 → EDA & Statistical Learning
⭐ If you found this useful
Give the repo a ⭐ and feel free to connect or reach out!
