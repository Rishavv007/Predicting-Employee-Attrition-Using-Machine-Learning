import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from ctgan import CTGAN

warnings.filterwarnings("ignore")

# ============================================================
# 1. Data Loading and Preprocessing
# ============================================================

# File path for the IBM HR Analytics dataset
file_path = "HR-Analytics-Employee-Attrition-and-Performance.csv"  # Adjust if needed

# Load the dataset
data = pd.read_csv(file_path)

# Drop unnecessary columns if they exist
cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
for col in cols_to_drop:
    if col in data.columns:
        data = data.drop(columns=[col])

# Our target is 'Attrition'. All other columns are features.
feature_names = [col for col in data.columns if col != "Attrition"]
target_column = "Attrition"

# Split into features and target
X = data[feature_names]
y = data[target_column]

# Split data into training and test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

# ============================================================
# 2. Model Training and Evaluation
# ============================================================

# Define individual models
log_reg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, eval_metric='logloss', use_label_encoder=False, random_state=42)
svm = SVC(probability=True, random_state=42)
knn = KNeighborsClassifier()
cat = CatBoostClassifier(iterations=100, verbose=0, random_state=42)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='Yes')
    rec = recall_score(y_test, y_pred, pos_label='Yes')
    f1 = f1_score(y_test, y_pred, pos_label='Yes')
    y_test_numeric = y_test.map({'No': 0, 'Yes': 1})
    auc_val = roc_auc_score(y_test_numeric, y_prob)
    return acc, prec, rec, f1, auc_val

# Train individual models
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf,
    "XGBoost": xgb,
    "SVM": svm,
    "KNN": knn,
    "CatBoost": cat
}
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)

# Define a stacking ensemble using CatBoost, XGBoost, and Random Forest as base learners
base_learners = [('catboost', cat), ('xgb', xgb), ('rf', rf)]
meta_model = LogisticRegression(random_state=42)
stacking_clf = StackingClassifier(estimators=base_learners, 
                                  final_estimator=meta_model, 
                                  passthrough=True, cv=5)
stacking_clf.fit(X_train, y_train)
results["Stacking Ensemble"] = evaluate_model(stacking_clf, X_train, y_train, X_test, y_test)

results_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]).T
print("Performance Comparison:")
print(results_df)

# ============================================================
# 3. Visualization
# ============================================================

# Accuracy Comparison Bar Chart
plt.figure(figsize=(10,6))
sns.barplot(x=results_df.index, y=results_df["Accuracy"], palette="viridis")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.show()

# ROC Curve Comparison
plt.figure(figsize=(10,8))
def plot_roc(model, name, X_test, y_test):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    y_test_numeric = y_test.map({'No': 0, 'Yes': 1})
    fpr, tpr, _ = roc_curve(y_test_numeric, y_prob)
    auc_val = roc_auc_score(y_test_numeric, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.2f})")
    
for name, model in {**models, "Stacking Ensemble": stacking_clf}.items():
    plot_roc(model, name, X_test, y_test)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix for the Stacking Ensemble
from sklearn.metrics import confusion_matrix
y_pred_stack = stacking_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted No", "Predicted Yes"],
            yticklabels=["Actual No", "Actual Yes"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Stacking Ensemble")
plt.show()

# ============================================================
# 4. CTGAN-based Data Augmentation (GenAI) and Retraining
# ============================================================

# Combine X_train and y_train into one DataFrame for CTGAN training
train_df = X_train.copy()
train_df["Attrition"] = y_train
train_df = train_df.dropna()
print("Original target distribution:")
print(train_df["Attrition"].value_counts())

# Define discrete (categorical) columns. Adjust as needed.
discrete_columns = ["Attrition", "BusinessTravel", "Department", "EducationField",
                    "EnvironmentSatisfaction", "Gender", "JobRole", "JobSatisfaction",
                    "MaritalStatus", "OverTime", "RelationshipSatisfaction", "WorkLifeBalance"]

# Convert each discrete column to string
for col in discrete_columns:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str)
        print(f"Column '{col}' converted to string. Unique values: {train_df[col].unique()}")

# Train CTGAN on the training data, specifying discrete columns
ctgan_model = CTGAN(epochs=300)
ctgan_model.fit(train_df, discrete_columns=discrete_columns)

# Generate synthetic data equal to the original training set size
synthetic_data = ctgan_model.sample(len(train_df))
print("Synthetic target sample (first 10):")
print(synthetic_data["Attrition"].head(10))

# Create an augmented dataset
augmented_df = pd.concat([train_df, synthetic_data], axis=0).reset_index(drop=True)
X_train_aug = augmented_df.drop("Attrition", axis=1)
y_train_aug = augmented_df["Attrition"]

# Convert augmented target values to discrete labels if necessary (thresholding if numeric)
if not pd.api.types.is_string_dtype(y_train_aug):
    y_train_aug = y_train_aug.apply(lambda x: "Yes" if x >= 0.5 else "No")
if np.issubdtype(y_test.dtype, np.number):
    y_test = pd.Series(y_test).map({0: "No", 1: "Yes"})

# Retrain the stacking ensemble on the augmented training data
stacking_clf.fit(X_train_aug, y_train_aug)
y_pred_aug = stacking_clf.predict(X_test)
aug_acc = accuracy_score(y_test, y_pred_aug)
print(f"Accuracy after CTGAN-based augmentation: {aug_acc:.4f}")

# ============================================================
# 5. Explainable AI (XAI) using SHAP on the Stacking Ensemble
# ============================================================
if not isinstance(X_test, pd.DataFrame):
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
else:
    X_test_df = X_test.copy()

explainer = shap.Explainer(stacking_clf.predict_proba, X_test_df)
shap_values = explainer(X_test_df)
shap.summary_plot(shap_values, X_test_df)
