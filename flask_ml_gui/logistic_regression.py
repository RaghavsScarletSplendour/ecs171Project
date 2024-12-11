import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle

#set the seed for reproducibility
np.random.seed(42)

#load the scaled and encoded dataset
data_scaled = pd.read_csv("../ECS171_FINAL_DATASET.csv")

#define the features (X) and target variable (y)
X = data_scaled[['VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor',
                 'Quarter_Q1', 'Quarter_Q2', 'Quarter_Q3', 'Quarter_Q4',
                   'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'TrafficType']]
y = data_scaled['Revenue']  # Target variable

#apply smote to handle class imbalance by oversampling the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

#split the resampled data into training and test sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

#initialize logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

#train the logistic regression model
log_reg.fit(X_train, y_train)

#predictions on the test set (probabilities)
y_test_pred_proba = log_reg.predict_proba(X_test)[:, 1]  #probs for the positive class (Revenue = 1)

#chose a custom threshold and it works well
threshold = 0.3
y_test_pred = (y_test_pred_proba >= threshold).astype(int)

#evaluation metrics
print("\nModel Performance After Oversampling with Threshold Adjustment:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1 Score:", f1_score(y_test, y_test_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_test_pred_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))


with open("log_reg_model.pkl", "wb") as file:
    pickle.dump(log_reg, file)