Fraud Detection in Financial Transactions

Project Overview

This project focuses on detecting fraudulent financial transactions using machine learning techniques. The dataset used is the creditcard.csv, which contains anonymized transaction data.

Dataset Overview

Rows: 284,807

Columns: 31 (Features: V1 to V28, Time, Amount, and Class)

Class Distribution:

Non-Fraudulent Transactions (0): 284,315

Fraudulent Transactions (1): 492

Missing Values: 0

Data Preprocessing

Standardized the features using StandardScaler.

Applied SMOTE to handle class imbalance with a sampling strategy of 10%.

Split the dataset into training (80%) and testing (20%).

Models Used

Random Forest Classifier

XGBoost Classifier

Isolation Forest (Anomaly Detection)

Model Evaluation

Random Forest Classifier:

Confusion Matrix:

[[56830    11]
 [   67  5642]]

Classification Report:

              precision    recall  f1-score   support

          0       1.00      1.00      1.00     56841
          1       1.00      0.99      0.99      5709

   accuracy                           1.00     62550
  macro avg       1.00      0.99      1.00     62550

weighted avg       1.00      1.00      1.00     62550

- **ROC AUC Score:** 0.9940

### XGBoost Classifier:
- **Confusion Matrix:**

[[56832     9]
[   11  5698]]

- **Classification Report:**

            precision    recall  f1-score   support

        0       1.00      1.00      1.00     56841
        1       1.00      1.00      1.00      5709

 accuracy                           1.00     62550
macro avg       1.00      1.00      1.00     62550

weighted avg       1.00      1.00      1.00     62550

- **ROC AUC Score:** 0.9989

## Significance and Observations
### Confusion Matrix:
- The confusion matrix provides a detailed breakdown of model performance in terms of correctly and incorrectly classified instances.
- For both classifiers, false negatives (fraudulent transactions incorrectly classified as non-fraudulent) are minimal, indicating strong fraud detection capabilities.
- False positives are also minimal, meaning legitimate transactions are rarely misclassified as fraud.

### Classification Report:
- **Precision:** Measures how many transactions classified as fraud are actually fraud. A high precision indicates fewer false positives.
- **Recall:** Measures how many actual fraudulent transactions were correctly classified. High recall ensures minimal false negatives, which is critical in fraud detection.
- **F1-Score:** The harmonic mean of precision and recall, providing a balanced measure of model performance.

### ROC AUC Score:
- The ROC AUC score measures the model’s ability to distinguish between fraudulent and non-fraudulent transactions.
- A score close to 1.0 (e.g., 0.9989 for XGBoost) suggests near-perfect classification capability, confirming that the model performs exceptionally well in fraud detection.

## Visualizations
- **Anomaly Scores Distribution (Isolation Forest)**
- **PCA Projection of Fraud and Non-Fraud Transactions**

## How to Run
1. Clone the repository:
 ```sh
 git clone https://github.com/Shrungal-DSAI/Fraud-Detection-Financial.git

Install dependencies:

pip install -r requirements.txt

Run the Python script:

python fraud_detection.py

Conclusion

Both the Random Forest and XGBoost classifiers performed exceptionally well, with XGBoost achieving a near-perfect ROC AUC score of 0.9989. Future work could include additional anomaly detection techniques and feature engineering to further improve fraud detection performance.

Author

Shrungal Kulkarni

For any queries, feel free to reach out or raise an issue in this repository!

