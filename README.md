# 🛡️ Fraud Detection in Financial Transactions  

🚀 **Detecting fraudulent financial transactions using machine learning techniques**  

---

## 📌 Project Overview  
This project focuses on detecting fraud in financial transactions using machine learning techniques.  
We analyze a highly imbalanced dataset and apply **SMOTE** to balance the class distribution.  
The models implemented include **Random Forest, XGBoost, and Isolation Forest**, achieving high accuracy in fraud detection.  

---

## 📂 Dataset Overview  
| Feature | Description |  
|---------|------------|  
| 📊 **Rows** | 284,807 |  
| 📊 **Columns** | 31 (V1 to V28, Time, Amount, Class) |  
| ✅ **Non-Fraud Transactions (0)** | 284,315 |  
| 🚨 **Fraudulent Transactions (1)** | 492 |  
| ❌ **Missing Values** | 0 |  

---

## 🔄 Data Preprocessing  
✔️ Standardized features using `StandardScaler`  
✔️ Applied **SMOTE** (Synthetic Minority Over-sampling) for class balance (sampling strategy: **10%**)  
✔️ Split dataset into **80% training** and **20% testing**  

---

## 🤖 Models & Performance  
| Model | Precision | Recall | F1-Score | ROC AUC Score |  
|--------|----------|--------|----------|---------------|  
[[56830 11]
[ 67 5642]]
✔️ **Classification Report**  
           precision    recall  f1-score   support  
0          1.00        1.00       1.00      56841  
1          1.00        0.99       0.99       5709  
accuracy 1.00 62550
macro avg 1.00 0.99 1.00 62550
weighted avg 1.00 1.00 1.00 62550
✔️ **ROC AUC Score:** `0.9940`  

#### 🔹 **XGBoost Classifier**  
✔️ **Confusion Matrix**  
[[56832 9]
[ 11 5698]]
✔️ **Classification Report**  
           precision    recall  f1-score   support  
0          1.00        1.00       1.00      56841  
1          1.00        1.00       1.00       5709  
accuracy 1.00 62550
macro avg 1.00 1.00 1.00 62550
weighted avg 1.00 1.00 1.00 62550
✔️ **ROC AUC Score:** `0.9989`  

---

## 📈 Key Observations  
✔️ **Low False Negatives:** Fraudulent transactions are rarely misclassified as non-fraudulent.  
✔️ **Low False Positives:** Genuine transactions are rarely misclassified as fraud.  
✔️ **High ROC AUC Score:** Near-perfect classification capability, ensuring strong fraud detection performance.  

---

## 📊 Visualizations  
📌 **Anomaly Scores Distribution (Isolation Forest)**  
📌 **PCA Projection of Fraud vs. Non-Fraud Transactions**  

---

## ⚙️ How to Run  

### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/Shrungal-DSAI/Fraud-Detection-Financial.git
cd Fraud-Detection-Financial
| **Random Forest** | 1.00 | 0.99 | 0.99 | 0.9940 |  
| **XGBoost** | 1.00 | 1.00 | 1.00 | 0.9989 |  
| **Isolation Forest** | Used for anomaly detection | - | - | - |  

### 📊 Model Evaluation  
#### 🔹 **Random Forest Classifier**  
✔️ **Confusion Matrix**

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Script
python fraud_detection.py

🔮 Future Improvements
🚀 Try Deep Learning models (e.g., LSTM, Autoencoders) for anomaly detection
🚀 Optimize hyperparameters using Bayesian Optimization
🚀 Deploy as a real-time fraud detection API using Flask or FastAPI

✨ Author
👤 Shrungal Kulkarni
📧 Email
🔗 GitHub

⭐ If you found this project helpful, please consider giving it a star! 🌟
