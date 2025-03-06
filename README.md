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
| **Random Forest** | 1.00 | 0.99 | 0.99 | 0.9940 |  
| **XGBoost** | 1.00 | 1.00 | 1.00 | 0.9989 |  
| **Isolation Forest** | Used for anomaly detection | - | - | - |  

### 📊 Model Evaluation  
#### 🔹 **Random Forest Classifier**  
✔️ **Confusion Matrix**  
