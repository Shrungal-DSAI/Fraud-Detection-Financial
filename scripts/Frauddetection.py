import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.impute import KNNImputer
from sklearn.metrics import make_scorer
import warnings

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
CONTAMINATION = 0.01
N_NEIGHBORS_KNN = 5

# Suppress warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_data(filepath="C:/Users/Admin/Downloads/creditcard.csv"): 
    """Loads data, handles missing values, and duplicates."""
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Number of duplicate rows: {data.duplicated().sum()}")
    data.drop_duplicates(inplace=True)
    print(f"Shape after removing duplicates: {data.shape}")

    if data.isnull().sum().any():
        print("Missing values detected. Imputing using KNN.")
        imputer = KNNImputer(n_neighbors=N_NEIGHBORS_KNN)
        data[:] = imputer.fit_transform(data)

    return data

def perform_eda(data):
    """Performs exploratory data analysis."""
    print("\nData Shape:", data.shape)
    print("\nClass Distribution:\n", data['Class'].value_counts())

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    sns.countplot(x='Class', data=data, ax=axes[0, 0]).set_title('Class Distribution')
    sns.boxplot(x='Class', y='Amount', data=data, ax=axes[0, 1]).set(title='Amount Distribution by Class', yscale='log')
    data['Hour'] = data['Time'].apply(lambda x: x / 3600)  # Convert seconds to hours
    sns.histplot(x='Hour', hue='Class', data=data, ax=axes[1, 0]).set_title('Transactions per Hour')
    sns.heatmap(data.corr(), cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1, 1]).set_title('Correlation Matrix')
    data.drop('Hour', axis=1, inplace=True)  # Remove the temporary 'Hour' column
    plt.tight_layout()
    plt.show()

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Trains and evaluates classification models using GridSearchCV."""
    models = {
        "Logistic Regression": (LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, solver='liblinear', penalty='l1', class_weight='balanced'), {}),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'), {'n_estimators': [150, 250], 'max_depth':[10, 15], 'min_samples_split':[2,5]}),
        "MLPClassifier": (MLPClassifier(random_state=RANDOM_STATE, max_iter=5000, early_stopping=True), {'hidden_layer_sizes': [(100,50), (50, 25)], 'alpha':[0.0001, 0.00001], 'learning_rate':['constant', 'adaptive']}),
    }

    results = {}
    scoring = {'f1': make_scorer(f1_score), 'precision': make_scorer(precision_score), 'recall': make_scorer(recall_score), 'roc_auc': make_scorer(roc_auc_score), 'average_precision': make_scorer(average_precision_score)}

    for name, (model, param_grid) in models.items():
        print(f"\nTraining and Evaluating {name}...")
        grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=N_FOLDS), scoring=scoring, refit='recall', n_jobs=-1, return_train_score=True, verbose=1)
        grid_search.fit(X_train, y_train)
        results[name] = grid_search

        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best Recall for {name}: {grid_search.best_score_}")

        y_prob = grid_search.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})')

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label = f'{name} (AP = {average_precision_score(y_test, y_prob):.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.show()
    return results

def perform_anomaly_detection(X_scaled, data):
    """Performs anomaly detection using Isolation Forest and One-Class SVM."""
    iso_forest = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    iso_forest.fit(X_scaled)
    data['iso_forest_anomaly'] = iso_forest.predict(X_scaled)

    one_class_svm = OneClassSVM(nu=0.001, kernel="rbf", gamma='auto')
    one_class_svm.fit(X_scaled)
    data['one_class_svm_anomaly'] = one_class_svm.predict(X_scaled)

    # Anomaly Analysis and Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(x='iso_forest_anomaly', y='Amount', data=data, ax=axes[0]).set_title('Amount by Isolation Forest Anomaly')
    sns.boxplot(x='one_class_svm_anomaly', y='Amount', data=data, ax=axes[1]).set_title('Amount by One-Class SVM Anomaly')
    plt.show()

    print("\nIsolation Forest Anomaly Counts:\n", data['iso_forest_anomaly'].value_counts())
    print("\nOne-Class SVM Anomaly Counts:\n", data['one_class_svm_anomaly'].value_counts())
    return data

def main():
    """Main function to orchestrate the project."""
    print("Project: Fraud Detection")
    data = load_and_preprocess_data() 
    perform_eda(data)

    X = data.drop(['Class', 'Time'], axis=1)
    y = data['Class']

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=RANDOM_STATE, n_jobs=-1, sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_resampled)

    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    best_model_name = max(results, key=lambda k: results[k].cv_results_['mean_test_recall'][results[k].best_index_])