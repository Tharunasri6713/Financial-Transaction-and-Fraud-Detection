"""
Financial Transaction Fraud Detection
-------------------------------------
A machine learning project to detect fraudulent financial transactions
using a Random Forest Classifier and comparing it with a simple Rule-Based System.
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Generate or Load Dataset
# -----------------------------
# Here we create a simulated dataset for demo purposes.
# In a real-world project, replace this with your actual dataset file.
num_samples = 10000

data = {
    'transaction_id': np.arange(1, num_samples + 1),
    'description': np.random.choice(
        ['Payment', 'Refund', 'Withdrawal', 'Deposit'], num_samples
    ),
    'amount': np.random.uniform(50, 500, num_samples),
    'is_fraud': np.random.choice([0, 1], num_samples, p=[0.97, 0.03])  # 3% fraud rate
}

df = pd.DataFrame(data)
print("✅ Dataset created successfully!")
print(df.head())

# -----------------------------
# Step 2: Preprocessing
# -----------------------------
from sklearn.feature_extraction.text import CountVectorizer

X_text = df['description']
vectorizer = CountVectorizer()
X_text_encoded = vectorizer.fit_transform(X_text)

# Combine encoded text with numeric features
X = np.hstack((X_text_encoded.toarray(), df[['amount']].values))
y = df['is_fraud']

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Step 4: Random Forest Model
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)

# -----------------------------
# Step 5: Predictions
# -----------------------------
y_pred = rf_model.predict(X_test)

# -----------------------------
# Step 6: Evaluation Metrics
# -----------------------------
rf_accuracy = accuracy_score(y_test, y_pred)
rf_precision = precision_score(y_test, y_pred)
rf_recall = recall_score(y_test, y_pred)
rf_f1 = f1_score(y_test, y_pred)

print("\n📊 Random Forest Model Performance:")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")

# -----------------------------
# Step 7: Confusion Matrix
# -----------------------------
rf_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# Step 8: Simple Rule-Based Model (for comparison)
# -----------------------------
def rule_based_model(df):
    fraud_keywords = ['Withdrawal', 'Deposit']
    predictions = []
    for desc in df['description']:
        if any(keyword in desc for keyword in fraud_keywords):
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)

rule_preds = rule_based_model(df)
rule_acc = accuracy_score(y, rule_preds)
rule_prec = precision_score(y, rule_preds)
rule_rec = recall_score(y, rule_preds)
rule_f1 = f1_score(y, rule_preds)

print("\n🧩 Rule-Based System Performance:")
print(f"Accuracy:  {rule_acc:.4f}")
print(f"Precision: {rule_prec:.4f}")
print(f"Recall:    {rule_rec:.4f}")
print(f"F1 Score:  {rule_f1:.4f}")

# -----------------------------
# Step 9: Compare Both Models
# -----------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rule_metrics = [rule_acc, rule_prec, rule_rec, rule_f1]
rf_metrics = [rf_accuracy, rf_precision, rf_recall, rf_f1]

plt.figure(figsize=(8, 5))
x = np.arange(len(metrics))
plt.bar(x - 0.2, rule_metrics, width=0.4, label='Rule-Based', color='skyblue')
plt.bar(x + 0.2, rf_metrics, width=0.4, label='Random Forest', color='salmon')
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Model Comparison: Rule-Based vs Random Forest")
plt.legend()
plt.show()

print("\n✅ Comparison completed successfully!")
