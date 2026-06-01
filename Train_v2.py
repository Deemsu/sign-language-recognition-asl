"""
3_Train_v2.py — Enhanced Model Training (A-Z, 26 classes)
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib
import sklearn
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# --- Paths ---
DATA_DIR    = "sign_data"
DATA_FILE   = os.path.join(DATA_DIR, "augmented_data.csv")
MODEL_PATH  = "sign_model.pkl"
REPORTS_DIR = "training_reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 60)
print("  Sign Language Model Training — A-Z (26 classes)")
print("=" * 60)

if not os.path.exists(DATA_FILE):
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")] if os.path.exists(DATA_DIR) else []
    if not csv_files:
        print(f"[ERROR] No data file found in '{DATA_DIR}'.")
        print("Please run previous steps first.")
        exit()
    DATA_FILE = os.path.join(DATA_DIR, csv_files[0])
    print(f"[INFO] Using: {DATA_FILE}")

print(f"\n[1/6] Loading data from '{DATA_FILE}'...")
df = pd.read_csv(DATA_FILE)

for col in df.columns:
    if col != 'label':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

X = df.drop("label", axis=1).astype(np.float32).values
y = df["label"].values
labels = sorted(set(y))

print(f"    Total samples : {len(df):,}")
print(f"    Total signs   : {len(labels)}")
print(f"    Signs         : {labels}")

print("\n[2/6] Sample distribution per sign:")
for label in labels:
    count = (y == label).sum()
    bar   = "█" * (count // 20)
    print(f"    {label:5s}: {count:5d}  {bar}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

X_train_s = np.nan_to_num(X_train_s.astype(np.float32))
X_test_s  = np.nan_to_num(X_test_s.astype(np.float32))

# Models
models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=1,
        max_depth=10,
    ),
    "SVM":SVC (
        kernel='rbf',
        C=10,
        gamma='scale',
        random_state=42,
        probability =True 
    ),
    "KNN": KNeighborsClassifier (
        n_neighbors=5 , n_jobs=1
    ),
}

print("\n[3/6] Cross-validation...")
y_train = np.array(y_train, dtype=str)
os.environ["Loky_MAX_CPU_COUNT"] = "1"  # Avoid thread issues with joblib in some environments
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_results = {}

for name, clf in models.items():
    print(f"Training {name}...")
    scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    cv_results[name] = scores
    print(f"{name}: {scores.mean():.4f} (±{scores.std():.4f})")

best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")

best_model.fit(X_train_s, y_train)

print ("\n[4/6] Training all models on full training set ")
test_results= {}
for name,clf in models.items():
    clf.fit(X_train_s, y_train)
    pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    test_results [name] = {"acc": acc, "pred": pred}
print(f"Accuracy: {acc:.4f}")

# Save model
joblib.dump({
    "model": best_model,
    "scaler": scaler,
    "labels": labels
}, MODEL_PATH)

print(f"Model saved: {MODEL_PATH}")

# ── 1. Model Comparison Bar Chart 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

model_names = list(test_results.keys())
cv_means = [cv_results[n].mean() for n in model_names]
test_accs = [test_results[n]["acc"] for n in model_names]

x = range(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar([i - width/2 for i in x], cv_means, width, label='CV Accuracy (3-fold)', color='gray')
bars2 = ax.bar([i + width/2 for i in x], test_accs, width, label='Test Accuracy', color='black')

ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Model Comparison — CV vs Test Accuracy', fontsize=12, fontweight='bold')
ax.set_xticks(list(x))
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0.7, 1.0)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

for bar in bars1:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'model_comparison.png'), dpi=150)
plt.close()
print("Saved: model_comparison.png")

# ── 2. Confusion Matrix (best model) 
y_pred = best_model.predict(X_test_s)
test_results[best_model_name]["pred"] = y_pred
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys',
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor='gray')
plt.title(f'Confusion Matrix — {best_model_name} (28 Classes)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# ── 3. Accuracy vs n_estimators (RF only) ──────────
print("\nGenerating accuracy curve...")
estimator_range = [10, 20, 30, 50, 75, 100, 150, 200]
accuracies = []

for n in estimator_range:
    rf = RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42, n_jobs=1)
    rf.fit(X_train_s, y_train)
    acc_n = accuracy_score(y_test, rf.predict(X_test_s))
    accuracies.append(acc_n)
    print(f"  n={n}: {acc_n:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(estimator_range, accuracies, color='black', linewidth=1.5, marker='o', markersize=5)
plt.axvline(x=50, color='gray', linestyle='--', linewidth=1)
plt.annotate('n=50 (selected)',
             xy=(50, accuracies[estimator_range.index(50)]),
             xytext=(60, accuracies[estimator_range.index(50)] - 0.002),
             fontsize=9, color='black')
plt.xlabel('Number of Trees (n_estimators)', fontsize=11)
plt.ylabel('Test Accuracy', fontsize=11)
plt.title('Random Forest — Accuracy vs Number of Trees', fontsize=12, fontweight='bold')
plt.ylim(0.95, 1.0)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'accuracy_curve.png'), dpi=150)
plt.close()
print("Saved: accuracy_curve.png")