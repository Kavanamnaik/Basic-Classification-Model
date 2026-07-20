
import os
 
os.makedirs("output", exist_ok=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================
# Step 1: Load Dataset
# ============================================

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

# Convert numeric labels into names
df["species_name"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================
# Step 2: Data Visualization
# ============================================

# Count Plot
plt.figure(figsize=(6,4))
sns.countplot(x="species_name", data=df)
plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.savefig(
    "output/count_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Pair Plot
sns.pairplot(df, hue="species_name")
plt.savefig(
    "output/pair_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))

correlation = df.iloc[:, :-2].corr()

sns.heatmap(correlation,
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.savefig(
    "output/heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ============================================
# Step 3: Split Dataset
# ============================================

X = df.iloc[:, :-2]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ============================================
# Step 4: Logistic Regression
# ============================================

lr_model = LogisticRegression(max_iter=200)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n===================================")
print("LOGISTIC REGRESSION")
print("===================================")

print("Accuracy:", lr_accuracy)

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

lr_cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    lr_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(
    "output/logistic_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ============================================
# Step 5: Decision Tree
# ============================================

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\n===================================")
print("DECISION TREE")
print("===================================")

print("Accuracy:", dt_accuracy)

print("\nClassification Report")
print(classification_report(y_test, dt_pred))

dt_cm = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    dt_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(
    "output/decision_tree_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ============================================
# Step 6: Model Comparison
# ============================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree"
    ],
    "Accuracy": [
        lr_accuracy,
        dt_accuracy
    ]
})

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(comparison)

# Best Model
best_model = comparison.loc[
    comparison["Accuracy"].idxmax(),
    "Model"
]

print("\nBest Performing Model:", best_model)