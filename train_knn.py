import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load CSV
df = pd.read_csv("rice_features.csv")

# Features and labels
X = df[['area', 'perimeter', 'major_axis', 'minor_axis', 'aspect_ratio', 
        'eccentricity', 'solidity', 'roundness',
        'R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std']]
y = df['label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize KNN
knn = KNeighborsClassifier()  # start with k=5

# Define hyperparameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13],
    'weights': ['distance'],
    'metric': ['euclidean']
}

# Grid search with 5-fold cross-validation
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

# Best parameters
print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

best_knn = grid.best_estimator_

# Predict
y_pred = best_knn.predict(X_test_scaled)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {acc:.4f}")

# Detailed report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred, labels=df['label'].unique())
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=df['label'].unique(), yticklabels=df['label'].unique())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - KNN")
plt.show()