import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("rice_features.csv")

print(df.columns)

# Features and labels
X = df[['area', 'perimeter', 'major_axis', 'minor_axis', 
        'aspect_ratio', 'eccentricity', 'solidity', 'roundness','R_mean',
       'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std']]
y = df['label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# --- Plot 1: with labels ---
plt.figure(figsize=(8,6))
for label in y.unique():
    idx = y == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Rice Shape Features (With Labels)")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot 2: without labels ---
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], color='gray', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Rice Shape Features (No Labels)")
plt.grid(True)
plt.show()
