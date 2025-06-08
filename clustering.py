
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset (Replace 'your_dataset.csv' with the actual file)
df = pd.read_csv("your_dataset.csv")

# Check dataset structure
print(df.head())

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

from sklearn.cluster import KMeans

# Apply K-Means with an initial guess of 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Centroids
print("Cluster Centers:\n", kmeans.cluster_centers_)



# Test different values of K
inertia_values = []
k_values = range(1, 10)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia_values.append(model.inertia_)

# Plot Elbow Curve
plt.plot(k_values, inertia_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

from sklearn.decomposition import PCA
import seaborn as sns

# Reduce dimensions to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA results to DataFrame
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Scatter Plot with Clusters
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df, palette="Set2")
plt.title("K-Means Clustering Visualization (2D PCA)")
plt.show()

from sklearn.metrics import silhouette_score

# Compute Silhouette Score
score = silhouette_score(X_scaled, df["Cluster"])
print(f"Silhouette Score: {score:.2f}")
