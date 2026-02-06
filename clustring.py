import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# ==========================================
# Load the Dataset
# ==========================================
df = pd.read_csv("single_genre_artists.csv")

# ==========================================
# Basic Data Exploration
# ==========================================
print("Shape:", df.shape)
df.info()
print(df.isnull().sum())

# ==========================================
# Drop Non-Audio Columns & Handle NaN
# ==========================================
df_features = df.drop(
    columns=[
        "id_songs",
        "name_song",
        "name_artists",
        "id_artists",
        "release_date",
        "genres"
    ]
)

df_features = df_features.dropna()
df = df.loc[df_features.index]

# ==========================================
# Feature Scaling
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

print("Final scaled shape:", X_scaled.shape)

# ==========================================
# Elbow Method
# ==========================================
inertia = []

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# ==========================================
# Silhouette Score for K Selection
# ==========================================
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K = {k} | Silhouette Score = {score:.3f}")

# ==========================================
# Apply K-Means
# ==========================================
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

df["kmeans_cluster"] = kmeans_labels

# ==========================================
# K-Means Evaluation
# ==========================================
print("K-Means Silhouette Score:",
      silhouette_score(X_scaled, kmeans_labels))

print("K-Means Davies-Bouldin Index:",
      davies_bouldin_score(X_scaled, kmeans_labels))

# ==========================================
# Cluster Size Balance
# ==========================================
print(df["kmeans_cluster"].value_counts())

# ==========================================
# Feature Interpretability (Cluster Profiling)
# ==========================================
cluster_profile = df_features.copy()
cluster_profile["kmeans_cluster"] = kmeans_labels

cluster_summary = cluster_profile.groupby("kmeans_cluster").mean()
print(cluster_summary)

# ==========================================
# Auto Label Clusters (INTERPRETATION)
# ==========================================
cluster_names = {
    0: "Party / Workout",
    1: "Chill / Acoustic",
    2: "Happy / Pop",
    3: "Instrumental / Focus",
    4: "Balanced / Mixed Mood"
}

df["cluster_label"] = df["kmeans_cluster"].map(cluster_names)

# ==========================================
# PCA Visualization (Using Labels)
# ==========================================
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
pca_df["cluster_label"] = df["cluster_label"]

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="cluster_label",
    palette="tab10"
)
plt.title("K-Means Clustering with Interpreted Labels (PCA)")
plt.show()

# ==========================================
# Export Final CSV
# ==========================================
df.to_csv("amazon_music_clustered_output.csv", index=False)
