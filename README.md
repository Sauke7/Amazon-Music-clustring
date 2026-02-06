# 🎵 Amazon Music Clustering

## 📌 Project Overview

With millions of songs available on music streaming platforms, manually categorizing tracks into genres is not scalable. This project uses **unsupervised machine learning** to automatically group Amazon Music songs based on their **audio characteristics**, without using predefined genre labels.

The goal is to identify meaningful clusters that represent different **musical moods or styles** such as party tracks, chill acoustic songs, instrumental music, etc.

---

## 🎯 Objectives

* Perform data cleaning and preprocessing on music audio features
* Apply clustering algorithms to group similar songs
* Evaluate clustering quality using standard metrics
* Interpret and label clusters using audio feature averages
* Visualize clusters using dimensionality reduction

---

## 🧠 Techniques & Algorithms Used

* **Data Preprocessing**: Handling missing values, feature selection
* **Feature Scaling**: StandardScaler
* **Clustering Algorithms**:

  * K-Means (primary algorithm)
  * DBSCAN (density-based, noise detection)
  * Hierarchical Clustering (Agglomerative)
* **Evaluation Metrics**:

  * Silhouette Score
  * Davies–Bouldin Index
* **Dimensionality Reduction**:

  * PCA (for visualization only)

---

## 📂 Dataset Description

* **File Name**: `single_genre_artists.csv`

* **Audio Features Used**:

  * danceability
  * energy
  * loudness
  * speechiness
  * acousticness
  * instrumentalness
  * liveness
  * valence
  * tempo
  * duration_ms

* **Dropped Columns** (identifiers / non-audio):

  * song ID, song name, artist name, artist ID, release date, genres

---

## ⚙️ Project Workflow

1. Load and explore the dataset
2. Drop non-audio columns and handle missing values
3. Scale features using StandardScaler
4. Determine optimal number of clusters using Elbow Method & Silhouette Score
5. Apply K-Means clustering
6. Evaluate clusters using quantitative metrics
7. Interpret clusters using mean feature values
8. Assign human-readable labels to clusters
9. Visualize clusters using PCA
10. Export final dataset with cluster labels

---

## 📊 Cluster Interpretation Example

* **Cluster 0**: High energy, high danceability → *Party / Workout Tracks*
* **Cluster 1**: Low energy, high acousticness → *Chill / Acoustic*
* **Cluster 3**: High instrumentalness → *Instrumental / Focus Music*

---

## 📈 Results

* Songs were grouped into distinct clusters based on audio similarity
* PCA visualization showed clear separation between clusters
* Cluster labels improved interpretability for recommendation use cases

---

## 📁 Output Files

* `amazon_music_clustered_output.csv`

  * Contains original song data
  * Includes cluster ID and human-readable cluster label

---

## 🛠 Tools & Libraries

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

## 🚀 Future Enhancements

* Build a Streamlit web app for interactive exploration
* Use t-SNE or UMAP for advanced visualization
* Integrate with recommendation systems

---

## 👨‍💻 Author

**Kammila Jaya Sai Haranadh**

---

⭐ If you find this project useful, feel free to star the repository!
