import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# -------------------- Load data --------------------
df = final.copy()
df = df.dropna().reset_index(drop=True)

# -------------------- FIXED feature list --------------------
feature_cols = [
    "EDA",
    "LeftPupilDiameter",
    "RightPupilDiameter",
    "LeftPupilPosInSensorX",
    "LeftPupilPosInSensorY",
    "RightPupilPosInSensorX",
    "RightPupilPosInSensorY",
    "NrmSRLeftEyeGazeDirX",
    "NrmSRLeftEyeGazeDirY",
    "NrmSRLeftEyeGazeDirZ",
    "NrmSRRightEyeGazeDirX",
    "NrmSRRightEyeGazeDirY",
    "NrmSRRightEyeGazeDirZ"
]

target_col = "participant"
sequence_length = 100

# -------------------- Windowing --------------------
def segment_data(data, target, seq_len):
    X, y = [], []
    for i in range(0, len(data) - seq_len, seq_len):
        X.append(data[i:i+seq_len])
        y.append(target[i])  # participant constant → take start label
    return np.array(X), np.array(y)

# -------------------- Build dataset --------------------
X_all, y_all = [], []

for pid, g in df.groupby("participant"):
    g = g.dropna().reset_index(drop=True)

    scaler = MinMaxScaler()
    g[feature_cols] = scaler.fit_transform(g[feature_cols])

    X = g[feature_cols].values
    y = g[target_col].values

    X_seg, y_seg = segment_data(X, y, sequence_length)

    X_all.append(X_seg)
    y_all.append(y_seg)

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print("Windowed shape:", X_all.shape)

# -------------------- Flatten windows --------------------
X_flat = X_all.reshape(X_all.shape[0], -1)

# -------------------- PCA --------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flat)

print("Explained variance:", pca.explained_variance_ratio_)

# -------------------- KMeans --------------------
k = len(np.unique(y_all))
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

sil = silhouette_score(X_pca, clusters)
print(f"\nSilhouette Score: {sil:.4f}")

# -------------------- Visualization (FIXED) --------------------
num_participants = len(np.unique(y_all))
palette = sns.color_palette("hls", num_participants)
color_map = mcolors.ListedColormap(palette)

plt.figure(figsize=(12, 8))

# IMPORTANT: show CLUSTERS (not participant labels)
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="tab10",
    alpha=0.8,
    s=50,
    edgecolors='k'
)

plt.title("PCA + KMeans Clusters", fontsize=14)
plt.xlabel("PC1")
plt.ylabel("PC2")

# Legend for clusters
handles = [
    Patch(color=plt.cm.tab10(i / k), label=f"P{i+1}")
    for i in range(k)
]

plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()