import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import kruskal, chi2_contingency

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================================================
# 1. LOAD DATA
# =========================================================
df = final.copy()
df = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------
# CLEAN PARTICIPANT IDS
# handles:
# 24(Left Early)
# 25(Fragment)
# etc.
# ---------------------------------------------------------
df["participant"] = (
    df["participant"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype(int)
)

metadata["participant"] = (
    metadata["participant"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .astype(int)
)


# =========================================================
# 2. FEATURES
# =========================================================
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

sequence_length = 100


# =========================================================
# 3. WINDOW FUNCTION
# =========================================================
def segment_data(data, seq_len):

    X = []

    for i in range(0, len(data) - seq_len, seq_len):
        X.append(data[i:i + seq_len])

    return np.array(X)


# =========================================================
# 4. BUILD WINDOW DATASET
# =========================================================
window_X = []
window_pid = []

participants_in_X = []

for pid, g in df.groupby("participant"):

    g = g.reset_index(drop=True)

    if len(g) < sequence_length:
        continue

    scaler = MinMaxScaler()

    g[feature_cols] = scaler.fit_transform(
        g[feature_cols]
    )

    X_seg = segment_data(
        g[feature_cols].values,
        sequence_length
    )

    if len(X_seg) == 0:
        continue

    window_X.append(X_seg)

    # IMPORTANT
    window_pid.extend([pid] * len(X_seg))

    participants_in_X.append(pid)

# ---------------------------------------------------------
# CONCATENATE
# ---------------------------------------------------------
X_all = np.concatenate(window_X, axis=0)
pid_all = np.array(window_pid)

print("\nParticipants in physiological data:", len(participants_in_X))
print("Windowed shape:", X_all.shape)
print("PID shape:", pid_all.shape)

print("\nUnique participants:")
print(np.unique(pid_all))


# =========================================================
# 5. PCA
# =========================================================
X_flat = X_all.reshape(X_all.shape[0], -1)

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_flat)

print("\nExplained variance:")
print(pca.explained_variance_ratio_)


# =========================================================
# 6. FILTER METADATA
# =========================================================
metadata_filtered = metadata[
    metadata["participant"].isin(participants_in_X)
].copy()

metadata_filtered = (
    metadata_filtered
    .drop_duplicates(subset="participant")
)

print("\nMetadata rows after filtering:")
print(len(metadata_filtered))


# =========================================================
# 7. SAFE NUMERIC CONVERSION
# =========================================================
ssq_cols = [
    c for c in metadata_filtered.columns
    if "SSQ" in c
]

metadata_filtered[ssq_cols] = (
    metadata_filtered[ssq_cols]
    .apply(pd.to_numeric, errors="coerce")
)


# =========================================================
# 8. KENNEDY SSQ COMPUTATION
# =========================================================
def compute_ssq(df):

    def safe(cols):

        cols = [c for c in cols if c in df.columns]

        return (
            df[cols]
            .apply(pd.to_numeric, errors="coerce")
            .sum(axis=1)
        )

    # -----------------------------------------------------
    # PRE
    # -----------------------------------------------------
    nausea_pre = [
        "SSQ [Nausea]",
        "SSQ [Salivation increasing]",
        "SSQ [Sweating]",
        "SSQ [*** Stomach awareness]",
        "SSQ [Burping]"
    ]

    oculo_pre = [
        "SSQ [General discomfort]",
        "SSQ [Fatigue]",
        "SSQ [Headache]",
        "SSQ [Eye strain]",
        "SSQ [Difficulty focusing]",
        "SSQ [Difficulty concentrating]",
        "SSQ [Blurred vision]"
    ]

    dis_pre = [
        "SSQ [Difficulty focusing]",
        "SSQ [Nausea]",
        "SSQ [*Fullness of the Head]",
        "SSQ [Blurred vision]",
        "SSQ [Dizziness with eyes open]",
        "SSQ [Dizziness with eyes closed]",
        "SSQ [** Vertigo]"
    ]

    # -----------------------------------------------------
    # POST
    # -----------------------------------------------------
    nausea_post = [c + " Number" for c in nausea_pre]
    oculo_post = [c + " Number" for c in oculo_pre]
    dis_post = [c + " Number" for c in dis_pre]

    # -----------------------------------------------------
    # KENNEDY WEIGHTS
    # -----------------------------------------------------
    df["pre_nausea"] = safe(nausea_pre) * 9.54
    df["pre_oculomotor"] = safe(oculo_pre) * 7.58
    df["pre_disorientation"] = safe(dis_pre) * 13.92

    df["pre_total"] = (
        safe(nausea_pre) +
        safe(oculo_pre) +
        safe(dis_pre)
    ) * 3.74

    df["post_nausea"] = safe(nausea_post) * 9.54
    df["post_oculomotor"] = safe(oculo_post) * 7.58
    df["post_disorientation"] = safe(dis_post) * 13.92

    df["post_total"] = (
        safe(nausea_post) +
        safe(oculo_post) +
        safe(dis_post)
    ) * 3.74

    return df


metadata_filtered = compute_ssq(metadata_filtered)


# =========================================================
# 9. EFFECT SIZE
# =========================================================
def epsilon_squared(H, n, k):

    return (H - k + 1) / (n - k)


# =========================================================
# 10. SAFE KRUSKAL
# =========================================================
def safe_kruskal(groups_data):

    all_vals = np.concatenate(groups_data)

    if np.nanstd(all_vals) == 0:
        return None, None

    try:
        return kruskal(*groups_data)

    except:
        return None, None


# =========================================================
# 11. STATS
# =========================================================
def run_stats(df):

    results = {}

    groups = sorted(df["cluster"].unique())

    k = len(groups)
    n = len(df)

    ssq_vars = [
        "pre_nausea",
        "pre_oculomotor",
        "pre_disorientation",
        "pre_total",
        "post_nausea",
        "post_oculomotor",
        "post_disorientation",
        "post_total"
    ]

    for var in ssq_vars:

        samples = [
            df[df["cluster"] == g][var].dropna().values
            for g in groups
        ]

        if any(len(s) < 2 for s in samples):
            continue

        H, p = safe_kruskal(samples)

        if H is None:
            continue

        results[var] = {
            "p_value": round(p, 4),
            "epsilon_sq": round(
                epsilon_squared(H, n, k), 4
            )
        }

    return results


# =========================================================
# 12. CLUSTERING LOOP
# =========================================================
k_values = [2, 3, 4, 5]

for k in k_values:

    print("\n" + "=" * 80)
    print(f"K = {k}")
    print("=" * 80)

    # -----------------------------------------------------
    # KMEANS
    # -----------------------------------------------------
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    win_clusters = kmeans.fit_predict(X_pca)

    sil = silhouette_score(
        X_pca,
        win_clusters
    )

    print(f"\nSilhouette Score: {sil:.4f}")

    # -----------------------------------------------------
    # MAJORITY VOTE PER PARTICIPANT
    # -----------------------------------------------------
    participant_clusters = {}

    for pid in np.unique(pid_all):

        mask = pid_all == pid

        votes = win_clusters[mask]

        if len(votes) == 0:
            continue

        majority = pd.Series(votes).mode()[0]

        participant_clusters[pid] = majority

    # -----------------------------------------------------
    # MERGE WITH METADATA
    # -----------------------------------------------------
    temp = metadata_filtered.copy()

    temp["cluster"] = temp["participant"].map(
        participant_clusters
    )

    # -----------------------------------------------------
    # PRINT ASSIGNMENTS
    # -----------------------------------------------------
    print("\nParticipant → Cluster assignment:")
    print(
        temp[
            ["participant", "cluster"]
        ]
        .sort_values("cluster")
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # CLUSTER COUNTS
    # -----------------------------------------------------
    print("\nCluster counts:")
    print(
        temp["cluster"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # REMOVE SINGLETONS
    # -----------------------------------------------------
    counts = temp["cluster"].value_counts()

    outliers = counts[
        counts == 1
    ].index

    temp = temp[
        ~temp["cluster"].isin(outliers)
    ].copy()

    if temp["cluster"].nunique() < 2:

        print("\nSkipped after singleton removal")
        continue

    # -----------------------------------------------------
    # RELABEL
    # -----------------------------------------------------
    mapping = {
        old: i
        for i, old in enumerate(
            sorted(temp["cluster"].unique())
        )
    }

    temp["cluster"] = temp["cluster"].map(mapping)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary = temp.groupby("cluster").agg(

    N=("cluster", "count"),

    pre_nausea=("pre_nausea", "mean"),
    pre_oculomotor=("pre_oculomotor", "mean"),
    pre_disorientation=("pre_disorientation", "mean"),
    pre_total=("pre_total", "mean"),

    post_nausea=("post_nausea", "mean"),
    post_oculomotor=("post_oculomotor", "mean"),
    post_disorientation=("post_disorientation", "mean"),
    post_total=("post_total", "mean"),

    age=("Age", "mean"),

    gender=(
        "Sex",
        lambda x: x.value_counts().to_dict()
    )

).reset_index()

    summary["Group"] = [
        f"Group {i+1}"
        for i in summary["cluster"]
    ]

    print("\nSUMMARY:")
    print(summary)

    # -----------------------------------------------------
    # STATS
    # -----------------------------------------------------
    stats = run_stats(temp)

    print("\nSTATS:")
    for var, res in stats.items():
        print(f"{var}: {res}")

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------
    win_majority = np.array([
        participant_clusters.get(pid, -1)
        for pid in pid_all
    ])

    plt.figure(figsize=(10, 7))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=win_majority,
        cmap="tab10",
        alpha=0.7,
        s=40,
        edgecolors="k",
        linewidths=0.3
    )

    plt.title(
        f"PCA + KMeans (K={k})"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    handles = [
        Patch(
            color=plt.cm.tab10(i / k),
            label=f"Cluster {i+1}"
        )
        for i in range(k)
    ]

    plt.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

print("\nDONE")