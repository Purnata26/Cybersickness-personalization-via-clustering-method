# =========================================================
# Unsupervised Cybersickness Detection Pipeline
# Label  : FMS at end of window > median → 1, else 0
#          GT per participant = majority vote across their windows
# Features: mean + std of windowed physiological signals per participant
# Methods : PCA+HDBSCAN | UMAP+HDBSCAN | UMAP+KMeans | UMAP+Agglomerative
# Eval    : Silhouette, ARI, NMI
# Scope   : Global + per physio-cluster (0 and 1), re-fit independently
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score,
                             adjusted_rand_score,
                             normalized_mutual_info_score)
from scipy.stats import mode
import hdbscan

try:
    import umap
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn", "-q"])
    import umap


# =========================================================
# A. CONFIG
# =========================================================
feature_cols = [
    "EDA",
    "LeftPupilDiameter",     "RightPupilDiameter",
    "LeftPupilPosInSensorX", "LeftPupilPosInSensorY",
    "RightPupilPosInSensorX","RightPupilPosInSensorY",
    "NrmSRLeftEyeGazeDirX",  "NrmSRLeftEyeGazeDirY",  "NrmSRLeftEyeGazeDirZ",
    "NrmSRRightEyeGazeDirX", "NrmSRRightEyeGazeDirY", "NrmSRRightEyeGazeDirZ"
]

cluster_map = {
    1: 0,  7: 0,  6: 0, 16: 0, 15: 0, 13: 0, 12: 0,  9: 0,
    8: 1,  5: 1,  4: 1,  2: 1,  3: 1, 10: 1, 14: 1, 17: 1,
   18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 11: 1
}

TARGET_COL   = "FMS"
SEQUENCE_LEN = 100
RANDOM_STATE = 42


# =========================================================
# B. BUILD WINDOWS + FMS LABELS PER WINDOW
# =========================================================
def build_feature_matrix(df_physio, feature_cols, target_col, seq_len):
    """
    Returns
    -------
    feat_df     : DataFrame (n_participants × 2F)
    labels      : Series   (n_participants,)  majority FMS class (0/1)
    valid_pids  : list
    window_info : dict  pid → {n_windows, win_labels, maj_label, fms_sick_pct}
    """
    df = df_physio.copy()
    df["participant"] = (df["participant"].astype(str)
                         .str.extract(r"(\d+)", expand=False).astype(int))
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.sort_values(["participant", "Timestamp"]).reset_index(drop=True)

    # global FMS median across ALL valid rows — fixed before any filtering
    all_fms    = df[target_col].dropna()
    fms_median = all_fms.median()
    print(f"Global FMS median (all rows): {fms_median:.2f}")
    print(f"FMS range : [{all_fms.min():.0f} – {all_fms.max():.0f}]")
    print(f"FMS unique: {sorted(all_fms.unique())}")

    rows, valid_pids, window_info = [], [], {}

    for pid, g in df.groupby("participant"):
        g = g.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

        if len(g) < seq_len:
            print(f"  PID {pid}: only {len(g)} valid rows — skipping")
            continue


        scaler    = MinMaxScaler()
        feat_vals = scaler.fit_transform(g[feature_cols]).astype(np.float32)
        fms_arr   = g[target_col].values.astype(np.float32)

        windows, win_labels = [], []
        for i in range(0, len(feat_vals) - seq_len, seq_len):
            win_fms   = fms_arr[i + seq_len - 1]        # FMS at END of window
            win_label = int(win_fms > fms_median)        # 1 = sick
            windows.append(feat_vals[i : i + seq_len])
            win_labels.append(win_label)

        if len(windows) == 0:
            print(f"  PID {pid}: no windows produced — skipping")
            continue

        windows    = np.array(windows)      # (N, seq_len, F)
        win_labels = np.array(win_labels)   # (N,)

        mean_vec = windows.mean(axis=(0, 1))
        std_vec  = windows.std(axis=(0, 1))
        rows.append(np.concatenate([mean_vec, std_vec]))

        maj_label = int(mode(win_labels, keepdims=True).mode[0])
        valid_pids.append(pid)
        window_info[pid] = {
            "n_windows":    len(windows),
            "win_labels":   win_labels,
            "maj_label":    maj_label,
            "fms_sick_pct": win_labels.mean() * 100
        }

        print(f"  PID {pid:>3}: {len(windows):>4} windows | "
              f"sick%={win_labels.mean()*100:.1f}% | "
              f"GT(majority)={maj_label} | "
              f"cluster={cluster_map.get(pid,'?')}")

    col_names = ([f"{c}_mean" for c in feature_cols] +
                 [f"{c}_std"  for c in feature_cols])
    feat_df   = pd.DataFrame(rows, columns=col_names, index=valid_pids)
    labels    = pd.Series(
        {pid: window_info[pid]["maj_label"] for pid in valid_pids},
        name="GT_label"
    )

    print(f"\nParticipants retained : {len(valid_pids)}  →  {valid_pids}")
    print(f"Label distribution    : {labels.value_counts().to_dict()}  "
          f"(0=low FMS, 1=high FMS)")

    return feat_df, labels, valid_pids, window_info


print("Building feature matrix...")
feat_df, labels, valid_pids, window_info = build_feature_matrix(
    final, feature_cols, TARGET_COL, SEQUENCE_LEN
)
print(f"\nFeature matrix shape: {feat_df.shape}")


# =========================================================
# C. METRICS HELPER
# =========================================================
def compute_metrics(true_labels, pred_labels, embedding,
                    method_name, scope_name):
    pred = np.array(pred_labels, dtype=int).copy()
    true = np.array(true_labels, dtype=int).copy()

    # reassign HDBSCAN noise (−1) to nearest cluster centroid
    noise_mask = pred == -1
    n_noise    = int(noise_mask.sum())

    if n_noise > 0:
        unique_cl = np.unique(pred[~noise_mask])
        if len(unique_cl) == 0:
            print(f"  [{scope_name}|{method_name}] ALL noise — skipped")
            return None
        centroids = np.array([embedding[pred == c].mean(axis=0)
                              for c in unique_cl])
        for idx in np.where(noise_mask)[0]:
            dists     = np.linalg.norm(centroids - embedding[idx], axis=1)
            pred[idx] = unique_cl[dists.argmin()]

    n_clusters = len(np.unique(pred))
    if n_clusters < 2:
        print(f"  [{scope_name}|{method_name}] <2 clusters — skipped")
        return None

    # align cluster indices to GT via majority vote
    aligned = pred.copy()
    for c in np.unique(pred):
        mask = pred == c
        maj  = int(mode(true[mask], keepdims=True).mode[0])
        aligned[mask] = maj

    sil = silhouette_score(embedding, pred)
    ari = adjusted_rand_score(true, aligned)
    nmi = normalized_mutual_info_score(true, aligned)

    return {
        "scope":          scope_name,
        "method":         method_name,
        "n_clusters":     n_clusters,
        "n_noise":        n_noise,
        "Silhouette":     round(sil, 4),
        "ARI":            round(ari, 4),
        "NMI":            round(nmi, 4),
        "pred_labels":    pred,
        "aligned_labels": aligned,
    }


# =========================================================
# D. PIPELINE RUNNER
# =========================================================
METHODS = ["PCA+HDBSCAN", "UMAP+HDBSCAN", "UMAP+KMeans", "UMAP+Agglomerative"]

def run_pipeline(feat_matrix, true_labels, scope_name,
                 random_state=RANDOM_STATE):

    X = StandardScaler().fit_transform(feat_matrix)
    y = np.array(true_labels, dtype=int)
    n = len(X)

    results    = {}
    embeddings = {}
    min_cs     = max(2, n // 5)

    # ── PCA + HDBSCAN ──────────────────────────────────────
    pca_emb = PCA(n_components=2,
                  random_state=random_state).fit_transform(X)
    embeddings["PCA+HDBSCAN"] = pca_emb

    hdb1 = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=1,
                            prediction_data=True)
    r = compute_metrics(y, hdb1.fit_predict(pca_emb),
                        pca_emb, "PCA+HDBSCAN", scope_name)
    if r:
        results["PCA+HDBSCAN"] = r

    # ── UMAP (shared embedding for 3 methods) ──────────────
    n_neighbors = max(2, min(15, n - 1))
    umap_emb    = umap.UMAP(n_components=2,
                             n_neighbors=n_neighbors,
                             min_dist=0.1,
                             random_state=random_state).fit_transform(X)
    for key in ["UMAP+HDBSCAN", "UMAP+KMeans", "UMAP+Agglomerative"]:
        embeddings[key] = umap_emb

    hdb2 = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=1,
                            prediction_data=True)
    r = compute_metrics(y, hdb2.fit_predict(umap_emb),
                        umap_emb, "UMAP+HDBSCAN", scope_name)
    if r:
        results["UMAP+HDBSCAN"] = r

    r = compute_metrics(
        y,
        KMeans(n_clusters=2, random_state=random_state,
               n_init=20).fit_predict(umap_emb),
        umap_emb, "UMAP+KMeans", scope_name
    )
    if r:
        results["UMAP+KMeans"] = r

    r = compute_metrics(
        y,
        AgglomerativeClustering(n_clusters=2).fit_predict(umap_emb),
        umap_emb, "UMAP+Agglomerative", scope_name
    )
    if r:
        results["UMAP+Agglomerative"] = r

    for method, res in results.items():
        print(f"  {method:<22} | clusters={res['n_clusters']} "
              f"noise={res['n_noise']} | "
              f"Sil={res['Silhouette']:.4f}  "
              f"ARI={res['ARI']:.4f}  "
              f"NMI={res['NMI']:.4f}")

    return results, embeddings


# =========================================================
# E. GLOBAL RUN
# =========================================================
print("\n" + "="*65)
print("GLOBAL  (all retained participants)")
print("="*65)
global_results, global_embs = run_pipeline(
    feat_df.values, labels.values, "Global"
)


# =========================================================
# F. PER PHYSIO-CLUSTER RUN
# =========================================================
cluster_results = {}
cluster_embs    = {}
cluster_pids    = {}
cluster_labels  = {}

for c in [0, 1]:
    pids_c  = [p for p in valid_pids
               if cluster_map.get(p) == c and p in labels.index]
    idx_c   = [valid_pids.index(p) for p in pids_c]
    feat_c  = feat_df.values[idx_c]
    label_c = labels.values[idx_c]

    print(f"\n{'='*65}")
    print(f"PHYSIO-CLUSTER {c}  |  n={len(pids_c)}  PIDs={pids_c}")
    print(f"  FMS label dist: "
          f"{dict(zip(*np.unique(label_c, return_counts=True)))}")
    print(f"  Per-PID detail:")
    for pid in pids_c:
        wi = window_info[pid]
        print(f"    PID {pid:>3}: {wi['n_windows']} windows | "
              f"sick%={wi['fms_sick_pct']:.1f}% | GT={wi['maj_label']}")
    print(f"{'='*65}")

    if len(np.unique(label_c)) < 2:
        print("  Only one FMS class present — skipping")
        continue
    if len(pids_c) < 4:
        print("  Too few participants — skipping")
        continue

    res, embs = run_pipeline(feat_c, label_c, f"Cluster-{c}")
    cluster_results[c] = res
    cluster_embs[c]    = embs
    cluster_pids[c]    = pids_c
    cluster_labels[c]  = label_c


# =========================================================
# G. SUMMARY TABLE
# =========================================================
all_rows = []
for res in [global_results] + list(cluster_results.values()):
    for method, r in res.items():
        all_rows.append({k: v for k, v in r.items()
                         if k not in ("pred_labels", "aligned_labels")})

summary_df = pd.DataFrame(all_rows)
print("\n" + "="*65)
print("FULL SUMMARY")
print("="*65)
print(summary_df.to_string(index=False))


# =========================================================
# H. WINDOW-LEVEL FMS DISTRIBUTION PLOT
# =========================================================
fig0, ax0 = plt.subplots(figsize=(14, 4))
pids_sorted = sorted(valid_pids)
sick_pcts   = [window_info[p]["fms_sick_pct"] for p in pids_sorted]
gt_labs     = [window_info[p]["maj_label"]    for p in pids_sorted]
bar_colors  = ["#F44336" if l == 1 else "#4CAF50" for l in gt_labs]

ax0.bar(range(len(pids_sorted)), sick_pcts,
        color=bar_colors, edgecolor="k", linewidth=0.5)
ax0.axhline(50, color="gray", ls="--", lw=1.2, label="50% threshold")
ax0.set_xticks(range(len(pids_sorted)))
ax0.set_xticklabels([f"P{p}" for p in pids_sorted], rotation=45, ha="right")
ax0.set_ylabel("% windows with FMS > median")
ax0.set_title("Per-Participant FMS Sick% across Windows\n"
              "(color = majority GT: red=sick, green=not sick)")
ax0.legend()
ax0.grid(axis="y", alpha=0.3)

for i, pid in enumerate(pids_sorted):
    cl = cluster_map.get(pid, "?")
    ax0.text(i, sick_pcts[i] + 1.5, f"C{cl}",
             ha="center", fontsize=7, color="navy")

plt.tight_layout()
plt.savefig("fms_window_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# =========================================================
# I. SCATTER PLOTS
# =========================================================
scope_info = [
    ("Global", global_results, global_embs, valid_pids, labels.values),
]
for c in [0, 1]:
    if c not in cluster_results:
        continue
    scope_info.append((
        f"Cluster-{c}",
        cluster_results[c], cluster_embs[c],
        cluster_pids[c],    cluster_labels[c]
    ))

n_rows = len(scope_info)
n_cols = len(METHODS)
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5.5*n_cols, 4.5*n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

gt_colors  = {0: "#4CAF50", 1: "#F44336"}
cl_markers = {0: "o", 1: "s", 2: "^"}

for row, (scope_name, res_dict, emb_dict, pids_s, labels_s) in \
        enumerate(scope_info):
    for col, method in enumerate(METHODS):
        ax  = axes[row, col]
        r   = res_dict.get(method)
        emb = emb_dict.get(method)

        if r is None or emb is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        pred_cl = r["pred_labels"]
        for i, (pid, gt, pc) in enumerate(zip(pids_s, labels_s, pred_cl)):
            ax.scatter(emb[i, 0], emb[i, 1],
                       color=gt_colors[int(gt)],
                       marker=cl_markers.get(int(pc % 3), "o"),
                       s=110, edgecolors="k", linewidths=0.6, zorder=3)
            spct = f"{window_info[pid]['fms_sick_pct']:.0f}%"
            ax.annotate(f"P{pid}\n{spct}", (emb[i, 0], emb[i, 1]),
                        fontsize=5.5, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points")

        ax.set_title(
            f"{scope_name} · {method}\n"
            f"Sil={r['Silhouette']:.3f}  "
            f"ARI={r['ARI']:.3f}  "
            f"NMI={r['NMI']:.3f}",
            fontsize=8.5
        )
        ax.set_xlabel("Dim 1", fontsize=7)
        ax.set_ylabel("Dim 2", fontsize=7)
        ax.grid(alpha=0.25)

legend_handles = [
    mpatches.Patch(color="#4CAF50", label="GT: Low FMS majority (0)"),
    mpatches.Patch(color="#F44336", label="GT: High FMS majority (1)"),
    plt.Line2D([0],[0], marker="o", color="gray", ls="None",
               ms=8, label="Pred cluster A"),
    plt.Line2D([0],[0], marker="s", color="gray", ls="None",
               ms=8, label="Pred cluster B"),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=4, bbox_to_anchor=(0.5, -0.015), fontsize=9)
plt.suptitle(
    "Unsupervised Cybersickness Detection (FMS-based GT)\n"
    "Color = majority FMS class   |   Marker = Predicted cluster   |   "
    "Annotation = PID + sick%",
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig("cybersickness_scatter.png", dpi=150, bbox_inches="tight")
plt.show()


# =========================================================
# J. METRICS BAR CHART
# =========================================================
metric_names = ["Silhouette", "ARI", "NMI"]
scope_colors = {
    "Global":    "#2196F3",
    "Cluster-0": "#FF9800",
    "Cluster-1": "#9C27B0"
}

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric in zip(axes2, metric_names):
    x       = np.arange(len(METHODS))
    bar_w   = 0.22
    scopes  = ["Global", "Cluster-0", "Cluster-1"]
    offsets = np.linspace(-bar_w, bar_w, len(scopes))

    for off, scope in zip(offsets, scopes):
        if scope == "Global":
            res_dict = global_results
        else:
            c = int(scope[-1])
            res_dict = cluster_results.get(c, {})

        vals = [res_dict[m][metric] if m in res_dict else 0.0
                for m in METHODS]
        ax.bar(x + off, vals, width=bar_w,
               label=scope, color=scope_colors[scope],
               alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f"{metric} by Method & Scope", fontsize=10)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle(
    "Unsupervised Cybersickness (FMS) — Silhouette / ARI / NMI\n"
    "Global vs Physio-Cluster 0 vs Physio-Cluster 1",
    fontsize=12
)
plt.tight_layout()
plt.savefig("cybersickness_metrics.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDONE")