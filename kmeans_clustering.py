# kmeans_clustering.py — Phase 3: K-Means Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── 1. Load RFM output from Phase 2 ────────────────────────────────────────
rfm = pd.read_csv('rfm_output.csv')
print(f"Loaded {len(rfm)} customers")

# ── 2. Prepare features for clustering ─────────────────────────────────────
# Log-transform Monetary & Frequency (reduces skew from outliers)
rfm['log_Monetary']  = np.log1p(rfm['Monetary'])
rfm['log_Frequency'] = np.log1p(rfm['Frequency'])

features = rfm[['Recency', 'log_Frequency', 'log_Monetary']]

# Standardize so no feature dominates
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ── 3. Elbow Method — find optimal k ───────────────────────────────────────
inertia = []
silhouette = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertia, 'bo-')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].axvline(x=4, color='red', linestyle='--', label='Optimal k=4')
axes[0].legend()

axes[1].plot(K_range, silhouette, 'go-')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].axvline(x=4, color='red', linestyle='--', label='Optimal k=4')
axes[1].legend()

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150)
plt.show()

# ── 4. Train final K-Means model with k=4 ──────────────────────────────────
# (adjust k based on what elbow/silhouette tells you)
OPTIMAL_K = 4

kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# ── 5. Profile each cluster ─────────────────────────────────────────────────
cluster_profile = rfm.groupby('Cluster').agg(
    Count      = ('CustomerID',  'count'),
    Avg_Recency    = ('Recency',     'mean'),
    Avg_Frequency  = ('Frequency',   'mean'),
    Avg_Monetary   = ('Monetary',    'mean'),
    Total_Revenue  = ('Monetary',    'sum')
).round(1)

print("\nCluster Profiles:")
print(cluster_profile)

# Auto-label clusters based on their profile
def label_cluster(row):
    if row['Avg_Recency'] < 30 and row['Avg_Frequency'] > 5:
        return 'High Value'
    elif row['Avg_Recency'] < 60 and row['Avg_Monetary'] > 1000:
        return 'Promising'
    elif row['Avg_Recency'] > 200:
        return 'Lost'
    else:
        return 'Needs Attention'

cluster_profile['Label'] = cluster_profile.apply(label_cluster, axis=1)
print("\nCluster Labels:")
print(cluster_profile[['Count','Label','Avg_Monetary','Total_Revenue']])

# Map labels back to main dataframe
label_map = cluster_profile['Label'].to_dict()
rfm['Cluster_Label'] = rfm['Cluster'].map(label_map)

# ── 6. Visualize clusters with PCA (2D) ────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

rfm['PCA1'] = X_pca[:, 0]
rfm['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
for i, label in enumerate(rfm['Cluster_Label'].unique()):
    mask = rfm['Cluster_Label'] == label
    plt.scatter(rfm[mask]['PCA1'], rfm[mask]['PCA2'],
                label=label, alpha=0.6, s=30, color=colors[i % len(colors)])

plt.title('Customer Clusters (PCA View)', fontsize=14, fontweight='bold')
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.legend()
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=150)
plt.show()

# Save final output 
rfm.to_csv('rfm_clustered.csv', index=False)
print("\nSaved rfm_clustered.csv") 
print(f"Variance explained by 2 PCA components: {sum(pca.explained_variance_ratio_)*100:.1f}%")
print(rfm.groupby('Cluster_Label')[['Recency','Frequency','Monetary']].mean().round(1))
