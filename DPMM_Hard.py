# Hard assignment of data to cluster.

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import chi2
from matplotlib.patches import Ellipse

# --- Data Generation & DPMM Fitting ---
data_1, _ = make_blobs(n_samples=500, centers=[[10, 5], [15, 8], [8, 12]], cluster_std=[1.5, 1.2, 1.0], random_state=42)
data_2, _ = make_blobs(n_samples=100, centers=[[20, 15], [5, 5]], cluster_std=[0.5, 0.7], random_state=1)
X = np.vstack((data_1, data_2))
df = pd.DataFrame(X, columns=['Luminosity (L)', 'Color Index (C)'])

# DPGMM setup: Uses DP prior to automatically find the number of clusters
dpmm = BayesianGaussianMixture(
    n_components=10, 
    covariance_type='full', 
    weight_concentration_prior_type='dirichlet_process', 
    max_iter=500, 
    random_state=42
)
dpmm.fit(df) # Training determines the cluster centers and weights

# --- CRITICAL STEP: HARD ASSIGNMENT ---
# .predict(df) returns the single Cluster_ID (hard assignment) with the 
# highest posterior probability for each data point. This uses the optimal 
# number of clusters determined by the DP prior during .fit().
hard_assignments = dpmm.predict(df) 
df['Cluster_ID'] = hard_assignments

# --- Analysis & Ellipse Setup (Uses the Hard Assignments) ---
active_clusters_mask = dpmm.weights_ > 0.01
component_means = dpmm.means_[active_clusters_mask]
component_covariances = dpmm.covariances_[active_clusters_mask]

# Color Synchronization
full_seaborn_palette = sns.color_palette('deep', n_colors=10)
all_indices = np.arange(dpmm.n_components)
active_component_ids = all_indices[active_clusters_mask]

# --- Ellipse Function ---
def plot_gaussian_ellipse(ax, mean, covariance, color, z=2):
    scale_factor = np.sqrt(chi2.ppf(0.95, 2))
    lambda_, v = np.linalg.eigh(covariance)
    lambda_ = np.sqrt(lambda_)
    angle = np.degrees(np.arctan2(*v[:, 0][::-1]))
    
    ellipse = Ellipse(
        xy=mean,
        width=lambda_[0] * scale_factor,
        height=lambda_[1] * scale_factor,
        angle=angle,
        color=color,
        alpha=0.3,
        linestyle='--',
        linewidth=4,
        fill=False
    )
    ax.add_patch(ellipse)

# --- PLOTTING (Matplotlib Only) ---
fig, ax = plt.subplots(figsize=(10, 7))

# 1. Manual Color Mapping for Scatter Plot
data_colors = [full_seaborn_palette[i] for i in df['Cluster_ID']]
ax.scatter(
    x=df['Color Index (C)'],
    y=df['Luminosity (L)'],
    c=data_colors,
    alpha=0.7,
    s=50 
)

# 2. Plot ellipses
for i, (mean, cov) in enumerate(zip(component_means, component_covariances)):
    cluster_id = active_component_ids[i]
    ellipse_color = full_seaborn_palette[cluster_id]

    # Correct Axis Handling: (x=C, y=L)
    center = (mean[1], mean[0]) 
    new_cov = cov[::-1, ::-1]   
    
    plot_gaussian_ellipse(ax, center, new_cov, ellipse_color)

# 3. Final Touches
ax.set_title('DPGMM: Hard Assignment with Optimized Cluster Count')
ax.set_xlabel('Color Index (C)')
ax.set_ylabel('Luminosity (L)')

# 4. Manual Legend
unique_ids = np.unique(df['Cluster_ID'])
legend_handles = []
for id_val in unique_ids:
    if id_val in active_component_ids:
        handle = plt.Line2D(
            [0], [0], 
            marker='o', 
            color='w', 
            markerfacecolor=full_seaborn_palette[id_val], 
            label=f'Population ID {id_val}',
            markersize=10
        )
        legend_handles.append(handle)

ax.legend(handles=legend_handles, title='Population ID', loc='best')
plt.show()