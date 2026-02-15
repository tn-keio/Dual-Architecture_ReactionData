
# Abnormal Reaction Detection System
# Copyright (C) 2026
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(history, save_path='training_loss.png'):
    """Plots training and validation loss."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_reconstruction_with_error(original, reconstruction, mse, sample_idx_label, save_path=None):
    """
    Plots a single sample: Original, Reconstruction, and Error area.
    """
    plt.figure(figsize=(10, 4))
    
    # Trace
    plt.plot(original, 'b-', label='Original')
    plt.plot(reconstruction, 'r--', label='Reconstruction')
    
    # Fill error
    error = np.abs(original - reconstruction)
    plt.fill_between(range(len(original)), original, reconstruction, color='lightcoral', alpha=0.5, label='Error')
    
    plt.title(f"Sample: {sample_idx_label} | MSE: {mse:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_anomaly_heatmap(original, error_vector, sample_idx_label, save_path=None):
    """
    Plots the original trace colored by the magnitude of reconstruction error.
    This serves as the 'Error Map' / Anomaly Localization.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    x = np.arange(len(original))
    y = original
    
    # Scatter plot with color mapped to error
    # Using scatter to show intensity at each point
    sc = ax.scatter(x, y, c=error_vector, cmap='jet', s=50, edgecolors='none', label='Error Intensity')
    ax.plot(x, y, 'k-', alpha=0.3) # Faint line to show connectivity
    
    plt.colorbar(sc, label='Squared Error')
    
    ax.set_title(f"Anomaly Localization (Sample: {sample_idx_label})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Normalized Value")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_top_anomalies(X, reconstructions, errors, indices, df_labels, top_n=5, save_prefix='anomaly'):
    """
    Plots top anomalies with both reconstruction view and heatmap view.
    """
    for i in range(min(top_n, len(indices))):
        idx = indices[i]
        orig = X[idx].flatten()
        recon = reconstructions[idx].flatten()
        err_vec = errors[idx] # vector of squared errors for this sample
        # logic to determine sample_label
        if 'NO' in df_labels.columns:
             sample_label = df_labels.iloc[idx]['NO']
        elif 'id' in df_labels.columns:
             sample_label = df_labels.iloc[idx]['id']
        else:
             sample_label = str(idx)
        mse = np.mean(err_vec)
        
        # Plot reconstruction comparison
        plot_reconstruction_with_error(
            orig, recon, mse, 
            sample_idx_label=f"{sample_label} (Rank {i+1})", 
            save_path=f"{save_prefix}_rank{i+1}_recon.png"
        )
        
        # Plot Heatmap
        plot_anomaly_heatmap(
            orig, err_vec, 
            sample_idx_label=f"{sample_label} (Rank {i+1})", 
            save_path=f"{save_prefix}_rank{i+1}_heatmap.png"
        )

def plot_mse_distribution(mse_scores, threshold, save_path=None):
    """
    Plots the distribution of MSE scores with the threshold.
    """
    plt.figure(figsize=(12, 6))
    
    # Scatter plot
    indices = np.arange(len(mse_scores))
    is_anomaly = mse_scores > threshold
    
    plt.scatter(indices[~is_anomaly], mse_scores[~is_anomaly], c='blue', alpha=0.5, label='Normal', s=10)
    plt.scatter(indices[is_anomaly], mse_scores[is_anomaly], c='red', alpha=0.8, label='Anomaly', s=20)
    
    # Threshold line
    plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.4f})')
    
    plt.title('Reconstruction Error (MSE) Distribution')
    plt.xlabel('Sample Index')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_group_comparison(X, is_anomaly_mask, save_path=None):
    """
    Plots the Mean +/- StdDev for Normal vs Anomaly groups.
    """
    X = X.squeeze() # Remove channel dim if present: (N, 38, 1) -> (N, 38)
    
    normal_data = X[~is_anomaly_mask]
    anomaly_data = X[is_anomaly_mask]
    
    # Calculate stats
    mean_normal = np.mean(normal_data, axis=0)
    std_normal = np.std(normal_data, axis=0)
    
    mean_anomaly = np.mean(anomaly_data, axis=0)
    std_anomaly = np.std(anomaly_data, axis=0)
    
    x_axis = np.arange(X.shape[1])
    
    plt.figure(figsize=(12, 6))
    
    # Plot Normal
    plt.plot(x_axis, mean_normal, 'b-', label='Normal Mean')
    plt.fill_between(x_axis, mean_normal - std_normal, mean_normal + std_normal, color='blue', alpha=0.1, label='Normal ±1 Std')
    
    # Plot Anomaly
    if len(anomaly_data) > 0:
        plt.plot(x_axis, mean_anomaly, 'r-', label='Anomaly Mean')
        plt.fill_between(x_axis, mean_anomaly - std_anomaly, mean_anomaly + std_anomaly, color='red', alpha=0.1, label='Anomaly ±1 Std')
    else:
        print("No anomalies found for group comparison plot.")
    
    plt.title('Reaction Trace Comparison: Normal vs Anomaly')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
