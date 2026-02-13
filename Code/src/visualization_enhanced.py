
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def plot_combined_report(s_id, 
                         waveforms_dict, 
                         reconstructions_dict, 
                         errors_dict, 
                         is_anomaly_dict,
                         save_path):
    """
    Generate a comprehensive report figure for a single sample (S_ID).
    
    Args:
        s_id: Sample ID
        waveforms_dict: {'Main': (N,), 'Sub': (N,), 'Calc': (N,)}
        reconstructions_dict: {'Main': (N,), 'Sub': (N,), 'Calc': (N,)}
        errors_dict: {'Main': (N,), 'Sub': (N,), 'Calc': (N,)} - Element-wise errors
        is_anomaly_dict: {'Main': bool, 'Sub': bool, 'Calc': bool} - Anomaly flags
        save_path: Path to save the figure
    """
    
    modes = ['Main', 'Sub', 'Calc']
    colors = {'Main': 'blue', 'Sub': 'green', 'Calc': 'red'}
    
    fig = plt.figure(figsize=(20, 12))
    plt.suptitle(f"Anomaly Report: Sample ID {s_id}", fontsize=20, weight='bold')
    
    # Create grid
    # Row 1: Original Waveforms (Superimposed or Side-by-Side?) -> Side-by-Side for clarity
    # Row 2: Reconstruction vs Original + Error Heatmap (The "GradCAM" equivalent)
    # Row 3: Residuals (Error magnitude)
    
    gs = fig.add_gridspec(3, 3)
    
    for i, mode in enumerate(modes):
        if mode not in waveforms_dict:
            continue
            
        wave = waveforms_dict[mode]
        recon = reconstructions_dict[mode]
        error = errors_dict[mode]
        is_anom = is_anomaly_dict.get(mode, False)
        
        status_color = 'red' if is_anom else 'green'
        status_text = "ANOMALY" if is_anom else "Normal"
        
        # --- Row 1: Basic Waveform ---
        ax1 = fig.add_subplot(gs[0, i])
        ax1.plot(wave, color=colors[mode], label=f'{mode} Input')
        ax1.set_title(f"{mode} Mode\nStatus: {status_text}", color=status_color, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # --- Row 2: "Error Heatmap" (Reconstruction Focus) ---
        # Plot Original line, but colored by Error magnitude
        ax2 = fig.add_subplot(gs[1, i])
        
        # Create line segments for multicolor line
        points = np.array([np.arange(len(wave)), wave]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Error for coloring (smooth it slightly for better visuals?)
        # Use element-wise error for coloring segments
        # Align error with segments (take avg of adjacent points)
        seg_errors = (error[:-1] + error[1:]) / 2
        
        # Normalize error to 0-1 range locally or globally? 
        # Locally emphasizes the "bad part" of THIS wave.
        norm = Normalize(vmin=0, vmax=np.max(error) if np.max(error) > 0 else 1)
        lc = LineCollection(segments, cmap='hot', norm=norm)
        lc.set_array(seg_errors)
        lc.set_linewidth(2)
        
        line = ax2.add_collection(lc)
        ax2.set_xlim(0, len(wave))
        ax2.set_ylim(np.min(wave)*1.1, np.max(wave)*1.1)
        
        # Also plot reconstruction as dashed gray
        ax2.plot(recon, color='gray', linestyle='--', alpha=0.5, label='Reconstruction')
        
        ax2.set_title(f"{mode} Error Map (Hot=High Error)")
        plt.colorbar(line, ax=ax2, label='Reconstruction Error')
        ax2.legend()
        
        # --- Row 3: Residuals ---
        ax3 = fig.add_subplot(gs[2, i])
        ax3.fill_between(range(len(error)), error, color='orange', alpha=0.5)
        ax3.plot(error, color='darkorange', linewidth=1)
        ax3.set_title(f"{mode} Residuals")
        ax3.grid(True, alpha=0.3)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def plot_venn_diagram(main_set, sub_set, calc_set, save_path):
    """
    Plot Venn diagram for 3 sets of anomaly indices.
    Requires matplotlib-venn (may need installation).
    If not available, fallback to basic text or simple circles if possible.
    """
    try:
        from matplotlib_venn import venn3
    except ImportError:
        print("matplotlib-venn not found. Skipping Venn diagram.")
        return

    plt.figure(figsize=(10, 10))
    venn3([main_set, sub_set, calc_set], ('Main', 'Sub', 'Calc'))
    plt.title("Overlap of Detected Anomalies by Mode")
    plt.savefig(save_path)
    plt.close()

