
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

import os
import sys
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# Add src path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
CODE_DIR = os.path.join(SCRIPT_DIR, '../Code')
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from src.logger import setup_logger
from src.visualization_enhanced import plot_combined_report, plot_venn_diagram
# Note: extract_features_by_mode likely isn't needed if we trust the NPY/CSV alignment from run_autoencoder
from src.data_loader import load_data, extract_features_by_mode, normalize_data
from src import config  # New config module

# Directories are now dynamic based on output_dirs passed to generate_report
# Removed global constants REPORT_DIR, STATS_DIR, INDIV_DIR, MODEL_EVAL_DIR

# Removed global logger setup causing immediate log file creation
# logger = setup_logger(...)

def load_results(output_dirs, modes=['Main', 'Sub', 'Calc']):
    dfs = {}
    logger = setup_logger(log_file=os.path.join(output_dirs['LOGS'], 'execution.log'))
    for mode in modes:
        # Look in DATA directory
        path = os.path.join(output_dirs['DATA'], f'dual_verification_results_{mode}.csv')
        if os.path.exists(path):
            dfs[mode] = pd.read_csv(path)
        else:
            logger.warning(f"Results for {mode} not found at {path}")
    return dfs

def _load_params(output_dirs, mode):
    """Load model parameters JSON for a given mode."""
    param_file = os.path.join(output_dirs['MODELS'], f'model_params_{mode}.json')
    if os.path.exists(param_file):
        try:
            with open(param_file, 'r') as pf:
                return json.load(pf)
        except Exception:
            pass
    return {}

def _load_norm_params(output_dirs, mode):
    """Load normalization parameters JSON for a given mode."""
    norm_file = os.path.join(output_dirs['MODELS'], f'norm_params_{mode}.json')
    if os.path.exists(norm_file):
        try:
            with open(norm_file, 'r') as nf:
                return json.load(nf)
        except Exception:
            pass
    return {}

def _compute_mse_stats(df):
    """Compute publication-ready MSE statistics from a results DataFrame."""
    if 'MSE' not in df.columns:
        return {}
    mse = df['MSE'].values
    mad = np.median(np.abs(mse - np.median(mse)))
    return {
        'n': len(mse),
        'mean': float(np.mean(mse)),
        'median': float(np.median(mse)),
        'std': float(np.std(mse)),
        'min': float(np.min(mse)),
        'max': float(np.max(mse)),
        'mad': float(mad),
        'q25': float(np.percentile(mse, 25)),
        'q75': float(np.percentile(mse, 75)),
    }

def _compute_separation_ratio(df):
    """Compute Separation Ratio: mean MSE of anomalies / mean MSE of normals."""
    if 'MSE' not in df.columns or 'Is_Anomaly' not in df.columns:
        return None
    normals = df[df['Is_Anomaly'] == 0]['MSE']
    anomalies = df[df['Is_Anomaly'] == 1]['MSE']
    if len(normals) == 0 or len(anomalies) == 0:
        return None
    return float(anomalies.mean() / normals.mean())

def generate_report(output_dirs=None):
    if output_dirs is None:
        output_dirs = config.get_output_dirs() # Default timestamped directory
        
    logger = setup_logger(log_file=os.path.join(output_dirs['LOGS'], 'execution.log'))
    logger.info("Generating Final Report (Publication Quality)...")
    
    # Define sub-directories
    STATS_DIR = os.path.join(output_dirs['REPORTS'], '02_Anomaly_Stats')
    os.makedirs(STATS_DIR, exist_ok=True)
    
    INDIV_DIR = os.path.join(output_dirs['FIGURES'], '03_Individual_Anomalies')
    os.makedirs(INDIV_DIR, exist_ok=True)
    
    MODEL_EVAL_DIR = os.path.join(output_dirs['FIGURES'], 'Model_Evaluation')
    os.makedirs(MODEL_EVAL_DIR, exist_ok=True)
    
    # Calculate relative path from Reports dir to Figures dir (for Markdown links)
    rel_figures_dir = os.path.relpath(output_dirs['FIGURES'], output_dirs['REPORTS']).replace(os.sep, '/')
    
    modes = ['Main', 'Sub', 'Calc']
    dfs = load_results(output_dirs, modes)
    
    if not dfs:
        logger.error("No results found to generate report.")
        return

    # =========================================================================
    # Data Collection Phase: Gather all statistics before writing
    # =========================================================================
    
    # --- Per-Mode Statistics ---
    mode_stats = {}
    for mode in modes:
        if mode not in dfs:
            continue
        df = dfs[mode]
        params = _load_params(output_dirs, mode)
        norm_params = _load_norm_params(output_dirs, mode)
        mse_stats = _compute_mse_stats(df)
        sep_ratio = _compute_separation_ratio(df)
        
        # Count anomalies by phase
        p1_candidates = int(df['Is_Diff_Candidate'].sum()) if 'Is_Diff_Candidate' in df.columns else 0
        p1_anomalies = int(df['Is_Anomaly'].sum()) if 'Is_Anomaly' in df.columns else 0
        p2_anomalies = int(df['P2_Is_Anomaly'].sum()) if 'P2_Is_Anomaly' in df.columns else 0
        
        dual_col = 'Is_Dual_Anomaly' if 'Is_Dual_Anomaly' in df.columns else None
        dual_anomalies = int(df[dual_col].sum()) if dual_col else 0
        
        # False positive rate (P1 anomalies not confirmed by P2)
        if p1_anomalies > 0 and dual_col:
            p1_only = int(((df.get('Is_Anomaly', 0) == 1) & (df.get(dual_col, 0) == 0)).sum())
            fp_reduction = (1 - dual_anomalies / p1_anomalies) * 100 if p1_anomalies > 0 else 0
        else:
            p1_only = 0
            fp_reduction = 0
        
        mode_stats[mode] = {
            'params': params,
            'norm_params': norm_params,
            'mse': mse_stats,
            'sep_ratio': sep_ratio,
            'p1_candidates': p1_candidates,
            'p1_anomalies': p1_anomalies,
            'p2_anomalies': p2_anomalies,
            'dual_anomalies': dual_anomalies,
            'p1_only': p1_only,
            'fp_reduction': fp_reduction,
            'total_n': len(df),
        }

    # --- Cross-Mode Anomaly Overlap ---
    anom_indices = {}
    for mode in modes:
        if mode in dfs:
            col = 'Is_Dual_Anomaly' if 'Is_Dual_Anomaly' in dfs[mode].columns else 'Is_Anomaly'
            anom_indices[mode] = set(dfs[mode][dfs[mode][col] == 1]['S_ID'])
    
    # Save Venn
    venn_path = os.path.join(STATS_DIR, 'anomaly_overlap_venn.png')
    plot_venn_diagram(
        anom_indices.get('Main', set()),
        anom_indices.get('Sub', set()),
        anom_indices.get('Calc', set()),
        venn_path
    )
    
    # --- Score Top Anomalies ---
    all_anoms = set().union(*anom_indices.values()) if anom_indices else set()
    anom_scores = []
    for s_id in all_anoms:
        score = 0
        detected_modes = []
        for mode in modes:
            if s_id in anom_indices.get(mode, set()):
                score += 1
                detected_modes.append(mode)
        anom_scores.append({'S_ID': s_id, 'Score': score, 'Modes': detected_modes})
    sorted_anoms = sorted(anom_scores, key=lambda x: x['Score'], reverse=True)
    top_anoms = sorted_anoms[:10]

    # --- Generate Individual Anomaly Plots ---
    logger.info(f"Generating detailed plots for top {len(top_anoms)} anomalies...")
    
    data_cache = {}
    for mode in modes:
        if mode in dfs:
            recon_path = os.path.join(output_dirs['DATA'], f'reconstructions_{mode}.npy')
            err_path = os.path.join(output_dirs['DATA'], f'errors_{mode}.npy')
            
            if os.path.exists(recon_path) and os.path.exists(err_path):
                try:
                    reconstructions = np.load(recon_path)
                    errors = np.load(err_path)
                    df = dfs[mode]
                    X = extract_features_by_mode(df, mode=mode, start_idx=11, end_idx=38)
                    # V5-2 Fix: Use saved norm_params for consistent normalization
                    norm_params = _load_norm_params(output_dirs, mode)
                    if norm_params and 'data_min' in norm_params and 'data_max' in norm_params:
                        dmin, dmax = norm_params['data_min'], norm_params['data_max']
                        X = (X - dmin) / (dmax - dmin) if dmax != dmin else np.zeros_like(X)
                    else:
                        X = normalize_data(X)
                    
                    if len(X) != len(reconstructions):
                        logger.warning(f"Shape mismatch for {mode}: CSV {len(X)} vs NPY {len(reconstructions)}")
                        continue
                    data_cache[mode] = (X, reconstructions, errors, df)
                except Exception as e:
                    logger.error(f"Error loading cache for {mode}: {e}")
            else:
                logger.warning(f"NPY files not found for {mode}. Skipping detailed plotting.")

    for item in top_anoms:
        s_id = item['S_ID']
        waveforms = {}
        recons = {}
        errors_dict = {}
        is_anomaly = {}
        
        for mode in modes:
            if mode in data_cache:
                X, recon, err, df = data_cache[mode]
                row_idx_list = df.index[df['S_ID'] == s_id].tolist()
                if not row_idx_list:
                    continue
                idx = row_idx_list[0]
                waveforms[mode] = X[idx]
                recons[mode] = recon[idx].flatten()
                errors_dict[mode] = err[idx].flatten()
                is_anomaly[mode] = (s_id in anom_indices.get(mode, set()))

        if waveforms:
            save_name = f"Report_SID_{s_id}.png"
            plot_combined_report(s_id, waveforms, recons, errors_dict, is_anomaly, os.path.join(INDIV_DIR, save_name))

    # =========================================================================
    # Report Writing Phase
    # =========================================================================
    md_path = os.path.join(output_dirs['REPORTS'], 'Summary_Report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        
        # --- Header ---
        f.write("# Anomaly Detection Analysis Report\n\n")
        f.write(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")
        
        # =====================================================================
        # Section 1: Dataset Overview
        # =====================================================================
        f.write("## 1. Dataset Overview\n\n")
        
        # Use first available mode for global stats
        first_mode = next(iter(mode_stats), None)
        if first_mode:
            ms = mode_stats[first_mode]
            total_n = ms['total_n']
            f.write(f"| Parameter | Value |\n")
            f.write(f"| :--- | :--- |\n")
            f.write(f"| Analyte | Copper (Cu) |\n")
            f.write(f"| Reagent | Quick Auto Neo Cu (Shino-Test) |\n")
            f.write(f"| Analyzer | Labospect 008α (Hitachi) |\n")
            f.write(f"| Collection Period | Nov 2022 - Apr 2023 |\n")
            f.write(f"| Effective N (after NaN exclusion) | {total_n} |\n")
            f.write(f"| Features per Sample | 28 time points (indices 11-38) |\n")
            f.write(f"| Analysis Modes | Main (RAM), Sub (RAS), Calc (RAM-RAS) |\n")
            
            # Normalization params
            if ms['norm_params']:
                f.write(f"| Normalization Range (Main) | [{ms['norm_params'].get('data_min', 'N/A'):.6f}, {ms['norm_params'].get('data_max', 'N/A'):.6f}] |\n")
            f.write(f"\n")
        
        # =====================================================================
        # Section 2: Model Configuration
        # =====================================================================
        f.write("## 2. Model Configuration (Hyperparameters)\n\n")
        f.write("### 2.1 Autoencoder Settings\n\n")
        f.write("| Parameter | Value |\n")
        f.write("| :--- | :--- |\n")
        if first_mode and mode_stats[first_mode]['params']:
            p = mode_stats[first_mode]['params']
            f.write(f"| Epochs | {p.get('epochs', 'N/A')} |\n")
            f.write(f"| Batch Size | {p.get('batch_size', 'N/A')} |\n")
            f.write(f"| Similarity Threshold | {p.get('similarity_threshold', 'N/A')} |\n")
            f.write(f"| Min Neighbors | {p.get('min_neighbors', 'N/A')} |\n")
        f.write(f"| Loss Function | Mean Squared Error (MSE) |\n")
        f.write(f"| Optimizer | Adam (lr=0.001) |\n")
        f.write(f"| Threshold Method | MAD (Median + 3.0 × 1.4826 × MAD) |\n")
        f.write(f"\n")
        
        f.write("### 2.2 CNN + Isolation Forest Settings\n\n")
        f.write("| Parameter | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write("| CNN Model | ResNet50 (ImageNet, frozen) |\n")
        f.write("| Feature Dimension | 2048 (Global Avg Pooling) |\n")
        f.write("| Input Image Size | 224 × 224 px |\n")
        f.write("| Isolation Forest n_estimators | 100 |\n")
        f.write("| Isolation Forest contamination | 0.02 |\n")
        f.write("| Random State | 42 |\n")
        f.write(f"\n")
        
        # =====================================================================
        # Section 3: Learning Curves
        # =====================================================================
        f.write("## 3. Training Loss Curves\n\n")
        for mode in modes:
            if mode in mode_stats:
                loss_img = f"{rel_figures_dir}/Model_Evaluation/training_loss_{mode}.png"
                f.write(f"### {mode} Mode\n")
                f.write(f"![Training Loss - {mode}]({loss_img})\n\n")
        
        # =====================================================================
        # Section 4: MSE Distribution Statistics (Per Mode)
        # =====================================================================
        f.write("## 4. MSE Distribution Statistics\n\n")
        f.write("Reconstruction error statistics across all samples for each analysis mode.\n\n")
        
        f.write("| Statistic | Main | Sub | Calc |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        stat_rows = [
            ('N', 'n', '{:,}'),
            ('Mean', 'mean', '{:.8f}'),
            ('Median', 'median', '{:.8f}'),
            ('Std Dev', 'std', '{:.8f}'),
            ('MAD', 'mad', '{:.8f}'),
            ('Min', 'min', '{:.8f}'),
            ('Max', 'max', '{:.8f}'),
            ('Q25 (25th %ile)', 'q25', '{:.8f}'),
            ('Q75 (75th %ile)', 'q75', '{:.8f}'),
        ]
        
        for label, key, fmt in stat_rows:
            vals = []
            for mode in modes:
                if mode in mode_stats and key in mode_stats[mode].get('mse', {}):
                    vals.append(fmt.format(mode_stats[mode]['mse'][key]))
                else:
                    vals.append('N/A')
            f.write(f"| {label} | {' | '.join(vals)} |\n")
        
        # Threshold row
        vals = []
        for mode in modes:
            if mode in mode_stats and mode_stats[mode]['params']:
                t = mode_stats[mode]['params'].get('mse_threshold', 'N/A')
                vals.append(f"{t:.8f}" if isinstance(t, float) else str(t))
            else:
                vals.append('N/A')
        f.write(f"| **Threshold (MAD)** | {' | '.join(vals)} |\n")
        f.write(f"\n")
        
        # MSE distribution plots
        f.write("### MSE Distribution Plots\n\n")
        for mode in modes:
            if mode in mode_stats:
                dist_img = f"{rel_figures_dir}/Model_Evaluation/mse_distribution_{mode}.png"
                f.write(f"![MSE Distribution - {mode}]({dist_img})\n\n")
        
        # =====================================================================
        # Section 5: Anomaly Detection Results (Per Mode)
        # =====================================================================
        f.write("## 5. Anomaly Detection Results\n\n")
        f.write("### 5.1 Detection Counts\n\n")
        
        f.write("| Metric | Main | Sub | Calc |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        count_rows = [
            ('Total Samples (N)', 'total_n'),
            ('Phase 1: High-MSE Candidates', 'p1_candidates'),
            ('Phase 1: Unique Anomalies (Post-Filter)', 'p1_anomalies'),
            ('Phase 2: IF Anomalies', 'p2_anomalies'),
            ('Dual Verified (P1 ∩ P2)', 'dual_anomalies'),
        ]
        
        for label, key in count_rows:
            vals = []
            for mode in modes:
                if mode in mode_stats:
                    vals.append(str(mode_stats[mode].get(key, 'N/A')))
                else:
                    vals.append('N/A')
            f.write(f"| {label} | {' | '.join(vals)} |\n")
        f.write(f"\n")
        
        # =====================================================================
        # Section 5.2: Separation Ratio
        # =====================================================================
        f.write("### 5.2 Model Discrimination (Separation Ratio)\n\n")
        f.write("Separation Ratio = Mean MSE (Anomalies) / Mean MSE (Normal Samples)\n\n")
        f.write("| Mode | Separation Ratio |\n")
        f.write("| :--- | :---: |\n")
        for mode in modes:
            if mode in mode_stats:
                sr = mode_stats[mode].get('sep_ratio')
                sr_str = f"{sr:.2f}" if sr is not None else "N/A (0 anomalies)"
                f.write(f"| {mode} | {sr_str} |\n")
        f.write(f"\n")
        
        # =====================================================================
        # Section 5.3: False Positive Reduction
        # =====================================================================
        f.write("### 5.3 False Positive Reduction by Dual Verification\n\n")
        f.write("| Mode | P1 Only (Not confirmed by P2) | Dual Verified | FP Reduction (%) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for mode in modes:
            if mode in mode_stats:
                ms = mode_stats[mode]
                f.write(f"| {mode} | {ms['p1_only']} | {ms['dual_anomalies']} | {ms['fp_reduction']:.1f}% |\n")
        f.write(f"\n")
        
        # Normal vs Anomaly comparison plots
        f.write("### Normal vs Anomaly Comparison\n\n")
        for mode in modes:
            if mode in mode_stats:
                comp_img = f"{rel_figures_dir}/Model_Evaluation/normal_vs_anomaly_comparison_{mode}.png"
                f.write(f"![Normal vs Anomaly - {mode}]({comp_img})\n\n")
        
        # =====================================================================
        # Section 6: Cross-Mode Anomaly Overlap
        # =====================================================================
        f.write("## 6. Cross-Mode Anomaly Overlap\n\n")
        
        f.write("| Mode | Anomaly Count | Sample IDs |\n")
        f.write("| :--- | :---: | :--- |\n")
        for mode in modes:
            ids = anom_indices.get(mode, set())
            id_str = ', '.join([str(int(x)) if x == int(x) else str(x) for x in sorted(ids)]) if ids else 'None'
            f.write(f"| {mode} | {len(ids)} | {id_str} |\n")
        
        # Intersection
        # V5-1 Fix: Correct cross-mode intersection (empty sets naturally yield empty intersection)
        anom_sets = [anom_indices.get(mode, set()) for mode in modes]
        all_modes_intersection = set.intersection(*anom_sets) if all(len(s) > 0 for s in anom_sets) else set()
        inter_str = ', '.join([str(int(x)) if x == int(x) else str(x) for x in sorted(all_modes_intersection)]) if all_modes_intersection else 'None'
        f.write(f"| **All Modes (∩)** | {len(all_modes_intersection)} | {inter_str} |\n")
        f.write(f"\n")
        
        f.write("### Venn Diagram\n")
        f.write("![Venn Diagram](02_Anomaly_Stats/anomaly_overlap_venn.png)\n\n")
        
        # =====================================================================
        # Section 7: Detailed Anomaly Profiles (Top 10)
        # =====================================================================
        f.write("## 7. Top Anomaly Profiles (Detailed)\n\n")
        if top_anoms:
            f.write(f"Detailed reconstruction analysis for the top {len(top_anoms)} anomalies ranked by cross-mode consistency.\n\n")
            
            for item in top_anoms:
                s_id = item['S_ID']
                modes_str = ", ".join(item['Modes'])
                consistency = f"{item['Score']}/{len(modes)}"
                
                f.write(f"### Sample ID: {s_id}\n")
                f.write(f"- **Detected in**: {modes_str}\n")
                f.write(f"- **Cross-Mode Consistency**: {consistency}\n")
                
                # Per-mode MSE for this sample
                for mode in modes:
                    if mode in dfs:
                        row = dfs[mode][dfs[mode]['S_ID'] == s_id]
                        if not row.empty and 'MSE' in row.columns:
                            mse_val = row['MSE'].values[0]
                            threshold = mode_stats.get(mode, {}).get('params', {}).get('mse_threshold', 0)
                            ratio = mse_val / threshold if threshold > 0 else 0
                            f.write(f"- **{mode} MSE**: {mse_val:.8f} (Threshold: {threshold:.8f}, Ratio: {ratio:.2f}×)\n")
                
                img_path = f"{rel_figures_dir}/03_Individual_Anomalies/Report_SID_{s_id}.png"
                f.write(f"\n![Anomaly Profile {s_id}]({img_path})\n\n")
        else:
            f.write("No anomalies were detected across all modes.\n\n")
        
        # =====================================================================
        # Section 8: Summary Statistics (Export-Ready JSON)
        # =====================================================================
        f.write("---\n\n")
        f.write("## Appendix: Normalization Parameters\n\n")
        f.write("| Mode | Data Min | Data Max |\n")
        f.write("| :--- | :---: | :---: |\n")
        for mode in modes:
            if mode in mode_stats:
                np_data = mode_stats[mode].get('norm_params', {})
                dmin = np_data.get('data_min', 'N/A')
                dmax = np_data.get('data_max', 'N/A')
                dmin_str = f"{dmin:.6f}" if isinstance(dmin, (int, float)) else str(dmin)
                dmax_str = f"{dmax:.6f}" if isinstance(dmax, (int, float)) else str(dmax)
                f.write(f"| {mode} | {dmin_str} | {dmax_str} |\n")
        f.write(f"\n")
    
    # Save statistics as JSON for programmatic access
    stats_json_path = os.path.join(output_dirs['REPORTS'], 'analysis_statistics.json')
    export_stats = {}
    for mode, ms in mode_stats.items():
        export_stats[mode] = {
            'total_n': ms['total_n'],
            'mse_statistics': ms['mse'],
            'separation_ratio': ms['sep_ratio'],
            'threshold': ms['params'].get('mse_threshold'),
            'phase1_candidates': ms['p1_candidates'],
            'phase1_anomalies': ms['p1_anomalies'],
            'phase2_anomalies': ms['p2_anomalies'],
            'dual_verified': ms['dual_anomalies'],
            'fp_reduction_pct': ms['fp_reduction'],
        }
    with open(stats_json_path, 'w', encoding='utf-8') as jf:
        json.dump(export_stats, jf, indent=4, ensure_ascii=False)
    
    # Save anomaly counts CSV
    stats_csv = []
    for mode in modes:
        count = len(anom_indices.get(mode, set()))
        stats_csv.append({'Mode': mode, 'Anomaly_Count': count})
    stats_df = pd.DataFrame(stats_csv)
    stats_df.to_csv(os.path.join(STATS_DIR, 'anomaly_counts.csv'), index=False)
    
    logger.info(f"Report generated at {md_path}")
    logger.info(f"Statistics JSON saved at {stats_json_path}")

if __name__ == "__main__":
    generate_report()
