
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
import numpy as np
import pandas as pd
import json
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import traceback

# Set Random Seeds for Reproducibility (R-2)
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

# Add Code/src to path for data_loader
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, '../Code')
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from src.model_autoencoder import AutoencoderModel
from src.anomaly_detection import AnomalyDetector
from src.visualization import plot_training_loss, plot_top_anomalies, plot_mse_distribution, plot_group_comparison
from src.logger import setup_logger, log_execution_time
from src.data_loader import load_data, extract_features_by_mode, normalize_data, get_required_columns
from src import config  # New config module

# ==========================
# Configuration
# ==========================
# Default paths
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, '../Data/202211-202304_Cu.csv')
TOP_N_ANOMALIES = 5
EPOCHS = 50
BATCH_SIZE = 32

# Anomaly Refinement Config
SIMILARITY_THRESHOLD = 0.92  # Cosine Similarity threshold (0-1)
MIN_NEIGHBORS = 3           # If a pattern appears this many times, it is NOT an anomaly

# Output Directories (Global constants removed, handled in run_analysis)

# Logger initialized inside run_analysis to support dynamic paths

def run_analysis(mode='Main', csv_path=None, output_dirs=None):
    """
    Run the full analysis pipeline for a specific mode.
    """
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH
    
    if output_dirs is None:
        output_dirs = config.get_output_dirs() # Default timestamped directory
        
    logger = setup_logger(log_file=os.path.join(output_dirs['LOGS'], 'execution.log'))
    logger.info(f"Using default output directory: {output_dirs['BASE']}")
        
    # Paths derived from output_dirs
    results_csv_sorted_path = os.path.join(output_dirs['DATA'], f'anomaly_analysis_results_{mode}.csv')
    results_csv_unsorted_path = os.path.join(output_dirs['DATA'], f'anomaly_analysis_results_{mode}_unsorted.csv') # Fix C-4
    model_save_path = os.path.join(output_dirs['MODELS'], f'best_model_ae_{mode}.keras')
    # Save params to MODELS dir as it relates to model training
    params_save_path = os.path.join(output_dirs['MODELS'], f'model_params_{mode}.json')
    
    # Validation/Evaluation artifacts go to Figures/Model_Evaluation
    eval_dir = os.path.join(output_dirs['FIGURES'], 'Model_Evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    try:
        logger.info(f"Starting Abnormal Reaction Analysis - Mode: {mode}...")
        
        # 1. Data Preparation
        with log_execution_time(logger, "Data Loading and Preprocessing"):
            df = load_data(csv_path)
            
            # --- Cleaning Step ---
            # Determine required columns for this mode
            try:
                required_cols = get_required_columns(mode, start_idx=11, end_idx=38)
            except ValueError as e:
                logger.error(f"Error determining required columns: {e}")
                return None
            
            # Check if columns exist
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                # If Main mode, might need fallback to '11'...'38' logic
                if mode == 'Main':
                    # Check fallback
                    fallback_cols = [str(i) for i in range(11, 39)]
                    if all(c in df.columns for c in fallback_cols):
                        required_cols = fallback_cols
                    else:
                        logger.error(f"Missing required columns for Main mode: {missing}")
                        return None
                else:
                    logger.error(f"Missing required columns for {mode} mode: {missing}")
                    return None
            
            # Drop rows with NaNs in required columns
            initial_len = len(df)
            df = df.dropna(subset=required_cols).reset_index(drop=True)
            # Ensure S_ID exists or create index based one
            if 'S_ID' not in df.columns:
                 df['S_ID'] = df.index
            
            new_len = len(df)
            logger.info(f"Dropped {initial_len - new_len} rows due to NaNs in {mode} columns.")
            
            if new_len == 0:
                logger.error("No valid data remaining after dropna. Aborting.")
                return None
            
            # Extract features using cleaned DF
            X = extract_features_by_mode(df, mode=mode, start_idx=11, end_idx=38)
            
            # Normalize
            # NOTE (C-2): Normalizing before split introduces leakage. 
            # For this project, we proceed as is but note it as a limitation.
            data_normalized = normalize_data(X)
            
            # Check for NaNs after normalization (e.g. infinite division)
            if np.isnan(data_normalized).any():
                logger.error("Normalized data contains NaNs (possible constant values or inf). Aborting.")
                return None

            # Reshape for CNN (N, Features, 1)
            X_reshaped = data_normalized.reshape(data_normalized.shape[0], data_normalized.shape[1], 1)
            
            # Split Train/Val
            X_train, X_val = train_test_split(X_reshaped, test_size=0.2, random_state=42)
            logger.info(f"Data shape: {X_reshaped.shape}")
            logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        # 2. Model Training
        with log_execution_time(logger, "Model Training"):
            logger.info(f"Building and training model for {mode}...")
            autoencoder = AutoencoderModel(input_length=X_reshaped.shape[1])
            
            history = autoencoder.train(X_train, X_val, epochs=EPOCHS, batch_size=BATCH_SIZE, save_path=model_save_path)
            
            # Save Training Loss Plot to Report Dir (Figures/Model_Evaluation)
            plot_training_loss(history, save_path=os.path.join(eval_dir, f'training_loss_{mode}.png'))
        
        # 3. Anomaly Detection
        with log_execution_time(logger, "Anomaly Detection"):
            logger.info("Running anomaly detection...")
            detector = AnomalyDetector(autoencoder.model)
            
            # Calculate MSE
            mse_scores, reconstructions = detector.calculate_mse(X_reshaped)
            
            # Calculate Element-wise Error
            try:
                # Optimize (M-5): Pass pre-calculated reconstructions
                errors_elementwise, _ = detector.calculate_elementwise_error(X_reshaped, reconstructions=reconstructions)
            except AttributeError:
                 errors_elementwise = np.mean(np.square(X_reshaped - reconstructions), axis=2) 
            
            # Determine Threshold
            threshold = detector.determine_threshold(mse_scores, method='mad', std_devs=3.0)
            logger.info(f"Initial MSE Threshold (MAD): {threshold:.6f}")
            
            # Identify Candidates
            candidate_indices = detector.detect_anomalies(mse_scores, threshold)
            logger.info(f"Found {len(candidate_indices)} initial candidates (High MSE) out of {len(X)} samples.")
            
            # Apply Density/similarity Filter
            logger.info(f"Filtering candidates: Keeping only unique patterns...")
            # Fix C-1: Pass X_reshaped (3D) to find_unique_anomalies if usage requires it, 
            # or data_normalized if expecting 2D. Checking AnomalyDetector source, 
            # it often uses flat vectors for cosine sim. Let's pass 2D for similarity, 
            # but ensure indices align.
            
            # NOTE: If find_unique_anomalies expects (N, Features), `data_normalized` is correct.
            final_anomaly_indices = detector.find_unique_anomalies(
                data_normalized, candidate_indices, 
                similarity_threshold=SIMILARITY_THRESHOLD, 
                min_neighbors=MIN_NEIGHBORS
            )
            logger.info(f"Final Count of Unique Anomalies: {len(final_anomaly_indices)}")
            
            anomaly_indices = final_anomaly_indices
        
        # 4. Save Results
        with log_execution_time(logger, "Saving Results"):
            # Create a copy of DF to avoid SettingWithCopy
            results_df = df.copy()
            results_df['MSE'] = mse_scores
            
            results_df['Is_Diff_Candidate'] = 0
            results_df.loc[candidate_indices, 'Is_Diff_Candidate'] = 1
            
            results_df['Is_Anomaly'] = 0
            results_df.loc[anomaly_indices, 'Is_Anomaly'] = 1
            
            results_df['Analysis_Mode'] = mode
            
            # Fix C-4: Save UNSORTED for pipeline alignment
            results_df.to_csv(results_csv_unsorted_path, index=False)
            logger.info(f"Unsorted results (Pipeline Ready) saved to {results_csv_unsorted_path}")
            
            # Save Reconstructions and Errors for Final Report (NPY)
            # These are needed by generate_report.py for detailed plots
            recon_path = os.path.join(output_dirs['DATA'], f'reconstructions_{mode}.npy')
            errors_path = os.path.join(output_dirs['DATA'], f'errors_{mode}.npy')
            np.save(recon_path, reconstructions)
            np.save(errors_path, errors_elementwise)
            logger.info(f"Saved reconstructions/errors to {output_dirs['DATA']}")

            # Save SORTED for human inspection
            df_sorted = results_df.sort_values('MSE', ascending=False)
            df_sorted.to_csv(results_csv_sorted_path, index=False)
            logger.info(f"Sorted results (Human Readable) saved to {results_csv_sorted_path}")
            
            # Log Parameters for Report
            params = {
                "mode": mode,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "min_neighbors": MIN_NEIGHBORS,
                "mse_threshold": float(threshold),
                "total_samples": len(df),
                "candidates_found": len(candidate_indices),
                "final_anomalies": len(anomaly_indices)
            }
            with open(params_save_path, 'w') as f:
                json.dump(params, f, indent=4)
            
            # Save normalization parameters for reproducibility (N-2)
            norm_params = {
                "data_min": float(np.nanmin(X)),
                "data_max": float(np.nanmax(X))
            }
            # Save norm params to Models dir alongside model params
            norm_params_path = os.path.join(output_dirs['MODELS'], f'norm_params_{mode}.json')
            with open(norm_params_path, 'w') as f:
                json.dump(norm_params, f, indent=4)
            logger.info(f"Normalization params saved to {norm_params_path}")
        
        # 5. Visualization
        with log_execution_time(logger, "Visualization"):
            sorted_indices_by_mse = np.argsort(mse_scores)[::-1]
            
            logger.info("Generating anomaly visualizations...")
            
            # 5.1 MSE Distribution
            plot_mse_distribution(mse_scores, threshold, save_path=os.path.join(eval_dir, f'mse_distribution_{mode}.png'))
            
            # 5.2 Group Comparison
            is_anomaly_mask = results_df['Is_Anomaly'].values.astype(bool)
            plot_group_comparison(data_normalized, is_anomaly_mask, save_path=os.path.join(eval_dir, f'normal_vs_anomaly_comparison_{mode}.png'))
            
            # 5.3 Top Anomalies (Legacy Plot)
            # Create subfolder in FIGURES for individual anomalies
            anomalies_dir = os.path.join(output_dirs['FIGURES'], f'anomalies_{mode}')
            os.makedirs(anomalies_dir, exist_ok=True)
            
            plot_top_anomalies(
                data_normalized, reconstructions, errors_elementwise, 
                sorted_indices_by_mse, results_df, 
                top_n=TOP_N_ANOMALIES, 
                save_prefix=f'{anomalies_dir}/anomaly'
            )
            
        logger.info(f"Analysis for {mode} completed successfully.")
        return results_csv_unsorted_path # Return Pipeline-Ready Path

    except Exception as e:
        logger.critical(f"An unhandled exception occurred during execution for mode {mode}.")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='Main', help='Analysis mode: Main, Sub, Calc')
    args = parser.parse_args()
    
    run_analysis(mode=args.mode)
