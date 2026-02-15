
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
from sklearn.ensemble import IsolationForest

# Add src path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
CODE_DIR = os.path.join(SCRIPT_DIR, '../Code')
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from src.logger import setup_logger, log_execution_time
from src import config # New config module

def detect_and_compare(features_path, phase1_results_path, output_csv=None, return_df=False, output_dirs=None):
    """
    Run Isolation Forest on features and merge with Phase 1 results.
    Strictly assumes features_path (npy) and phase1_results_path (csv) are row-aligned.
    """
    if output_dirs is None:
        output_dirs = config.get_output_dirs() # Default timestamped directory
        
    logger = setup_logger(log_file=os.path.join(output_dirs['LOGS'], 'execution.log'))
    logger.info("Starting Phase 2 Anomaly Detection (Isolation Forest on Image Features)...")
    
    # 1. Load Features
    if not os.path.exists(features_path):
        logger.error(f"Features file not found: {features_path}")
        return None
        
    try:
        features = np.load(features_path)
        # Check if features is 2D
        if len(features.shape) > 2:
             # If for some reason it's 3D, flatten? ResNet avg pool should be 1D per sample (N, 2048)
             features = features.reshape(features.shape[0], -1)
             
        logger.info(f"Loaded features: {features.shape}")
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        return None
    
    # 2. Isolation Forest
    logger.info("Training Isolation Forest...")
    try:
        # R-3: Contamination hardcoded. Noted as limitation in plan/report.
        # R-2: Set random_state
        clf = IsolationForest(n_estimators=100, contamination=0.02, random_state=42, n_jobs=-1)
        preds = clf.fit_predict(features)
        scores = clf.decision_function(features) # Lower is more abnormal
    except Exception as e:
        logger.error(f"Isolation Forest failed: {e}")
        return None
    
    # preds: 1 for inliers, -1 for outliers
    anomaly_indices = np.where(preds == -1)[0]
    
    logger.info(f"Isolation Forest found {len(anomaly_indices)} anomalies (Contamination=0.02).")
    
    # 3. Load Phase 1 Results for comparison
    df_p1 = None
    if os.path.exists(phase1_results_path):
        df_p1 = pd.read_csv(phase1_results_path)
        logger.info(f"Loaded Phase 1 results from {phase1_results_path}")
    else:
        logger.error(f"Phase 1 results NOT found at {phase1_results_path}. Cannot proceed with alignment.")
        return None
        
    if df_p1 is not None:
        # Check alignment (Strict)
        if len(df_p1) != len(features):
            logger.error(f"CRITICAL LENGTH MISMATCH: Phase 1 ({len(df_p1)}) vs Phase 2 ({len(features)}).")
            logger.error("Data alignment cannot be guaranteed. Aborting Phase 2 merge.")
            return None
            
    # 4. Merge Results
    results_df = df_p1.copy()
    
    # Append Phase 2 columns
    results_df['P2_IF_Score'] = scores
    results_df['P2_Is_Anomaly'] = 0
    results_df.iloc[anomaly_indices, results_df.columns.get_loc('P2_Is_Anomaly')] = 1
    
    if output_csv:
        dirname = os.path.dirname(output_csv)
        if dirname:  # Fix N-8: Guard against empty dirname
            os.makedirs(dirname, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Phase 2 results saved to {output_csv}")
    
    if return_df:
        return results_df
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--phase1_results_path', type=str, default='')
    parser.add_argument('--output_csv', type=str, default=None)
    
    args = parser.parse_args()
    
    detect_and_compare(
        features_path=args.features_path, 
        phase1_results_path=args.phase1_results_path, 
        output_csv=args.output_csv
    )
