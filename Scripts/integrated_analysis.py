
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
import numpy as np
import traceback

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

CODE_DIR = os.path.join(SCRIPT_DIR, '../Code')
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from src.logger import setup_logger, log_execution_time
# Import modules
import run_autoencoder
import generate_images 
import extract_features 
import detect_and_compare
import generate_report
from src import config # New config module

# Config
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, '../Data/202211-202304_Cu.csv')

def main():
    # Initialize output structure (Base Results Dir)
    dirs = config.get_output_dirs()
    
    logger = setup_logger(log_file=os.path.join(dirs['LOGS'], 'execution.log'))
    logger.info("==================================================")
    logger.info("STARTING INTEGRATED ANALYSIS (REVIEWED VERSION)")
    logger.info(f"Output Directory: {dirs['BASE']}")
    logger.info("Modes: Main (RAM), Sub (RAS), Calc (RAM-RAS)")
    logger.info("==================================================")

    modes = ['Main', 'Sub', 'Calc']
    
    # Define target datasets (List allows future expansion)
    target_datasets = [DEFAULT_CSV_PATH]
    
    for csv_path in target_datasets:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        logger.info(f"=== Processing Dataset: {dataset_name} ===")
        
        # Create Dataset-Specific Subdirectories to prevent conflict
        ds_dirs = dirs.copy()
        ds_dirs['DATA'] = os.path.join(dirs['DATA'], dataset_name)
        ds_dirs['FIGURES'] = os.path.join(dirs['FIGURES'], dataset_name)
        ds_dirs['MODELS'] = os.path.join(dirs['MODELS'], dataset_name)
        ds_dirs['REPORTS'] = os.path.join(dirs['REPORTS'], dataset_name)
        
        for k in ['DATA', 'FIGURES', 'MODELS', 'REPORTS']:
            os.makedirs(ds_dirs[k], exist_ok=True)
            
        for mode in modes:
            try:
                logger.info(f"--- Processing Mode: {mode} (Dataset: {dataset_name}) ---")
                
                # Step 1: Run Method 1 (Autoencoder)
                # Returns path to UNSORTED CSV
                p1_results_path = None
                with log_execution_time(logger, f"Phase 1: Autoencoder Analysis ({mode})"):
                    # Pass dataset-specific dirs and specific csv_path
                    p1_results_path = run_autoencoder.run_analysis(mode=mode, csv_path=csv_path, output_dirs=ds_dirs)
                    
                    if not p1_results_path:
                        logger.error(f"Phase 1 failed for mode {mode}. Skipping subsequent steps for this mode.")
                        continue
                
                logger.info(f"Phase 1 complete for {mode}. Unsorted Results at: {p1_results_path}")
                
                # Step 2: Run Method 2 Preparation (Image Generation)
                logger.info(f"START: Phase 2 Prep: Image Generation ({mode})")
                img_dir = os.path.join(ds_dirs['DATA'], f'Images_{mode}')
                
                if not generate_images.generate_images(mode=mode, csv_path=p1_results_path, output_dir=img_dir, output_dirs=ds_dirs):
                    logger.error("Image generation failed. Aborting Phase 2.")
                    continue
    
                # Step 3: Run Method 2 Preparation (Feature Extraction)
                logger.info(f"START: Phase 2 Prep: Feature Extraction ({mode})")
                feat_file = os.path.join(ds_dirs['DATA'], f'image_features_{mode}.npy')
                # Pass output_dirs (optional for logger config)
                if not extract_features.extract_features(image_dir=img_dir, output_file=feat_file, output_dirs=ds_dirs):
                    logger.error("Feature extraction failed. Aborting Phase 2.")
                    continue
    
                # Step 4: Run Method 2 (Isolation Forest) & Merge
                logger.info(f"START: Phase 2: Isolation Forest & Comparison ({mode})")
                combined_csv_path = os.path.join(ds_dirs['DATA'], f'dual_verification_results_{mode}.csv')
                
                # This function now performs STRICT length check.
                combined_df = detect_and_compare.detect_and_compare(
                    features_path=feat_file,
                    phase1_results_path=p1_results_path,
                    output_csv=combined_csv_path,
                    return_df=True,
                    output_dirs=ds_dirs
                )
                
                if combined_df is None:
                    logger.error("Phase 2 detection/merge failed (likely alignment mismatch). Check logs.")
                    continue
    
                # Step 5: Dual Verification (Intersection)
                with log_execution_time(logger, f"Phase 3: Dual Verification ({mode})"):
                    logger.info("Identifying True Anomalies (Detected by BOTH methods)...")
                    
                    if 'Is_Anomaly' in combined_df.columns and 'P2_Is_Anomaly' in combined_df.columns:
                        combined_df['Is_Dual_Anomaly'] = 0
                        
                        dual_anomalies = combined_df[
                            (combined_df['Is_Anomaly'] == 1) & 
                            (combined_df['P2_Is_Anomaly'] == 1)
                        ]
                        
                        combined_df.loc[dual_anomalies.index, 'Is_Dual_Anomaly'] = 1
                        
                        # Save Final Verified Results
                        combined_df.to_csv(combined_csv_path, index=False)
                        
                        logger.info(f"Dual Verification Complete for {mode}.")
                        logger.info(f"Total True Anomalies: {len(dual_anomalies)}")
                    else:
                        logger.warning("Columns Is_Anomaly or P2_Is_Anomaly missing. Skipping Dual check.")
    
            except Exception as e:
                logger.critical(f"Integrated analysis failed for mode {mode}: {e}")
                logger.error(traceback.format_exc())
    
        # Step 6: Generate Final Report for THIS Dataset
        # Since outputs are isolated in ds_dirs, report generation works on them
        logger.info("==================================================")
        logger.info(f"GENERATING FINAL REPORT for {dataset_name}")
        logger.info("==================================================")
        generate_report.generate_report(output_dirs=ds_dirs)
    
    logger.info(f"ALL PROCESSING COMPLETE. Results saved to: {dirs['BASE']}")

if __name__ == "__main__":
    main()
