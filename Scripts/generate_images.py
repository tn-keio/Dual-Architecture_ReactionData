
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
import matplotlib.pyplot as plt

# Add current directory to path to allow importing from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
CODE_DIR = os.path.join(SCRIPT_DIR, '../Code')
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from src import config  # New config module
from src.logger import setup_logger, log_execution_time
from src.data_loader import load_data, get_required_columns, extract_features_by_mode

# Config
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, '../Data/202211-202304_Cu.csv')
IMG_SIZE_INCHES = (2.24, 2.24) 
# Constants
DPI = 100

def generate_images(mode='Main', csv_path=None, output_dir=None, output_dirs=None):
    if output_dirs is None:
        output_dirs = config.get_output_dirs()

    logger = setup_logger(log_file=os.path.join(output_dirs['LOGS'], 'execution.log'))
    
    if output_dir is None:
        # Default to Data/Images_{mode} inside output hierarchy
        output_dir = os.path.join(output_dirs['DATA'], f'Images_{mode}')
        
    if csv_path is None:
        # Default to seek p1 results in output DATA if not provided
        # Or fall back to DEFAULT_CSV_PATH?
        # Integrated analysis passes csv_path so this is fall back for standalone
        csv_path = DEFAULT_CSV_PATH 
        
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating images for mode {mode} from {csv_path} to {output_dir}...")
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data using consolidated logic
    try:
        df = load_data(csv_path)
        
        # --- Data Cleaning (Must match run_autoencoder.py) ---
        required_cols = get_required_columns(mode, start_idx=11, end_idx=38)
        
        # Check if required columns exist in df; if not, and if this is a Phase 1 results CSV,
        # the columns are already present from the original data.
        
        # Check if columns exist
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
             # Look for fallback (Main only)
             if mode == 'Main':
                 fallback_cols = [str(i) for i in range(11, 39)]
                 if all(c in df.columns for c in fallback_cols):
                     required_cols = fallback_cols
                 else:
                     logger.error(f"Missing required columns for Main: {missing}")
                     return False
             else:
                 logger.error(f"Missing required columns for {mode}: {missing}")
                 return False

        # Drop NaNs
        initial_len = len(df)
        df_clean = df.dropna(subset=required_cols).reset_index(drop=True)
        new_len = len(df_clean)
        
        if new_len < initial_len:
            logger.info(f"Dropped {initial_len - new_len} rows due to NaNs (matching Autoencoder logic).")
            
        if new_len == 0:
            logger.error("No valid data remaining. Aborting.")
            return False
            
        # Extract Features
        X = extract_features_by_mode(df_clean, mode=mode, start_idx=11, end_idx=38)
        
        # Scale for plotting (Global Min/Max of the clean dataset)
        global_min = X.min()
        global_max = X.max()
        
        logger.info(f"Data Loaded & Cleaned. Shape: {X.shape}")
        logger.info(f"Global Scale - Min: {global_min}, Max: {global_max}")
        
    except Exception as e:
        logger.error(f"Failed to load/process data: {e}")
        return False

    logger.info(f"Generating {len(X)} images in {output_dir}...")
    
    # Check if images already exist to avoid re-generation? 
    # For safety, let's overwrite or skip if count matches?
    # User might have changed data. Let's regenerate.
    
    cnt = 0
    for i in range(len(X)):
        row = X[i]
        
        save_path = os.path.join(output_dir, f"img_{i}.png")
        
        # Optimization: Skip if exists? No, better safe.
        
        # Create figure without frame
        fig = plt.figure(figsize=IMG_SIZE_INCHES, dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1]) # Full frame
        
        # Plot
        ax.plot(row, color='black', linewidth=1.5)
        
        # Set consistent limits
        ax.set_ylim(global_min, global_max)
        ax.set_xlim(0, len(row)-1)
        
        # Hide axes
        ax.axis('off')
        
        # Save
        plt.savefig(save_path, dpi=DPI)
        plt.close(fig)
        
        cnt += 1
        if cnt % 500 == 0:
             logger.info(f"Generated {cnt} images...")

    logger.info(f"Image generation completed. Saved {cnt} images to {output_dir}")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='Main')
    args = parser.parse_args()
    
    generate_images(mode=args.mode)
