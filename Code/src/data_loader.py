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

import pandas as pd
import numpy as np

def load_data(csv_path):
    """
    Load the dataset from CSV.
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File {csv_path} not found.")

def get_required_columns(mode='Main', start_idx=11, end_idx=38):
    """
    Get the list of column names required for a specific mode.
    Useful for cleaning data (dropping NaNs).
    """
    indices = range(start_idx, end_idx + 1)
    if mode == 'Main':
        # Prefer RAM format, but if not present, caller needs to handle?
        # We return the list of expected columns in the new format since we are working with new data.
        return [f'RAM{i}' for i in indices]
    elif mode == 'Sub':
        return [f'RAS{i}' for i in indices]
    elif mode == 'Calc':
        # Requires both
        return [f'RAM{i}' for i in indices] + [f'RAS{i}' for i in indices]
    else:
        raise ValueError(f"Unknown mode: {mode}")

def extract_features_by_mode(df, mode='Main', start_idx=11, end_idx=38):
    """
    Extract features based on the mode (Main, Sub, Calc).
    Assumes df has the necessary columns (use get_required_columns + dropna first).
    """
    indices = range(start_idx, end_idx + 1)
    
    if mode == 'Main':
        cols = [f'RAM{i}' for i in indices]
        if cols[0] not in df.columns:
             # Fallback
             cols = [str(i) for i in indices]
             
        X = df[cols].values
        
    elif mode == 'Sub':
        cols = [f'RAS{i}' for i in indices]
        X = df[cols].values
        
    elif mode == 'Calc':
        cols_main = [f'RAM{i}' for i in indices]
        cols_sub = [f'RAS{i}' for i in indices]
        
        X_main = df[cols_main].values
        X_sub = df[cols_sub].values
        X = X_main - X_sub
             
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return X

def normalize_data(X):
    """
    Global MinMax Normalization.
    """
    data_min = np.nanmin(X)
    data_max = np.nanmax(X)
    if data_max == data_min:
        return np.zeros_like(X)
    X_norm = (X - data_min) / (data_max - data_min)
    return X_norm
