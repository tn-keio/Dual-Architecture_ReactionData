
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_alignment(mode):
    csv_path = os.path.join(SCRIPT_DIR, f'anomaly_analysis_results_{mode}_unsorted.csv')
    npy_path = os.path.join(SCRIPT_DIR, f'reconstructions_{mode}.npy')
    
    if not os.path.exists(csv_path):
        print(f"Skipping {mode}: CSV not found.")
        return
    if not os.path.exists(npy_path):
        print(f"Skipping {mode}: NPY not found.")
        return
        
    df = pd.read_csv(csv_path)
    arr = np.load(npy_path)
    
    print(f"[{mode}] Checking alignment...")
    print(f"  CSV Rows: {len(df)}")
    print(f"  NPY Rows: {len(arr)}")
    
    if len(df) == len(arr):
        print(f"  {mode}: PASSED (Lengths match)")
    else:
        print(f"  {mode}: FAILED (Length mismatch)")

if __name__ == "__main__":
    check_alignment('Main')
    check_alignment('Sub')
    check_alignment('Calc')
