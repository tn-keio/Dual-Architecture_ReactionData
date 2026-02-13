
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
