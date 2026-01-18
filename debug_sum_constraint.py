#!/usr/bin/env python3
"""
Debug why the sum constraint is not being satisfied.
"""

import pandas as pd
import numpy as np

def debug_sum_constraint():
    """Debug the sum constraint issue."""
    print("Debugging Sum Constraint Issue")
    print("="*40)
    
    # Load M=3 dataset and check first few samples
    df3 = pd.read_csv("noma_training_data_M3.csv")
    
    print("M=3 - First 5 samples:")
    for i in range(5):
        row = df3.iloc[i]
        alpha1, alpha2, alpha3 = row['alpha1'], row['alpha2'], row['alpha3']
        alpha_sum = alpha1 + alpha2 + alpha3
        
        print(f"Sample {i+1}:")
        print(f"  α1={alpha1:.6f}, α2={alpha2:.6f}, α3={alpha3:.6f}")
        print(f"  Sum={alpha_sum:.6f} (should be 1.0)")
        print(f"  Deficit: {1.0 - alpha_sum:.6f}")
        print()
    
    # Check overall statistics
    alpha_sums = df3['alpha1'] + df3['alpha2'] + df3['alpha3']
    print(f"M=3 Sum statistics:")
    print(f"  Mean: {alpha_sums.mean():.6f}")
    print(f"  Std:  {alpha_sums.std():.6f}")
    print(f"  Min:  {alpha_sums.min():.6f}")
    print(f"  Max:  {alpha_sums.max():.6f}")
    
    print("\n" + "="*40)
    print("M=4 - First 5 samples:")
    
    df4 = pd.read_csv("noma_training_data_M4.csv")
    for i in range(5):
        row = df4.iloc[i]
        alpha1, alpha2, alpha3, alpha4 = row['alpha1'], row['alpha2'], row['alpha3'], row['alpha4']
        alpha_sum = alpha1 + alpha2 + alpha3 + alpha4
        
        print(f"Sample {i+1}:")
        print(f"  α1={alpha1:.6f}, α2={alpha2:.6f}, α3={alpha3:.6f}, α4={alpha4:.6f}")
        print(f"  Sum={alpha_sum:.6f} (should be 1.0)")
        print(f"  Deficit: {1.0 - alpha_sum:.6f}")
        print()
    
    # Check overall statistics
    alpha_sums = df4['alpha1'] + df4['alpha2'] + df4['alpha3'] + df4['alpha4']
    print(f"M=4 Sum statistics:")
    print(f"  Mean: {alpha_sums.mean():.6f}")
    print(f"  Std:  {alpha_sums.std():.6f}")
    print(f"  Min:  {alpha_sums.min():.6f}")
    print(f"  Max:  {alpha_sums.max():.6f}")
    
    print("\n" + "="*40)
    print("ANALYSIS:")
    print("The sum constraint is NOT being satisfied!")
    print("This indicates that the min(α', α'') operation is not preserving the sum constraint.")
    print("This is actually expected behavior when α'' has negative values and we fallback to α',")
    print("but here it seems like we're using α'' values that don't sum to 1.0.")
    print("\nThis suggests there might be an issue with the SIC constraint matrix construction")
    print("or the linear system solution.")

if __name__ == "__main__":
    debug_sum_constraint()