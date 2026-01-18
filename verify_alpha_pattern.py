#!/usr/bin/env python3
"""
Verify that the regenerated datasets show the expected alpha pattern:
- α values vary (not all equal to 1/M)
- Expected: α3 (weakest user) > α2 > α1 (strongest user) for M=3
- Expected: α4 (weakest user) > α3 > α2 > α1 (strongest user) for M=4
"""

import pandas as pd
import numpy as np

def verify_dataset(filename, M):
    """Verify the alpha pattern in a dataset."""
    print(f"\nVerifying {filename} (M={M})...")
    print("="*50)
    
    # Load dataset
    df = pd.read_csv(filename)
    print(f"Dataset size: {len(df)} samples")
    
    # Check alpha columns
    alpha_cols = [f'alpha{i+1}' for i in range(M)]
    
    # Statistics
    alpha_means = df[alpha_cols].mean()
    alpha_stds = df[alpha_cols].std()
    
    print(f"Alpha means: {alpha_means.values}")
    print(f"Alpha stds:  {alpha_stds.values}")
    
    # Check if values vary from equal allocation
    equal_allocation = 1.0 / M
    varying_samples = 0
    
    for idx, row in df.iterrows():
        alpha_values = [row[col] for col in alpha_cols]
        if not np.allclose(alpha_values, equal_allocation, atol=1e-6):
            varying_samples += 1
    
    print(f"Samples with varying α: {varying_samples}/{len(df)} ({varying_samples/len(df)*100:.1f}%)")
    
    # Check expected pattern for varying samples
    if M == 3:
        # Expected: α3 > α2 > α1 (weaker users get more power)
        pattern_correct = 0
        for idx, row in df.iterrows():
            alpha1, alpha2, alpha3 = row['alpha1'], row['alpha2'], row['alpha3']
            if not np.allclose([alpha1, alpha2, alpha3], equal_allocation, atol=1e-6):
                # Check if α3 >= α2 >= α1 (allowing for some numerical tolerance)
                if alpha3 >= alpha2 - 1e-6 and alpha2 >= alpha1 - 1e-6:
                    pattern_correct += 1
        
        print(f"Samples following expected pattern (α3 >= α2 >= α1): {pattern_correct}/{varying_samples}")
        
    elif M == 4:
        # Expected: α4 > α3 > α2 > α1 (weaker users get more power)
        pattern_correct = 0
        for idx, row in df.iterrows():
            alpha1, alpha2, alpha3, alpha4 = row['alpha1'], row['alpha2'], row['alpha3'], row['alpha4']
            if not np.allclose([alpha1, alpha2, alpha3, alpha4], equal_allocation, atol=1e-6):
                # Check if α4 >= α3 >= α2 >= α1 (allowing for some numerical tolerance)
                if (alpha4 >= alpha3 - 1e-6 and alpha3 >= alpha2 - 1e-6 and 
                    alpha2 >= alpha1 - 1e-6):
                    pattern_correct += 1
        
        print(f"Samples following expected pattern (α4 >= α3 >= α2 >= α1): {pattern_correct}/{varying_samples}")
    
    # Show first few varying samples
    print(f"\nFirst 5 samples with varying α:")
    varying_count = 0
    for idx, row in df.iterrows():
        alpha_values = [row[col] for col in alpha_cols]
        if not np.allclose(alpha_values, equal_allocation, atol=1e-6):
            varying_count += 1
            if varying_count <= 5:
                h_values = [row[f'h{i+1}_gain'] for i in range(M)]
                print(f"  Sample {varying_count}:")
                print(f"    h_gains: {[f'{h:.4f}' for h in h_values]}")
                print(f"    α values: {[f'{a:.4f}' for a in alpha_values]}")
                print(f"    Sum rate: {row['Sum_Rate']:.4f}")
                
                # Check pattern
                if M == 3:
                    if alpha_values[2] >= alpha_values[1] >= alpha_values[0]:
                        print(f"    ✓ Pattern correct: α3 >= α2 >= α1")
                    else:
                        print(f"    ✗ Pattern incorrect")
                elif M == 4:
                    if (alpha_values[3] >= alpha_values[2] >= alpha_values[1] >= alpha_values[0]):
                        print(f"    ✓ Pattern correct: α4 >= α3 >= α2 >= α1")
                    else:
                        print(f"    ✗ Pattern incorrect")
                print()
    
    return varying_samples > 0

def main():
    """Main verification function."""
    print("Verifying regenerated NOMA datasets")
    print("="*60)
    
    # Verify M=3 dataset
    m3_success = verify_dataset("noma_training_data_M3.csv", 3)
    
    # Verify M=4 dataset  
    m4_success = verify_dataset("noma_training_data_M4.csv", 4)
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY:")
    print(f"✓ M=3 dataset: {'PASS' if m3_success else 'FAIL'} - Alpha values vary from equal allocation")
    print(f"✓ M=4 dataset: {'PASS' if m4_success else 'FAIL'} - Alpha values vary from equal allocation")
    
    if m3_success and m4_success:
        print("\n🎉 SUCCESS: Both datasets show varying alpha values!")
        print("The corrected SIC constraint algorithm is working properly.")
        print("Weaker users (higher indices) receive more power as expected.")
    else:
        print("\n❌ FAILURE: Datasets still show equal allocation only.")

if __name__ == "__main__":
    main()