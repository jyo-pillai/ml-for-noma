#!/usr/bin/env python3
"""
Test dataset generation with corrected SIC constraint.
"""

import sys
sys.path.insert(0, '.')

from noma_corrected import NOMAConfig, generate_channels, compute_power_allocation
import pandas as pd
import numpy as np

def generate_small_dataset(config, n_samples=100):
    """Generate a small dataset to test the corrected implementation."""
    data_records = []
    skipped_count = 0
    varying_count = 0
    
    print(f"Generating {n_samples} samples with corrected SIC constraint...")
    
    for i in range(n_samples):
        # Generate sample
        h_gains = generate_channels(config)
        alpha_star = compute_power_allocation(h_gains, config)
        sum_rate = compute_sum_rate(h_gains, alpha_star, config)
        
        # Validate
        if validate_sample(h_gains, alpha_star, sum_rate):
            # Check if power allocation varies from equal allocation
            if not np.allclose(alpha_star, 1/config.M):
                varying_count += 1
                
            # Create record
            record = {}
            for m in range(config.M):
                record[f'h{m+1}_gain'] = h_gains[m]
            for m in range(config.M):
                record[f'alpha{m+1}'] = alpha_star[m]
            record['Sum_Rate'] = sum_rate
            
            data_records.append(record)
        else:
            skipped_count += 1
    
    # Create DataFrame
    df = pd.DataFrame(data_records)
    
    print(f"Generated {len(df)} valid samples")
    print(f"Skipped {skipped_count} invalid samples")
    print(f"Samples with varying power allocation: {varying_count}/{len(df)} ({varying_count/len(df)*100:.1f}%)")
    
    if varying_count > 0:
        print("\nSUCCESS: Corrected SIC constraint produces varying power allocations!")
        
        # Show some examples
        varying_samples = df[~np.isclose(df['alpha1'], 1/config.M)]
        if len(varying_samples) > 0:
            print("\nExamples of varying power allocations:")
            print(varying_samples.head(3))
    else:
        print("All samples still use equal allocation (1/M)")
    
    return df

def compute_sum_rate(h_gains, alpha, config):
    """Simple sum rate calculation for testing."""
    total_rate = 0.0
    for m in range(len(h_gains)):
        # Simple SINR calculation
        signal = alpha[m] * config.P * h_gains[m]
        if m == 0:
            interference = 0
        else:
            interference = np.sum(alpha[:m]) * config.P * h_gains[m]
        sinr = signal / (interference + config.N0)
        rate = np.log2(1 + sinr)
        total_rate += rate
    return total_rate

def validate_sample(h_gains, alpha, sum_rate):
    """Simple validation for testing."""
    return (np.all(h_gains[:-1] >= h_gains[1:]) and 
            np.all(h_gains > 0) and 
            np.all(alpha >= 0) and 
            np.sum(alpha) <= 1.01 and 
            sum_rate >= 0)

if __name__ == "__main__":
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    df = generate_small_dataset(config, n_samples=200)
    
    if len(df) > 0:
        print(f"\nDataset statistics:")
        print(f"Mean sum rate: {df['Sum_Rate'].mean():.4f}")
        print(f"Alpha1 range: [{df['alpha1'].min():.4f}, {df['alpha1'].max():.4f}]")
        print(f"Alpha2 range: [{df['alpha2'].min():.4f}, {df['alpha2'].max():.4f}]")
        print(f"Alpha3 range: [{df['alpha3'].min():.4f}, {df['alpha3'].max():.4f}]")