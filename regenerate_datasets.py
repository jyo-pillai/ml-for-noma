#!/usr/bin/env python3
"""
Script to regenerate NOMA datasets with corrected algorithm.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class NOMAConfig:
    M: int = 3
    P: float = 1.0
    N0: float = 0.001
    Pg: float = 0.01
    N_samples: int = 10000
    path_loss_exp: float = 2.0

def generate_channels(config: NOMAConfig) -> np.ndarray:
    """Generate M channel gains sorted in descending order using Rayleigh fading model."""
    h_real = np.random.randn(config.M)
    h_imag = np.random.randn(config.M)
    h_complex = (h_real + 1j * h_imag) / np.sqrt(2)
    h_gains = np.abs(h_complex) ** 2
    h_gains_sorted = np.sort(h_gains)[::-1]
    return h_gains_sorted

def compute_alpha_prime(M: int) -> np.ndarray:
    """Compute equal allocation bound."""
    return np.ones(M) / M

def compute_alpha_double_prime(h_gains: np.ndarray, config: NOMAConfig) -> np.ndarray:
    """
    CORRECTED SIC constraint bound implementation.
    
    SIC constraint: α_i - Σ(α_k, k=i+1..M) = Pg / (P · |h_i|²) for i=1 to M-1
    """
    M = config.M
    P = config.P
    Pg = config.Pg
    
    # Matrix A construction (M×M)
    A = np.zeros((M, M))
    B = np.zeros(M)
    
    # Row 0: Sum constraint (Σα = 1)
    A[0, :] = 1.0
    B[0] = 1.0
    
    # Rows 1 to M-1: SIC constraints
    for row in range(1, M):
        user_i = row  # User i in 1-based indexing (1 to M-1)
        user_i_idx = user_i - 1  # User i in 0-based indexing (0 to M-2)
        
        # Coefficient +1 at position i-1 (user i in 0-based indexing)
        A[row, user_i_idx] = 1.0
        
        # Coefficients -1 at positions i to M-1 (users i+1 to M in 0-based indexing)
        if user_i < M:
            A[row, user_i:] = -1.0
            
        # Vector B: Pg / (P · |h_i|²)
        B[row] = Pg / (P * h_gains[user_i_idx])

    # Solve linear system A · α'' = B
    try:
        alpha_double_prime = np.linalg.solve(A, B)
        
        # Safety check: if any α'' < 0, fallback to α'
        if np.any(alpha_double_prime < 0):
            return None
            
        return alpha_double_prime
        
    except np.linalg.LinAlgError:
        return None

def select_alpha_star(alpha_prime: np.ndarray, alpha_double_prime: np.ndarray) -> np.ndarray:
    """Select final power allocation as element-wise minimum."""
    return np.minimum(alpha_prime, alpha_double_prime)

def compute_power_allocation(h_gains: np.ndarray, config: NOMAConfig) -> np.ndarray:
    """Compute optimal power allocation using dual-bound method."""
    M = config.M
    
    # Step 1: Compute equal allocation bound (α')
    alpha_prime = compute_alpha_prime(M)
    
    # Step 2: Compute SIC constraint bound (α'')
    alpha_double_prime = compute_alpha_double_prime(h_gains, config)
    
    # Step 3: Handle fallback cases
    if alpha_double_prime is None:
        return alpha_prime

    if np.any(alpha_double_prime < 0):
        return alpha_prime
    
    # Step 4: Compute final α*
    alpha_star = select_alpha_star(alpha_prime, alpha_double_prime)
    
    return alpha_star

def compute_sinr(m: int, h_gains: np.ndarray, alpha: np.ndarray, config: NOMAConfig) -> float:
    """Compute SINR for user m using NOMA formula."""
    signal_power = alpha[m] * config.P * h_gains[m]
    
    if m == 0:
        interference_power = 0.0
    else:
        interference_sum_alpha = np.sum(alpha[:m])
        interference_power = interference_sum_alpha * config.P * h_gains[m]
        
    sinr = signal_power / (interference_power + config.N0)
    return float(max(0.0, sinr))

def compute_rate(sinr: float) -> float:
    """Compute Shannon rate from SINR."""
    return np.log2(1 + sinr)

def compute_sum_rate(h_gains: np.ndarray, alpha: np.ndarray, config: NOMAConfig) -> float:
    """Compute the total system sum rate."""
    M = len(h_gains)
    total_rate = 0.0
    
    for m in range(M):
        sinr = compute_sinr(m, h_gains, alpha, config)
        rate = compute_rate(sinr)
        total_rate += rate
        
    return total_rate

def validate_sample(h_gains: np.ndarray, alpha: np.ndarray, sum_rate: float) -> bool:
    """Validate a generated data sample."""
    if not np.all(h_gains[:-1] >= h_gains[1:]):
        return False
    if not np.all(h_gains > 0):
        return False
    if not np.all(alpha >= 0):
        return False
    if not (np.sum(alpha) <= 1.0 + 1e-6):
        return False
    if sum_rate < 0:
        return False
    return True

def generate_dataset(config: NOMAConfig, filename: str) -> pd.DataFrame:
    """Generate the NOMA training dataset."""
    data_records = []
    skipped_count = 0
    varying_count = 0
    
    print(f"Generating {config.N_samples} samples for M={config.M} users...")
    
    for i in range(config.N_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{config.N_samples} samples")
            
        # Generate sample
        h_gains = generate_channels(config)
        alpha_star = compute_power_allocation(h_gains, config)
        sum_rate = compute_sum_rate(h_gains, alpha_star, config)
        
        # Check if alpha values vary from equal allocation
        if not np.allclose(alpha_star, 1/config.M, atol=1e-6):
            varying_count += 1
        
        # Validate and store
        if validate_sample(h_gains, alpha_star, sum_rate):
            record = {}
            
            # Add channel gains
            for m in range(config.M):
                record[f'h{m+1}_gain'] = h_gains[m]
                
            # Add power allocations
            for m in range(config.M):
                record[f'alpha{m+1}'] = alpha_star[m]
                
            # Add sum rate
            record['Sum_Rate'] = sum_rate
            
            data_records.append(record)
        else:
            skipped_count += 1
    
    # Create DataFrame and save
    df = pd.DataFrame(data_records)
    df.to_csv(filename, index=False)
    
    print(f"Generation complete!")
    print(f"  Valid samples: {len(df)}")
    print(f"  Skipped samples: {skipped_count}")
    print(f"  Samples with varying α: {varying_count} ({varying_count/config.N_samples*100:.1f}%)")
    print(f"  Dataset saved to: {filename}")
    
    # Show sample statistics
    if len(df) > 0:
        print(f"\nSample statistics:")
        alpha_cols = [f'alpha{m+1}' for m in range(config.M)]
        print(f"  Alpha means: {df[alpha_cols].mean().values}")
        print(f"  Alpha stds:  {df[alpha_cols].std().values}")
        print(f"  Sum rate mean: {df['Sum_Rate'].mean():.4f}")
        
        # Show first few samples
        print(f"\nFirst 3 samples:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            h_vals = [row[f'h{m+1}_gain'] for m in range(config.M)]
            a_vals = [row[f'alpha{m+1}'] for m in range(config.M)]
            print(f"  Sample {i+1}: h={h_vals}, α={a_vals}, rate={row['Sum_Rate']:.4f}")
    
    return df

def main():
    """Main function to regenerate both M=3 and M=4 datasets."""
    print("Regenerating NOMA datasets with corrected algorithm")
    print("="*60)
    
    # Generate M=3 dataset
    print("\n1. Generating M=3 dataset...")
    config_m3 = NOMAConfig(M=3, N_samples=10000)
    df_m3 = generate_dataset(config_m3, "noma_training_data_M3.csv")
    
    # Generate M=4 dataset
    print("\n2. Generating M=4 dataset...")
    config_m4 = NOMAConfig(M=4, N_samples=10000)
    df_m4 = generate_dataset(config_m4, "noma_training_data_M4.csv")
    
    print("\n" + "="*60)
    print("Dataset regeneration complete!")
    print("Files created:")
    print("  - noma_training_data_M3.csv")
    print("  - noma_training_data_M4.csv")

if __name__ == "__main__":
    main()