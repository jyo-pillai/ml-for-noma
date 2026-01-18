#!/usr/bin/env python3
"""
Corrected NOMA implementation with working SIC constraint.
"""

import numpy as np
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

def test_corrected_implementation():
    """Test the corrected implementation."""
    print("Testing corrected NOMA implementation")
    print("="*50)
    
    # Test with specific good channel conditions
    h_gains = np.array([2.0, 1.0, 0.5])
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print(f"Channel gains: {h_gains}")
    print(f"Config: M={config.M}, P={config.P}, Pg={config.Pg}")
    print()
    
    # Test α''
    alpha_double_prime = compute_alpha_double_prime(h_gains, config)
    print(f"α'' (SIC constraint): {alpha_double_prime}")
    
    if alpha_double_prime is not None:
        print(f"Sum of α'': {np.sum(alpha_double_prime):.6f}")
        print(f"Any negative in α''? {np.any(alpha_double_prime < 0)}")
        
        # Test full power allocation
        alpha_star = compute_power_allocation(h_gains, config)
        print(f"α* (final allocation): {alpha_star}")
        print(f"Sum of α*: {np.sum(alpha_star):.6f}")
        print(f"All equal to 1/M? {np.allclose(alpha_star, 1/config.M)}")
        
        if not np.allclose(alpha_star, 1/config.M):
            print("SUCCESS: Power allocation varies from equal allocation!")
            print(f"Expected pattern: α1 < α2 < α3 (weaker users get more power)")
            print(f"Actual: α1={alpha_star[0]:.4f}, α2={alpha_star[1]:.4f}, α3={alpha_star[2]:.4f}")
        else:
            print("Still using equal allocation (α'' had negative values)")
    else:
        print("α'' is None (singular matrix or negative values)")
    
    print()
    print("Testing with multiple random samples:")
    
    # Test with multiple samples to find valid cases
    valid_count = 0
    for i in range(20):
        h_gains_random = generate_channels(config)
        alpha_double_prime_random = compute_alpha_double_prime(h_gains_random, config)
        
        if alpha_double_prime_random is not None and not np.any(alpha_double_prime_random < 0):
            valid_count += 1
            alpha_star_random = compute_power_allocation(h_gains_random, config)
            
            print(f"Valid sample {valid_count}:")
            print(f"  h_gains: {h_gains_random}")
            print(f"  α*: {alpha_star_random}")
            print(f"  Varies? {not np.allclose(alpha_star_random, 1/config.M)}")
            
            if valid_count >= 3:  # Show first 3 valid cases
                break
    
    print(f"\nFound {valid_count} valid cases out of 20 trials")

if __name__ == "__main__":
    test_corrected_implementation()