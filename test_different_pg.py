#!/usr/bin/env python3
"""
Test with different power gap values to find valid α'' solutions.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, generate_channels, compute_alpha_double_prime

def test_different_pg():
    """Test with different Pg values."""
    
    # Use a fixed seed for reproducible results
    np.random.seed(42)
    h_gains = generate_channels(NOMAConfig(M=3))
    
    print(f"Channel gains: {h_gains}")
    print()
    
    # Test different Pg values
    pg_values = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    for pg in pg_values:
        config = NOMAConfig(M=3, P=1.0, Pg=pg, N0=0.001)
        alpha_double_prime = compute_alpha_double_prime(h_gains, config)
        
        print(f"Pg = {pg}:")
        if alpha_double_prime is None:
            print("  α'' = None (singular matrix)")
        else:
            print(f"  α'' = {alpha_double_prime}")
            print(f"  Sum = {np.sum(alpha_double_prime):.6f}")
            print(f"  Any negative? {np.any(alpha_double_prime < 0)}")
            print(f"  Valid? {not np.any(alpha_double_prime < 0)}")
        print()

if __name__ == "__main__":
    test_different_pg()