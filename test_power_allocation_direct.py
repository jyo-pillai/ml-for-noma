#!/usr/bin/env python3
"""
Test the compute_power_allocation function directly.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, generate_channels, compute_power_allocation, compute_alpha_prime, compute_alpha_double_prime

def test_power_allocation_direct():
    """Test compute_power_allocation directly."""
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Testing compute_power_allocation directly...")
    
    # Use a fixed seed for reproducible results
    np.random.seed(42)
    
    for i in range(3):
        h_gains = generate_channels(config)
        
        print(f"\nSample {i+1}:")
        print(f"  Channel gains: {h_gains}")
        
        # Step by step
        alpha_prime = compute_alpha_prime(config.M)
        print(f"  α' (equal): {alpha_prime}")
        
        alpha_double_prime = compute_alpha_double_prime(h_gains, config)
        print(f"  α'' (SIC): {alpha_double_prime}")
        
        if alpha_double_prime is None:
            print("  α'' is None (singular matrix or negative values)")
            alpha_star = alpha_prime
        else:
            print(f"  Any negative in α''? {np.any(alpha_double_prime < 0)}")
            alpha_star = np.minimum(alpha_prime, alpha_double_prime)
        
        print(f"  α* (final): {alpha_star}")
        
        # Now test the full function
        alpha_from_function = compute_power_allocation(h_gains, config)
        print(f"  Function result: {alpha_from_function}")
        print(f"  Match? {np.allclose(alpha_star, alpha_from_function)}")

if __name__ == "__main__":
    test_power_allocation_direct()