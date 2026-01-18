#!/usr/bin/env python3
"""
Test compute_alpha_double_prime with specific channel gains.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, compute_alpha_double_prime, compute_power_allocation

def test_specific_channels():
    """Test with specific channel gains that should work."""
    
    # Use the same channel gains that worked in manual calculation
    h_gains = np.array([2.0, 1.0, 0.5])
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Testing compute_alpha_double_prime with specific channel gains")
    print(f"Channel gains: {h_gains}")
    print(f"Config: M={config.M}, P={config.P}, Pg={config.Pg}")
    print()
    
    # Test our implementation
    alpha_double_prime = compute_alpha_double_prime(h_gains, config)
    
    if alpha_double_prime is None:
        print("compute_alpha_double_prime returned None")
    else:
        print(f"α'' from implementation: {alpha_double_prime}")
        print(f"Sum: {np.sum(alpha_double_prime):.6f}")
        print(f"Any negative? {np.any(alpha_double_prime < 0)}")
        
        # Test full power allocation
        alpha_star = compute_power_allocation(h_gains, config)
        print(f"α* from compute_power_allocation: {alpha_star}")
        print(f"All equal to 1/M? {np.allclose(alpha_star, 1/config.M)}")
    
    print()
    
    # Also test with even better channel conditions
    h_gains_better = np.array([5.0, 2.0, 1.0])
    print(f"Testing with better channels: {h_gains_better}")
    
    alpha_double_prime_better = compute_alpha_double_prime(h_gains_better, config)
    
    if alpha_double_prime_better is None:
        print("compute_alpha_double_prime returned None")
    else:
        print(f"α'' from implementation: {alpha_double_prime_better}")
        print(f"Sum: {np.sum(alpha_double_prime_better):.6f}")
        print(f"Any negative? {np.any(alpha_double_prime_better < 0)}")
        
        alpha_star_better = compute_power_allocation(h_gains_better, config)
        print(f"α* from compute_power_allocation: {alpha_star_better}")
        print(f"All equal to 1/M? {np.allclose(alpha_star_better, 1/config.M)}")

if __name__ == "__main__":
    test_specific_channels()