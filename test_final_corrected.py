#!/usr/bin/env python3
"""
Test the final corrected implementation.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, compute_alpha_double_prime, compute_power_allocation

def test_final_corrected():
    """Test the final corrected implementation."""
    
    # Test with good channel conditions
    h_gains_good = np.array([2.0, 1.0, 0.5])
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Testing with good channel conditions:")
    print(f"Channel gains: {h_gains_good}")
    
    alpha_double_prime = compute_alpha_double_prime(h_gains_good, config)
    print(f"α'': {alpha_double_prime}")
    
    if alpha_double_prime is not None:
        print(f"Sum: {np.sum(alpha_double_prime):.6f}")
        print(f"Any negative? {np.any(alpha_double_prime < 0)}")
        
        alpha_star = compute_power_allocation(h_gains_good, config)
        print(f"α* (final): {alpha_star}")
        print(f"All equal to 1/M? {np.allclose(alpha_star, 1/config.M)}")
        print(f"Varies from equal allocation? {not np.allclose(alpha_star, 1/config.M)}")
    else:
        print("α'' is None (negative values)")
    
    print()
    
    # Test with even better channel conditions
    h_gains_better = np.array([5.0, 2.0, 1.0])
    print("Testing with even better channel conditions:")
    print(f"Channel gains: {h_gains_better}")
    
    alpha_double_prime_better = compute_alpha_double_prime(h_gains_better, config)
    print(f"α'': {alpha_double_prime_better}")
    
    if alpha_double_prime_better is not None:
        print(f"Sum: {np.sum(alpha_double_prime_better):.6f}")
        print(f"Any negative? {np.any(alpha_double_prime_better < 0)}")
        
        alpha_star_better = compute_power_allocation(h_gains_better, config)
        print(f"α* (final): {alpha_star_better}")
        print(f"All equal to 1/M? {np.allclose(alpha_star_better, 1/config.M)}")
        print(f"Varies from equal allocation? {not np.allclose(alpha_star_better, 1/config.M)}")
    else:
        print("α'' is None (negative values)")

if __name__ == "__main__":
    test_final_corrected()