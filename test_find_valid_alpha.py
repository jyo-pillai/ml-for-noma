#!/usr/bin/env python3
"""
Test multiple channel realizations to find cases where α'' is valid.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, generate_channels, compute_alpha_double_prime, compute_power_allocation

def test_find_valid_alpha():
    """Test multiple channel realizations to find valid α'' cases."""
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Searching for channel realizations with valid α''...")
    print(f"Configuration: M={config.M}, P={config.P}, Pg={config.Pg}")
    print()
    
    valid_count = 0
    total_count = 100
    
    for i in range(total_count):
        h_gains = generate_channels(config)
        alpha_double_prime = compute_alpha_double_prime(h_gains, config)
        
        if alpha_double_prime is not None and not np.any(alpha_double_prime < 0):
            valid_count += 1
            alpha_star = compute_power_allocation(h_gains, config)
            
            print(f"Valid case {valid_count}:")
            print(f"  Channel gains: {h_gains}")
            print(f"  α'': {alpha_double_prime}")
            print(f"  α* (final): {alpha_star}")
            print(f"  All equal to 1/M? {np.allclose(alpha_star, 1/config.M)}")
            print()
            
            if valid_count >= 5:  # Show first 5 valid cases
                break
    
    print(f"Found {valid_count} valid cases out of {total_count} trials ({valid_count/total_count*100:.1f}%)")
    
    if valid_count == 0:
        print("No valid α'' found. This suggests the power gap Pg=0.01 might be too large for typical channel conditions.")
        print("Let's try with a smaller Pg...")
        
        # Try with smaller Pg
        config_small = NOMAConfig(M=3, P=1.0, Pg=0.001, N0=0.001)
        print(f"\nTrying with Pg={config_small.Pg}:")
        
        for i in range(20):
            h_gains = generate_channels(config_small)
            alpha_double_prime = compute_alpha_double_prime(h_gains, config_small)
            
            if alpha_double_prime is not None and not np.any(alpha_double_prime < 0):
                alpha_star = compute_power_allocation(h_gains, config_small)
                
                print(f"Valid case with Pg={config_small.Pg}:")
                print(f"  Channel gains: {h_gains}")
                print(f"  α'': {alpha_double_prime}")
                print(f"  α* (final): {alpha_star}")
                print(f"  All equal to 1/M? {np.allclose(alpha_star, 1/config.M)}")
                break

if __name__ == "__main__":
    test_find_valid_alpha()