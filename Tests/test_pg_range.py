#!/usr/bin/env python3
"""
Test with Pg values >= 0.01 to find valid α'' solutions.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, generate_channels, compute_alpha_double_prime, compute_power_allocation

def test_pg_range():
    """Test with different Pg values >= 0.01."""
    
    # Test Pg values >= 0.01
    pg_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.1]
    
    print("Testing Pg values >= 0.01 for valid α'' solutions...")
    print()
    
    for pg in pg_values:
        config = NOMAConfig(M=3, P=1.0, Pg=pg, N0=0.001)
        print(f"Testing Pg = {pg}:")
        
        valid_count = 0
        total_trials = 100
        
        for i in range(total_trials):
            h_gains = generate_channels(config)
            alpha_double_prime = compute_alpha_double_prime(h_gains, config)
            
            if alpha_double_prime is not None and not np.any(alpha_double_prime < 0):
                valid_count += 1
                
                # Show first valid case for this Pg
                if valid_count == 1:
                    alpha_star = compute_power_allocation(h_gains, config)
                    print(f"  First valid case:")
                    print(f"    Channel gains: {h_gains}")
                    print(f"    α'': {alpha_double_prime}")
                    print(f"    α* (final): {alpha_star}")
                    print(f"    Varies from 1/M? {not np.allclose(alpha_star, 1/config.M)}")
        
        print(f"  Valid cases: {valid_count}/{total_trials} ({valid_count/total_trials*100:.1f}%)")
        print()
    
    # Also test what happens when we increase the total power P
    print("Testing with increased total power P (keeping Pg=0.01):")
    p_values = [1.0, 2.0, 5.0, 10.0]
    
    for p in p_values:
        config = NOMAConfig(M=3, P=p, Pg=0.01, N0=0.001)
        print(f"Testing P = {p}W:")
        
        valid_count = 0
        total_trials = 50
        
        for i in range(total_trials):
            h_gains = generate_channels(config)
            alpha_double_prime = compute_alpha_double_prime(h_gains, config)
            
            if alpha_double_prime is not None and not np.any(alpha_double_prime < 0):
                valid_count += 1
                
                if valid_count == 1:
                    alpha_star = compute_power_allocation(h_gains, config)
                    print(f"  First valid case:")
                    print(f"    Channel gains: {h_gains}")
                    print(f"    α'': {alpha_double_prime}")
                    print(f"    α* (final): {alpha_star}")
                    print(f"    Varies from 1/M? {not np.allclose(alpha_star, 1/config.M)}")
        
        print(f"  Valid cases: {valid_count}/{total_trials} ({valid_count/total_trials*100:.1f}%)")
        print()

if __name__ == "__main__":
    test_pg_range()