#!/usr/bin/env python3
"""
Test script to verify the corrected SIC constraint formula produces varying power allocations.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, generate_channels, compute_power_allocation

def test_corrected_algorithm():
    """Test that the corrected algorithm produces varying power allocations."""
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Testing corrected SIC constraint formula...")
    print(f"Configuration: M={config.M}, P={config.P}, Pg={config.Pg}, N0={config.N0}")
    print()
    
    # Generate several samples to see variation
    for i in range(5):
        h_gains = generate_channels(config)
        alpha_star = compute_power_allocation(h_gains, config)
        
        print(f"Sample {i+1}:")
        print(f"  Channel gains: {h_gains}")
        print(f"  Power allocation: {alpha_star}")
        print(f"  Sum of alphas: {np.sum(alpha_star):.6f}")
        print(f"  All equal to 1/M? {np.allclose(alpha_star, 1/config.M)}")
        print()
    
    # Test with M=4 as well
    print("Testing with M=4:")
    config_m4 = NOMAConfig(M=4, P=1.0, Pg=0.01, N0=0.001)
    
    for i in range(3):
        h_gains = generate_channels(config_m4)
        alpha_star = compute_power_allocation(h_gains, config_m4)
        
        print(f"Sample {i+1} (M=4):")
        print(f"  Channel gains: {h_gains}")
        print(f"  Power allocation: {alpha_star}")
        print(f"  Sum of alphas: {np.sum(alpha_star):.6f}")
        print(f"  All equal to 1/M? {np.allclose(alpha_star, 1/config_m4.M)}")
        print()

if __name__ == "__main__":
    test_corrected_algorithm()