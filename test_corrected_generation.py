#!/usr/bin/env python3
"""
Test script to verify the corrected algorithm generates varying alpha values.
"""

import numpy as np
import sys
sys.path.append('.')
from noma_dataset_generator import NOMAConfig, generate_channels, compute_power_allocation, compute_sum_rate

def test_corrected_algorithm():
    """Test that the corrected algorithm produces varying alpha values."""
    print("Testing corrected algorithm with sample generations...")
    print("="*60)
    
    config = NOMAConfig(M=3, N_samples=10)
    
    varying_count = 0
    total_samples = 20
    
    for i in range(total_samples):
        h_gains = generate_channels(config)
        alpha_star = compute_power_allocation(h_gains, config)
        sum_rate = compute_sum_rate(h_gains, alpha_star, config)
        
        # Check if alpha values vary from equal allocation
        is_equal = np.allclose(alpha_star, 1/3, atol=1e-6)
        if not is_equal:
            varying_count += 1
            
        if i < 5:  # Show first 5 samples
            print(f"Sample {i+1}:")
            print(f"  h_gains: [{h_gains[0]:.4f}, {h_gains[1]:.4f}, {h_gains[2]:.4f}]")
            print(f"  alpha*:  [{alpha_star[0]:.4f}, {alpha_star[1]:.4f}, {alpha_star[2]:.4f}]")
            print(f"  Equal allocation? {is_equal}")
            print(f"  Sum rate: {sum_rate:.4f}")
            
            if not is_equal:
                print(f"  SUCCESS: α1={alpha_star[0]:.4f}, α2={alpha_star[1]:.4f}, α3={alpha_star[2]:.4f}")
                # Check expected pattern: weaker users should get more power
                if alpha_star[2] > alpha_star[1] and alpha_star[1] >= alpha_star[0]:
                    print(f"  PATTERN CORRECT: α3 > α2 >= α1 (weaker users get more power)")
                else:
                    print(f"  PATTERN: α1={alpha_star[0]:.4f}, α2={alpha_star[1]:.4f}, α3={alpha_star[2]:.4f}")
            print()
    
    print(f"Results: {varying_count}/{total_samples} samples had varying alpha values")
    print(f"Success rate: {varying_count/total_samples*100:.1f}%")
    
    if varying_count > 0:
        print("✓ CORRECTED ALGORITHM IS WORKING - Alpha values vary from equal allocation!")
        return True
    else:
        print("✗ Algorithm still using equal allocation only")
        return False

if __name__ == "__main__":
    test_corrected_algorithm()