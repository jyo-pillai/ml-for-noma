#!/usr/bin/env python3
"""
Verify that power allocations match expected pattern:
- Weaker users (higher index) should get more power
- α3 (weakest user) > α2 > α1 (strongest user) for typical cases
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, generate_channels, compute_power_allocation

def verify_power_allocation_pattern():
    """Verify power allocation pattern across multiple samples."""
    config = NOMAConfig(M=3, P=1.0, Pg=0.01)
    
    print("Verifying power allocation pattern...")
    print("Expected: Weaker users (higher index) get more power")
    print("Pattern: α3 (weakest) > α2 > α1 (strongest) for typical cases")
    print()
    
    # Generate multiple samples to verify pattern
    samples_with_pattern = 0
    total_samples = 100
    
    for i in range(total_samples):
        # Generate channels and power allocation
        h_gains = generate_channels(config)
        alpha_star = compute_power_allocation(h_gains, config)
        
        # Check if pattern holds: α3 > α2 > α1
        if alpha_star[2] > alpha_star[1] > alpha_star[0]:
            samples_with_pattern += 1
            
        # Print first 10 samples for inspection
        if i < 10:
            print(f"Sample {i+1}:")
            print(f"  Channel gains: h1={h_gains[0]:.4f}, h2={h_gains[1]:.4f}, h3={h_gains[2]:.4f}")
            print(f"  Power alloc:   α1={alpha_star[0]:.4f}, α2={alpha_star[1]:.4f}, α3={alpha_star[2]:.4f}")
            print(f"  Pattern α3>α2>α1: {alpha_star[2] > alpha_star[1] > alpha_star[0]}")
            print(f"  Sum: {np.sum(alpha_star):.4f}")
            print()
    
    pattern_percentage = (samples_with_pattern / total_samples) * 100
    print(f"Pattern verification results:")
    print(f"  Samples with α3 > α2 > α1 pattern: {samples_with_pattern}/{total_samples} ({pattern_percentage:.1f}%)")
    
    # Verify that power allocations are not all equal (1/M)
    equal_allocation_samples = 0
    for i in range(total_samples):
        h_gains = generate_channels(config)
        alpha_star = compute_power_allocation(h_gains, config)
        
        # Check if all alphas are approximately equal (within small tolerance)
        if np.allclose(alpha_star, 1/3, atol=1e-3):
            equal_allocation_samples += 1
    
    equal_percentage = (equal_allocation_samples / total_samples) * 100
    print(f"  Samples with equal allocation (α ≈ 1/3): {equal_allocation_samples}/{total_samples} ({equal_percentage:.1f}%)")
    
    # Verify the algorithm is working correctly
    if pattern_percentage > 50:
        print("✓ PASS: Power allocation pattern is correct - weaker users get more power")
    else:
        print("✗ FAIL: Power allocation pattern is incorrect")
        
    if equal_percentage < 90:
        print("✓ PASS: Power allocations vary based on channel conditions (not all equal)")
    else:
        print("✗ FAIL: Power allocations are mostly equal - algorithm may not be working")

if __name__ == "__main__":
    verify_power_allocation_pattern()