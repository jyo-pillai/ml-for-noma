#!/usr/bin/env python3
"""
Verify the corrected power allocation pattern.
The correct pattern is not necessarily α3 > α2 > α1, but rather:
- α1 is often constrained by the equal allocation bound (1/3)
- α2 and α3 are determined by SIC constraints
- The sum should be less than 1 (not equal to 1)
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, generate_channels, compute_power_allocation, compute_alpha_prime, compute_alpha_double_prime

def verify_corrected_algorithm():
    """Verify the corrected algorithm behavior."""
    config = NOMAConfig(M=3, P=1.0, Pg=0.01)
    
    print("Verifying corrected NOMA power allocation algorithm...")
    print("Expected behavior:")
    print("- α1 often equals 1/3 (equal allocation bound)")
    print("- α2, α3 determined by SIC constraints")
    print("- Sum(α) < 1 (not equal to 1)")
    print("- Power allocations vary with channel conditions")
    print()
    
    # Statistics tracking
    alpha1_equals_third = 0
    sum_less_than_one = 0
    varying_allocations = 0
    total_samples = 100
    
    alpha_sums = []
    alpha1_values = []
    alpha2_values = []
    alpha3_values = []
    
    for i in range(total_samples):
        # Generate channels and power allocation
        h_gains = generate_channels(config)
        alpha_star = compute_power_allocation(h_gains, config)
        
        # Track statistics
        if np.isclose(alpha_star[0], 1/3, atol=1e-6):
            alpha1_equals_third += 1
            
        alpha_sum = np.sum(alpha_star)
        if alpha_sum < 1.0:
            sum_less_than_one += 1
            
        # Check if allocations are not all equal
        if not np.allclose(alpha_star, alpha_star[0], atol=1e-3):
            varying_allocations += 1
            
        # Store values for analysis
        alpha_sums.append(alpha_sum)
        alpha1_values.append(alpha_star[0])
        alpha2_values.append(alpha_star[1])
        alpha3_values.append(alpha_star[2])
        
        # Print first 5 samples for inspection
        if i < 5:
            print(f"Sample {i+1}:")
            print(f"  h_gains: [{h_gains[0]:.4f}, {h_gains[1]:.4f}, {h_gains[2]:.4f}]")
            print(f"  α*:      [{alpha_star[0]:.4f}, {alpha_star[1]:.4f}, {alpha_star[2]:.4f}]")
            print(f"  Sum:     {alpha_sum:.4f}")
            print(f"  α1=1/3:  {np.isclose(alpha_star[0], 1/3, atol=1e-6)}")
            print()
    
    # Calculate statistics
    alpha1_third_pct = (alpha1_equals_third / total_samples) * 100
    sum_less_pct = (sum_less_than_one / total_samples) * 100
    varying_pct = (varying_allocations / total_samples) * 100
    
    print("Algorithm verification results:")
    print(f"  α1 equals 1/3: {alpha1_equals_third}/{total_samples} ({alpha1_third_pct:.1f}%)")
    print(f"  Sum(α) < 1.0:  {sum_less_than_one}/{total_samples} ({sum_less_pct:.1f}%)")
    print(f"  Varying alloc: {varying_allocations}/{total_samples} ({varying_pct:.1f}%)")
    print()
    
    # Statistical summary
    print("Statistical summary:")
    print(f"  α1: mean={np.mean(alpha1_values):.4f}, std={np.std(alpha1_values):.4f}")
    print(f"  α2: mean={np.mean(alpha2_values):.4f}, std={np.std(alpha2_values):.4f}")
    print(f"  α3: mean={np.mean(alpha3_values):.4f}, std={np.std(alpha3_values):.4f}")
    print(f"  Sum: mean={np.mean(alpha_sums):.4f}, std={np.std(alpha_sums):.4f}")
    print()
    
    # Verify expected behavior
    print("Verification results:")
    
    if alpha1_third_pct > 80:
        print("✓ PASS: α1 frequently equals 1/3 (equal allocation bound constraint)")
    else:
        print("? INFO: α1 doesn't always equal 1/3 - this may be normal depending on channel conditions")
        
    if sum_less_pct > 90:
        print("✓ PASS: Sum(α) < 1.0 for most samples (correct SIC constraint behavior)")
    else:
        print("✗ FAIL: Sum(α) should be < 1.0 for most samples")
        
    if varying_pct > 90:
        print("✓ PASS: Power allocations vary with channel conditions")
    else:
        print("✗ FAIL: Power allocations should vary with channel conditions")
        
    # Check that the algorithm is not just returning equal allocation
    if np.std(alpha2_values) > 0.01 and np.std(alpha3_values) > 0.01:
        print("✓ PASS: α2 and α3 show significant variation (SIC constraints working)")
    else:
        print("✗ FAIL: α2 and α3 should show variation based on SIC constraints")

def compare_with_equal_allocation():
    """Compare with pure equal allocation to show the algorithm is working."""
    config = NOMAConfig(M=3, P=1.0, Pg=0.01)
    
    print("\nComparing with pure equal allocation:")
    
    # Generate a sample
    h_gains = generate_channels(config)
    alpha_star = compute_power_allocation(h_gains, config)
    alpha_equal = np.ones(3) / 3
    
    print(f"Channel gains: {h_gains}")
    print(f"Equal allocation (1/3 each): {alpha_equal}")
    print(f"Optimized allocation:         {alpha_star}")
    print(f"Difference from equal:        {alpha_star - alpha_equal}")
    print(f"Sum (equal):                  {np.sum(alpha_equal):.4f}")
    print(f"Sum (optimized):              {np.sum(alpha_star):.4f}")

if __name__ == "__main__":
    verify_corrected_algorithm()
    compare_with_equal_allocation()