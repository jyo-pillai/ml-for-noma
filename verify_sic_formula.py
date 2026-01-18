#!/usr/bin/env python3
"""
Verify the SIC constraint formula implementation against manual calculation.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig

def verify_sic_formula():
    """Verify SIC constraint formula implementation."""
    
    # Use specific channel gains for verification
    h_gains = np.array([2.0, 1.0, 0.5])  # Strong > Medium > Weak
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Verifying SIC constraint formula implementation")
    print(f"Channel gains: {h_gains}")
    print(f"P = {config.P}, Pg = {config.Pg}")
    print()
    
    # Manual matrix construction according to the task specification:
    # Correct SIC constraint: α_i - Σ(α_k, k=i+1..M) = Pg / (P · |h_i|²) for i=1 to M-1
    
    M = 3
    A = np.zeros((M, M))
    B = np.zeros(M)
    
    # Row 0: Sum constraint
    A[0, :] = 1.0
    B[0] = 1.0
    print("Row 0 (sum constraint): Σα = 1")
    print(f"A[0] = {A[0]}, B[0] = {B[0]}")
    print()
    
    # Row 1: SIC constraint for user 1 (i=1)
    # α_1 - Σ(α_k, k=2..3) = α_1 - (α_2 + α_3) = Pg / (P · |h_1|²)
    A[1, 0] = 1.0   # α_1 coefficient
    A[1, 1] = -1.0  # α_2 coefficient  
    A[1, 2] = -1.0  # α_3 coefficient
    B[1] = config.Pg / (config.P * h_gains[0])  # Pg / (P · |h_1|²)
    print("Row 1 (SIC constraint for user 1): α_1 - α_2 - α_3 = Pg/(P·|h_1|²)")
    print(f"A[1] = {A[1]}, B[1] = {B[1]:.6f}")
    print()
    
    # Row 2: SIC constraint for user 2 (i=2)  
    # α_2 - Σ(α_k, k=3..3) = α_2 - α_3 = Pg / (P · |h_2|²)
    A[2, 1] = 1.0   # α_2 coefficient
    A[2, 2] = -1.0  # α_3 coefficient
    B[2] = config.Pg / (config.P * h_gains[1])  # Pg / (P · |h_2|²)
    print("Row 2 (SIC constraint for user 2): α_2 - α_3 = Pg/(P·|h_2|²)")
    print(f"A[2] = {A[2]}, B[2] = {B[2]:.6f}")
    print()
    
    print("Complete system A·α = B:")
    print("A =")
    print(A)
    print(f"B = {B}")
    print()
    
    # Solve the system
    try:
        alpha_solution = np.linalg.solve(A, B)
        print(f"Solution α = {alpha_solution}")
        print(f"Sum = {np.sum(alpha_solution):.6f}")
        print(f"Any negative? {np.any(alpha_solution < 0)}")
        print()
        
        # Verify the constraints
        print("Verification:")
        print(f"Sum constraint: Σα = {np.sum(alpha_solution):.6f} (should be 1.0)")
        
        # SIC constraint 1: α_1 - α_2 - α_3 = Pg/(P·|h_1|²)
        lhs1 = alpha_solution[0] - alpha_solution[1] - alpha_solution[2]
        rhs1 = config.Pg / (config.P * h_gains[0])
        print(f"SIC constraint 1: {lhs1:.6f} = {rhs1:.6f} (diff: {abs(lhs1-rhs1):.2e})")
        
        # SIC constraint 2: α_2 - α_3 = Pg/(P·|h_2|²)
        lhs2 = alpha_solution[1] - alpha_solution[2]
        rhs2 = config.Pg / (config.P * h_gains[1])
        print(f"SIC constraint 2: {lhs2:.6f} = {rhs2:.6f} (diff: {abs(lhs2-rhs2):.2e})")
        
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error: {e}")
    
    # Now test our implementation
    print("\n" + "="*50)
    print("Testing our implementation:")
    
    from noma_dataset_generator import compute_alpha_double_prime
    alpha_impl = compute_alpha_double_prime(h_gains, config)
    
    if alpha_impl is None:
        print("Our implementation returned None (negative values detected)")
    else:
        print(f"Our implementation result: {alpha_impl}")
        print(f"Matches manual calculation? {np.allclose(alpha_solution, alpha_impl) if 'alpha_solution' in locals() else 'N/A'}")

if __name__ == "__main__":
    verify_sic_formula()