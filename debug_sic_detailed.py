#!/usr/bin/env python3
"""
Debug the SIC constraint calculation in detail to understand why
the algorithm is falling back to equal allocation.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, generate_channels, compute_alpha_double_prime, compute_alpha_prime

def debug_sic_calculation():
    """Debug a single SIC constraint calculation."""
    config = NOMAConfig(M=3, P=1.0, Pg=0.01)
    
    print("Debugging SIC constraint calculation...")
    print(f"Config: M={config.M}, P={config.P}, Pg={config.Pg}")
    print()
    
    # Generate a sample
    h_gains = generate_channels(config)
    print(f"Channel gains: {h_gains}")
    print(f"Sorted order: h1={h_gains[0]:.4f} >= h2={h_gains[1]:.4f} >= h3={h_gains[2]:.4f}")
    print()
    
    # Compute equal allocation bound
    alpha_prime = compute_alpha_prime(config.M)
    print(f"Equal allocation bound (α'): {alpha_prime}")
    print()
    
    # Debug the SIC constraint matrix construction
    M = config.M
    P = config.P
    Pg = config.Pg
    
    print("Constructing SIC constraint matrix A and vector B:")
    
    # Matrix A construction (M×M)
    A = np.zeros((M, M))
    B = np.zeros(M)
    
    # Row 0: Sum constraint (Σα = 1)
    A[0, :] = 1.0
    B[0] = 1.0
    print(f"Row 0 (sum constraint): A[0] = {A[0]}, B[0] = {B[0]}")
    
    # Rows 1 to M-1: SIC constraints
    for row in range(1, M):
        user_i = row  # User i in 1-based indexing (1 to M-1)
        user_i_idx = user_i - 1  # User i in 0-based indexing (0 to M-2)
        
        print(f"\nRow {row} (SIC constraint for user {user_i}):")
        print(f"  user_i = {user_i} (1-based), user_i_idx = {user_i_idx} (0-based)")
        
        # Coefficient +1 at position i-1 (user i in 0-based indexing)
        A[row, user_i_idx] = 1.0
        print(f"  Set A[{row}, {user_i_idx}] = +1.0")
        
        # Coefficients -1 at positions i to M-1 (users i+1 to M in 0-based indexing)
        if user_i < M:  # If there are users after user i
            A[row, user_i:] = -1.0
            print(f"  Set A[{row}, {user_i}:{M}] = -1.0")
            
        # Vector B: Pg / (P · |h_i|²)
        B[row] = Pg / (P * h_gains[user_i_idx])
        print(f"  B[{row}] = Pg / (P * h_gains[{user_i_idx}]) = {Pg} / ({P} * {h_gains[user_i_idx]:.4f}) = {B[row]:.6f}")
        
        print(f"  Final A[{row}] = {A[row]}")
    
    print(f"\nFinal matrix A:")
    print(A)
    print(f"\nFinal vector B:")
    print(B)
    
    # Solve the system
    try:
        alpha_double_prime = np.linalg.solve(A, B)
        print(f"\nSolution α'': {alpha_double_prime}")
        print(f"Sum of α'': {np.sum(alpha_double_prime):.6f}")
        
        # Check for negative values
        if np.any(alpha_double_prime < 0):
            print("⚠️  WARNING: α'' contains negative values - will fallback to α'")
            negative_indices = np.where(alpha_double_prime < 0)[0]
            print(f"   Negative values at indices: {negative_indices}")
            for idx in negative_indices:
                print(f"   α''[{idx}] = {alpha_double_prime[idx]:.6f}")
        else:
            print("✓ All α'' values are non-negative")
            
        # Verify SIC constraints
        print(f"\nVerifying SIC constraints:")
        for i in range(M - 1):
            user_i_idx = i  # 0-based index for user i+1 in 1-based indexing
            
            # Left side: α_i - Σ(α_k, k=i+1..M)
            alpha_i = alpha_double_prime[user_i_idx]
            sum_weaker_users = np.sum(alpha_double_prime[user_i_idx + 1:]) if user_i_idx + 1 < M else 0.0
            lhs = alpha_i - sum_weaker_users
            
            # Right side: Pg / (P · |h_i|²)
            rhs = Pg / (P * h_gains[user_i_idx])
            
            print(f"  User {i+1}: α{i+1} - Σ(α_k, k={i+2}..{M}) = {alpha_i:.6f} - {sum_weaker_users:.6f} = {lhs:.6f}")
            print(f"           Should equal Pg/(P*|h{i+1}|²) = {rhs:.6f}")
            print(f"           Difference: {abs(lhs - rhs):.8f}")
            
    except np.linalg.LinAlgError as e:
        print(f"\n❌ LinAlgError: {e}")
        alpha_double_prime = None
    
    # Final selection
    if alpha_double_prime is None:
        print(f"\nFinal selection: Using α' (equal allocation) due to singular matrix")
        final_alpha = alpha_prime
    elif np.any(alpha_double_prime < 0):
        print(f"\nFinal selection: Using α' (equal allocation) due to negative α''")
        final_alpha = alpha_prime
    else:
        final_alpha = np.minimum(alpha_prime, alpha_double_prime)
        print(f"\nFinal selection: α* = min(α', α'')")
        print(f"  α' = {alpha_prime}")
        print(f"  α'' = {alpha_double_prime}")
        print(f"  α* = {final_alpha}")
    
    print(f"\nFinal power allocation: {final_alpha}")
    print(f"Sum: {np.sum(final_alpha):.6f}")

if __name__ == "__main__":
    debug_sic_calculation()