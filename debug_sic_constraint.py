#!/usr/bin/env python3
"""
Debug script to understand why SIC constraint is not working.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig, generate_channels, compute_alpha_prime, compute_alpha_double_prime

def debug_sic_constraint():
    """Debug the SIC constraint calculation."""
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Debugging SIC constraint calculation...")
    print(f"Configuration: M={config.M}, P={config.P}, Pg={config.Pg}")
    print()
    
    # Generate a sample
    h_gains = generate_channels(config)
    print(f"Channel gains: {h_gains}")
    
    # Compute α'
    alpha_prime = compute_alpha_prime(config.M)
    print(f"α' (equal allocation): {alpha_prime}")
    
    # Debug α'' calculation step by step
    M = config.M
    P = config.P
    Pg = config.Pg
    
    print(f"\nDebugging α'' calculation:")
    print(f"M={M}, P={P}, Pg={Pg}")
    
    # Matrix A construction
    A = np.zeros((M, M))
    B = np.zeros(M)
    
    # Row 0: Sum constraint
    A[0, :] = 1.0
    B[0] = 1.0
    print(f"Row 0 (sum constraint): A[0] = {A[0]}, B[0] = {B[0]}")
    
    # Rows 1 to M-1: SIC constraints
    for row in range(1, M):
        user_i = row  # User i in 1-based indexing (1 to M-1)
        user_i_idx = user_i - 1  # User i in 0-based indexing (0 to M-2)
        
        print(f"\nRow {row} (SIC constraint for user {user_i}):")
        print(f"  user_i = {user_i}, user_i_idx = {user_i_idx}")
        
        # Coefficient +1 at position i-1 (user i in 0-based indexing)
        A[row, user_i_idx] = 1.0
        print(f"  Set A[{row}, {user_i_idx}] = 1.0")
        
        # Coefficients -1 at positions i to M-1 (users i+1 to M in 0-based indexing)
        if user_i < M:
            A[row, user_i:] = -1.0
            print(f"  Set A[{row}, {user_i}:] = -1.0")
            
        # Vector B: Pg / (P · |h_i|²)
        B[row] = Pg / (P * h_gains[user_i_idx])
        print(f"  B[{row}] = {Pg} / ({P} * {h_gains[user_i_idx]}) = {B[row]}")
        
        print(f"  A[{row}] = {A[row]}")
    
    print(f"\nFinal matrix A:")
    print(A)
    print(f"Final vector B:")
    print(B)
    
    # Solve the system
    try:
        alpha_double_prime = np.linalg.solve(A, B)
        print(f"\nSolved α'': {alpha_double_prime}")
        print(f"Sum of α'': {np.sum(alpha_double_prime)}")
        print(f"Any negative values? {np.any(alpha_double_prime < 0)}")
        
        if np.any(alpha_double_prime < 0):
            print("α'' has negative values, will fallback to α'")
        else:
            # Compute final α*
            alpha_star = np.minimum(alpha_prime, alpha_double_prime)
            print(f"α* = min(α', α''): {alpha_star}")
            
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error: {e}")
        print("Matrix is singular, will fallback to α'")

if __name__ == "__main__":
    debug_sic_constraint()