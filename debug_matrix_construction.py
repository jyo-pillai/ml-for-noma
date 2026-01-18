#!/usr/bin/env python3
"""
Debug the matrix construction in our implementation.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig

def debug_matrix_construction():
    """Debug matrix construction step by step."""
    
    h_gains = np.array([2.0, 1.0, 0.5])
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Debugging matrix construction in our implementation")
    print(f"Channel gains: {h_gains}")
    print()
    
    # Replicate the implementation logic
    M = config.M
    P = config.P
    Pg = config.Pg
    
    A = np.zeros((M, M))
    B = np.zeros(M)
    
    # Row 0: Sum constraint
    A[0, :] = 1.0
    B[0] = 1.0
    print(f"Row 0: A[0] = {A[0]}, B[0] = {B[0]}")
    
    # Rows 1 to M-1: SIC constraints
    for row in range(1, M):
        user_i = row  # User i in 1-based indexing (1 to M-1)
        user_i_idx = user_i - 1  # User i in 0-based indexing (0 to M-2)
        
        print(f"\nRow {row} (user_i = {user_i}, user_i_idx = {user_i_idx}):")
        
        # Coefficient +1 at position i-1 (user i in 0-based indexing)
        A[row, user_i_idx] = 1.0
        print(f"  Set A[{row}, {user_i_idx}] = 1.0")
        
        # Coefficients -1 at positions i to M-1 (users i+1 to M in 0-based indexing)
        if user_i < M:
            A[row, user_i:] = -1.0
            print(f"  Set A[{row}, {user_i}:] = -1.0")
            
        # Vector B
        B[row] = Pg / (P * h_gains[user_i_idx])
        print(f"  B[{row}] = {Pg} / ({P} * {h_gains[user_i_idx]}) = {B[row]}")
        
        print(f"  Final A[{row}] = {A[row]}")
    
    print(f"\nFinal matrix A:")
    print(A)
    print(f"Final vector B: {B}")
    
    # Compare with manual calculation
    print(f"\nExpected matrix A (from manual calculation):")
    A_expected = np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [0.0, 1.0, -1.0]
    ])
    print(A_expected)
    
    B_expected = np.array([1.0, 0.005, 0.01])
    print(f"Expected vector B: {B_expected}")
    
    print(f"\nMatrices match? {np.allclose(A, A_expected)}")
    print(f"Vectors match? {np.allclose(B, B_expected)}")
    
    # Solve both systems
    try:
        alpha_impl = np.linalg.solve(A, B)
        alpha_expected = np.linalg.solve(A_expected, B_expected)
        
        print(f"\nImplementation solution: {alpha_impl}")
        print(f"Expected solution: {alpha_expected}")
        print(f"Solutions match? {np.allclose(alpha_impl, alpha_expected)}")
        
    except np.linalg.LinAlgError as e:
        print(f"Error solving system: {e}")

if __name__ == "__main__":
    debug_matrix_construction()