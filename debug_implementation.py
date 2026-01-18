#!/usr/bin/env python3
"""
Debug the actual implementation by copying and adding debug prints.
"""

import numpy as np
from noma_dataset_generator import NOMAConfig

def debug_compute_alpha_double_prime(h_gains: np.ndarray, config: NOMAConfig) -> np.ndarray:
    """Debug version of compute_alpha_double_prime with prints."""
    M = config.M
    P = config.P
    Pg = config.Pg
    
    print(f"Debug: M={M}, P={P}, Pg={Pg}")
    print(f"Debug: h_gains={h_gains}")
    
    # Matrix A construction (M×M)
    A = np.zeros((M, M))
    B = np.zeros(M)
    
    # Row 0: Sum constraint (Σα = 1)
    A[0, :] = 1.0
    B[0] = 1.0
    print(f"Debug: Row 0 - A[0]={A[0]}, B[0]={B[0]}")
    
    # Rows 1 to M-1: SIC constraints
    for row in range(1, M):
        user_i = row  # User i in 1-based indexing (1 to M-1)
        user_i_idx = user_i - 1  # User i in 0-based indexing (0 to M-2)
        
        print(f"Debug: Row {row} - user_i={user_i}, user_i_idx={user_i_idx}")
        
        # Coefficient +1 at position i-1 (user i in 0-based indexing)
        A[row, user_i_idx] = 1.0
        print(f"Debug: Set A[{row}, {user_i_idx}] = 1.0")
        
        # Coefficients -1 at positions i to M-1 (users i+1 to M in 0-based indexing)
        if user_i < M:  # If there are users after user i
            A[row, user_i:] = -1.0
            print(f"Debug: Set A[{row}, {user_i}:] = -1.0")
            
        # Vector B: Pg / (P · |h_i|²)
        B[row] = Pg / (P * h_gains[user_i_idx])
        print(f"Debug: B[{row}] = {Pg} / ({P} * {h_gains[user_i_idx]}) = {B[row]}")
        print(f"Debug: A[{row}] = {A[row]}")

    print(f"Debug: Final A=\n{A}")
    print(f"Debug: Final B={B}")

    # Solve linear system A · α'' = B
    try:
        alpha_double_prime = np.linalg.solve(A, B)
        print(f"Debug: Solved alpha_double_prime = {alpha_double_prime}")
        
        # Safety check: if any α'' < 0, fallback to α'
        if np.any(alpha_double_prime < 0):
            print(f"Debug: Found negative values, returning None")
            return None
            
        return alpha_double_prime
        
    except np.linalg.LinAlgError as e:
        print(f"Debug: LinAlgError: {e}")
        return None

def test_debug():
    """Test the debug version."""
    h_gains = np.array([2.0, 1.0, 0.5])
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Testing debug version:")
    result = debug_compute_alpha_double_prime(h_gains, config)
    print(f"Final result: {result}")

if __name__ == "__main__":
    test_debug()