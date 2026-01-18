#!/usr/bin/env python3
"""
Corrected SIC constraint implementation.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class NOMAConfig:
    M: int = 3
    P: float = 1.0
    N0: float = 0.001
    Pg: float = 0.01
    N_samples: int = 10000
    path_loss_exp: float = 2.0

def compute_alpha_double_prime_corrected(h_gains: np.ndarray, config: NOMAConfig) -> np.ndarray:
    """
    CORRECTED implementation of SIC constraint bound.
    """
    print(f"CORRECTED FUNCTION CALLED WITH h_gains={h_gains}, config.Pg={config.Pg}")
    
    M = config.M
    P = config.P
    Pg = config.Pg
    
    # Matrix A construction (M×M)
    A = np.zeros((M, M))
    B = np.zeros(M)
    
    # Row 0: Sum constraint (Σα = 1)
    A[0, :] = 1.0
    B[0] = 1.0
    
    # Rows 1 to M-1: SIC constraints
    for row in range(1, M):
        user_i = row  # User i in 1-based indexing (1 to M-1)
        user_i_idx = user_i - 1  # User i in 0-based indexing (0 to M-2)
        
        # Coefficient +1 at position i-1 (user i in 0-based indexing)
        A[row, user_i_idx] = 1.0
        
        # Coefficients -1 at positions i to M-1 (users i+1 to M in 0-based indexing)
        if user_i < M:  # If there are users after user i
            A[row, user_i:] = -1.0
            
        # Vector B: Pg / (P · |h_i|²)
        B[row] = Pg / (P * h_gains[user_i_idx])

    print(f"CORRECTED: A=\n{A}")
    print(f"CORRECTED: B={B}")

    # Solve linear system A · α'' = B
    try:
        alpha_double_prime = np.linalg.solve(A, B)
        print(f"CORRECTED: Solved alpha_double_prime = {alpha_double_prime}")
        
        # Safety check: if any α'' < 0, fallback to α'
        if np.any(alpha_double_prime < 0):
            print(f"CORRECTED: Found negative values, returning None")
            return None
            
        return alpha_double_prime
        
    except np.linalg.LinAlgError:
        print(f"CORRECTED: LinAlgError")
        return None

def test_corrected():
    """Test the corrected function."""
    h_gains = np.array([2.0, 1.0, 0.5])
    config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
    
    print("Testing corrected implementation:")
    result = compute_alpha_double_prime_corrected(h_gains, config)
    print(f"Result: {result}")
    print(f"Any negative: {np.any(result < 0) if result is not None else 'None'}")

if __name__ == "__main__":
    test_corrected()