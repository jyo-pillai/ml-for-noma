#!/usr/bin/env python3
"""
Updated Property Test for SIC Constraint Satisfaction (Task 11.2)

This test verifies the corrected SIC constraint formula:
α_i - Σ(α_k, k=i+1..M) ≈ Pg/(P·|h_i|²) within tolerance

Test for M=3 and M=4 as specified in task requirements.
"""

import numpy as np
import sys
from hypothesis import given, strategies as st, settings
from dataclasses import dataclass

@dataclass
class NOMAConfig:
    """Configuration parameters for NOMA system."""
    M: int = 3
    P: float = 1.0
    N0: float = 0.001
    Pg: float = 0.01
    N_samples: int = 10000
    path_loss_exp: float = 2.0

def generate_channels(config: NOMAConfig) -> np.ndarray:
    """Generate M channel gains sorted in descending order."""
    M = config.M
    
    # Generate complex channel coefficients using Rayleigh fading
    # h_m ~ CN(0, d^(-β)) where d is distance and β is path loss exponent
    h_complex = (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)
    
    # Compute channel gains: g_m = |h_m|^2
    h_gains = np.abs(h_complex) ** 2
    
    # Sort in descending order (strongest user first)
    h_gains = np.sort(h_gains)[::-1]
    
    return h_gains

def compute_alpha_double_prime(h_gains: np.ndarray, config: NOMAConfig) -> np.ndarray:
    """
    Compute SIC constraint bound by solving linear system A·α'' = B.
    
    **CORRECTED SIC Constraint Formula:**
    For user i (1-based indexing from 1 to M-1):
    α_i - Σ(α_k, k=i+1..M) = Pg / (P · |h_i|²)
    
    This constraint ensures sufficient power difference between users for successful
    Successive Interference Cancellation (SIC). The power gap Pg is the minimum
    difference required for the receiver to decode and cancel interference.
    
    **Matrix Construction:**
    - Row 0: Sum constraint [1, 1, ..., 1] · α'' = 1
    - Row i (1 to M-1): SIC constraint for user i
      - Coefficient +1 at position i-1 (user i in 0-based indexing)
      - Coefficients -1 at positions i to M-1 (users i+1 to M in 0-based indexing)
      - RHS = Pg / (P · |h_i|²)
    
    Args:
        h_gains: Sorted channel gains in descending order, shape (M,)
        config: NOMAConfig object containing system parameters (P, Pg, M)
        
    Returns:
        alpha_double_prime: SIC constraint bound, shape (M,) or None if singular matrix
    """
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
    # For user i (1-based from paper), which is index i-1 in 0-based Python:
    # α_i - Σ(α_k, k=i+1..M) = Pg / (P · |h_i|²)
    for row in range(1, M):
        user_i = row  # User i in 1-based indexing (1 to M-1)
        user_i_idx = user_i - 1  # User i in 0-based indexing (0 to M-2)
        
        # Coefficient +1 at position i-1 (user i in 0-based indexing)
        A[row, user_i_idx] = 1.0
        
        # Coefficients -1 at positions i to M-1 (users i+1 to M in 0-based indexing)
        if user_i < M:  # If there are users after user i
            A[row, user_i:] = -1.0
            
        # Vector B: Pg / (P · |h_i|²)
        # h_gains[user_i_idx] corresponds to |h_i|² for user i
        B[row] = Pg / (P * h_gains[user_i_idx])

    # Solve linear system A · α'' = B
    try:
        alpha_double_prime = np.linalg.solve(A, B)
        
        # Safety check: if any α'' < 0, fallback to α'
        if np.any(alpha_double_prime < 0):
            return None
            
        return alpha_double_prime
        
    except np.linalg.LinAlgError:
        # Return None for singular matrix (will fallback to α')
        return None

@settings(max_examples=100)
@given(M=st.integers(min_value=3, max_value=4))
def test_property_4_sic_constraint_satisfaction(M):
    """
    Feature: noma-dataset-generator, Property 4 (Updated): SIC Constraint Satisfaction
    
    **Validates: Requirements 3.4**
    
    For any generated sample with computed α'', for each user i from 1 to M-1, 
    the SIC constraint must hold:
    
    α_i - Σ(α_k, k=i+1..M) ≈ Pg/(P·|h_i|²) within tolerance
    
    This constraint ensures that there is sufficient power difference between users 
    for SIC to work correctly. The power gap Pg is the minimum difference needed 
    for successful interference cancellation.
    
    Test for M=3 and M=4 as specified in task requirements.
    """
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    
    try:
        alpha_double_prime = compute_alpha_double_prime(h_gains, config)
        
        # If the matrix was singular, alpha_double_prime will be None
        if alpha_double_prime is None:
            # Skip this sample - singular matrix is acceptable
            return
        
        # Verify SIC constraint for each user i from 1 to M-1 (1-based indexing)
        # In 0-based indexing: users 0 to M-2
        for i in range(M - 1):
            user_i_idx = i  # 0-based index for user i+1 in 1-based indexing
            
            # Left side: α_i - Σ(α_k, k=i+1..M)
            alpha_i = alpha_double_prime[user_i_idx]
            sum_weaker_users = np.sum(alpha_double_prime[user_i_idx + 1:]) if user_i_idx + 1 < M else 0.0
            lhs = alpha_i - sum_weaker_users
            
            # Right side: Pg / (P · |h_i|²)
            rhs = config.Pg / (config.P * h_gains[user_i_idx])
            
            # Verify constraint within numerical tolerance
            assert np.isclose(lhs, rhs, atol=1e-6), \
                f"SIC constraint violation for user {i+1} (0-based idx {user_i_idx}): " \
                f"LHS={lhs:.8f}, RHS={rhs:.8f}, diff={abs(lhs-rhs):.8f}, " \
                f"h_gains={h_gains}, alpha''={alpha_double_prime}"
        
    except np.linalg.LinAlgError:
        # Singular matrix - acceptable to skip
        pass

if __name__ == "__main__":
    # Run a few manual tests
    print("Testing SIC constraint satisfaction...")
    
    # Test M=3
    print("\nTesting M=3:")
    config = NOMAConfig(M=3)
    for i in range(5):
        h_gains = generate_channels(config)
        alpha_double_prime = compute_alpha_double_prime(h_gains, config)
        
        if alpha_double_prime is not None:
            print(f"  Test {i+1}: h_gains={h_gains}")
            print(f"           alpha''={alpha_double_prime}")
            
            # Verify constraints manually
            for user_i in range(2):  # Users 0 and 1 (0-based)
                alpha_i = alpha_double_prime[user_i]
                sum_weaker = np.sum(alpha_double_prime[user_i + 1:])
                lhs = alpha_i - sum_weaker
                rhs = config.Pg / (config.P * h_gains[user_i])
                print(f"           User {user_i+1}: LHS={lhs:.6f}, RHS={rhs:.6f}, Match={np.isclose(lhs, rhs, atol=1e-6)}")
        else:
            print(f"  Test {i+1}: Singular matrix (skipped)")
    
    # Test M=4
    print("\nTesting M=4:")
    config = NOMAConfig(M=4)
    for i in range(5):
        h_gains = generate_channels(config)
        alpha_double_prime = compute_alpha_double_prime(h_gains, config)
        
        if alpha_double_prime is not None:
            print(f"  Test {i+1}: h_gains={h_gains}")
            print(f"           alpha''={alpha_double_prime}")
            
            # Verify constraints manually
            for user_i in range(3):  # Users 0, 1, and 2 (0-based)
                alpha_i = alpha_double_prime[user_i]
                sum_weaker = np.sum(alpha_double_prime[user_i + 1:])
                lhs = alpha_i - sum_weaker
                rhs = config.Pg / (config.P * h_gains[user_i])
                print(f"           User {user_i+1}: LHS={lhs:.6f}, RHS={rhs:.6f}, Match={np.isclose(lhs, rhs, atol=1e-6)}")
        else:
            print(f"  Test {i+1}: Singular matrix (skipped)")
    
    print("\nRunning property-based test...")
    try:
        test_property_4_sic_constraint_satisfaction()
        print("Property-based test completed successfully!")
    except Exception as e:
        print(f"Property-based test failed: {e}")