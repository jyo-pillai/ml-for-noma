"""
Property-Based Tests for NOMA Dataset Generator

These tests verify universal correctness properties across many randomly generated inputs
using the Hypothesis library for property-based testing.

Each test runs a minimum of 100 iterations with different random inputs to ensure
the properties hold for all valid configurations.
"""

import os
import sys
import numpy as np
import pytest
from unittest.mock import patch
from hypothesis import given, strategies as st, settings, assume
from noma_dataset_generator import (
    NOMAConfig, 
    generate_channels, 
    generate_dataset,
    compute_sinr,
    compute_sum_rate,
    validate_sample,
    compute_power_allocation,
    parse_args
)


@settings(max_examples=100)
@given(M=st.integers(min_value=2, max_value=10))
def test_channel_ordering(M):
    """
    Feature: noma-dataset-generator, Property 1: Channel Gain Ordering Invariant
    
    **Validates: Requirements 2.3**
    
    For any generated sample, the channel gains must be sorted in descending order:
    h1_gain >= h2_gain >= ... >= hM_gain.
    
    This property is fundamental to NOMA operation. The SIC decoding order depends
    on users being ordered by signal strength. If this property is violated, the
    entire power allocation and rate calculation becomes invalid.
    """
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    
    # Verify descending order: each element >= next element
    assert np.all(h_gains[:-1] >= h_gains[1:]), \
        f"Channel gains not sorted descending: {h_gains}"
    
    # Verify correct shape
    assert h_gains.shape == (M,), \
        f"Expected shape ({M},), got {h_gains.shape}"
    
    # Verify all gains are positive (channel gains are squared magnitudes)
    assert np.all(h_gains > 0), \
        f"All channel gains must be positive, got: {h_gains}"



@settings(max_examples=100)
@given(M=st.integers(min_value=2, max_value=10))
def test_equal_allocation_bound(M):
    """
    Feature: noma-dataset-generator, Property 2: Equal Allocation Bound Correctness
    
    **Validates: Requirements 3.1**
    
    For any value of M, when computing the equal allocation bound α', all elements
    must equal 1/M exactly.
    
    The equal allocation bound represents the trivial solution where all users get
    equal power. This serves as an upper bound in the optimization.
    """
    from noma_dataset_generator import compute_alpha_prime
    
    alpha_prime = compute_alpha_prime(M)
    
    # Verify all elements equal 1/M
    expected = np.ones(M) / M
    assert np.allclose(alpha_prime, expected), \
        f"Expected all elements to be {1/M}, got {alpha_prime}"
    
    # Verify correct shape
    assert alpha_prime.shape == (M,), \
        f"Expected shape ({M},), got {alpha_prime.shape}"
    
    # Verify sum equals 1
    assert np.isclose(np.sum(alpha_prime), 1.0), \
        f"Sum of alpha_prime should be 1.0, got {np.sum(alpha_prime)}"



@settings(max_examples=100)
@given(M=st.integers(min_value=2, max_value=10))
def test_sic_constraint_satisfaction(M):
    """
    Feature: noma-dataset-generator, Property 4: SIC Constraint Satisfaction
    
    **Validates: Requirements 3.4**
    
    For any generated sample with computed α'', the linear system constraints must hold.
    The implementation solves A·α'' = B where:
    - Row 0: Sum constraint (all alphas sum to 1)
    - Rows 1 to M-1: SIC constraints for each user boundary
    
    For row m (1 to M-1), the constraint is:
    2*P*h[m-1]*sum(alpha[0:m]) + P*h[m-1]*sum(alpha[m:M]) = P*h[m-1] - Pg
    
    This ensures sufficient power difference between users for SIC to work correctly.
    """
    from noma_dataset_generator import compute_alpha_double_prime
    
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    
    try:
        alpha_double_prime = compute_alpha_double_prime(h_gains, config)
        
        # If the matrix was singular, alpha_double_prime will be None
        if alpha_double_prime is None:
            # This is acceptable - skip this sample
            return
        
        # Verify SIC constraint for each row m from 1 to M-1 (0-indexed loop variable)
        # This corresponds to users 2 to M in 1-indexed notation
        for m in range(1, M):
            # Sum of alphas for indices 0 to m-1 (coefficient 2*P*h[m-1])
            sum_prev = np.sum(alpha_double_prime[:m])
            
            # Sum of alphas for indices m to M-1 (coefficient P*h[m-1])
            sum_curr_and_after = np.sum(alpha_double_prime[m:])
            
            # Channel gain h[m-1] (0-indexed)
            h_gain = h_gains[m - 1]
            
            # Left-hand side: 2*P*h*sum_prev + P*h*sum_curr_and_after
            lhs = 2.0 * config.P * h_gain * sum_prev + config.P * h_gain * sum_curr_and_after
            
            # Right-hand side: P*h - Pg
            rhs = config.P * h_gain - config.Pg
            
            # Verify constraint holds within numerical tolerance
            assert np.isclose(lhs, rhs, atol=1e-6), \
                f"SIC constraint violated for row {m}: LHS={lhs}, RHS={rhs}, diff={abs(lhs-rhs)}"
        
    except np.linalg.LinAlgError:
        # Singular matrix - acceptable to skip
        pass



@settings(max_examples=100)
@given(M=st.integers(min_value=2, max_value=10))
def test_minimum_selection_correctness(M):
    """
    Feature: noma-dataset-generator, Property 5: Minimum Selection Correctness
    
    **Validates: Requirements 3.6**
    
    For any generated sample, the final power allocation must satisfy
    α*_m ≤ α'_m AND α*_m ≤ α''_m for all users m.
    
    Algorithm 1 selects the minimum of the two bounds element-wise. This property
    verifies that the selection logic is implemented correctly.
    """
    from noma_dataset_generator import compute_alpha_prime, compute_alpha_double_prime, select_alpha_star
    
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    
    alpha_prime = compute_alpha_prime(M)
    
    try:
        alpha_double_prime = compute_alpha_double_prime(h_gains, config)
        
        # If the matrix was singular, alpha_double_prime will be None
        if alpha_double_prime is None:
            # Skip this sample
            return
        
        alpha_star = select_alpha_star(alpha_prime, alpha_double_prime)
        
        # Verify α* ≤ α' element-wise (with small tolerance for numerical errors)
        assert np.all(alpha_star <= alpha_prime + 1e-9), \
            f"α* should be ≤ α': α*={alpha_star}, α'={alpha_prime}"
        
        # Verify α* ≤ α'' element-wise (with small tolerance for numerical errors)
        assert np.all(alpha_star <= alpha_double_prime + 1e-9), \
            f"α* should be ≤ α'': α*={alpha_star}, α''={alpha_double_prime}"
        
        # Verify α* is the element-wise minimum
        expected_alpha_star = np.minimum(alpha_prime, alpha_double_prime)
        assert np.allclose(alpha_star, expected_alpha_star), \
            f"α* should be min(α', α''): got {alpha_star}, expected {expected_alpha_star}"
        
    except np.linalg.LinAlgError:
        # Singular matrix - acceptable to skip
        pass



@settings(max_examples=100)
@given(M=st.integers(min_value=2, max_value=10))
def test_power_allocation_feasibility(M):
    """
    Feature: noma-dataset-generator, Property 3: Power Allocation Feasibility
    
    **Validates: Requirements 3.3, 7.2, 7.3**
    
    For any generated sample, the final power allocation α* must satisfy:
    - All elements α*_m ≥ 0 (non-negativity)
    - Σ(α*_m) ≤ 1 (total power constraint)
    
    Power allocations must be physically realizable. Negative power is meaningless,
    and the sum cannot exceed the total available power.
    """
    from noma_dataset_generator import compute_power_allocation
    
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    alpha_star = compute_power_allocation(h_gains, config)
    
    # Verify non-negativity: all α* ≥ 0
    assert np.all(alpha_star >= 0), \
        f"All power allocations must be non-negative, got {alpha_star}"
    
    # Verify power constraint: sum(α*) ≤ 1.0
    alpha_sum = np.sum(alpha_star)
    assert alpha_sum <= 1.0 + 1e-6, \
        f"Sum of power allocations must be ≤ 1.0, got {alpha_sum}"
    
    # Verify correct shape
    assert alpha_star.shape == (M,), \
        f"Expected shape ({M},), got {alpha_star.shape}"
    


# --- Strategies for Hypothesis ---

@st.composite
def noma_inputs(draw, min_m=2, max_m=10):
    """Generates valid NOMA inputs: sorted h_gains and normalized alpha."""
    M = draw(st.integers(min_value=min_m, max_value=max_m))
    
    # Generate sorted channel gains (positive)
    h_gains = draw(st.lists(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=M, max_size=M
    ))
    h_gains.sort(reverse=True) # Sort descending
    
    # Generate power allocation (sums to <= 1)
    raw_alpha = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=M, max_size=M
    ))
    
    # Normalize alpha to sum to exactly 1.0 (or less)
    total_raw = sum(raw_alpha)
    if total_raw > 0:
        alpha = [x / total_raw for x in raw_alpha]
    else:
        alpha = [1.0/M] * M # Fallback
        
    return M, np.array(h_gains), np.array(alpha)

# --- Task 5: Property Tests ---

@settings(max_examples=100)
@given(M=st.integers(min_value=2, max_value=10))
def test_sinr_non_negativity(M):
    """
    Feature: noma-dataset-generator, Property 6: SINR Non-Negativity
    
    **Validates: Requirements 7.5**
    
    For any generated sample, all computed SINR values γ_m must be non-negative (γ_m ≥ 0).
    
    SINR is a ratio of signal power to interference-plus-noise power. Since all terms
    are non-negative (power allocations, channel gains, noise), the SINR must also be
    non-negative. Negative SINR indicates a computation error.
    """
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    alpha_star = compute_power_allocation(h_gains, config)
    
    for m in range(M):
        sinr = compute_sinr(m, h_gains, alpha_star, config)
        assert sinr >= -1e-9, \
            f"SINR for user {m} should be non-negative, got {sinr}"

@settings(max_examples=100)
@given(M=st.integers(min_value=3, max_value=10), user_idx=st.integers(min_value=1, max_value=9))
def test_interference_term_correctness(M, user_idx):
    """
    Feature: noma-dataset-generator, Property 8: Interference Term Correctness
    
    **Validates: Requirements 4.1**
    
    For any user m > 0, interference must only include users 0 to m-1.
    
    In downlink NOMA with SIC, user m decodes and removes the signals of all stronger
    users (0 to m-1) before decoding its own signal. Users m+1 to M-1 are treated as
    noise. This property ensures the interference term reflects the correct decoding order.
    """
    # Ensure user_idx is valid for M
    assume(user_idx < M)
    
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    alpha_star = compute_power_allocation(h_gains, config)
    
    # Manually compute expected interference for user_idx
    # Interference only includes users 0 to user_idx-1 (stronger users)
    expected_interference = 0.0
    for i in range(user_idx):
        expected_interference += alpha_star[i] * config.P * h_gains[user_idx]
    
    # Compute SINR using the function
    sinr = compute_sinr(user_idx, h_gains, alpha_star, config)
    
    # Extract actual interference from SINR calculation
    # SINR = signal / (interference + noise)
    # Therefore: interference = signal / SINR - noise
    signal = alpha_star[user_idx] * config.P * h_gains[user_idx]
    
    # Handle edge case where SINR might be 0 (avoid division by zero)
    if sinr > 0:
        denominator = signal / sinr
        actual_interference = denominator - config.N0
        
        assert np.isclose(actual_interference, expected_interference, atol=1e-6), \
            f"Interference mismatch for user {user_idx}: expected {expected_interference}, got {actual_interference}"

@settings(max_examples=100)
@given(inputs=noma_inputs())
def test_sum_rate_computation_correctness(inputs):
    """
    Feature: noma-dataset-generator, Property 7: Sum Rate Computation Correctness
    Validates: Requirements 4.1, 4.2, 4.3
    """
    M, h_gains, alpha = inputs
    config = NOMAConfig(M=M)
    
    # 1. Compute via function
    calculated_sum_rate = compute_sum_rate(h_gains, alpha, config)
    
    # 2. Independently recompute
    manual_sum_rate = 0.0
    for m in range(M):
        # Re-implement logic locally to verify
        sig = alpha[m] * config.P * h_gains[m]
        if m == 0:
            inter = 0.0
        else:
            inter = np.sum(alpha[:m]) * config.P * h_gains[m]
            
        gamma = sig / (inter + config.N0)
        rate = np.log2(1 + gamma)
        manual_sum_rate += rate
        
    assert np.isclose(calculated_sum_rate, manual_sum_rate, atol=1e-6), \
        "Sum rate calculation does not match independent verification"

# --- Task 6: Unit Tests for Validation ---

def test_validate_sample_valid():
    """Test valid sample passes validation."""
    h_gains = np.array([1.0, 0.5, 0.1])
    alpha = np.array([0.2, 0.3, 0.5]) # Sums to 1
    sum_rate = 5.0
    assert validate_sample(h_gains, alpha, sum_rate) is True

def test_validate_sample_unsorted_channels():
    """Test unsorted channels fail validation."""
    h_gains = np.array([0.5, 1.0, 0.1]) # Not descending
    alpha = np.array([0.2, 0.3, 0.5])
    sum_rate = 5.0
    assert validate_sample(h_gains, alpha, sum_rate) is False

def test_validate_sample_negative_alpha():
    """Test negative alpha fails validation."""
    h_gains = np.array([1.0, 0.5, 0.1])
    alpha = np.array([0.2, -0.1, 0.9]) # Negative value
    sum_rate = 5.0
    assert validate_sample(h_gains, alpha, sum_rate) is False

def test_validate_sample_alpha_sum_gt_one():
    """Test sum(alpha) > 1 fails validation."""
    h_gains = np.array([1.0, 0.5, 0.1])
    alpha = np.array([0.5, 0.5, 0.5]) # Sums to 1.5
    sum_rate = 5.0
    assert validate_sample(h_gains, alpha, sum_rate) is False

def test_validate_sample_negative_sum_rate():
    """Test negative sum_rate fails validation."""
    h_gains = np.array([1.0, 0.5, 0.1])
    alpha = np.array([0.2, 0.3, 0.5])
    sum_rate = -1.0
    assert validate_sample(h_gains, alpha, sum_rate) is False




def test_dataset_generation_integration():
    """
    Integration test for dataset generation (Task 7.2).
    
    Verifies that generate_dataset produces a valid DataFrame and CSV file.
    - Generate small dataset (N=100)
    - Verify DataFrame has correct shape
    - Verify all columns present
    - Verify CSV file created
    """
    # 1. Setup small config (N=100 as per task requirement)
    M = 3
    N = 100
    config = NOMAConfig(M=M, N_samples=N)
    
    # Store original CSV filename that generate_dataset creates
    csv_filename = 'noma_training_data.csv'
    
    # 2. Run Generation
    df = generate_dataset(config)
    
    # 3. Verify DataFrame has correct shape
    # Expected columns: M channels + M alphas + 1 Sum_Rate = 2*M + 1
    expected_cols = (2 * M) + 1
    assert df.shape[1] == expected_cols, \
        f"Expected {expected_cols} columns, got {df.shape[1]}"
    # Allow for some skipped samples, but we expect most to be valid
    assert df.shape[0] > 0, "DataFrame should have at least one row"
    assert df.shape[0] <= N, f"DataFrame should have at most {N} rows"
    
    # 4. Verify all columns present
    expected_names = ['h1_gain', 'h2_gain', 'h3_gain', 
                      'alpha1', 'alpha2', 'alpha3', 
                      'Sum_Rate']
    assert list(df.columns) == expected_names, \
        f"Expected columns {expected_names}, got {list(df.columns)}"
    
    # 5. Verify CSV file created by generate_dataset
    assert os.path.exists(csv_filename), \
        f"CSV file '{csv_filename}' should be created by generate_dataset"
    
    # Cleanup
    if os.path.exists(csv_filename):
        os.remove(csv_filename)


def test_dataset_generation_integration_M4():
    """
    Integration test for M=4 dataset generation.
    
    Verifies generation works correctly for 4-user NOMA system.
    """
    config = NOMAConfig(M=4, N_samples=50)
    csv_filename = 'noma_training_data.csv'
    
    df = generate_dataset(config)
    
    # Verify M=4 specific columns exist
    assert 'h4_gain' in df.columns, "h4_gain column should exist for M=4"
    assert 'alpha4' in df.columns, "alpha4 column should exist for M=4"
    
    # Verify correct number of columns: 4 channels + 4 alphas + 1 Sum_Rate = 9
    assert len(df.columns) == 9, f"Expected 9 columns for M=4, got {len(df.columns)}"
    
    # Verify all expected columns
    expected_names = ['h1_gain', 'h2_gain', 'h3_gain', 'h4_gain',
                      'alpha1', 'alpha2', 'alpha3', 'alpha4',
                      'Sum_Rate']
    assert list(df.columns) == expected_names, \
        f"Expected columns {expected_names}, got {list(df.columns)}"
    
    # Verify CSV file created
    assert os.path.exists(csv_filename), \
        f"CSV file '{csv_filename}' should be created"
    
    # Cleanup
    if os.path.exists(csv_filename):
        os.remove(csv_filename)



def test_cli_default_args():
    """Test that default M is 3."""
    test_args = ['prog_name']
    with patch.object(sys, 'argv', test_args):
        args = parse_args()
        assert args.users == 3

def test_cli_custom_args():
    """Test setting M to 4."""
    test_args = ['prog_name', '--users', '4']
    with patch.object(sys, 'argv', test_args):
        args = parse_args()
        assert args.users == 4

def test_cli_invalid_args():
    """Test that invalid M values are rejected."""
    # argparse exits the program on error, so we expect SystemExit
    test_args = ['prog_name', '--users', '5']
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit):
            parse_args()
