# Implementation Plan: NOMA Dataset Generator

## Overview

This implementation plan breaks down the NOMA dataset generator into discrete coding tasks. The system will generate 10,000 training samples of channel conditions, optimal power allocations, and sum rates for a downlink NOMA network. Implementation follows a bottom-up approach: configuration → channel generation → power allocation → rate calculation → dataset generation → testing.

## Tasks

- [x] 1. Set up project structure and configuration
  - Create main script file `noma_dataset_generator.py`
  - Define `NOMAConfig` dataclass with parameters: M, P, N0, Pg, N_samples, path_loss_exp
  - Set default values: M=3, P=1.0, N0=0.001, Pg=0.01, N_samples=10000
  - Import required libraries: numpy, pandas, dataclasses
  - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6_

- [x] 2. Implement channel generation
  - [x] 2.1 Write `generate_channels()` function
    - Generate M complex channel coefficients using Rayleigh fading: `h = np.random.randn() + 1j*np.random.randn()`
    - Compute channel gains: `g = np.abs(h)**2`
    - Sort gains in descending order
    - Return sorted gains array
    - Add docstring explaining Rayleigh fading model from paper
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 2.2 Write property test for channel ordering
    - **Property 1: Channel Gain Ordering Invariant**
    - **Validates: Requirements 2.3**
    - Use Hypothesis to generate random M values (2-10)
    - Verify all generated channels are sorted descending
    - Run 100 iterations minimum

- [x] 3. Implement power allocation optimization
  - [x] 3.1 Write `compute_alpha_prime()` function
    - Return array of M elements, each equal to 1/M
    - Add comment: "Algorithm 1 - Equal allocation bound"
    - _Requirements: 3.1_

  - [x] 3.2 Write property test for equal allocation bound
    - **Property 2: Equal Allocation Bound Correctness**
    - **Validates: Requirements 3.1**
    - Test for various M values that all elements equal 1/M

  - [x] 3.3 Write `compute_alpha_double_prime()` function
    - Construct matrix A (M×M) and vector B (M×1)
    - Row 1: Sum constraint [1, 1, ..., 1] · α'' = 1
    - Rows 2 to M: SIC constraints using formula from paper
    - Solve using `np.linalg.solve(A, B)`
    - Wrap in try-except for `LinAlgError`
    - Return None if singular matrix
    - Add detailed comments explaining SIC constraint formula
    - _Requirements: 3.2, 3.3, 3.4, 3.5_

  - [x] 3.4 Write property test for SIC constraint satisfaction
    - **Property 4: SIC Constraint Satisfaction**
    - **Validates: Requirements 3.4**
    - Generate random channels and compute α''
    - Verify SIC constraint equation holds for each user m (2 to M)
    - Handle LinAlgError gracefully

  - [x] 3.5 Write `select_alpha_star()` function
    - Compute element-wise minimum: `np.minimum(alpha_prime, alpha_double_prime)`
    - Return α* array
    - _Requirements: 3.6_

  - [x] 3.6 Write property test for minimum selection
    - **Property 5: Minimum Selection Correctness**
    - **Validates: Requirements 3.6**
    - Verify α* ≤ α' and α* ≤ α'' element-wise

  - [x] 3.7 Write `compute_power_allocation()` orchestrator function
    - Call `compute_alpha_prime(M)`
    - Call `compute_alpha_double_prime(h_gains, config)`
    - If α'' is None, return α'
    - Otherwise call `select_alpha_star(alpha_prime, alpha_double_prime)`
    - Return α*
    - _Requirements: 3.1, 3.2, 3.6_

  - [x] 3.8 Write property test for power allocation feasibility
    - **Property 3: Power Allocation Feasibility**
    - **Validates: Requirements 3.3, 7.2, 7.3**
    - Verify all α* ≥ 0
    - Verify sum(α*) ≤ 1.0

- [x] 4. Checkpoint - Verify power allocation logic
  - Ensure all tests pass, ask the user if questions arise.

- [-] 5. Implement sum rate calculation
  - [x] 5.1 Write `compute_sinr()` function
    - Calculate signal power: `alpha[m] * P * h_gains[m]`
    - Calculate interference: sum of `alpha[i] * P * h_gains[m]` for i=0 to m-1
    - Calculate SINR: signal / (interference + N0)
    - Clip SINR to minimum of 0
    - Add comment: "NOMA SINR formula - interference from users 1 to m-1"
    - _Requirements: 4.1, 7.5_

  - [x] 5.2 Write property test for SINR non-negativity
    - **Property 6: SINR Non-Negativity**
    - **Validates: Requirements 7.5**
    - Generate random samples and verify all SINR ≥ 0

  - [x] 5.3 Write property test for interference term correctness
    - **Property 8: Interference Term Correctness**
    - **Validates: Requirements 4.1**
    - Verify interference only includes users 0 to m-1

  - [x] 5.4 Write `compute_rate()` function
    - Calculate Shannon rate: `np.log2(1 + sinr)`
    - Return rate
    - _Requirements: 4.2_

  - [x] 5.5 Write `compute_sum_rate()` function
    - Loop through all M users
    - Compute SINR for each using `compute_sinr()`
    - Compute rate for each using `compute_rate()`
    - Sum all rates
    - Return sum_rate
    - _Requirements: 4.3_

  - [x] 5.6 Write property test for sum rate computation correctness
    - **Property 7: Sum Rate Computation Correctness**
    - **Validates: Requirements 4.1, 4.2, 4.3**
    - Generate sample, compute sum rate
    - Independently recompute and verify match

- [ ] 6. Implement validation and error handling
  - [x] 6.1 Write `validate_sample()` function
    - Check channel gains sorted descending
    - Check all gains > 0
    - Check all alpha ≥ 0
    - Check sum(alpha) ≤ 1 + 1e-6
    - Check sum_rate ≥ 0
    - Return boolean
    - _Requirements: 7.2, 7.3, 7.4_

  - [x] 6.2 Write unit tests for validation function
    - Test valid sample passes
    - Test unsorted channels fail
    - Test negative alpha fails
    - Test sum > 1 fails

- [-] 7. Implement main dataset generation loop
  - [x] 7.1 Write `generate_dataset()` function
    - Initialize empty lists for storing data
    - Create loop for N_samples iterations
    - Inside loop: generate channels, compute power allocation, compute sum rate
    - Call `validate_sample()` - if invalid, skip and continue
    - Append valid samples to lists
    - Track number of skipped samples
    - After loop, create DataFrame with appropriate column names
    - For M=3: columns = ['h1_gain', 'h2_gain', 'h3_gain', 'alpha1', 'alpha2', 'alpha3', 'Sum_Rate']
    - For M=4: columns = ['h1_gain', 'h2_gain', 'h3_gain', 'h4_gain', 'alpha1', 'alpha2', 'alpha3', 'alpha4', 'Sum_Rate']
    - Save DataFrame to CSV: `df.to_csv('noma_training_data.csv', index=False)`
    - Print summary: samples generated, samples skipped
    - Return DataFrame
    - _Requirements: 1.6, 5.1, 5.2, 5.3, 5.4, 5.5, 7.4_

  - [x] 7.2 Write integration test for dataset generation
    - Generate small dataset (N=100)
    - Verify DataFrame has correct shape
    - Verify all columns present
    - Verify CSV file created

- [ ] 8. Add main execution block and CLI support
  - [x] 8.1 Add `if __name__ == "__main__":` block
    - Create default config for M=3
    - Call `generate_dataset(config)`
    - Print completion message
    - _Requirements: 1.1, 1.6_

  - [x] 8.2 Add support for M=4 via command-line argument
    - Use `argparse` to accept `--users` argument
    - Allow values 3 or 4
    - Create config with specified M
    - _Requirements: 1.2_

  - [x] 8.3 Write unit tests for CLI argument parsing
    - Test default M=3
    - Test M=4 argument
    - Test invalid M value

- [ ] 9. Add documentation and comments
  - [x] 9.1 Add module-level docstring
    - Explain purpose: training dataset for ML-based NOMA power allocation
    - Reference "Sum Rate Maximization Under SIC Constraint" method
    - List key formulas and paper references
    - _Requirements: 6.1_

  - [ ] 9.2 Add inline comments for all formulas
    - Mark each formula with paper reference
    - Explain SIC constraint derivation
    - Explain SINR calculation with interference term
    - _Requirements: 6.1, 6.5_

- [x] 10. Final checkpoint and validation
  - Run all unit tests and property tests
  - Generate full dataset with M=3 (10,000 samples)
  - Generate full dataset with M=4 (10,000 samples)
  - Verify CSV files created successfully
  - Print summary statistics (mean/std of channels, alphas, sum rates)
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Each task references specific requirements for traceability
- Property tests use Hypothesis library with minimum 100 iterations
- The implementation follows Algorithm 1 from the paper for power allocation
- Error handling ensures numerical stability for edge cases (singular matrices, negative values)
- Validation ensures all generated samples are physically realizable
