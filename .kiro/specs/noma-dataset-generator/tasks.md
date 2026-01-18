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

- [x] 5. Implement sum rate calculation
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

- [x] 6. Implement validation and error handling
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

- [x] 7. Implement main dataset generation loop
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

- [x] 8. Add main execution block and CLI support
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

- [x] 9. Add documentation and comments
  - [x] 9.1 Add module-level docstring
    - Explain purpose: training dataset for ML-based NOMA power allocation
    - Reference "Sum Rate Maximization Under SIC Constraint" method
    - List key formulas and paper references
    - _Requirements: 6.1_

  - [x] 9.2 Add inline comments for all formulas
    - Mark each formula with paper reference
    - Explain SIC constraint derivation
    - Explain SINR calculation with interference term
    - _Requirements: 6.1, 6.5_

- [x] 10. Initial checkpoint and validation
  - Run all unit tests and property tests
  - Generate full dataset with M=3 (10,000 samples)
  - Generate full dataset with M=4 (10,000 samples)
  - Verify CSV files created successfully

---

## Phase 2: Critical Algorithm Correction

- [ ] 11. Fix SIC constraint formula (CRITICAL)
  - [x] 11.1 Re-implement `compute_alpha_double_prime()` with correct SIC formula
    - **Problem:** Current formula produces α ≈ 1/M for all users (incorrect)
    - **Correct SIC constraint:** `α_i - Σ(α_k, k=i+1..M) = Pg / (P · |h_i|²)` for i=1 to M-1
    - Construct matrix A dynamically for any M:
      - Row 0: [1, 1, ..., 1] for sum constraint (Σα = 1)
      - Row i (1 to M-1): coefficient +1 at position i-1, coefficients -1 at positions i to M-1
    - Vector B: [1, Pg/(P·|h1|²), Pg/(P·|h2|²), ..., Pg/(P·|h_{M-1}|²)]
    - Add safety check: if any α'' < 0, fallback to α'
    - _Requirements: 3.2, 3.3, 3.4_

  - [x] 11.2 Update property test for SIC constraint
    - **Property 4 (Updated): SIC Constraint Satisfaction**
    - Verify equation: `α_i - Σ(α_k, k=i+1..M) ≈ Pg/(P·|h_i|²)` within tolerance
    - Test for M=3 and M=4
    - _Requirements: 3.4_

  - [x] 11.3 Regenerate datasets with corrected algorithm
    - Regenerate M=3 dataset (10,000 samples)
    - Regenerate M=4 dataset (10,000 samples)
    - Verify α values vary (not all equal to 1/M)
    - Expected: α3 (weakest user) > α2 > α1 (strongest user)
    - _Requirements: 5.5, 7.4_

- [x] 12. Checkpoint - Verify corrected algorithm
  - Run all property tests with updated SIC formula
  - Verify power allocations match expected pattern (weaker users get more power)
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 3: Verification Plots (Monte Carlo Simulation)

- [ ] 13. Create verification plots script
  - [x] 13.1 Create `verify_noma_plots.py` script
    - Import corrected power allocation functions
    - Set up matplotlib for 3 subplots
    - _Requirements: 6.1_

  - [x] 13.2 Implement Plot 1: Sum Rate vs. Transmit Power (Fig 3)
    - X-axis: P from 0.1 to 3.0 Watts
    - Y-axis: Average Sum Rate (bps/Hz)
    - Monte Carlo: 1000 channel realizations per P value
    - Series: M=3 (blue) and M=4 (red)
    - Parameters: Pg=0.01W, N0=0.001W
    - Expected: Logarithmic growth, ~8-9 bps/Hz at P=1W for M=3
    - _Requirements: 4.3_

  - [x] 13.3 Implement Plot 2: PA Coefficients vs. Power Gap (Fig 5)
    - X-axis: Pg from 0 to 0.04 Watts
    - Y-axis: Average α values
    - Monte Carlo: 1000 channel realizations per Pg value
    - Parameters: P=1W, M=3
    - Series: α1 (User 1), α2 (User 2), α3 (User 3)
    - Expected: α3 (weakest) stays high/flat, α1 & α2 decrease with Pg
    - _Requirements: 3.4, 3.6_

  - [ ] 13.4 Implement Plot 3: Sum Rate vs. Power Gap (Fig 6)
    - X-axis: Pg from 0 to 0.04 Watts
    - Y-axis: Average Sum Rate (bps/Hz)
    - Monte Carlo: 1000 channel realizations per Pg value
    - Parameters: P=1W, M=3, M=4
    - Expected: Sum Rate decreases as Pg increases
    - _Requirements: 4.3_

  - [ ] 13.5 Save plots and validate against paper
    - Save as `noma_verification_plots.png`
    - Compare with paper's Fig 3, 5, 6
    - Validate against paper's Table II (P=1W, Pg=0.01W, M=3):
      - Expected Sum Rate: ~8.68 bps/Hz
      - Expected α: u1~0.18, u2~0.21, u3~0.33
    - Document any discrepancies
    - _Requirements: 4.3, 3.6_

- [ ] 14. Checkpoint - Verify plots match paper
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 4: ML Model Training Pipeline

- [ ] 15. Create ML training script
  - [ ] 15.1 Create `train_noma_model.py` script
    - Load CSV dataset (corrected version)
    - Split into train/validation/test (70/15/15)
    - Normalize channel gains (optional)
    - _Requirements: 5.1, 5.2_

  - [ ] 15.2 Define neural network architecture
    - Input layer: M neurons (channel gains)
    - Hidden layer 1: 64 neurons, ReLU activation
    - Hidden layer 2: 32 neurons, ReLU activation
    - Output layer: M neurons, Softmax (ensures Σα = 1)
    - Framework: TensorFlow/Keras or PyTorch
    - _Requirements: 5.3_

  - [ ] 15.3 Implement training loop
    - Loss function: MSE between predicted and optimal α
    - Optimizer: Adam (lr=0.001)
    - Epochs: 200
    - Batch size: 32
    - Early stopping: patience=20
    - _Requirements: 5.3_

  - [ ] 15.4 Evaluate model performance
    - Compute MSE on test set
    - Compare Sum Rate: ML-predicted α vs. optimal α
    - Measure inference time (ms per sample)
    - _Requirements: 5.4_

  - [ ] 15.5 Generate ML insight visualizations
    - Plot: Predicted vs. Actual α values (scatter plot)
    - Plot: Sum Rate distribution (ML vs. Optimal)
    - Plot: Training/Validation loss curves
    - Plot: Feature importance (which h_i matters most)
    - _Requirements: 5.4_

  - [ ] 15.6 Document findings
    - How close does ML get to optimal?
    - Speed comparison: ML inference vs. matrix solve
    - Recommendations for deployment
    - _Requirements: 6.1_

- [ ] 16. Final checkpoint
  - Ensure all tests pass
  - Verify ML model achieves acceptable accuracy
  - Document complete pipeline from data generation to ML training

---

## Notes

- Tasks 1-10 are completed (initial implementation)
- Task 11 is CRITICAL - the current SIC formula produces incorrect results (all α ≈ 1/M)
- After fixing Task 11, expect varying α values based on channel conditions
- The weakest user (User M) should receive the MOST power to compensate for poor channel conditions
- Verification plots (Task 13) are essential to confirm algorithm matches paper results
- ML model (Tasks 15-16) should learn: given h → predict optimal α without solving linear system

## Expected Results After Correction

### Power Allocation (P=1W, Pg=0.01W, M=3)
Based on paper's Table II:
- α1 (strongest user): ~0.18
- α2 (middle user): ~0.21  
- α3 (weakest user): ~0.33
- Sum Rate: ~8.68 bps/Hz

### Key Insight
The weakest user (User M) should receive the MOST power to compensate for poor channel conditions. This is the fundamental principle of NOMA - allocate more power to weaker users while using SIC to decode stronger users first.
