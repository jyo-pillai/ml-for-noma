# NOMA Verification & ML Training Task List

## Overview

This document separates completed work from remaining tasks, focusing on:
1. Fixing the critical SIC constraint formula
2. Creating verification plots to cross-check with the source paper
3. Building an ML training pipeline

---

## ✅ COMPLETED TASKS

### Core Infrastructure
- [x] Project structure with `noma_dataset_generator.py`
- [x] `NOMAConfig` dataclass (M, P, N0, Pg, N_samples, path_loss_exp)
- [x] CLI support via argparse (--users 3 or 4)

### Channel Generation
- [x] `generate_channels()` - Rayleigh fading model
- [x] Channel gains sorted descending (|h1|² ≥ |h2|² ≥ ... ≥ |hM|²)
- [x] Property test: Channel ordering invariant

### Power Allocation (Partial)
- [x] `compute_alpha_prime()` - Equal allocation bound (α' = 1/M)
- [x] `select_alpha_star()` - Element-wise minimum
- [x] `compute_power_allocation()` - Orchestrator function
- [x] Property tests: Equal allocation, minimum selection, feasibility

### Rate Calculation
- [x] `compute_sinr()` - SINR formula with interference
- [x] `compute_rate()` - Shannon capacity
- [x] `compute_sum_rate()` - Total system rate
- [x] Property tests: SINR non-negativity, interference correctness, sum rate

### Dataset Generation
- [x] `validate_sample()` - Physical constraint checks
- [x] `generate_dataset()` - Main loop with CSV export
- [x] Integration tests for M=3 and M=4

### Generated Data Files
- [x] `noma_training_data_M3.csv` (10,000 samples)
- [x] `noma_training_data_M4.csv` (10,000 samples)

---

## 🔴 CRITICAL FIX REQUIRED

### Task 1: Fix SIC Constraint Formula in `compute_alpha_double_prime()`

**Problem:** The current implementation uses an incorrect matrix formulation.

**Correct Mathematical Formulation:**

The SIC constraint ensures User i's signal power exceeds the sum of weaker users by the power gap:

```
α_i - Σ(α_k, k=i+1 to M) = Pg / (P · |h_i|²)   for i = 1 to M-1
```

Plus the sum constraint:
```
Σ(α_i, i=1 to M) = 1
```

**Matrix System Ax = B:**

For M=3:
```
A = | 1   1   1  |    B = |    1           |
    | 1  -1  -1  |        | Pg/(P·|h1|²)   |
    | 0   1  -1  |        | Pg/(P·|h2|²)   |
```

For M=4:
```
A = | 1   1   1   1  |    B = |    1           |
    | 1  -1  -1  -1  |        | Pg/(P·|h1|²)   |
    | 0   1  -1  -1  |        | Pg/(P·|h2|²)   |
    | 0   0   1  -1  |        | Pg/(P·|h3|²)   |
```

**General Pattern:**
- Row 0: All 1s (sum constraint)
- Row i (1 to M-1): Position i-1 has +1, positions i to M-1 have -1

---

## 🆕 NEW TASKS

### Phase 1: Algorithm Correction

- [ ] 1.1 Re-implement `compute_alpha_double_prime()` with correct SIC formula
  - Construct matrix A dynamically for any M
  - Row 0: [1, 1, ..., 1] for sum constraint
  - Row i: coefficient +1 at position i-1, coefficients -1 at positions i to M-1
  - Vector B: [1, Pg/(P·|h1|²), Pg/(P·|h2|²), ..., Pg/(P·|h_{M-1}|²)]
  - Add safety check: if any α'' < 0, fallback to α'
  - _Validates: Requirements 3.2, 3.3, 3.4_

- [ ] 1.2 Update property test for SIC constraint
  - Verify the equation: α_i - Σ(α_k, k=i+1..M) = Pg/(P·|h_i|²)
  - Test for M=3 and M=4

- [ ] 1.3 Regenerate datasets with corrected algorithm
  - Generate M=3 dataset (10,000 samples)
  - Generate M=4 dataset (10,000 samples)
  - Verify α values vary (not all equal to 1/M)

---

### Phase 2: Verification Plots (Monte Carlo Simulation)

- [ ] 2.1 Create `verify_noma_plots.py` script
  - Import corrected power allocation functions
  - Set up matplotlib for 3 subplots

- [ ] 2.2 Implement Plot 1: Sum Rate vs. Transmit Power (Fig 3)
  - X-axis: P from 0.1 to 3.0 Watts (avoid P=0)
  - Y-axis: Average Sum Rate (bps/Hz)
  - Monte Carlo: 1000 channel realizations per P value
  - Series: M=3 (blue) and M=4 (red)
  - Expected: Logarithmic growth, ~8-9 bps/Hz at P=1W for M=3
  - Parameters: Pg=0.01W, N0=0.001W

- [ ] 2.3 Implement Plot 2: PA Coefficients vs. Power Gap (Fig 5)
  - X-axis: Pg from 0 to 0.04 Watts
  - Y-axis: Average α values
  - Monte Carlo: 1000 channel realizations per Pg value
  - Parameters: P=1W, M=3
  - Series: α1 (User 1), α2 (User 2), α3 (User 3)
  - Expected: α3 (weakest) stays high/flat, α1 & α2 decrease with Pg

- [ ] 2.4 Implement Plot 3: Sum Rate vs. Power Gap (Fig 6)
  - X-axis: Pg from 0 to 0.04 Watts
  - Y-axis: Average Sum Rate (bps/Hz)
  - Monte Carlo: 1000 channel realizations per Pg value
  - Parameters: P=1W, M=3
  - Expected: Sum Rate decreases as Pg increases

- [ ] 2.5 Save plots and validate against paper
  - Save as `noma_verification_plots.png`
  - Compare with paper's Fig 3, 5, 6
  - Document any discrepancies

---

### Phase 3: ML Model Training Pipeline

- [ ] 3.1 Create `train_noma_model.py` script
  - Load CSV dataset
  - Split into train/validation/test (70/15/15)
  - Normalize channel gains (optional)

- [ ] 3.2 Define neural network architecture
  - Input layer: M neurons (channel gains)
  - Hidden layer 1: 64 neurons, ReLU
  - Hidden layer 2: 32 neurons, ReLU
  - Output layer: M neurons, Softmax (ensures Σα = 1)
  - Framework: TensorFlow/Keras or PyTorch

- [ ] 3.3 Implement training loop
  - Loss function: MSE between predicted and optimal α
  - Optimizer: Adam (lr=0.001)
  - Epochs: 200
  - Batch size: 32
  - Early stopping: patience=20

- [ ] 3.4 Evaluate model performance
  - Compute MSE on test set
  - Compare Sum Rate: ML-predicted α vs. optimal α
  - Measure inference time (ms per sample)

- [ ] 3.5 Generate ML insight visualizations
  - Plot: Predicted vs. Actual α (scatter plot)
  - Plot: Sum Rate distribution (ML vs. Optimal)
  - Plot: Training/Validation loss curves
  - Plot: Feature importance (which h_i matters most)

- [ ] 3.6 Document findings
  - How close does ML get to optimal?
  - Speed comparison: ML inference vs. matrix solve
  - Recommendations for deployment

---

## Execution Priority

1. **Task 1.1** - Fix SIC formula (CRITICAL - blocks verification)
2. **Task 1.2** - Update property test
3. **Task 1.3** - Regenerate datasets
4. **Task 2.1-2.5** - Verification plots (validates correctness)
5. **Task 3.1-3.6** - ML training pipeline

---

## Expected Results After Correction

### Power Allocation (P=1W, Pg=0.01W, M=3)
Based on paper's Table II:
- α1 (strongest user): ~0.18
- α2 (middle user): ~0.21  
- α3 (weakest user): ~0.33
- Sum Rate: ~8.68 bps/Hz

### Key Insight
The weakest user (User M) should receive the MOST power to compensate for poor channel conditions. This is the fundamental principle of NOMA - allocate more power to weaker users while using SIC to decode stronger users first.

---

## Notes

- Current datasets show α ≈ 1/M for all users (incorrect - SIC formula bug)
- After fix, expect varying α based on channel conditions
- ML model should learn: given h → predict optimal α without solving linear system
- Verification plots are essential to confirm algorithm matches paper results
