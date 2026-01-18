# Extended Task List: NOMA Dataset Generator Verification & ML Training

## Status Summary

### ✅ Already Completed
1. Project structure and configuration (`NOMAConfig` dataclass)
2. Channel generation with Rayleigh fading (sorted descending)
3. Equal allocation bound α' = 1/M
4. SINR and Sum Rate calculation
5. Dataset generation loop with validation
6. CSV export functionality
7. CLI support for M=3 and M=4
8. Basic property tests (8 tests)
9. Unit tests for validation

### ⚠️ CRITICAL FIX REQUIRED
1. **SIC Constraint Formula (α'') is INCORRECT**
   - Current: Uses wrong matrix formulation
   - Required: `α_i - Σ(α_k, k=i+1..M) = Pg / (P · |h_i|²)` for i=1 to M-1

### 🆕 New Tasks Required

---

## Phase 1: Fix Core Algorithm

### Task 1.1: Re-implement `compute_alpha_double_prime()` with CORRECT SIC formula
**Priority: CRITICAL**

The correct SIC constraint from the paper:
- For user i (1 to M-1): `α_i - Σ(α_k, k=i+1..M) = Pg / (P · |h_i|²)`
- Plus sum constraint: `Σ(α_i) = 1`

Matrix formulation:
- Row 0: [1, 1, 1, ..., 1] · α = 1 (sum constraint)
- Row i (for i=1 to M-1): α_i - α_{i+1} - α_{i+2} - ... - α_M = Pg/(P·|h_i|²)

**Acceptance Criteria:**
- Matrix A correctly encodes SIC constraints
- Vector B correctly encodes RHS values
- Fallback to α' if α'' has negative values

### Task 1.2: Update property test for SIC constraint
**Priority: HIGH**

Update `test_sic_constraint_satisfaction` to verify the CORRECT formula.

### Task 1.3: Regenerate datasets with corrected algorithm
**Priority: HIGH**

- Regenerate M=3 dataset (10,000 samples)
- Regenerate M=4 dataset (10,000 samples)
- Verify power allocations are no longer all equal (1/M)

---

## Phase 2: Verification Plots (Monte Carlo Simulation)

### Task 2.1: Create `verify_noma_plots.py` with corrected algorithm
**Priority: HIGH**

Generate 3 plots using Monte Carlo simulations (≥1000 channel realizations):

#### Plot 1: Sum Rate vs. Transmit Power (Fig 3)
- X-axis: P from 0 to 3 Watts
- Y-axis: Sum Rate (bps/Hz)
- Series: M=3 and M=4
- Expected: Logarithmic growth, ~8-9 bps/Hz at P=1W, M=3

#### Plot 2: PA Coefficients vs. Power Gap (Fig 5)
- X-axis: Pg from 0 to 0.04 Watts
- Y-axis: α values
- Parameters: P=1W, M=3
- Expected: User 3 (weakest) high/flat, Users 1&2 decrease with Pg

#### Plot 3: Sum Rate vs. Power Gap (Fig 6)
- X-axis: Pg from 0 to 0.04 Watts
- Y-axis: Sum Rate
- Parameters: P=1W, M=3
- Expected: Sum Rate decreases as Pg increases

### Task 2.2: Validate against paper's Table II
**Priority: MEDIUM**

For P=1W, Pg=0.01W, M=3:
- Expected Sum Rate: ~8.68 bps/Hz
- Expected α: u1~0.18, u2~0.21, u3~0.33

---

## Phase 3: ML Model Training Pipeline

### Task 3.1: Create ML training script `train_noma_model.py`
**Priority: MEDIUM**

**Objective:** Train a neural network to predict optimal power allocation from channel gains.

**Architecture:**
- Input: M channel gains [h1, h2, ..., hM]
- Output: M power allocation coefficients [α1, α2, ..., αM]
- Hidden layers: 2-3 dense layers with ReLU activation
- Output activation: Softmax (ensures sum = 1)

**Training:**
- Loss: MSE between predicted and optimal α
- Optimizer: Adam
- Validation split: 20%
- Epochs: 100-200

### Task 3.2: Model evaluation and insights
**Priority: MEDIUM**

**Metrics:**
- MSE on test set
- Sum Rate comparison: ML-predicted α vs. optimal α
- Inference time comparison

**Insights to extract:**
1. How well does ML approximate the optimization?
2. What's the trade-off between accuracy and speed?
3. Feature importance: Which channel gains matter most?

### Task 3.3: Create visualization for ML results
**Priority: LOW**

- Plot: Predicted vs. Actual α values
- Plot: Sum Rate distribution (ML vs. Optimal)
- Plot: Learning curves (loss vs. epochs)

---

## Phase 4: Documentation & Cleanup

### Task 4.1: Add inline comments for corrected formulas
**Priority: LOW**

Document the correct SIC constraint derivation in code comments.

### Task 4.2: Update design.md with corrected formulas
**Priority: LOW**

Ensure design document reflects the correct mathematical formulation.

---

## Execution Order

1. **Task 1.1** - Fix SIC formula (CRITICAL - blocks everything)
2. **Task 1.2** - Update property test
3. **Task 1.3** - Regenerate datasets
4. **Task 2.1** - Generate verification plots
5. **Task 2.2** - Validate against paper
6. **Task 3.1** - ML training script
7. **Task 3.2** - Model evaluation
8. **Task 3.3** - ML visualizations
9. **Task 4.1-4.2** - Documentation

---

## Notes

- The current implementation always returns α' = 1/M because the incorrect SIC formula produces invalid results
- After fixing, expect to see varying α values based on channel conditions
- ML model should learn the mapping: h → α* without solving the linear system
