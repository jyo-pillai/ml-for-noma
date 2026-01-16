# Design Document: NOMA Dataset Generator

## Overview

The NOMA Dataset Generator is a Python-based system that produces training data for Machine Learning models optimizing power allocation in Downlink NOMA networks. The system implements the "Sum Rate Maximization Under SIC Constraint" algorithm to generate 10,000 samples of channel conditions, optimal power allocations, and corresponding sum rates.

The core workflow follows these steps:
1. Generate random channel coefficients using Rayleigh fading
2. Sort users by channel gain (strongest to weakest)
3. Compute optimal power allocation using a dual-bound optimization (Algorithm 1)
4. Calculate SINR and sum rate for each configuration
5. Store results in a structured CSV dataset

## Architecture

The system follows a procedural architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                   Main Generation Loop                   │
│                    (N iterations)                        │
└───────────────┬─────────────────────────────────────────┘
                │
                ├──► Channel Generator
                │    └─► Rayleigh fading model
                │    └─► Sort by channel gain
                │
                ├──► Power Allocation Optimizer
                │    └─► Compute α' (equal allocation bound)
                │    └─► Compute α'' (SIC constraint bound)
                │    └─► Solve linear system A·α'' = B
                │    └─► Select α* = min(α', α'')
                │
                ├──► Sum Rate Calculator
                │    └─► Compute SINR for each user
                │    └─► Compute individual rates R_m
                │    └─► Sum all rates
                │
                └──► Data Storage
                     └─► Append to DataFrame
                     └─► Export to CSV
```

## Components and Interfaces

### 1. Configuration Module

**Purpose:** Store and provide access to system parameters.

**Interface:**
```python
class NOMAConfig:
    M: int              # Number of users (3 or 4)
    P: float            # Total transmit power (1.0 W)
    N0: float           # Noise power (0.001 W)
    Pg: float           # Power gap for SIC (0.01 W)
    N_samples: int      # Number of samples (10,000)
    path_loss_exp: float # Path loss exponent β
```

### 2. Channel Generator

**Purpose:** Generate and sort channel coefficients according to Rayleigh fading.

**Interface:**
```python
def generate_channels(config: NOMAConfig) -> np.ndarray:
    """
    Generate M channel gains sorted in descending order.
    
    Returns:
        h_gains: Array of shape (M,) with |h_1|^2 >= |h_2|^2 >= ... >= |h_M|^2
    """
```

**Implementation Details:**
- Generate complex channel coefficients: h_m ~ CN(0, d^(-β))
- Compute channel gains: g_m = |h_m|^2
- Sort in descending order (strongest user first)
- Return sorted gains

### 3. Power Allocation Optimizer

**Purpose:** Compute optimal power allocation using Algorithm 1 from the paper.

**Interface:**
```python
def compute_power_allocation(h_gains: np.ndarray, config: NOMAConfig) -> np.ndarray:
    """
    Compute optimal power allocation α* using dual-bound method.
    
    Args:
        h_gains: Sorted channel gains (descending order)
        config: System configuration
        
    Returns:
        alpha_star: Array of shape (M,) with optimal power fractions
    """
```

**Sub-components:**

**3a. Equal Allocation Bound (α')**
```python
def compute_alpha_prime(M: int) -> np.ndarray:
    """
    Compute equal allocation bound: α'_m = 1/M for all m.
    
    Returns:
        alpha_prime: Array of shape (M,) with all elements = 1/M
    """
```

**3b. SIC Constraint Bound (α'')**
```python
def compute_alpha_double_prime(h_gains: np.ndarray, config: NOMAConfig) -> np.ndarray:
    """
    Compute SIC constraint bound by solving linear system A·α'' = B.
    
    The linear system encodes:
    - Sum constraint: Σ α''_i = 1
    - SIC constraints for m=2 to M:
      (2·Σ(α''_i, i=1..m-1) + Σ(α''_i, i=m+1..M))·P·|h_(m-1)|^2 = P·|h_(m-1)|^2 - P_g
    
    Returns:
        alpha_double_prime: Array of shape (M,) satisfying SIC constraints
    """
```

**Matrix Construction:**
- Row 1: Sum constraint [1, 1, 1, ..., 1] · α'' = 1
- Rows 2 to M: SIC constraints for each user boundary

**3c. Final Selection**
```python
def select_alpha_star(alpha_prime: np.ndarray, alpha_double_prime: np.ndarray) -> np.ndarray:
    """
    Select final allocation: α*_m = min(α'_m, α''_m) element-wise.
    
    Returns:
        alpha_star: Array of shape (M,) with optimal allocation
    """
```

### 4. Sum Rate Calculator

**Purpose:** Compute SINR and sum rate for a given channel and power allocation.

**Interface:**
```python
def compute_sum_rate(h_gains: np.ndarray, alpha: np.ndarray, config: NOMAConfig) -> float:
    """
    Compute sum rate R_sum = Σ R_m for all users.
    
    Args:
        h_gains: Sorted channel gains
        alpha: Power allocation fractions
        config: System configuration
        
    Returns:
        sum_rate: Total achievable rate in bits/s/Hz
    """
```

**Sub-components:**

**4a. SINR Calculation**
```python
def compute_sinr(m: int, h_gains: np.ndarray, alpha: np.ndarray, config: NOMAConfig) -> float:
    """
    Compute SINR for user m using NOMA formula:
    
    γ_m = (α_m · P · |h_m|^2) / (|h_m|^2 · Σ(α_i · P, i=1..m-1) + σ²)
    
    The interference term includes all users decoded before user m.
    
    Returns:
        sinr: Signal-to-Interference-plus-Noise Ratio for user m
    """
```

**4b. Rate Calculation**
```python
def compute_rate(sinr: float) -> float:
    """
    Compute Shannon rate: R = log₂(1 + γ).
    
    Returns:
        rate: Achievable rate in bits/s/Hz
    """
```

### 5. Dataset Generator (Main Loop)

**Purpose:** Orchestrate the generation of N samples and store results.

**Interface:**
```python
def generate_dataset(config: NOMAConfig, output_file: str = "noma_training_data.csv") -> pd.DataFrame:
    """
    Generate N samples of (channels, power_allocation, sum_rate).
    
    Returns:
        df: DataFrame with columns [h1_gain, h2_gain, ..., alpha1, alpha2, ..., Sum_Rate]
    """
```

**Implementation:**
```python
for i in range(N_samples):
    # Step A: Generate channels
    h_gains = generate_channels(config)
    
    # Step B: Compute power allocation
    alpha_star = compute_power_allocation(h_gains, config)
    
    # Step C: Compute sum rate
    sum_rate = compute_sum_rate(h_gains, alpha_star, config)
    
    # Store sample
    append_to_dataframe(h_gains, alpha_star, sum_rate)
```

### 6. Validation Module

**Purpose:** Ensure generated samples are mathematically valid.

**Interface:**
```python
def validate_sample(h_gains: np.ndarray, alpha: np.ndarray) -> bool:
    """
    Check if a sample satisfies basic constraints:
    - All channel gains > 0
    - All power allocations >= 0
    - Sum of power allocations <= 1
    - Channel gains are sorted descending
    
    Returns:
        is_valid: True if sample passes all checks
    """
```

## Data Models

### Channel State
```python
@dataclass
class ChannelState:
    h_gains: np.ndarray  # Shape (M,), sorted descending
    M: int               # Number of users
```

### Power Allocation
```python
@dataclass
class PowerAllocation:
    alpha: np.ndarray    # Shape (M,), power fractions
    alpha_prime: np.ndarray   # Equal allocation bound
    alpha_double_prime: np.ndarray  # SIC constraint bound
```

### Sample
```python
@dataclass
class NOMAample:
    h_gains: np.ndarray  # Channel gains
    alpha: np.ndarray    # Power allocation
    sum_rate: float      # Achievable sum rate
```

### Dataset Schema

For M=3:
```
| h1_gain | h2_gain | h3_gain | alpha1 | alpha2 | alpha3 | Sum_Rate |
|---------|---------|---------|--------|--------|--------|----------|
| float   | float   | float   | float  | float  | float  | float    |
```

For M=4:
```
| h1_gain | h2_gain | h3_gain | h4_gain | alpha1 | alpha2 | alpha3 | alpha4 | Sum_Rate |
|---------|---------|---------|---------|--------|--------|--------|--------|----------|
| float   | float   | float   | float   | float  | float  | float  | float  | float    |
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the acceptance criteria analysis, we identify the following correctness properties that must hold for all generated samples:

### Property 1: Channel Gain Ordering Invariant

*For any* generated sample, the channel gains must be sorted in descending order: h1_gain ≥ h2_gain ≥ ... ≥ hM_gain.

**Validates: Requirements 2.3**

**Rationale:** This is fundamental to NOMA operation. The SIC decoding order depends on users being ordered by signal strength. If this property is violated, the entire power allocation and rate calculation becomes invalid.

### Property 2: Equal Allocation Bound Correctness

*For any* value of M, when computing the equal allocation bound α', all elements must equal 1/M exactly.

**Validates: Requirements 3.1**

**Rationale:** The equal allocation bound represents the trivial solution where all users get equal power. This serves as an upper bound in the optimization.

### Property 3: Power Allocation Feasibility

*For any* generated sample, the final power allocation α* must satisfy:
- All elements α*_m ≥ 0 (non-negativity)
- Σ(α*_m) ≤ 1 (total power constraint)

**Validates: Requirements 3.3, 7.2, 7.3**

**Rationale:** Power allocations must be physically realizable. Negative power is meaningless, and the sum cannot exceed the total available power. This property combines the sum constraint from α'' with the non-negativity requirement for α*.

### Property 4: SIC Constraint Satisfaction

*For any* generated sample with computed α'', for each user m from 2 to M, the SIC constraint must hold:

(2·Σ(α''_i for i=1 to m-1) + Σ(α''_i for i=m+1 to M))·P·|h_(m-1)|^2 ≈ P·|h_(m-1)|^2 - P_g

(within numerical tolerance ε = 1e-6)

**Validates: Requirements 3.4**

**Rationale:** This constraint ensures that there is sufficient power difference between users for SIC to work correctly. The power gap P_g is the minimum difference needed for successful interference cancellation.

### Property 5: Minimum Selection Correctness

*For any* generated sample, the final power allocation must satisfy α*_m ≤ α'_m AND α*_m ≤ α''_m for all users m.

**Validates: Requirements 3.6**

**Rationale:** Algorithm 1 selects the minimum of the two bounds element-wise. This property verifies that the selection logic is implemented correctly.

### Property 6: SINR Non-Negativity

*For any* generated sample, all computed SINR values γ_m must be non-negative (γ_m ≥ 0).

**Validates: Requirements 7.5**

**Rationale:** SINR is a ratio of signal power to interference-plus-noise power. Since all terms are non-negative (power allocations, channel gains, noise), the SINR must also be non-negative. Negative SINR indicates a computation error.

### Property 7: Sum Rate Computation Correctness

*For any* generated sample, when independently recomputing the sum rate using:
1. γ_m = (α*_m·P·|h_m|^2) / (|h_m|^2·Σ(α*_i·P for i=1 to m-1) + σ²)
2. R_m = log₂(1 + γ_m)
3. R_sum = Σ(R_m)

The recomputed R_sum must match the stored Sum_Rate value (within numerical tolerance ε = 1e-6).

**Validates: Requirements 4.1, 4.2, 4.3**

**Rationale:** This property verifies the entire rate calculation pipeline. By independently recomputing from the stored channel gains and power allocations, we ensure the formulas are implemented correctly and the stored sum rate is accurate.

### Property 8: Interference Term Correctness

*For any* generated sample and user m, when computing SINR, the interference term must only include users 1 through m-1 (users decoded before user m in the SIC order).

**Validates: Requirements 4.1**

**Rationale:** In downlink NOMA with SIC, user m decodes and removes the signals of all stronger users (1 to m-1) before decoding its own signal. Users m+1 to M are treated as noise. This property ensures the interference term reflects the correct decoding order.

## Error Handling

### Numerical Stability

**Linear System Singularity:**
- **Issue:** The matrix A in the linear system A·α'' = B may be singular or ill-conditioned for certain channel realizations.
- **Handling:** Wrap `numpy.linalg.solve()` in a try-except block. If `LinAlgError` is raised, skip the current sample and generate a new channel realization.
- **Logging:** Count and report the number of skipped samples at the end of generation.

**Division by Zero in SINR:**
- **Issue:** For user 1 (strongest user), the interference term is zero. The denominator becomes σ² only.
- **Handling:** This is expected behavior. Ensure σ² > 0 in configuration to prevent division by zero.

**Negative SINR:**
- **Issue:** Due to numerical errors, SINR might become slightly negative (e.g., -1e-15).
- **Handling:** Clip SINR values to a minimum of 0 before computing rates: `sinr = max(0, sinr)`.

### Invalid Power Allocations

**Negative α Values:**
- **Issue:** The linear system solution might produce negative α'' values for certain channel conditions.
- **Handling:** After computing α*, validate that all elements are ≥ 0. If validation fails, skip the sample.

**Sum Exceeds 1:**
- **Issue:** Due to numerical errors, Σ(α*) might slightly exceed 1.
- **Handling:** If Σ(α*) > 1 + ε (where ε = 1e-6), skip the sample. If 1 < Σ(α*) ≤ 1 + ε, normalize: α* = α* / Σ(α*).

### Sample Validation

**Validation Function:**
```python
def validate_sample(h_gains, alpha, sum_rate):
    # Check channel gains are sorted descending
    if not np.all(h_gains[:-1] >= h_gains[1:]):
        return False
    
    # Check all values are positive
    if np.any(h_gains <= 0) or np.any(alpha < 0):
        return False
    
    # Check power constraint
    if np.sum(alpha) > 1 + 1e-6:
        return False
    
    # Check sum rate is positive
    if sum_rate < 0:
        return False
    
    return True
```

**Retry Logic:**
- Maximum retries per sample: 10
- If 10 consecutive samples fail validation, raise an error (indicates systematic problem)

### Edge Cases

**All Users Have Same Channel Gain:**
- **Issue:** If h_1 = h_2 = ... = h_M, the SIC constraint matrix becomes singular.
- **Probability:** Extremely low with continuous Rayleigh distribution.
- **Handling:** Caught by linear system solver error handling.

**Very Weak Channels:**
- **Issue:** If all channel gains are very small, α'' might require more power than available.
- **Handling:** The min(α', α'') selection will choose α', resulting in equal power allocation.

**Very Strong Channels:**
- **Issue:** If channel gains are very large, numerical overflow in SINR calculation.
- **Handling:** Use `np.float64` precision. For extreme cases, clip SINR to a maximum value (e.g., 1e10) before computing log.

## Testing Strategy

The testing strategy employs a dual approach combining unit tests for specific cases and property-based tests for universal correctness guarantees.

### Unit Testing

Unit tests verify specific examples, edge cases, and integration points:

**Configuration Tests:**
- Test default configuration has M=3, P=1.0, N0=0.001, Pg=0.01
- Test configuration accepts M=4
- Test output file naming

**Channel Generation Tests:**
- Test that generated channels have correct shape (M,)
- Test that channels are sorted in descending order (specific example)
- Test edge case: M=1 (single user)

**Power Allocation Tests:**
- Test α' computation for M=3 and M=4 (should be [1/3, 1/3, 1/3] and [1/4, 1/4, 1/4, 1/4])
- Test matrix construction for M=3 (verify A and B have correct shapes)
- Test min selection with known α' and α'' values

**Sum Rate Tests:**
- Test SINR calculation for user 1 (no interference case)
- Test rate calculation with known SINR values
- Test sum rate aggregation

**Integration Tests:**
- Test end-to-end generation of 10 samples
- Test CSV output has correct columns for M=3 and M=4
- Test that generated CSV can be loaded back into pandas

**Error Handling Tests:**
- Test that singular matrix is caught and sample is skipped
- Test that negative α values cause sample rejection
- Test that invalid samples are replaced

### Property-Based Testing

Property-based tests verify universal properties across many randomly generated inputs. We will use the **Hypothesis** library for Python.

**Configuration:**
- Minimum 100 iterations per property test
- Random seed for reproducibility
- Each test tagged with feature name and property number

**Test 1: Channel Gain Ordering**
```python
@given(M=st.integers(min_value=2, max_value=10))
def test_channel_ordering(M):
    """
    Feature: noma-dataset-generator, Property 1: Channel Gain Ordering Invariant
    
    For any M, generated channel gains must be sorted descending.
    """
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    
    # Verify descending order
    assert np.all(h_gains[:-1] >= h_gains[1:])
```

**Test 2: Equal Allocation Bound**
```python
@given(M=st.integers(min_value=2, max_value=10))
def test_equal_allocation_bound(M):
    """
    Feature: noma-dataset-generator, Property 2: Equal Allocation Bound Correctness
    
    For any M, α' must have all elements equal to 1/M.
    """
    alpha_prime = compute_alpha_prime(M)
    
    expected = np.ones(M) / M
    assert np.allclose(alpha_prime, expected)
```

**Test 3: Power Allocation Feasibility**
```python
@given(M=st.integers(min_value=2, max_value=10))
def test_power_allocation_feasibility(M):
    """
    Feature: noma-dataset-generator, Property 3: Power Allocation Feasibility
    
    For any generated sample, α* must be non-negative and sum to ≤ 1.
    """
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    alpha_star = compute_power_allocation(h_gains, config)
    
    # Non-negativity
    assert np.all(alpha_star >= 0)
    
    # Power constraint
    assert np.sum(alpha_star) <= 1.0 + 1e-6
```

**Test 4: SIC Constraint Satisfaction**
```python
@given(M=st.integers(min_value=2, max_value=10))
def test_sic_constraint(M):
    """
    Feature: noma-dataset-generator, Property 4: SIC Constraint Satisfaction
    
    For any generated sample, α'' must satisfy SIC constraints.
    """
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    
    try:
        alpha_double_prime = compute_alpha_double_prime(h_gains, config)
        
        # Check SIC constraint for each user m from 2 to M
        for m in range(2, M + 1):
            # Sum of alphas before user m
            sum_before = np.sum(alpha_double_prime[:m-1])
            # Sum of alphas after user m
            sum_after = np.sum(alpha_double_prime[m:]) if m < M else 0
            
            lhs = (2 * sum_before + sum_after) * config.P * h_gains[m-2]
            rhs = config.P * h_gains[m-2] - config.Pg
            
            assert np.isclose(lhs, rhs, atol=1e-6)
    except np.linalg.LinAlgError:
        # Singular matrix - acceptable to skip
        pass
```

**Test 5: Minimum Selection Correctness**
```python
@given(M=st.integers(min_value=2, max_value=10))
def test_minimum_selection(M):
    """
    Feature: noma-dataset-generator, Property 5: Minimum Selection Correctness
    
    For any sample, α* must be element-wise minimum of α' and α''.
    """
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    
    alpha_prime = compute_alpha_prime(M)
    try:
        alpha_double_prime = compute_alpha_double_prime(h_gains, config)
        alpha_star = select_alpha_star(alpha_prime, alpha_double_prime)
        
        # Verify α* ≤ α' and α* ≤ α''
        assert np.all(alpha_star <= alpha_prime + 1e-9)
        assert np.all(alpha_star <= alpha_double_prime + 1e-9)
    except np.linalg.LinAlgError:
        pass
```

**Test 6: SINR Non-Negativity**
```python
@given(M=st.integers(min_value=2, max_value=10))
def test_sinr_non_negative(M):
    """
    Feature: noma-dataset-generator, Property 6: SINR Non-Negativity
    
    For any generated sample, all SINR values must be non-negative.
    """
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    alpha_star = compute_power_allocation(h_gains, config)
    
    for m in range(M):
        sinr = compute_sinr(m, h_gains, alpha_star, config)
        assert sinr >= -1e-9  # Allow tiny numerical errors
```

**Test 7: Sum Rate Computation Correctness**
```python
@given(M=st.integers(min_value=2, max_value=10))
def test_sum_rate_correctness(M):
    """
    Feature: noma-dataset-generator, Property 7: Sum Rate Computation Correctness
    
    For any sample, independently recomputed sum rate must match stored value.
    """
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    alpha_star = compute_power_allocation(h_gains, config)
    
    # Compute sum rate using the function
    sum_rate = compute_sum_rate(h_gains, alpha_star, config)
    
    # Independently recompute
    rates = []
    for m in range(M):
        sinr = compute_sinr(m, h_gains, alpha_star, config)
        rate = compute_rate(sinr)
        rates.append(rate)
    
    recomputed_sum_rate = np.sum(rates)
    
    assert np.isclose(sum_rate, recomputed_sum_rate, atol=1e-6)
```

**Test 8: Interference Term Correctness**
```python
@given(M=st.integers(min_value=3, max_value=10), user_idx=st.integers(min_value=1, max_value=9))
def test_interference_term(M, user_idx):
    """
    Feature: noma-dataset-generator, Property 8: Interference Term Correctness
    
    For any user m > 0, interference must only include users 0 to m-1.
    """
    assume(user_idx < M)  # Ensure user_idx is valid for M
    
    config = NOMAConfig(M=M)
    h_gains = generate_channels(config)
    alpha_star = compute_power_allocation(h_gains, config)
    
    # Manually compute expected interference for user_idx
    expected_interference = 0
    for i in range(user_idx):
        expected_interference += alpha_star[i] * config.P * h_gains[user_idx]
    
    # Compute SINR and extract interference from it
    sinr = compute_sinr(user_idx, h_gains, alpha_star, config)
    signal = alpha_star[user_idx] * config.P * h_gains[user_idx]
    denominator = signal / sinr
    actual_interference = denominator - config.N0
    
    assert np.isclose(actual_interference, expected_interference, atol=1e-6)
```

### Test Execution

**Running Tests:**
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run all property tests (100 iterations each)
pytest tests/properties/ -v --hypothesis-seed=42

# Run with more iterations for thorough testing
pytest tests/properties/ -v --hypothesis-seed=42 --hypothesis-iterations=1000
```

**Coverage Goals:**
- Unit test coverage: > 90% of lines
- Property test coverage: All 8 correctness properties
- Integration test coverage: End-to-end workflow

### Continuous Validation

**During Dataset Generation:**
- Run validation function on each sample before adding to DataFrame
- Log statistics: number of samples generated, number skipped, reasons for skipping
- If skip rate > 10%, warn user about potential configuration issues

**Post-Generation Validation:**
- Run property tests on a random sample of 100 rows from the generated CSV
- Verify all properties hold for the stored data
- Generate summary statistics: mean/std of channel gains, power allocations, sum rates
