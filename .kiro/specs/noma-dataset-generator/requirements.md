# Requirements Document

## Introduction

This document specifies the requirements for a Python-based training dataset generator for Machine Learning models that optimize Power Allocation (PA) in Downlink NOMA (Non-Orthogonal Multiple Access) networks. The system implements the "Sum Rate Maximization Under SIC Constraint" method to generate optimal power allocation configurations and their corresponding sum rates.

## Glossary

- **NOMA**: Non-Orthogonal Multiple Access - a wireless communication technique allowing multiple users to share the same frequency/time resource
- **SIC**: Successive Interference Cancellation - a decoding technique where stronger signals are decoded first and removed from the received signal
- **Channel_Gain**: The squared magnitude of the channel coefficient (|h_m|^2), representing signal strength
- **Power_Allocation**: The fraction of total transmit power (α_m) assigned to each user
- **SINR**: Signal-to-Interference-plus-Noise Ratio - the quality metric for each user's signal
- **Sum_Rate**: The total data rate achievable across all users in the system
- **Dataset_Generator**: The system that produces training samples for ML models
- **Rayleigh_Fading**: A statistical model for wireless channel variations
- **Power_Gap**: The minimum power difference required for successful SIC (P_g)

## Requirements

### Requirement 1: System Configuration

**User Story:** As a researcher, I want to configure the NOMA system parameters, so that I can generate datasets matching my network specifications.

#### Acceptance Criteria

1. THE Dataset_Generator SHALL support M=3 users as the default configuration
2. THE Dataset_Generator SHALL support M=4 users through a configurable parameter
3. THE Dataset_Generator SHALL use a total transmit power P=1.0 Watt
4. THE Dataset_Generator SHALL use a noise power σ²=0.001 Watt
5. THE Dataset_Generator SHALL use a power gap P_g=0.01 Watt for SIC constraints
6. THE Dataset_Generator SHALL generate N=10,000 samples per dataset

### Requirement 2: Channel Generation

**User Story:** As a researcher, I want realistic wireless channel coefficients, so that my dataset reflects real-world fading conditions.

#### Acceptance Criteria

1. WHEN generating channel coefficients, THE Dataset_Generator SHALL use the Rayleigh fading model with h_m ~ CN(0, d^(-β))
2. WHEN channel coefficients are generated, THE Dataset_Generator SHALL compute channel gains as |h_m|^2
3. WHEN channel gains are computed, THE Dataset_Generator SHALL sort them in descending order such that |h_1|^2 ≥ |h_2|^2 ≥ ... ≥ |h_M|^2
4. THE Dataset_Generator SHALL assign User 1 as the strongest user (highest channel gain)
5. THE Dataset_Generator SHALL assign User M as the weakest user (lowest channel gain)

### Requirement 3: Power Allocation Calculation

**User Story:** As a researcher, I want optimal power allocation computed using Algorithm 1, so that the dataset contains mathematically valid NOMA configurations.

#### Acceptance Criteria

1. WHEN computing power allocation, THE Dataset_Generator SHALL calculate the first bound α'_m = 1/M for all users
2. WHEN computing the second bound α'', THE Dataset_Generator SHALL construct a linear system A·α'' = B
3. THE Dataset_Generator SHALL enforce the sum constraint: Σ(α''_i) = 1 for i=1 to M
4. FOR each user m from 2 to M, THE Dataset_Generator SHALL enforce the SIC constraint: (2·Σ(α''_i for i=1 to m-1) + Σ(α''_i for i=m+1 to M))·P·|h_(m-1)|^2 = P·|h_(m-1)|^2 - P_g
5. WHEN solving the linear system, THE Dataset_Generator SHALL use numpy.linalg.solve
6. WHEN determining final power allocation, THE Dataset_Generator SHALL compute α*_m = min(α'_m, α''_m) for each user m

### Requirement 4: Sum Rate Computation

**User Story:** As a researcher, I want accurate sum rate calculations, so that my ML model learns the correct optimization objective.

#### Acceptance Criteria

1. WHEN calculating SINR for user m, THE Dataset_Generator SHALL use the formula: γ_m = (α*_m·P·|h_m|^2) / (|h_m|^2·Σ(α*_i·P for i=1 to m-1) + σ²)
2. WHEN calculating individual rates, THE Dataset_Generator SHALL use the formula: R_m = log₂(1 + γ_m)
3. WHEN calculating sum rate, THE Dataset_Generator SHALL compute R_sum = Σ(R_m) for all users
4. THE Dataset_Generator SHALL ensure the interference term reflects the downlink NOMA decoding order based on sorted channels

### Requirement 5: Data Storage and Output

**User Story:** As a researcher, I want the dataset saved in a standard format, so that I can easily load it into ML frameworks.

#### Acceptance Criteria

1. WHEN storing results, THE Dataset_Generator SHALL use a Pandas DataFrame
2. FOR M=3 users, THE Dataset_Generator SHALL create columns: [h1_gain, h2_gain, h3_gain, alpha1, alpha2, alpha3, Sum_Rate]
3. FOR M=4 users, THE Dataset_Generator SHALL create columns: [h1_gain, h2_gain, h3_gain, h4_gain, alpha1, alpha2, alpha3, alpha4, Sum_Rate]
4. WHEN saving the dataset, THE Dataset_Generator SHALL write to a CSV file named "noma_training_data.csv"
5. THE Dataset_Generator SHALL generate exactly N rows of data in the output file

### Requirement 6: Code Quality and Documentation

**User Story:** As a developer, I want well-documented code, so that I can understand and modify the implementation.

#### Acceptance Criteria

1. THE Dataset_Generator SHALL include comments explaining which formula from the paper is being used at each step
2. THE Dataset_Generator SHALL use numpy for matrix operations and random number generation
3. THE Dataset_Generator SHALL use pandas for data storage and CSV export
4. THE Dataset_Generator SHALL handle matrix construction dynamically based on the value of M
5. THE Dataset_Generator SHALL include variable names that clearly indicate their mathematical correspondence (e.g., alpha, h_gain, SINR)

### Requirement 7: Numerical Stability and Validation

**User Story:** As a researcher, I want numerically stable computations, so that my dataset doesn't contain invalid or degenerate samples.

#### Acceptance Criteria

1. WHEN solving linear systems, THE Dataset_Generator SHALL handle potential numerical instabilities
2. WHEN power allocations are computed, THE Dataset_Generator SHALL verify that Σ(α*_m) ≤ 1
3. WHEN power allocations are computed, THE Dataset_Generator SHALL verify that all α*_m ≥ 0
4. IF a sample produces invalid power allocations, THE Dataset_Generator SHALL skip that sample and generate a replacement
5. THE Dataset_Generator SHALL ensure all SINR values are non-negative before computing rates
