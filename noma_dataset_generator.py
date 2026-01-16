"""
NOMA Dataset Generator for Power Allocation Optimization
========================================================

**Paper Reference:**
Trankatwar, S., & Wali, P. K. (2024). "Power Allocation for Sum Rate Maximization 
Under SIC Constraint in NOMA Networks." 
*2024 16th International Conference on COMmunication Systems & NETworkS (COMSNETS)*.

**Purpose:**
This script generates a synthetic dataset to train Machine Learning (ML) models 
to predict optimal Power Allocation (PA) coefficients in a Downlink NOMA system. 
The goal is to maximize the Sum Rate while satisfying strict Successive Interference 
Cancellation (SIC) constraints.

**Methodology: Sum Rate Maximization Under SIC Constraint**
The implementation follows the "Algorithm 1" proposed in the paper, which determines 
the optimal power allocation ($\alpha^*$) by finding the intersection of two constraints:
1.  **Power Budget Constraint ($\alpha'$):** The physical limit of total transmit power.
2.  **SIC Constraint ($\alpha''$):** The minimum power difference required between 
    superposed signals to ensure the receiver can successfully decode and cancel interference.

The optimal allocation is derived as: $\alpha^*_m = \min(\alpha'_m, \alpha''_m)$

**Key Formulas:**

1.  **System Model (Sorted Channels):**
    Users are indexed by channel strength: $|h_1| \ge |h_2| \ge \dots \ge |h_M|$.
    - User 1: Strongest channel (Nearest to BS).
    - User M: Weakest channel (Farthest from BS).

2.  **SINR Calculation (Eq. 1 in Paper):**
    For user $m$, Signal-to-Interference-plus-Noise Ratio is:
    $$ \gamma_m = \frac{\alpha_m P |h_m|^2}{ |h_m|^2 \sum_{i=1}^{m-1} \alpha_i P + \sigma^2 } $$
    *Note: In Downlink NOMA, User $m$ performs SIC for weaker users ($m+1 \dots M$) 
    but treats stronger users ($1 \dots m-1$) as noise/interference.*

3.  **SIC Constraint (Eq. 3 in Paper):**
    To ensure successful decoding, the signal power difference must exceed the Power Gap ($P_g$):
    $$ \alpha_m P |h_{m-1}|^2 - \sum_{i=1}^{m-1} \alpha_i P |h_{m-1}|^2 \ge P_g, \quad \text{for } m=2 \dots M $$

4.  **Matrix Formulation for $\alpha''$ (Eq. 9 & 10 in Paper):**
    The upper bound $\alpha''$ is found by solving the linear system $A \cdot \alpha'' = B$, 
    which represents the tightest possible packing of signals that satisfies the SIC gap $P_g$.

**Dataset Structure for ML Training:**
The output CSV is structured for Supervised Learning:
- **Input Features ($X$):** Channel Gains (`h1_gain`, `h2_gain`...). 
  *The model should learn to observe the environment ($h$).*
- **Target Variables ($Y$):** Optimal Power Coefficients (`alpha1`, `alpha2`...). 
  *The model predicts the optimal allocation ($\alpha$).*
- **Performance Metric:** `Sum_Rate` (Theoretical maximum capacity).
"""

import numpy as np
import pandas as pd
import argparse
import sys
from dataclasses import dataclass


@dataclass
class NOMAConfig:
    """
    Configuration parameters for NOMA system.
    
    Attributes:
        M: Number of users (3 or 4)
        P: Total transmit power in Watts (default: 1.0 W)
        N0: Noise power in Watts (default: 0.001 W)
        Pg: Power gap for SIC in Watts (default: 0.01 W)
        N_samples: Number of samples to generate (default: 10,000)
        path_loss_exp: Path loss exponent β for channel model (default: 2.0)
    """
    M: int = 3
    P: float = 1.0
    N0: float = 0.001
    Pg: float = 0.01
    N_samples: int = 10000
    path_loss_exp: float = 2.0


def generate_channels(config: NOMAConfig) -> np.ndarray:
    """
    Generate M channel gains sorted in descending order using Rayleigh fading model.
    
    The Rayleigh fading model represents wireless channel variations in rich scattering
    environments. Each complex channel coefficient h_m follows a complex Gaussian
    distribution: h_m ~ CN(0, d^(-β)), where d is distance and β is path loss exponent.
    
    Channel gains are computed as g_m = |h_m|^2, representing the signal power attenuation.
    Users are ordered by channel gain (strongest to weakest) to establish the SIC
    decoding order required for NOMA operation.
    
    Formula from paper:
    - h_m = (randn + j*randn) / sqrt(2)  [Rayleigh fading]
    - g_m = |h_m|^2  [Channel gain]
    - Sort: g_1 >= g_2 >= ... >= g_M  [SIC order]
    
    Args:
        config: NOMAConfig object containing system parameters (M, path_loss_exp)
        
    Returns:
        h_gains: Array of shape (M,) with sorted channel gains in descending order
                 where h_gains[0] is the strongest user and h_gains[M-1] is the weakest
    """
    # Generate M complex channel coefficients using Rayleigh fading
    # Each coefficient: h = (randn + j*randn) / sqrt(2) for proper normalization
    h_real = np.random.randn(config.M)
    h_imag = np.random.randn(config.M)
    h_complex = (h_real + 1j * h_imag) / np.sqrt(2)
    
    # Compute channel gains: g = |h|^2
    h_gains = np.abs(h_complex) ** 2
    
    # Sort gains in descending order (strongest user first for SIC)
    h_gains_sorted = np.sort(h_gains)[::-1]
    
    return h_gains_sorted


def compute_alpha_prime(M: int) -> np.ndarray:
    """
    Compute equal allocation bound (Algorithm 1 - first bound).
    
    The equal allocation bound represents the trivial solution where all users
    receive equal power fractions. This serves as an upper bound in the
    dual-bound optimization approach.
    
    Formula from paper (Algorithm 1 - Equal allocation bound):
    α'_m = 1/M for all m = 1, 2, ..., M
    
    Args:
        M: Number of users
        
    Returns:
        alpha_prime: Array of shape (M,) with all elements equal to 1/M
    """
    # Algorithm 1 - Equal allocation bound
    return np.ones(M) / M



def compute_alpha_double_prime(h_gains: np.ndarray, config: NOMAConfig) -> np.ndarray:
    """
    Compute SIC constraint bound (Algorithm 1 - second bound).
    
    This function solves a linear system A·α'' = B that encodes:
    1. Sum constraint: All power fractions must sum to 1
    2. SIC constraints: Ensure sufficient power difference for successful interference cancellation
    
    The SIC constraint for user m ensures that after decoding users 1 to m-1,
    there is sufficient power remaining (power gap P_g) for user m-1 to be decoded.
    
    Formula from paper (Algorithm 1 - SIC constraint bound):
    - Row 1: Σ(α''_i) = 1  [Sum constraint]
    - Rows 2 to M: For each user m from 2 to M:
      (2·Σ(α''_i, i=1..m-1) + Σ(α''_i, i=m+1..M))·P·|h_(m-1)|^2 = P·|h_(m-1)|^2 - P_g
    
    The SIC constraint ensures that the interference power plus the desired signal power
    for users m to M does not exceed the available power after accounting for the
    power gap needed for successful SIC decoding.
    
    Args:
        h_gains: Sorted channel gains in descending order, shape (M,)
        config: NOMAConfig object containing P, Pg, and M
        
    Returns:
        alpha_double_prime: Array of shape (M,) satisfying SIC constraints,
                           or None if the linear system is singular
    """

    """
    Returns SIC constraint bound (alpha'') by solving linear system A * alpha = B.
    Derivation from Paper Eq (8), (9), (10).
    Ensures that for every user, the power difference is sufficient for SIC.
    """
    M = config.M
    A = np.zeros((M, M))
    B = np.zeros(M)
    
    # Constraint 1: Sum of alphas = 1 (Eq. 4c active constraint)
    A[0, :] = 1.0
    B[0] = 1.0
    
    # Constraint 2: SIC requirements for users 2 to M (Eq. 8 in paper)
    # Formula: (2*Sum(prev) + Sum(next)) * P * |h_{m-1}|^2 = P*|h_{m-1}|^2 - Pg
    for m in range(1, M):
        # Coefficients for sum(alpha_1 ... alpha_{m-1})
        A[m, :m] = 2.0 * config.P * h_gains[m-1]
        
        # Coefficients for sum(alpha_m ... alpha_M)
        A[m, m:] = config.P * h_gains[m-1] 
        
        B[m] = config.P * h_gains[m-1] - config.Pg

    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None
    


def select_alpha_star(alpha_prime: np.ndarray, alpha_double_prime: np.ndarray) -> np.ndarray:
    """
    Select final power allocation as element-wise minimum of two bounds.
    
    Algorithm 1 computes two bounds on the power allocation:
    - α': Equal allocation bound (1/M for all users)
    - α'': SIC constraint bound (from solving linear system)
    
    The optimal allocation α* is the element-wise minimum, ensuring both
    constraints are satisfied.
    
    Formula from paper (Algorithm 1 - Final selection):
    α*_m = min(α'_m, α''_m) for each user m
    
    Args:
        alpha_prime: Equal allocation bound, shape (M,)
        alpha_double_prime: SIC constraint bound, shape (M,)
        
    Returns:
        alpha_star: Optimal power allocation, shape (M,)
    """
    return np.minimum(alpha_prime, alpha_double_prime)



def compute_power_allocation(h_gains: np.ndarray, config: NOMAConfig) -> np.ndarray:
    """
    Compute optimal power allocation using dual-bound method (Algorithm 1).
    
    This orchestrator function implements the complete Algorithm 1 from the paper:
    1. Compute equal allocation bound α' = 1/M
    2. Compute SIC constraint bound α'' by solving linear system
    3. Select final allocation α* = min(α', α'')
    
    If the linear system for α'' is singular (rare edge case), the function
    falls back to using α' (equal allocation).
    
    If α'' contains negative values (SIC constraints too restrictive), the function
    also falls back to α' to ensure physical realizability.
    
    Args:
        h_gains: Sorted channel gains in descending order, shape (M,)
        config: NOMAConfig object containing system parameters
        
    Returns:
        alpha_star: Optimal power allocation fractions, shape (M,)
    """
    M = config.M
    
    # Step 1: Compute equal allocation bound (α')
    alpha_prime = compute_alpha_prime(M)
    
    # Step 2: Compute SIC constraint bound (α'')
    alpha_double_prime = compute_alpha_double_prime(h_gains, config)
    
    # Step 3: Validate and Select final allocation
    
    # Check 1: Handle Singular Matrix
    # If the linear system could not be solved, alpha_double_prime is None.
    if alpha_double_prime is None:
        return alpha_prime

    # Check 2: Handle Negative Values (Physical Feasibility)
    # The SIC calculation might return negative power values if the channel gains 
    # don't support the required Power Gap (Pg). Power cannot be negative.
    if np.any(alpha_double_prime < 0):
        return alpha_prime
    
    # Step 4: Compute Final α* # select_alpha_star computes element-wise min(α', α'')
    alpha_star = select_alpha_star(alpha_prime, alpha_double_prime)
    
    return alpha_star



def compute_sinr(m: int, h_gains: np.ndarray, alpha: np.ndarray, config: NOMAConfig) -> float:
    """
    Computes SINR for user m based on NOMA Downlink interference.
    
    Formula: γ_m = (α_m·P·|h_m|²) / (|h_m|²·Σ(α_i·P, i=0..m-1) + σ²)
    User 0 (strongest) has no interference.
    User m experiences interference from users 0 to m-1 (stronger users treated as noise).
    """
    # Calculate signal power: alpha[m] * P * h_gains[m]
    signal_power = alpha[m] * config.P * h_gains[m]
    
    # Calculate interference: sum of alpha[i] * P * h_gains[m] for i=0 to m-1
    # NOMA SINR formula - interference from users 1 to m-1 (mapped to 0..m-1 in 0-indexing)
    if m == 0:
        interference_power = 0.0
    else:
        # Sum of alphas for all users stronger than m (indices 0 to m-1)
        # Multiplied by Total Power P and current user's channel gain h_m
        interference_sum_alpha = np.sum(alpha[:m])
        interference_power = interference_sum_alpha * config.P * h_gains[m]
        
    # Calculate SINR: signal / (interference + N0)
    sinr = signal_power / (interference_power + config.N0)
    
    # Clip SINR to minimum of 0
    return float(max(0.0, sinr))

def compute_rate(sinr: float) -> float:
    """
    Computes Shannon rate from SINR.
    
    Returns:
        Rate in bits/s/Hz
    """
    return np.log2(1 + sinr)

def compute_sum_rate(h_gains: np.ndarray, alpha: np.ndarray, config: NOMAConfig) -> float:
    """
    Computes the total system sum rate by summing individual user rates.
    """
    M = len(h_gains)
    total_rate = 0.0
    
    # Loop through all M users (0 to M-1)
    for m in range(M):
        # Compute SINR for each
        sinr = compute_sinr(m, h_gains, alpha, config)
        
        # Compute rate for each
        rate = compute_rate(sinr)
        
        # Sum all rates
        total_rate += rate
        
    return total_rate



def validate_sample(h_gains: np.ndarray, alpha: np.ndarray, sum_rate: float) -> bool:
    """
    Validates a generated data sample against physical and logical constraints.
    
    Checks:
    1. Channels sorted descending
    2. All gains > 0
    3. All alphas >= 0
    4. Sum(alpha) <= 1 (with tolerance)
    5. Sum rate >= 0
    """
    # Check channel gains sorted descending: np.all(h_gains[:-1] >= h_gains[1:])
    if not np.all(h_gains[:-1] >= h_gains[1:]):
        return False
        
    # Check all gains > 0: np.all(h_gains > 0)
    if not np.all(h_gains > 0):
        return False
        
    # Check all alpha >= 0: np.all(alpha >= 0)
    if not np.all(alpha >= 0):
        return False
        
    # Check sum(alpha) <= 1 + 1e-6: np.sum(alpha) <= 1 + 1e-6
    if not (np.sum(alpha) <= 1.0 + 1e-6):
        return False
        
    # Check sum_rate >= 0
    if sum_rate < 0:
        return False
        
    return True



def generate_dataset(config: NOMAConfig) -> pd.DataFrame:
    """
    Generates the NOMA training dataset and saves it to a CSV file.
    
    Returns:
        pd.DataFrame containing channel gains, power allocations, and sum rate.
    """
    data_records = []
    skipped_count = 0
    
    print(f"Starting generation of {config.N_samples} samples for M={config.M} users...")
    
    for _ in range(config.N_samples):
        # 1. Generate Environment
        h_gains = generate_channels(config)
        
        # 2. Compute Optimization Target
        alpha_star = compute_power_allocation(h_gains, config)
        
        # 3. Compute Resulting Metric
        sum_rate = compute_sum_rate(h_gains, alpha_star, config)
        
        # 4. Validation
        if validate_sample(h_gains, alpha_star, sum_rate):
            # Create record dictionary
            record = {}
            
            # Add Channel Gains: h1_gain, h2_gain...
            for m in range(config.M):
                record[f'h{m+1}_gain'] = h_gains[m]
                
            # Add Power Allocations: alpha1, alpha2...
            for m in range(config.M):
                record[f'alpha{m+1}'] = alpha_star[m]
                
            # Add Target
            record['Sum_Rate'] = sum_rate
            
            data_records.append(record)
        else:
            skipped_count += 1
            
    # Create DataFrame
    df = pd.DataFrame(data_records)
    
    # Save to CSV (Added as per Task 7.1 requirement)
    filename = 'noma_training_data.csv'
    df.to_csv(filename, index=False)
    
    print(f"Generation Complete. Valid: {len(df)}, Skipped: {skipped_count}")
    print(f"Dataset saved to {filename}")
    
    return df


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="NOMA Dataset Generator")
    parser.add_argument(
        '--users', 
        type=int, 
        choices=[3, 4], 
        default=3,
        help='Number of users in the NOMA cluster (3 or 4)'
    )
    return parser.parse_args()



def main():
    """Main execution block."""
    args = parse_args()
    
    # Configure System
    config = NOMAConfig(M=args.users)
    
    # Generate Data
    # Note: CSV saving is handled inside generate_dataset now
    df = generate_dataset(config)
    
    # Summary Statistics
    if not df.empty:
        print("\n--- Dataset Summary ---")
        print(df.describe().loc[['mean', 'std', 'min', 'max']])
        print("\nSample Rows:")
        print(df.head())
