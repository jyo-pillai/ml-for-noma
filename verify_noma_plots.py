#!/usr/bin/env python3
"""
NOMA Verification Plots Script
==============================

This script creates verification plots to validate the corrected NOMA power allocation
algorithm against the theoretical results from the paper. It generates three key plots
using Monte Carlo simulation to match the paper's figures.

**Paper Reference:**
Trankatwar, S., & Wali, P. K. (2024). "Power Allocation for Sum Rate Maximization 
Under SIC Constraint in NOMA Networks." 
*2024 16th International Conference on COMmunication Systems & NETworkS (COMSNETS)*.

**Plots Generated:**
1. Plot 1: Sum Rate vs. Transmit Power (Fig 3 from paper)
2. Plot 2: PA Coefficients vs. Power Gap (Fig 5 from paper)  
3. Plot 3: Sum Rate vs. Power Gap (Fig 6 from paper)

**Requirements: 6.1**
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Import corrected power allocation functions
from noma_corrected import (
    NOMAConfig,
    generate_channels,
    compute_power_allocation
)

# Import SINR and rate computation from main module
from noma_dataset_generator import (
    compute_sinr,
    compute_rate
)


def compute_sum_rate(h_gains: np.ndarray, alpha: np.ndarray, config: NOMAConfig) -> float:
    """
    Compute sum rate for given channel gains and power allocation.
    
    Args:
        h_gains: Sorted channel gains in descending order
        alpha: Power allocation coefficients
        config: NOMA system configuration
        
    Returns:
        sum_rate: Total achievable sum rate in bps/Hz
    """
    M = len(h_gains)
    total_rate = 0.0
    
    for m in range(M):
        sinr = compute_sinr(m, h_gains, alpha, config)
        rate = compute_rate(sinr)
        total_rate += rate
        
    return total_rate


def setup_plots():
    """
    Set up matplotlib for 3 subplots with proper styling.
    
    Returns:
        fig: Figure object
        axes: Array of 3 subplot axes
    """
    # Create figure with 3 subplots arranged horizontally
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set overall figure title
    fig.suptitle('NOMA Power Allocation Verification Plots', fontsize=16, fontweight='bold')
    
    # Configure subplot spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig, axes


def plot_sum_rate_vs_transmit_power(ax):
    """
    Plot 1: Sum Rate vs. Transmit Power (Fig 3 from paper)
    
    X-axis: P from 0.1 to 3.0 Watts
    Y-axis: Average Sum Rate (bps/Hz)
    Monte Carlo: 1000 channel realizations per P value
    Series: M=3 (blue) and M=4 (red)
    Parameters: Pg=0.01W, N0=0.001W
    Expected: Logarithmic growth, ~8-9 bps/Hz at P=1W for M=3
    
    Requirements: 4.3
    """
    print("Implementing Plot 1: Sum Rate vs. Transmit Power...")
    
    # Define power range: 0.1 to 3.0 Watts
    P_values = np.linspace(0.1, 3.0, 30)  # 30 points for smooth curve
    
    # Fixed parameters
    Pg = 0.01  # Power gap (W)
    N0 = 0.001  # Noise power (W)
    n_realizations = 1000  # Monte Carlo iterations per P value
    
    # Storage for results
    sum_rates_M3 = []
    sum_rates_M4 = []
    
    print(f"Running Monte Carlo simulation with {n_realizations} realizations per P value...")
    
    # Monte Carlo simulation for each P value
    for i, P in enumerate(P_values):
        print(f"Progress: {i+1}/{len(P_values)} (P={P:.2f}W)", end='\r')
        
        # Results for current P value
        rates_M3 = []
        rates_M4 = []
        
        # Monte Carlo iterations
        for _ in range(n_realizations):
            # M=3 case
            config_M3 = NOMAConfig(M=3, P=P, Pg=Pg, N0=N0)
            h_gains_M3 = generate_channels(config_M3)
            alpha_M3 = compute_power_allocation(h_gains_M3, config_M3)
            sum_rate_M3 = compute_sum_rate(h_gains_M3, alpha_M3, config_M3)
            rates_M3.append(sum_rate_M3)
            
            # M=4 case
            config_M4 = NOMAConfig(M=4, P=P, Pg=Pg, N0=N0)
            h_gains_M4 = generate_channels(config_M4)
            alpha_M4 = compute_power_allocation(h_gains_M4, config_M4)
            sum_rate_M4 = compute_sum_rate(h_gains_M4, alpha_M4, config_M4)
            rates_M4.append(sum_rate_M4)
        
        # Store average sum rates
        sum_rates_M3.append(np.mean(rates_M3))
        sum_rates_M4.append(np.mean(rates_M4))
    
    print()  # New line after progress indicator
    
    # Convert to numpy arrays
    sum_rates_M3 = np.array(sum_rates_M3)
    sum_rates_M4 = np.array(sum_rates_M4)
    
    # Create the plot
    ax.plot(P_values, sum_rates_M3, 'b-', linewidth=2, label='M=3 users', marker='o', markersize=4)
    ax.plot(P_values, sum_rates_M4, 'r-', linewidth=2, label='M=4 users', marker='s', markersize=4)
    
    # Configure plot
    ax.set_xlabel('Transmit Power P (Watts)', fontsize=12)
    ax.set_ylabel('Average Sum Rate (bps/Hz)', fontsize=12)
    ax.set_title('Sum Rate vs. Transmit Power\n(Fig 3 from paper)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Set reasonable axis limits
    ax.set_xlim(0.1, 3.0)
    ax.set_ylim(0, max(np.max(sum_rates_M3), np.max(sum_rates_M4)) * 1.1)
    
    # Print validation results
    print(f"\nPlot 1 Results:")
    print(f"At P=1W:")
    print(f"  M=3: {sum_rates_M3[np.argmin(np.abs(P_values - 1.0))]:.2f} bps/Hz (expected ~8-9 bps/Hz)")
    print(f"  M=4: {sum_rates_M4[np.argmin(np.abs(P_values - 1.0))]:.2f} bps/Hz")
    print(f"Maximum sum rates:")
    print(f"  M=3: {np.max(sum_rates_M3):.2f} bps/Hz at P={P_values[np.argmax(sum_rates_M3)]:.2f}W")
    print(f"  M=4: {np.max(sum_rates_M4):.2f} bps/Hz at P={P_values[np.argmax(sum_rates_M4)]:.2f}W")
    
    return P_values, sum_rates_M3, sum_rates_M4


def plot_pa_coefficients_vs_power_gap(ax):
    """
    Plot 2: PA Coefficients vs. Power Gap (Fig 5 from paper)
    
    X-axis: Pg from 0 to 0.04 Watts
    Y-axis: Average α values
    Monte Carlo: 1000 channel realizations per Pg value
    Parameters: P=1W, M=3
    Series: α1 (User 1), α2 (User 2), α3 (User 3)
    Expected: α3 (weakest) stays high/flat, α1 & α2 decrease with Pg
    
    Requirements: 3.4, 3.6
    """
    print("Implementing Plot 2: PA Coefficients vs. Power Gap...")
    
    # Define power gap range: 0 to 0.04 Watts
    Pg_values = np.linspace(0, 0.04, 41)  # 41 points for smooth curve (0.001 step)
    
    # Fixed parameters
    P = 1.0  # Transmit power (W)
    N0 = 0.001  # Noise power (W)
    M = 3  # Number of users
    n_realizations = 1000  # Monte Carlo iterations per Pg value
    
    # Storage for results - average alpha values for each user
    alpha1_avg = []  # User 1 (strongest)
    alpha2_avg = []  # User 2 (middle)
    alpha3_avg = []  # User 3 (weakest)
    
    print(f"Running Monte Carlo simulation with {n_realizations} realizations per Pg value...")
    
    # Monte Carlo simulation for each Pg value
    for i, Pg in enumerate(Pg_values):
        print(f"Progress: {i+1}/{len(Pg_values)} (Pg={Pg:.4f}W)", end='\r')
        
        # Results for current Pg value
        alpha1_values = []
        alpha2_values = []
        alpha3_values = []
        
        # Monte Carlo iterations
        for _ in range(n_realizations):
            # Create configuration for current Pg
            config = NOMAConfig(M=M, P=P, Pg=Pg, N0=N0)
            
            # Generate channels and compute power allocation
            h_gains = generate_channels(config)
            alpha = compute_power_allocation(h_gains, config)
            
            # Store individual alpha values
            alpha1_values.append(alpha[0])  # User 1 (strongest)
            alpha2_values.append(alpha[1])  # User 2 (middle)
            alpha3_values.append(alpha[2])  # User 3 (weakest)
        
        # Store average alpha values for current Pg
        alpha1_avg.append(np.mean(alpha1_values))
        alpha2_avg.append(np.mean(alpha2_values))
        alpha3_avg.append(np.mean(alpha3_values))
    
    print()  # New line after progress indicator
    
    # Convert to numpy arrays
    alpha1_avg = np.array(alpha1_avg)
    alpha2_avg = np.array(alpha2_avg)
    alpha3_avg = np.array(alpha3_avg)
    
    # Create the plot
    ax.plot(Pg_values, alpha1_avg, 'b-', linewidth=2, label='α₁ (User 1 - Strongest)', marker='o', markersize=4)
    ax.plot(Pg_values, alpha2_avg, 'g-', linewidth=2, label='α₂ (User 2 - Middle)', marker='s', markersize=4)
    ax.plot(Pg_values, alpha3_avg, 'r-', linewidth=2, label='α₃ (User 3 - Weakest)', marker='^', markersize=4)
    
    # Configure plot
    ax.set_xlabel('Power Gap Pg (Watts)', fontsize=12)
    ax.set_ylabel('Average Power Allocation α', fontsize=12)
    ax.set_title('PA Coefficients vs. Power Gap\n(Fig 5 from paper)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Set reasonable axis limits
    ax.set_xlim(0, 0.04)
    ax.set_ylim(0, max(np.max(alpha1_avg), np.max(alpha2_avg), np.max(alpha3_avg)) * 1.1)
    
    # Print validation results
    print(f"\nPlot 2 Results:")
    print(f"At Pg=0W:")
    print(f"  α₁ (User 1): {alpha1_avg[0]:.4f}")
    print(f"  α₂ (User 2): {alpha2_avg[0]:.4f}")
    print(f"  α₃ (User 3): {alpha3_avg[0]:.4f}")
    print(f"At Pg=0.04W:")
    print(f"  α₁ (User 1): {alpha1_avg[-1]:.4f}")
    print(f"  α₂ (User 2): {alpha2_avg[-1]:.4f}")
    print(f"  α₃ (User 3): {alpha3_avg[-1]:.4f}")
    
    # Analyze trends
    alpha1_change = alpha1_avg[-1] - alpha1_avg[0]
    alpha2_change = alpha2_avg[-1] - alpha2_avg[0]
    alpha3_change = alpha3_avg[-1] - alpha3_avg[0]
    
    print(f"Trends (change from Pg=0 to Pg=0.04):")
    print(f"  α₁ change: {alpha1_change:+.4f} ({'decreasing' if alpha1_change < 0 else 'increasing'})")
    print(f"  α₂ change: {alpha2_change:+.4f} ({'decreasing' if alpha2_change < 0 else 'increasing'})")
    print(f"  α₃ change: {alpha3_change:+.4f} ({'decreasing' if alpha3_change < 0 else 'increasing'})")
    
    return Pg_values, alpha1_avg, alpha2_avg, alpha3_avg


def plot_sum_rate_vs_power_gap(ax):
    """
    Plot 3: Sum Rate vs. Power Gap (Fig 6 from paper)
    
    X-axis: Pg from 0 to 0.04 Watts
    Y-axis: Average Sum Rate (bps/Hz)
    Monte Carlo: 1000 channel realizations per Pg value
    Parameters: P=1W
    Series: M=3 (blue) and M=4 (red)
    Expected: Sum Rate decreases as Pg increases
    
    Requirements: 4.3
    """
    print("Implementing Plot 3: Sum Rate vs. Power Gap...")
    
    # Define power gap range: 0 to 0.04 Watts
    Pg_values = np.linspace(0, 0.04, 41)  # 41 points for smooth curve (0.001 step)
    
    # Fixed parameters
    P = 1.0  # Transmit power (W)
    N0 = 0.001  # Noise power (W)
    n_realizations = 1000  # Monte Carlo iterations per Pg value
    
    # Storage for results
    sum_rates_M3 = []
    sum_rates_M4 = []
    
    print(f"Running Monte Carlo simulation with {n_realizations} realizations per Pg value...")
    
    # Monte Carlo simulation for each Pg value
    for i, Pg in enumerate(Pg_values):
        print(f"Progress: {i+1}/{len(Pg_values)} (Pg={Pg:.4f}W)", end='\r')
        
        # Results for current Pg value
        rates_M3 = []
        rates_M4 = []
        
        # Monte Carlo iterations
        for _ in range(n_realizations):
            # M=3 case
            config_M3 = NOMAConfig(M=3, P=P, Pg=Pg, N0=N0)
            h_gains_M3 = generate_channels(config_M3)
            alpha_M3 = compute_power_allocation(h_gains_M3, config_M3)
            sum_rate_M3 = compute_sum_rate(h_gains_M3, alpha_M3, config_M3)
            rates_M3.append(sum_rate_M3)
            
            # M=4 case
            config_M4 = NOMAConfig(M=4, P=P, Pg=Pg, N0=N0)
            h_gains_M4 = generate_channels(config_M4)
            alpha_M4 = compute_power_allocation(h_gains_M4, config_M4)
            sum_rate_M4 = compute_sum_rate(h_gains_M4, alpha_M4, config_M4)
            rates_M4.append(sum_rate_M4)
        
        # Store average sum rates for current Pg
        sum_rates_M3.append(np.mean(rates_M3))
        sum_rates_M4.append(np.mean(rates_M4))
    
    print()  # New line after progress indicator
    
    # Convert to numpy arrays
    sum_rates_M3 = np.array(sum_rates_M3)
    sum_rates_M4 = np.array(sum_rates_M4)
    
    # Create the plot
    ax.plot(Pg_values, sum_rates_M3, 'b-', linewidth=2, label='M=3 users', marker='o', markersize=4)
    ax.plot(Pg_values, sum_rates_M4, 'r-', linewidth=2, label='M=4 users', marker='s', markersize=4)
    
    # Configure plot
    ax.set_xlabel('Power Gap Pg (Watts)', fontsize=12)
    ax.set_ylabel('Average Sum Rate (bps/Hz)', fontsize=12)
    ax.set_title('Sum Rate vs. Power Gap\n(Fig 6 from paper)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Set reasonable axis limits
    ax.set_xlim(0, 0.04)
    ax.set_ylim(0, max(np.max(sum_rates_M3), np.max(sum_rates_M4)) * 1.1)
    
    # Print validation results
    print(f"\nPlot 3 Results:")
    print(f"At Pg=0W:")
    print(f"  M=3: {sum_rates_M3[0]:.4f} bps/Hz")
    print(f"  M=4: {sum_rates_M4[0]:.4f} bps/Hz")
    print(f"At Pg=0.04W:")
    print(f"  M=3: {sum_rates_M3[-1]:.4f} bps/Hz")
    print(f"  M=4: {sum_rates_M4[-1]:.4f} bps/Hz")
    
    # Analyze trends
    sum_rate_change_M3 = sum_rates_M3[-1] - sum_rates_M3[0]
    sum_rate_change_M4 = sum_rates_M4[-1] - sum_rates_M4[0]
    print(f"Sum Rate changes (Pg=0 to Pg=0.04):")
    print(f"  M=3: {sum_rate_change_M3:+.4f} bps/Hz ({'Decreasing' if sum_rate_change_M3 < 0 else 'Increasing'})")
    print(f"  M=4: {sum_rate_change_M4:+.4f} bps/Hz ({'Decreasing' if sum_rate_change_M4 < 0 else 'Increasing'})")
    print(f"Expected trend: Decreasing for both M=3 and M=4")
    
    # Find maximum and minimum sum rates
    max_idx_M3 = np.argmax(sum_rates_M3)
    min_idx_M3 = np.argmin(sum_rates_M3)
    max_idx_M4 = np.argmax(sum_rates_M4)
    min_idx_M4 = np.argmin(sum_rates_M4)
    
    print(f"M=3 - Max: {sum_rates_M3[max_idx_M3]:.4f} bps/Hz at Pg={Pg_values[max_idx_M3]:.4f}W")
    print(f"M=3 - Min: {sum_rates_M3[min_idx_M3]:.4f} bps/Hz at Pg={Pg_values[min_idx_M3]:.4f}W")
    print(f"M=4 - Max: {sum_rates_M4[max_idx_M4]:.4f} bps/Hz at Pg={Pg_values[max_idx_M4]:.4f}W")
    print(f"M=4 - Min: {sum_rates_M4[min_idx_M4]:.4f} bps/Hz at Pg={Pg_values[min_idx_M4]:.4f}W")
    
    return Pg_values, sum_rates_M3, sum_rates_M4


def main():
    """
    Main function to create verification plots.
    
    This function sets up the plotting environment and implements all 3 plots.
    """
    print("NOMA Verification Plots Script")
    print("=" * 50)
    print("Setting up matplotlib for 3 subplots...")
    
    # Set up the plotting environment
    fig, axes = setup_plots()
    
    # Test that corrected power allocation functions are accessible
    print("\nTesting import of corrected power allocation functions...")
    
    try:
        # Test basic functionality
        config = NOMAConfig(M=3, P=1.0, Pg=0.01, N0=0.001)
        h_gains = generate_channels(config)
        alpha = compute_power_allocation(h_gains, config)
        sum_rate = compute_sum_rate(h_gains, alpha, config)
        
        print(f"✓ Successfully imported and tested power allocation functions")
        print(f"  Sample channel gains: {h_gains}")
        print(f"  Sample power allocation: {alpha}")
        print(f"  Sample sum rate: {sum_rate:.4f} bps/Hz")
        
    except Exception as e:
        print(f"✗ Error testing power allocation functions: {e}")
        return
    
    # Implement Plot 1: Sum Rate vs. Transmit Power
    try:
        P_values, sum_rates_M3, sum_rates_M4 = plot_sum_rate_vs_transmit_power(axes[0])
        print("✓ Plot 1 completed successfully")
    except Exception as e:
        print(f"✗ Error creating Plot 1: {e}")
        return
    
    # Implement Plot 2: PA Coefficients vs. Power Gap
    try:
        Pg_values, alpha1_avg, alpha2_avg, alpha3_avg = plot_pa_coefficients_vs_power_gap(axes[1])
        print("✓ Plot 2 completed successfully")
    except Exception as e:
        print(f"✗ Error creating Plot 2: {e}")
        return
    
    # Implement Plot 3: Sum Rate vs. Power Gap
    try:
        Pg_values_plot3, sum_rates_M3_plot3, sum_rates_M4_plot3 = plot_sum_rate_vs_power_gap(axes[2])
        print("✓ Plot 3 completed successfully")
    except Exception as e:
        print(f"✗ Error creating Plot 3: {e}")
        return
    
    # Save the plot
    plt.savefig('noma_verification_plots.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ All plots completed and saved as 'noma_verification_plots.png'")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()