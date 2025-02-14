import numpy as np
# Add parent directory to path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time_indep_algorithm as tia

# Function to run SOR with different omega values
def run_sor_experiment(N_values, omega_values, tia):
    results = []

    for N in N_values:
        best_iterations = float('inf')
        best_omega = None

        for omega in omega_values:
            _, iteration_sor, delta_sor, _ = tia.sor_seq(N=N, M=N, omega=omega, max_iterations=10000)

            if iteration_sor < best_iterations:
                best_iterations = iteration_sor
                best_omega = omega

        results.append((N, best_omega, best_iterations))

    return np.array(results)

# Define grid sizes and omega values to test
# N_values = np.arange(50, 150, 10)
N_values = [50, 60, 80, 100, 120, 150, 200, 300, 500, 1000]
omega_values = np.linspace(1.7, 2.0, 10)  # Search omega in [1.7, 2.0]

# Run the experiment
results = run_sor_experiment(N_values, omega_values, tia)
print(results)
