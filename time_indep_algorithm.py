import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def jacobi_diffusion_vectorized(N=50, M=50, D=1.0, dt=0.0001, dx=1/50, max_iterations=50000, epsilon=1e-7):
    """
    Jacobi interation for 2D diffusion equation with periodic boundary conditions.
    This is for Elliptic PDE: ∂c/∂t = D(∂^2c/∂x^2 + ∂^2c/∂y^2)

    Parameters:
    -----------
    N : int
        Number of grid points in x-direction
    M : int
        Number of grid points in y-direction
    D : float
        Diffusion coefficient, default to 1.0
    dt : float
        Time step size
    dx : float
        Grid spacing
    max_iterations : int
        Maximum number of iterations
    epsilon : float
        Convergence criterion

    Returns:
    --------
    c_k : 2D array
        Concentration distribution at time step k
    """

    c_k = np.zeros((N, M))
    c_kp1 = np.zeros_like(c_k)

    # set the left and right boundary values
    left_right_boundary = np.linspace(0, 1, N)
    print(left_right_boundary)

    # stationary boundary conditions
    c_k[:, 0] = left_right_boundary
    c_k[:, -1] = left_right_boundary
    c_k[-1, :] = 1
    c_k[0, :] = 0

    c_kp1[:, 0] = left_right_boundary
    c_kp1[:, -1] = left_right_boundary
    c_kp1[-1, :] = 1
    c_kp1[0, :] = 0

    # default alpha is 0.25
    alpha = D * dt / (dx ** 2)

    iteration = 0
    while iteration < max_iterations:
        # update interior points
        # # add a 0 column at the beginning and end of the array
        # left_shifted = np.pad(c_k[1:-1, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=0)  # pad left with a 0 column
        # right_shifted = np.pad(c_k[1:-1, 1:], ((0, 0), (0, 1)), mode='constant', constant_values=0)  # pad right with a 0 column

        c_kp1[1:-1, 1:-1] = alpha * (
            c_k[:-2, 1:-1] + c_k[2:, 1:-1] + c_k[1:-1, :-2] + c_k[1:-1, 2:]
        )

        # calculate the error
        delta = np.max(np.abs(c_kp1 - c_k))

        # check for convergence
        if delta < epsilon:
            break

        # swap arrays
        c_k, c_kp1 = c_kp1, c_k

        iteration += 1

    print(f"Converged after {iteration} iterations with error {delta:.6e}")
    return c_k

# run the Jacobi iteration
optimized_concentration = jacobi_diffusion_vectorized()

# plot the heatmap of the optimized concentration
plt.figure(figsize=(6, 4))
plt.imshow(optimized_concentration, cmap="hot", aspect="auto", extent=[0, 1, 0, 1])
plt.colorbar(label="Concentration")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Heatmap of Optimized Concentration $c(x, y)$")
plt.gca().invert_yaxis()  # invert y-axis to match the analytical solution
plt.show()


# calculate the numerical solution
print(optimized_concentration[:, 0])
print(optimized_concentration[:, -2])
c_y_numerical_optimized = np.mean(optimized_concentration, axis=1)
y_values_optimized = np.linspace(0, 1, len(c_y_numerical_optimized))

# calculate the maximum error
max_error_optimized = np.max(np.abs(c_y_numerical_optimized - y_values_optimized))
print(f"Maximum error: {max_error_optimized:.6e}")

# plot the numerical solution
plt.figure(figsize=(6, 4))
plt.plot(y_values_optimized, c_y_numerical_optimized, label="Optimized Numerical Solution (Jacobi)", linestyle="--", marker="o")
plt.plot(y_values_optimized, y_values_optimized, label="Analytical Solution $c(y) = y$", linestyle="-", color="r")
plt.xlabel("y")
plt.ylabel("Concentration $c(y)$")
plt.legend()
plt.title("Comparison of Optimized Numerical and Analytical Solution (Periodic BC)")
plt.grid(True)
plt.show()
