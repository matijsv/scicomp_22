import numpy as np
import matplotlib.pyplot as plt
from numba import prange, njit

# jacobi method
def jacobi_parallel(N=50, M=50, D=1.0, dt=0.0001, dx=1/50, max_iterations=50000, epsilon=1e-8):
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
    iteration : int
        Number of iterations
    delta : float
        Error at convergence
    """

    # initialize the concentration array
    c_k = np.zeros((N, M))

    # set the left and right boundary values
    left_right_boundary = np.linspace(0, 1, N)

    # stationary boundary conditions
    c_k[:, 0] = left_right_boundary
    c_k[:, -1] = left_right_boundary
    c_k[-1, :] = 1
    c_k[0, :] = 0

    # create a copy of the concentration array
    c_kp1 = np.copy(c_k)

    # default alpha is 0.25
    alpha = D * dt / (dx ** 2)

    iteration = 0
    while iteration < max_iterations:
        # update interior points
        c_kp1 = update_interior(c_k, c_kp1, alpha)
        
        # calculate the error
        delta = compute_max_diff(c_k, c_kp1)
        
        # check for convergence
        if delta < epsilon:
            break

        # swap arrays
        c_k, c_kp1 = c_kp1, c_k
        iteration += 1

    return c_k, iteration, delta

@njit(parallel=True)
def update_interior(c_k, c_kp1, alpha):
    rows, cols = c_k.shape
    for i in prange(1, rows - 1):
        for j in prange(1, cols - 1):
            c_kp1[i, j] = alpha * (
                c_k[i - 1, j] + c_k[i + 1, j] + c_k[i, j - 1] + c_k[i, j + 1]
            )
    return c_kp1

# calculate the maximum difference between two arrays
@njit(parallel=True)
def compute_max_diff(c_k, c_kp1):
    rows, cols = c_k.shape
    local_max = np.zeros(rows - 2)  # local max for each thread
    for i in prange(1, rows - 1):
        max_val = 0.0
        for j in range(1, cols - 1):
            diff = abs(c_kp1[i, j] - c_k[i, j])
            if diff > max_val:
                max_val = diff
        local_max[i - 1] = max_val  # store the local max for each thread
    return np.max(local_max)  # return the global max

# Gauss-Seidel method
@njit
def gauss_seidel_seq(N=50, M=50, D=1.0, dt=0.0001, dx=1/50, max_iterations=50000, epsilon=1e-5):
    """
    Gauss-Seidel interation for 2D diffusion equation with periodic boundary conditions.

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
    iteration : int
        Number of iterations
    delta : float
        Error at convergence
    """
    # initialize the concentration array
    c_k = np.zeros((N, M))

    # set the left and right boundary values
    left_right_boundary = np.linspace(0, 1, N)

    # stationary boundary conditions
    c_k[:, 0] = left_right_boundary
    c_k[:, -1] = left_right_boundary
    c_k[-1, :] = 1
    c_k[0, :] = 0

    # default alpha is 0.25
    alpha = D * dt / (dx ** 2)

    iteration = 0
    while iteration < max_iterations:
        delta = 0.0  # global error at each iteration
        
        # Gauss-Seidel in-place update
        for i in range(1, N - 1):
            for j in range(1, M - 1):
                old_value = c_k[i, j]
                c_k[i, j] = alpha * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, j+1] + c_k[i, j-1])
                delta = max(delta, abs(c_k[i, j] - old_value))

        # check for convergence
        if delta < epsilon:
            break

        iteration += 1

    return c_k, iteration, delta

@njit(parallel=True)
def gauss_seidel_wavefront(N=50, M=50, D=1.0, dt=0.0001, dx=1/50, max_iterations=50000, epsilon=1e-5):
    """
    Gauss-Seidel parallel implementation, known as "wave front", for 2D diffusion equation with periodic boundary conditions.

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
    iteration : int
        Number of iterations
    delta : float
        Error at convergence
    """
    # initialize the concentration array
    c_k = np.zeros((N, M))

    # set the left and right boundary values
    left_right_boundary = np.linspace(0, 1, N)

    # stationary boundary conditions
    c_k[:, 0] = left_right_boundary
    c_k[:, -1] = left_right_boundary
    c_k[-1, :] = 1
    c_k[0, :] = 0

    alpha = D * dt / (dx ** 2)

    iteration = 0
    while iteration < max_iterations:
        delta = 0.0

        # wavefront update, parallel processing
        # the wavefront is the diagonal line from right-top to left-bottom
        for wavefront in range(2, N + M - 2):  # wavefront index, recall that the first and last rows and columns are boundary, which are fixed. so we should -2 after N+M
            for i in prange(1, N - 1):  # parallel processing
                j = wavefront - i
                if 1 <= j < M - 1: # we should skip j = M-1, because it is the boundary
                    old_value = c_k[i, j]
                    c_k[i, j] = alpha * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, j+1] + c_k[i, j-1])
                    delta = max(delta, abs(c_k[i, j] - old_value))

        # check for convergence
        if delta < epsilon:
            break

        iteration += 1

    return c_k, iteration, delta

if __name__ == "__main__":
    # run the Jacobi iteration
    # optimized_concentration, iteration, delta = jacobi_parallel()
    # # check each column is the same symetrically
    # for i in range(1, optimized_concentration.shape[1] // 2):
    #     np.allclose(optimized_concentration[:, i], optimized_concentration[:, -i])
    #print("All columns are the same symetrically.")

    optimized_concentration, iteration, delta = gauss_seidel_wavefront()
    print(f"Converged after {iteration} iterations with error {delta:.6e}")
    
    # plot the heatmap of the optimized concentration
    plt.figure(figsize=(6, 4))
    plt.imshow(optimized_concentration, cmap="hot", aspect="auto", origin="upper",extent=[0, 1, 1, 0])
    plt.colorbar(label="Concentration")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Heatmap of Optimized Concentration $c(x, y)$")
    plt.gca().invert_yaxis()  # invert y-axis to match the analytical solution
    plt.show()

    #print(optimized_concentration[:, 0])
    print(optimized_concentration[:, 1])
    print(optimized_concentration[:, 2])
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