import numpy as np
import matplotlib.pyplot as plt
from numba import prange, njit

# jacobi method
def jacobi_parallel(N=50, M=50, D=1.0, dt=0.0001, dx=1/50, max_iterations=10000, epsilon=1e-5):
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
    delta_list : list
        List of errors at each iteration
    """

    # initialize the concentration array
    c_k = np.zeros((N, M))

    # # set the left and right boundary values
    # left_right_boundary = np.linspace(0, 1, N)
    # c_k[:, 0] = left_right_boundary
    # c_k[:, -1] = left_right_boundary

    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0

    # create a copy of the concentration array
    c_kp1 = np.copy(c_k)

    # default alpha is 0.25
    alpha = D * dt / (dx ** 2)
    delta_list = np.zeros(max_iterations)

    iteration = 0
    while iteration < max_iterations:
        # update interior points
        c_kp1 = update_interior(c_k, c_kp1, alpha)
        
        # calculate the error
        delta = compute_max_diff(c_k, c_kp1)
        delta_list[iteration] = delta

        # check for convergence
        if delta < epsilon:
            break

        # swap arrays
        c_k, c_kp1 = c_kp1, c_k
        iteration += 1

    return c_k, iteration, delta, delta_list

# jacobi method
def jacobi_parallel_within_time(N=50, M=50, D=1.0, dt=0.0001, dx=1/50, max_iterations=1000, epsilon=1e-5):
    # initialize the concentration array
    c_k = np.zeros((N, M))

    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0

    # create a copy of the concentration array
    c_kp1 = np.copy(c_k)
    c_k_list = [np.copy(c_k)]

    # default alpha is 0.25
    alpha = D * dt / (dx ** 2)
    delta_list = np.zeros(max_iterations)

    iteration = 0
    while iteration < max_iterations:
        # update interior points
        c_kp1 = update_interior_time_dependent(c_k, c_kp1, alpha)
        
        # calculate the error
        delta = compute_max_diff(c_k, c_kp1)
        delta_list[iteration] = delta

        # # check for convergence
        # if delta < epsilon:
        #     break

        # swap arrays
        c_k, c_kp1 = c_kp1, c_k
        ck_copy = np.copy(c_k)
        c_k_list.append(ck_copy)
        iteration += 1

    return c_k, iteration, delta, delta_list, c_k_list

@njit(parallel=True)
def update_interior(c_k, c_kp1, alpha):
    rows, cols = c_k.shape
    for i in prange(1, rows - 1):
        # for j in prange(1, cols - 1):
        #     c_kp1[i, j] = alpha * (
        #         c_k[i - 1, j] + c_k[i + 1, j] + c_k[i, j - 1] + c_k[i, j + 1]
        #     )
        for j in range(0, cols):
            # deal with the periodic boundary conditions
            if j == 0:
                # the left side of the left boundary is the right boundary
                c_kp1[i, j] = alpha * (
                    c_k[i - 1, j] + c_k[i + 1, j] + c_k[i, -1] + c_k[i, j + 1]
                )
            elif j == cols - 1:
                # the right side of the right boundary is the left boundary
                c_kp1[i, j] = alpha * (
                    c_k[i - 1, j] + c_k[i + 1, j] + c_k[i, j - 1] + c_k[i, 0]
                )
            else:
                # normal interior points
                c_kp1[i, j] = alpha * (
                    c_k[i - 1, j] + c_k[i + 1, j] + c_k[i, j - 1] + c_k[i, j + 1]
                )
    return c_kp1

@njit(parallel=True)
def update_interior_time_dependent(c_k, c_kp1, alpha):
    rows, cols = c_k.shape
    for i in prange(1, rows - 1):
        for j in range(0, cols):
            # deal with the periodic boundary conditions
            c_k[i, j] = alpha * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, (j+1) % cols] + c_k[i, (j-1) % cols] - 4 * c_k[i, j]) + c_k[i, j]
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
def gauss_seidel_seq(N=50, M=50, D=1.0, dt=0.0001, dx=1/50, max_iterations=10000, epsilon=1e-5):
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

    # # set the left and right boundary values
    # left_right_boundary = np.linspace(0, 1, N)
    # c_k[:, 0] = left_right_boundary
    # c_k[:, -1] = left_right_boundary

    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0

    # default alpha is 0.25
    alpha = D * dt / (dx ** 2)
    delta_list = np.zeros(max_iterations)

    iteration = 0
    while iteration < max_iterations:
        delta = 0.0  # global error at each iteration
        
        # Gauss-Seidel in-place update
        for i in range(1, N - 1):
            for j in range(0, M):
                old_value = c_k[i, j]
                # deal with the periodic boundary conditions
                if j == 0:
                    # the left side of the left boundary is the right boundary
                    c_k[i, j] = alpha * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, -1] + c_k[i, j+1])
                elif j == M - 1:
                    # the right side of the right boundary is the left boundary
                    c_k[i, j] = alpha * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, j-1] + c_k[i, 0])
                else:
                    # normal interior points
                    c_k[i, j] = alpha * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, j+1] + c_k[i, j-1])
                
                delta = max(delta, abs(c_k[i, j] - old_value))
        
        delta_list[iteration] = delta
        # check for convergence
        if delta < epsilon:
            break

        iteration += 1

    return c_k, iteration, delta, delta_list

@njit(parallel=True)
def gauss_seidel_wavefront(N=50, M=50, D=1.0, dt=0.0001, dx=1/50, max_iterations=10000, epsilon=1e-5):
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

    # # set the left and right boundary values
    # left_right_boundary = np.linspace(0, 1, N)
    # c_k[:, 0] = left_right_boundary
    # c_k[:, -1] = left_right_boundary

    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0

    alpha = D * dt / (dx ** 2)
    delta_list = np.zeros(max_iterations)

    iteration = 0
    while iteration < max_iterations:
        delta = 0.0

        # wavefront update, parallel processing
        # the wavefront is the diagonal line from right-top to left-bottom
        for wavefront in range(1, N + M - 2):  # wavefront index, recall that the first and last rows are boundary, which are fixed. so we should -2 after N+M
            delta_local_list = np.zeros(N)
            for i in prange(1, N - 1):  # parallel processing
                if i > wavefront or wavefront - i >= M:
                    continue

                j = wavefront - i
                if 0 <= j <= M - 1:
                    old_value = c_k[i, j]
                    # deal with the periodic boundary conditions
                    if j == 0:
                        # the left side of the left boundary is the right boundary
                        c_k[i, j] = alpha * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, -1] + c_k[i, j+1])
                    elif j == M - 1:
                        # the right side of the right boundary is the left boundary
                        c_k[i, j] = alpha * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, j-1] + c_k[i, 0])
                    else:
                        c_k[i, j] = alpha * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, j+1] + c_k[i, j-1])

                    # reduce the maximum error
                    delta_local_list[i] = abs(c_k[i, j] - old_value)
            delta = max(delta, np.max(delta_local_list))

        delta_list[iteration] = delta
        # check for convergence
        if delta < epsilon:
            break

        iteration += 1

    return c_k, iteration, delta, delta_list

# Successive Over Relaxation (SOR) method
@njit
def sor_seq(N=50, M=50, omega=1.0, max_iterations=50000, epsilon=1e-5):
    """
    Successive Over Relaxation (SOR) interation for 2D diffusion equation with periodic boundary conditions.

    Parameters:
    -----------
    N : int
        Number of grid points in x-direction
    M : int
        Number of grid points in y-direction
    omega : float
        Relaxation parameter, default to 1.0
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

    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0

    delta_list = np.zeros(max_iterations)
    iteration = 0
    while iteration < max_iterations:
        delta = 0.0  # global error at each iteration
        
        # general SOR update
        for i in range(1, N - 1):
            for j in range(0, M):
                old_value = c_k[i, j]
                # deal with the periodic boundary conditions
                # the left side of the left boundary is the right boundary, and vice versa
                # so we can use the modulo operator to deal with the periodic boundary conditions
                c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                delta = max(delta, abs(c_k[i, j] - old_value))

        delta_list[iteration] = delta
        # check for convergence
        if delta < epsilon:
            break

        iteration += 1

    return c_k, iteration, delta, delta_list

@njit(parallel=True)
def sor_red_black(N=50, M=50, omega=1.0, max_iterations=50000, epsilon=1e-5):
    """
    Red-Black SOR iteration for 2D diffusion equation with periodic boundary conditions.

    Parameters:
    -----------
    N : int
        Number of grid points in x-direction
    M : int
        Number of grid points in y-direction
    omega : float
        Relaxation parameter, default to 1.0
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
    delta_list : 1D array
        Error at each iteration
    """
    # initialize the concentration array
    c_k = np.zeros((N, M))

    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0

    delta_list = np.zeros(max_iterations)
    iteration = 0

    while iteration < max_iterations:
        delta = 0.0  # global error at each iteration
        delta_local_list = np.zeros(N)  # local error for each thread
        # update red points
        for i in prange(1, N - 1):  
            for j in range(0, M):
                if (i + j) % 2 == 0:  # choose red points
                    old_value = c_k[i, j]
                    # deal with the periodic boundary conditions
                    c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                    delta_local_list[i] = max(delta, abs(c_k[i, j] - old_value))
        
        # update the global error
        delta = np.max(delta_local_list)

        # update black after red updated
        for i in prange(1, N - 1):  
            for j in range(0, M):
                if (i + j) % 2 == 1:  # choose black points
                    old_value = c_k[i, j]
                    c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                    delta_local_list[i] = max(delta, abs(c_k[i, j] - old_value))
        
        # update the global error
        delta = np.max(delta_local_list)
        
        # store the global error
        delta_list[iteration] = delta

        # convergence check
        if delta < epsilon:
            break

        iteration += 1

    return c_k, iteration, delta, delta_list
@njit(parallel=True)
def sor_wavefront(N=50, M=50, omega=1.0, max_iterations=10000, epsilon=1e-5):
    """
    Successive Over Relaxation (SOR) interation for 2D diffusion equation with periodic boundary conditions.
    Parallel implementation, known as "wave front".

    Parameters:
    -----------
    N : int
        Number of grid points in x-direction
    M : int
        Number of grid points in y-direction
    omega : float
        Relaxation parameter, default to 1.0
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

    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0

    iteration = 0
    delta_list = np.zeros(max_iterations)
    while iteration < max_iterations:
        delta = 0.0

        # wavefront update, parallel processing
        # the wavefront is the diagonal line from right-top to left-bottom
        for wavefront in range(1, N + M - 2):  # wavefront index, recall that the first and last rows and columns are boundary, which are fixed. so we should -2 after N+M
            delta_local_list = np.zeros(N)
            for i in prange(1, N - 1):  # parallel processing
                if i > wavefront or wavefront - i >= M:
                    continue

                j = wavefront - i
                if 0 <= j <= M - 1:
                    old_value = c_k[i, j]
                    # deal with the periodic boundary conditions
                    if j == 0:
                        # the left side of the left boundary is the right boundary
                        c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, -1] + c_k[i, j+1]) + (1 - omega) * c_k[i, j]
                    elif j == M - 1:
                        # the right side of the right boundary is the left boundary
                        c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, j-1] + c_k[i, 0]) + (1 - omega) * c_k[i, j]
                    else:
                        c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, j+1] + c_k[i, j-1]) + (1 - omega) * c_k[i, j]
                    
                    delta_local_list[i] = abs(c_k[i, j] - old_value)
            delta = max(delta, np.max(delta_local_list))
        
        delta_list[iteration] = delta

        # check for convergence
        if delta < epsilon:
            break

        iteration += 1

    return c_k, iteration, delta, delta_list

@njit
def sor_with_rect_sinks(N=50, M=50, omega=1.0, max_iterations=50000, epsilon=1e-5, sink_list=[]):
    """
    Successive Over Relaxation (SOR) iteration with absorbing rectangular obstacles (sinks).

    Parameters:
    -----------
    N, M : int
        Grid size (N x M)
    omega : float
        Relaxation parameter (SOR factor)
    max_iterations : int
        Maximum iterations
    epsilon : float
        Convergence criterion
    sink_list : list of tuples
        List of rectangular sinks defined as [(x, y, l, w), ...], where:
        - (x, y) is the top-left corner
        - l, w are the length and width of the rectangle
    
    Returns:
    --------
    c_k : 2D array
        Concentration field at convergence
    iteration : int
        Number of iterations
    delta : float
        Final error at convergence
    delta_list : 1D array
        Error history over iterations
    """
    # initialize the concentration array
    c_k = np.zeros((N, M))

    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0

    # create a mask for the obstacles
    # 0 is normal cell, 1 is sink
    object_mask = np.zeros((N, M), dtype=np.int32)

    # mark the sink locations
    for (x, y, l, w) in sink_list:
        x_end = min(x + l, N)  # ensure the sink is within the grid
        y_end = min(y + w, M)

        assert x > 0 and x < N - 1 and x_end < N - 1, "Sink x-coordinate must be within the grid"
        assert y >= 0 and y < M and y_end <= M, "Sink y-coordinate must be within the grid"

        for i in range(x, x_end):
            for j in range(y, y_end):
                object_mask[i, j] = 1  # mark the sink

    delta_list = np.zeros(max_iterations)
    iteration = 0

    while iteration < max_iterations:
        delta = 0.0  # global error at each iteration
        
        for i in range(1, N - 1):
            for j in range(0, M):
                if object_mask[i, j] == 1:
                    # skip the sink cells
                    c_k[i, j] = 0
                    continue

                old_value = c_k[i, j]
                # deal with the periodic boundary conditions
                c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                delta = max(delta, abs(c_k[i, j] - old_value))

        delta_list[iteration] = delta

        # check for convergence
        if delta < epsilon:
            break

        iteration += 1

    return c_k, iteration, delta, delta_list

import numpy as np
from numba import njit

@njit
def sor_with_rect_insulators(N=50, M=50, omega=1.0, max_iterations=5000, epsilon=1e-5, insulator_list=[]):
    """
    Successive Over Relaxation (SOR) iteration with insulating rectangular obstacles (Neumann boundary).

    Parameters:
    -----------
    N, M : int
        Grid size (N x M)
    omega : float
        Relaxation parameter (SOR factor)
    max_iterations : int
        Maximum iterations
    epsilon : float
        Convergence criterion
    insulator_list : list of tuples
        List of rectangular insulators defined as [(x, y, l, w), ...], where:
        - (x, y) is the top-left corner
        - l, w are the length and width of the rectangle
    
    Returns:
    --------
    c_k : 2D array
        Concentration field at convergence
    iteration : int
        Number of iterations
    delta : float
        Final error at convergence
    delta_list : 1D array
        Error history over iterations
    """
    # initialize the concentration array
    c_k = np.zeros((N, M))

    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0

    # create a mask for the insulators 1 is insulator, 0 is normal cell
    insulator_mask = np.zeros((N, M), dtype=np.int32)

    # mark the insulator locations
    for (x, y, l, w) in insulator_list:
        x_end = min(x + l, N)  # ensure the insulator is within the grid
        y_end = min(y + w, M)

        assert x > 0 and x < N - 1 and x_end < N - 1, "Insulator x-coordinate must be within the grid"
        assert y >= 0 and y < M and y_end <= M, "Insulator y-coordinate must be within the grid"

        for i in range(x, x_end):
            for j in range(y, y_end):
                insulator_mask[i, j] = 1  # mark the insulator

    delta_list = np.zeros(max_iterations)
    iteration = 0

    while iteration < max_iterations:
        delta = 0.0  # global error at each iteration
        
        for i in range(1, N - 1):
            for j in range(0, M):
                if insulator_mask[i, j] == 1:
                    # insulator cells, just relect the heat
                    if insulator_mask[i-1, j] == 0:  # up is normal cell
                        c_k[i, j] = c_k[i-1, j]
                    elif insulator_mask[i, j-1] == 0:  # left is normal cell
                        c_k[i, j] = c_k[i, j-1]
                    elif insulator_mask[i, j+1] == 0:  # right is normal cell
                        c_k[i, j] = c_k[i, j+1]
                    elif insulator_mask[i+1, j] == 0:  # down is normal cell
                        c_k[i, j] = c_k[i+1, j]
                    continue  # skip the insulator cells

                old_value = c_k[i, j]
                
                # deal with the periodic boundary conditions
                c_k[i, j] = omega / 4.0 * (c_k[i+1, j] + c_k[i-1, j] + c_k[i, (j+1) % M] + c_k[i, (j-1) % M]) + (1 - omega) * c_k[i, j]
                delta = max(delta, abs(c_k[i, j] - old_value))

        delta_list[iteration] = delta

        # check for convergence
        if delta < epsilon:
            break

        iteration += 1

    return c_k, iteration, delta, delta_list


if __name__ == "__main__":
    # run the Jacobi iteration
    optimized_concentration, iteration, delta, _ = sor_with_rect_sinks(sink_list=[(20, 20, 10, 10)])
    # check each column is the same symetrically
    for i in range(1, optimized_concentration.shape[1]):
        np.allclose(optimized_concentration[:, i], optimized_concentration[:, 0])
    print("All columns are the same.")
    print(f"Converged after {iteration} iterations with error {delta:.6e}")

    # plot the heatmap of the optimized concentration
    plt.figure(figsize=(6, 4))
    plt.imshow(optimized_concentration, cmap="hot", aspect="auto", origin="lower", extent=[0, 1, 1, 0])
    plt.colorbar(label="Concentration")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Heatmap of Optimized Concentration $c(x, y)$")
    plt.gca().invert_yaxis()  # invert y-axis to match the analytical solution
    plt.show()

    print(optimized_concentration[0, :])
    print(optimized_concentration[:, 0])
    print(optimized_concentration[:, 1])
    print(optimized_concentration[:, -1])
    print(optimized_concentration[:, -2])
    c_y_numerical_optimized = np.mean(optimized_concentration, axis=1)
    y_values_optimized = np.linspace(1, 0, len(c_y_numerical_optimized))

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