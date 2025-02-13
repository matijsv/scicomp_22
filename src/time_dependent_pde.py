import numpy as np
from numba import njit, prange
from scipy.special import erfc

# Eq is dc/dt = D ( d^2 c / dx + d^2 c / dy)
# discretized version:
@njit(parallel= True)
def update_interior(c_k, Ddt_divby_dxsq):
    """
    Updates the matrix according to the discretized time
    dependent diffusion equation with a 5-point stencil.

    Parameters
    ----------
    c_k : 2DArray
        Initial array of concentrations.
    Ddt_divby_dxsq : Float
        Coefficient present in the time-dependent
        discretized diffusion equation.

    Returns
    -------
    2DArray
        Array stepped forward by one dt
    """
    rows, cols = c_k.shape
    c_k_new = np.copy(c_k) # initialize forward copy
    for i in prange(1, rows - 1): #skip top and bottom rows (constants)
        for j in range(0,cols):
            
            # this could be done with % operator but I am too scared of it
            if j == 0: # left periodic boundary condition
                left, right = c_k[i,-1], c_k[i,j+1]
            elif j == cols - 1: # right periodic boundary condition
                left, right = c_k[i,j-1], c_k[i,0]
            else:
                left, right = c_k[i,j-1], c_k[i,j+1]
            
            c_k_new[i,j] = c_k[i,j] + Ddt_divby_dxsq*(
                left + right + c_k[i-1, j] + c_k[i+1, j] - 4*c_k[i,j]
            )
    return c_k_new
    
    
def time_dependent_algorithm(N=50, M=50, D=1.0, dt=0.0001, t_max=1):
    """
    Runs the time-dependent diffusion equation on an NxM grid
    0 to t_max with step size of dt.

    Parameters
    ----------
    N : int, optional
        Number of grid rows, by default 50
    M : int, optional
        Number of grid collumns, by default 50
    D : float, optional
        Diffusion coefficient, by default 1.0
    dt : float, optional
        Size of time steps, by default 0.0001
    t_max : int, optional
        Maximum time step, by default 1

    Returns
    -------
    List of 2DArrays
        A list containing the array at every time step
    """
    results_list = []
    # initialize the concentration array
    c_k = np.zeros((N, M))
    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0
    
    dx = 1/N # dx = dy
    Ddt_divby_dxsq =  D * dt / (dx ** 2)
    
    print('4 * Coefficient in eq 7 (max 1):', 4*Ddt_divby_dxsq)
    #assert 4*Ddt_divby_dxsq <= 1 , f"4*Ddt_divby_dxsq > 1 : {4*Ddt_divby_dxsq}, will be unstable."
    
    results_list.append(c_k) # add initial array to results list
    t = 0
    while t <= t_max:
        c_k = update_interior(c_k, Ddt_divby_dxsq)
        results_list.append(c_k)
        t += dt
        
        # Print progress bar
        progress = t / t_max
        bar_length = 40
        block = int(round(bar_length * progress))
        bar = "#" * block + "-" * (bar_length - block)
        print(f"\rProgress: [{bar}] {progress * 100:.2f}%", end="")

    return results_list

def analytical_solution(x, t, D, max_terms=50):
    """
    Compute the analytical solution for the diffusion equation.
    
    Parameters:
    - x: spatial coordinate (numpy array)
    - t: time (scalar)
    - D: diffusion constant
    - max_terms: number of terms in the summation (truncated at a reasonable value)
    
    Returns:
    - c_x_t: analytical solution at given x and t
    """
    if t == 0:
        return np.zeros_like(x)  # Initial condition assumed to be zero
    
    sum_term = np.zeros_like(x)
    for i in range(max_terms):
        term1 = erfc((1 - x + 2 * i) / (2 * np.sqrt(D * t)))
        term2 = erfc((1 + x + 2 * i) / (2 * np.sqrt(D * t)))
        sum_term += term1 - term2
    
    return sum_term


if __name__ == "__main__":
    dt= 0.00005
    results_list = time_dependent_algorithm(dt=dt, t_max=1)
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    #plots for 0.001, 0.01, 0.1, and 1
    times = [0.001, 0.01, 0.1, 1]
    indexes = [int((1/dt)*time) for time in times]
    
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    for i, index in enumerate(indexes):
        exact_sol = analytical_solution(np.linspace(0,1,50),times[i],1)
        exact_sol = exact_sol[::-1]
        ax.plot(results_list[index][:, 25], label = f'T = {times[i]}')
        
        mse = np.mean((results_list[index][:, 25] - exact_sol) ** 2)
        ax.plot(exact_sol, 'kx', label=f'MSE at T = {times[i]}: {mse:.2e}')
        
        
    ax.set_title('Concentration c(y) at different times')
    ax.set_xlabel("y")
    ax.set_ylabel("c")
    ax.invert_xaxis()
    ax.legend()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(results_list[0], cmap="hot", aspect="auto", origin="lower", extent=[0, 1, 1, 0])
    fig.colorbar(cax, label="Concentration")
    ax.set_title('Concentration $c(x,y)')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()  # invert y-axis to match the analytical solution
    
    N_FRAMES = 101
    frames = np.linspace(0, len(results_list)-1, N_FRAMES).astype(int)
    def update_frame(frame):
        index = frames[frame]
        cax.set_array(results_list[index])
        ax.set_title(f"Concentration $c(x,y), time: {dt*index:.4f}")
        print(index)
        return cax

    ani = animation.FuncAnimation(fig, update_frame, frames=N_FRAMES, interval=50, blit=False)

    ani.save('concentration_evolution.mp4', writer='ffmpeg')
    plt.show()
    #0.00001, 0.00002, 0.00003