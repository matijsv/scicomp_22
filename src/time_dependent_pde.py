import numpy as np
from numba import njit, prange


# Eq is dc/dt = D ( d^2 c / dx + d^2 c / dy)
# discretized version:
@njit(parallel= True)
def update_interior(c_k, c_k_new, Ddt_divby_dxsq):
    rows, cols = c_k.shape
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
    
    
def time_dependent_algorithm(N=50, M=50, D=1.0, dt=0.0001, t_max=0.1):
    
    results_list = []
    # initialize the concentration array
    c_k = np.zeros((N, M))
    # stationary boundary conditions
    c_k[0, :] = 1
    c_k[-1, :] = 0
    
    dx = 1/N # dx = dy
    Ddt_divby_dxsq =  D * dt / (dx ** 2)
    
    #print('4 * Coefficient in eq 7 (max 1):', 4*Ddt_divby_dxsq)
    assert 4*Ddt_divby_dxsq <= 1 , f"4*Ddt_divby_dxsq > 1 : {4*Ddt_divby_dxsq}, will be unstable."
    
    t = 0
    while t < t_max:
        c_k_new = np.copy(c_k) # initialize forward copy
        c_k = update_interior(c_k, c_k_new, Ddt_divby_dxsq)
        results_list.append(c_k)
        t += dt
        
        # Print progress bar
        progress = t / t_max
        bar_length = 40
        block = int(round(bar_length * progress))
        bar = "#" * block + "-" * (bar_length - block)
        print(f"\rProgress: [{bar}] {progress * 100:.2f}%", end="")

    return results_list

if __name__ == "__main__":
    dt= 0.0001
    results_list = time_dependent_algorithm(dt=dt, t_max=1)
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(results_list[0], cmap="hot", aspect="auto", origin="lower", extent=[0, 1, 1, 0])
    fig.colorbar(cax, label="Concentration")
    ax.set_title('Concentration $c(x,y)')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()  # invert y-axis to match the analytical solution
    
    N_FRAMES = 101
    frames = np.linspace(0, len(results_list), N_FRAMES).astype(int)
    def update_frame(frame):
        index = frames[frame]
        cax.set_array(results_list[index])
        ax.set_title(f"Concentration $c(x,y), time: {dt*index:.4f}")
        return cax, #ax

    ani = animation.FuncAnimation(fig, update_frame, frames=N_FRAMES-1, interval=50, blit=False)

    ani.save('concentration_evolution.mp4', writer='ffmpeg')

    plt.show()