import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def initialize_simulation(L, N, dt, c, init='1'):
    """Initialize the wave simulation parameters."""
    dx = L / N  # Spatial step size
    r = (c * dt / dx) ** 2  # Stability condition
    x = np.linspace(0, L, N+1)  # Discretized x-domain

    # Initial conditions
    if init == '1':
        initial_Psi = np.sin(2 * np.pi * x) 
    elif init == '2':
        initial_Psi = np.sin(5 * np.pi * x)  
    elif init == '3':
        initial_Psi = np.where((x > 1/5) & (x < 2/5), np.sin(5 * np.pi * x), 0)

    return dx, r, x, initial_Psi

def next_timestep(Psi_old, Psi_current, Psi_new, N, r):
    """Compute the next timestep using the wave equation."""
    for i in range(1, N):  # Skip boundaries (fixed boundary conditions)
        Psi_new[i] = 2 * Psi_current[i] - Psi_old[i] + r * (Psi_current[i+1] - 2 * Psi_current[i] + Psi_current[i-1])

    # Update time levels
    Psi_old[:] = Psi_current[:]
    Psi_current[:] = Psi_new[:]

def animate_wave(N, timesteps, x, Psi_old, Psi_current, Psi_new, r, L, snapshot_intervals, mainfile=False):
    """Create an animation of the wave evolution and capture snapshots efficiently."""
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot(x, Psi_current, label="Wave Propagation", color="blue")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(0, L)
    ax.set_xlabel("Position x")
    ax.set_ylabel(r"Wave Amplitude $\Psi$")
    ax.set_title("Wave Evolution Over Time")
    ax.legend()

    # List to store snapshots
    snapshots = {}

    def update(frame):
        if frame >= timesteps - 1:
            ani.event_source.stop()  

        next_timestep(Psi_old, Psi_current, Psi_new, N, r)
        line.set_ydata(Psi_current)

        # Store snapshot at specified intervals
        if frame in snapshot_intervals:
            snapshots[frame] = Psi_current.copy()

        return line,

    ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=20, blit=True)

    if mainfile:
        plt.show()

    plt.close()
    
    return ani, snapshots  # Return both the animation and collected snapshots

def plot_snapshots(x, snapshots, snapshot_intervals):
    """Plot the wave at different timesteps using different colors."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_intervals)))

    for i, t in enumerate(snapshot_intervals):
        if t in snapshots:
            ax.plot(x, snapshots[t], label=f"Timestep {t}", color=colors[i])

    ax.set_xlabel("Position x")
    ax.set_ylabel(r"Wave Amplitude $\Psi$")
    ax.set_title("Wave Evolution at Different Timesteps")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Parameters
    L = 1  # Length of the string
    N = 100  # Number of intervals
    dt = 0.001  # Time step
    c = 1  # Wave speed
    timesteps = 200  # Number of time steps
    snapshot_intervals = [0, 40, 80, 120, 160, 199]  # Selected timesteps for snapshots

    # Initialize the simulation
    dx, r, x, initial_Psi = initialize_simulation(L, N, dt, c)

    # Initialize wave function arrays
    Psi_old = initial_Psi.copy()
    Psi_current = initial_Psi.copy()  # dPsi/dt = 0 at t=0
    Psi_new = np.zeros(N+1)

    # Run animation and collect snapshots
    ani, snapshots = animate_wave(N, timesteps, x, Psi_old, Psi_current, Psi_new, r, L, snapshot_intervals, mainfile=True)

    # Plot snapshots after the animation
    plot_snapshots(x, snapshots, snapshot_intervals)
