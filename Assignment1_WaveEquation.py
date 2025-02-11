import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 1  # Length of the string
N = 100  # Number of intervals
dx = L / N  # Spatial step size
dt = 0.001  # Time step
c = 1  # Wave speed
timesteps = 200  # Number of time steps

# Stability condition (CFL condition)
r = (c * dt / dx) ** 2  # Must be <= 1 for stability
print(r)

# Initial conditions
x = np.linspace(0, L, N+1)

# Case I, II, III
#initial_Psi = np.sin(2 * np.pi * x)
#initial_Psi = np.sin(5 * np.pi * x)
initial_Psi = np.where((x > 1/5) & (x < 2/5), np.sin(5 * np.pi * x), 0)

# Wave function arrays at j-1, j and j+1 
Psi_old = initial_Psi.copy()  
Psi_current = Psi_old.copy()
Psi_new = np.zeros(N+1)

# Time stepping function
def next_timestep():
    global Psi_old, Psi_current, Psi_new
    for i in range(1, N):  # Skip boundaries (fixed at 0)
        Psi_new[i] = 2 * Psi_current[i] - Psi_old[i] + r * (Psi_current[i+1] - 2*Psi_current[i] + Psi_current[i-1])
    
    Psi_old[:] = Psi_current[:]
    Psi_current[:] = Psi_new[:]


# Animation
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(x, Psi_current, label="Wave Propagation", color="blue")
ax.set_ylim(-1.2, 1.2)
ax.set_xlim(0, L)
ax.set_xlabel("Position x")
ax.set_ylabel(r"Wave Amplitude $\Psi$")
ax.legend()

def update(frame):
    next_timestep()
    line.set_ydata(Psi_current)
    return line,

ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=20, blit=True)

plt.show()
