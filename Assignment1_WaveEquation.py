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

# Initial conditions
x = np.linspace(0, L, N+1)  # Discretized x-domain

initial_Psi = np.sin(2 * np.pi * x)
# initial_Psi = np.sin(5 * np.pi * x)
initial_Psi = np.where((x > 1/5) & (x < 2/5), np.sin(5 * np.pi * x), 0)

# Initialize wave function arrays
Psi_old = initial_Psi.copy()  
Psi_current = initial_Psi.copy() # dPsi/dt = 0 at t=0 
Psi_new = np.zeros(N+1)

snapshot_intervals = [0, 40, 80, 120, 160, 199] 
snapshots = []

# Time stepping function
def next_timestep():
    global Psi_old, Psi_current, Psi_new
    for i in range(1, N):  # Skip boundaries
        Psi_new[i] = 2 * Psi_current[i] - Psi_old[i] + r * (Psi_current[i+1] - 2*Psi_current[i] + Psi_current[i-1])
    
    # Update time levels
    Psi_old[:] = Psi_current[:]
    Psi_current[:] = Psi_new[:]

# Generate snapshots before animation
for t in range(timesteps):
    next_timestep()
    if t in snapshot_intervals:
        snapshots.append(Psi_current.copy())

# Plot with colors for different timesteps
fig_snap, ax_snap = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_intervals)))

for i, snap in enumerate(snapshots):
    ax_snap.plot(x, snap, label=f"Timestep {snapshot_intervals[i]}", color=colors[i])

ax_snap.set_xlabel("Position x")
ax_snap.set_ylabel("Wave Amplitude Ψ")
ax_snap.set_title("Wave Evolution at Different Timesteps")
ax_snap.legend()
plt.tight_layout()
snapshots_filename = "wave_snapshots.png"
fig_snap.savefig(snapshots_filename, dpi=300)

# Animation over time
fig_anim, ax_anim = plt.subplots(figsize=(8, 5))
line, = ax_anim.plot(x, Psi_current, label="Wave Propagation", color="blue")
ax_anim.set_ylim(-1.2, 1.2)
ax_anim.set_xlim(0, L)
ax_anim.set_xlabel("Position x")
ax_anim.set_ylabel("Wave Amplitude Ψ")
ax_anim.set_title("Wave Evolution Over Time")
ax_anim.legend()

def update(frame):
    next_timestep()
    line.set_ydata(Psi_current)
    return line,


ani = animation.FuncAnimation(fig_anim, update, frames=timesteps, interval=20, blit=True)
plt.show()
