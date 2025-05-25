import cupy as cp
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from tqdm import tqdm

# Create frames directory
os.makedirs('frames', exist_ok=True)
os.makedirs('thrust_frames', exist_ok=True)

# Constants
L = cp.float64(2.25)  # Length of domain
z_start = cp.float64(-0.075)  # Start of z-axis
R_max = cp.float64(0.075)  # Maximum radial extent
nz = 630  # Number of grid points in z
nr = 21   # Number of grid points in r
dz = cp.float64((L - z_start) / (nz - 1))  # Grid spacing in z
dr = cp.float64(R_max / (nr - 1))  # Grid spacing in r
z = cp.linspace(z_start, L, nz, dtype=cp.float64)
r = cp.linspace(0, R_max, nr, dtype=cp.float64)
Z, R = cp.meshgrid(z, r, indexing='ij')
r_wall_inlet = 0.05  # free end of petal valve (m)

j_wall = cp.searchsorted(r, r_wall_inlet, side='left').item()  # Index of the inlet in the r array
u_wind = cp.float64(20.0)  # Constant z-wind velocity (m/s)
r_exhaust = cp.float64(0.03)  # Exhaust exit radius (m)

# Physical constants
gamma = cp.float64(1.4)  # Specific heat ratio
R_gas = cp.float64(287)  # Gas constant (J/kg·K)
mu = cp.float64(1.8e-5)  # Dynamic viscosity (kg/m·s)
k_fluid = cp.float64(0.026)  # Thermal conductivity (W/m·K)
D_i = cp.float64(2e-5)  # Diffusion coefficient (m²/s)
cv = R_gas / (gamma - 1)  # Specific heat at constant volume
cp_const = gamma * cv  # Specific heat at constant pressure
qr = cp.float64(46.3e6)  # Heat of reaction (J/kg)
A = cp.float64(4e12)  # Pre-exponential factor (1/s)
Ea = cp.float64(1.22e8)  # Activation energy (J/kmol)
R_u = cp.float64(8314)  # Universal gas constant (J/kmol·K)
M_fuel = cp.float64(44)  # Molar mass of fuel (kg/kmol)
M_O2 = cp.float64(32)  # Molar mass of oxygen (kg/kmol)
X_LFL = cp.float64(0.0237)
X_UFL = cp.float64(0.095)

Yf_LFL = X_LFL * M_fuel / (X_LFL * M_fuel + (1 - X_LFL) * M_O2)
Yf_UFL = X_UFL * M_fuel / (X_UFL * M_fuel + (1 - X_UFL) * M_O2)

# Initial conditions (sea-level air)
rho0 = cp.float64(1.225)  # Density (kg/m³)
T0 = cp.float64(300)  # Temperature (K)
p0 = rho0 * R_gas * T0  # Pressure (Pa)
E0 = cv * T0  # Specific internal energy (J/kg)
Yf0 = cp.float64(0.06)  # Fuel mass fraction inside
Yo0 = cp.float64(0.23)  # Oxygen mass fraction in air
Yp0 = cp.float64(0)  # Product mass fraction

# Time parameters
dt = cp.float64(1e-8)  # Time step (s)
t_max = cp.float64(0.02)  # Total simulation time (s)
n_steps = int(t_max / dt)  # Number of time steps

# Pulsejet geometry (walls from z=0 to z=1.5)
def pulsejet_wall(z):
    r_wall = cp.full_like(z, cp.inf, dtype=cp.float64)
    mask1 = (0 <= z) & (z <= 0.5)
    r_wall[mask1] = 0.05
    mask2 = (0.5 < z) & (z <= 0.7)
    r_wall[mask2] = 0.05 - (z[mask2] - 0.5) * (0.05 - 0.03) / 0.2
    mask3 = (0.7 < z) & (z <= 1.5)
    r_wall[mask3] = 0.03
    return r_wall

i_valve = cp.searchsorted(z, 0, side='left').item()  # z = 0

# Wall mask (including 'inlet' at z=0 as a solid wall)
r_wall = pulsejet_wall(z)
r_wall_arr = r_wall[:, None]
wall_mask = cp.zeros((nz, nr), dtype=cp.bool_)
j_wall_indices = cp.searchsorted(r, r_wall, side='left')
mask_z = (0 <= z) & (z <= 1.5)
valid_indices = mask_z & (j_wall_indices < nr)
wall_mask[valid_indices, j_wall_indices[valid_indices]] = True
wall_mask[i_valve, :j_wall + 1] = True

chamber_idx = (Z > 0) & (Z < 0.5) & (R < r_wall_arr)
r_inlet = cp.linspace(-r_wall_inlet, r_wall_inlet, 100)
z_inlet = cp.zeros_like(r_inlet)

# Functions
def compute_thermo_properties(U):
    rho = U[:, :, 0]
    E = U[:, :, 3] / (rho + 1e-10)
    ur = U[:, :, 1] / (rho + 1e-10)
    uz = U[:, :, 2] / (rho + 1e-10)
    Yf = U[:, :, 4] / (rho + 1e-10)
    Yo = U[:, :, 5] / (rho + 1e-10)
    Yp = U[:, :, 6] / (rho + 1e-10)
    M_prod = cp.float64(28.0)
    R_gas_mix = R_u * cp.clip((Yf / M_fuel + Yo / M_O2 + Yp / M_prod), 0, None)
    cv_mix = R_gas_mix / (gamma - 1)
    T = (E - 0.5 * (ur**2 + uz**2)) / cv_mix
    T = cp.clip(T, 1.0, None)
    p = rho * R_gas_mix * T
    return T, p, R_gas_mix

def combustion_source(U):
    rho = U[:, :, 0]
    Yf = U[:, :, 4] / (rho + 1e-10)
    Yo = U[:, :, 5] / (rho + 1e-10)
    T, p, R_gas_mix = compute_thermo_properties(U)
    T_ign = cp.clip(800 - 20 * (p / p0 - 1), 600, None)
    can_burn = (T >= T_ign) & (Yf >= Yf_LFL) & (Yf <= Yf_UFL)
    C_f = rho * Yf / M_fuel
    C_o2 = rho * Yo / M_O2
    k_arr = A * cp.exp(-Ea / (R_u * T))
    Rf_mol = cp.where(can_burn, k_arr * C_f * C_o2, 0.0)
    Ro_mol = 5 * Rf_mol
    Rf_mass = -Rf_mol * M_fuel
    Ro_mass = -Ro_mol * M_O2
    Rp_mass = -(Rf_mass + Ro_mass)
    S = cp.zeros_like(U)
    S[:, :, 4] = Rf_mass
    S[:, :, 5] = Ro_mass
    S[:, :, 6] = Rp_mass
    S[:, :, 3] = -Rf_mass * qr
    S[wall_mask, :] = 0
    return S, Rf_mass

def compute_fluxes(U, p):
    rho = U[:, :, 0]
    ur = U[:, :, 1] / (rho + 1e-10)
    uz = U[:, :, 2] / (rho + 1e-10)
    Fz = cp.zeros_like(U)
    Fr = cp.zeros_like(U)
    Fz[:, :, 0] = U[:, :, 2]  # rho * uz
    Fr[:, :, 0] = U[:, :, 1]  # rho * ur
    Fz[:, :, 1] = U[:, :, 1] * uz
    Fr[:, :, 1] = U[:, :, 1] * ur + p
    Fz[:, :, 2] = U[:, :, 2] * uz + p
    Fr[:, :, 2] = U[:, :, 2] * ur
    Fz[:, :, 3] = (U[:, :, 3] + p) * uz
    Fr[:, :, 3] = (U[:, :, 3] + p) * ur
    for i in range(4, 7):
        Fz[:, :, i] = U[:, :, i] * uz
        Fr[:, :, i] = U[:, :, i] * ur
    return Fz, Fr

def viscous_fluxes(U, T):
    rho = U[:, :, 0]
    ur = U[:, :, 1] / (rho + 1e-10)
    uz = U[:, :, 2] / (rho + 1e-10)
    dur_dz = cp.gradient(ur, dz, axis=0)
    dur_dr = cp.gradient(ur, dr, axis=1)
    duz_dz = cp.gradient(uz, dz, axis=0)
    duz_dr = cp.gradient(uz, dr, axis=1)
    dT_dz = cp.gradient(T, dz, axis=0)
    dT_dr = cp.gradient(T, dr, axis=1)
    R_safe = R + 1e-10
    tau_rz = mu * (dur_dz + duz_dr)
    div_v = dur_dr + ur / R_safe + duz_dz
    tau_zz = mu * (2 * duz_dz - 2 / 3 * div_v)
    tau_rr = mu * (2 * dur_dr - 2 / 3 * div_v)
    Fz_visc = cp.zeros_like(U)
    Fr_visc = cp.zeros_like(U)
    Fz_visc[:, :, 1] = tau_rz
    Fz_visc[:, :, 2] = tau_zz
    Fr_visc[:, :, 1] = tau_rr
    Fr_visc[:, :, 2] = tau_rz
    Fz_visc[:, :, 3] = tau_zz * uz + tau_rz * ur - k_fluid * dT_dz
    Fr_visc[:, :, 3] = tau_rz * uz + tau_rr * ur - k_fluid * dT_dr
    Fz_visc[wall_mask, :] = 0
    Fr_visc[wall_mask, :] = 0
    return Fz_visc, Fr_visc

E_max = rho0 * cv * 5000.0

def apply_boundary_conditions(U):
    T, p, R_gas_mix = compute_thermo_properties(U)
    rho = U[:, :, 0]
    ur = U[:, :, 1] / (rho + 1e-10)
    uz = U[:, :, 2] / (rho + 1e-10)
    U[i_valve, :, 1:3] = 0  # Zero radial and axial velocities
    U[i_valve, :, 0] = U[i_valve - 1, :, 0]  # Density from adjacent point
    U[i_valve, :, 3] = U[i_valve - 1, :, 3]  # Energy from adjacent point
    U[i_valve, :, 4:7] = U[i_valve - 1, :, 4:7]  # Species from adjacent point
    U[:, 0, 1] = 0
    for i in [0, 2, 3, 4, 5, 6]:
        U[:, 0, i] = U[:, 1, i]
    U[-1, :, :] = U[-2, :, :]  # Outflow
    U[0, :, :] = U[1, :, :]  # Outflow
    U[:, -1, :] = U[:, -2, :]  # Outflow
    U[:, :, 0] = cp.clip(U[:, :, 0], 1e-6, None)
    U[:, :, 1] = cp.clip(U[:, :, 1], -1e3, 1e3)
    U[:, :, 2] = cp.clip(U[:, :, 2], -1e3, 1e3)
    U[:, :, 3] = cp.clip(U[:, :, 3], 0, E_max)
    species_sum = U[:, :, 4] + U[:, :, 5] + U[:, :, 6]
    zero_species = species_sum < 1e-10
    U[zero_species, 4] = 0
    U[zero_species, 5] = U[zero_species, 0] * Yo0
    U[zero_species, 6] = 0
    U[:, :, 4:] = cp.clip(U[:, :, 4:], 0, U[:, :, 0][:, :, None])
    U = cp.where(cp.isfinite(U), U, cp.array([rho0, 0, 0, rho0 * E0, 0, rho0 * Yo0, 0])[None, None, :])
    return U

def compute_cfl(U, dt, dz, dr):
    rho = U[:, :, 0]
    ur = U[:, :, 1] / (rho + 1e-10)
    uz = U[:, :, 2] / (rho + 1e-10)
    T, p, R_gas_mix = compute_thermo_properties(U)
    c = cp.sqrt(gamma * R_gas_mix * T)  # Speed of sound
    max_speed_z = cp.max(cp.abs(uz) + c)
    max_speed_r = cp.max(cp.abs(ur) + c)
    cfl_z = dt * max_speed_z / dz
    cfl_r = dt * max_speed_r / dr
    return max(cfl_z, cfl_r)

# Initialization
U_ambient = cp.zeros((nz, nr, 7), dtype=cp.float64)
U_ambient[:, :, 0] = rho0
U_ambient[:, :, 1] = 0
U_ambient[:, :, 2] = 0
U_ambient[:, :, 3] = rho0 * E0
U_ambient[:, :, 4] = 0
U_ambient[:, :, 5] = rho0 * Yo0
U_ambient[:, :, 6] = 0

U = U_ambient.copy()

outside = (R >= r_wall_arr) & (Z >= 0)
reset_vals = ((R >= r_wall_arr) & (Z <= 1.5) & (Z >= 0)) | ((Z >= z_start) & (Z <= 0) & (R <= 0.075))

spark_profile = cp.exp(-((Z - 0.15)**2 / 0.02**2 + R**2 / 0.02**2))
spark_profile[outside] = 0
U[:, :, 3] = rho0 * cv * (300 + 1200 * spark_profile)

fuel_profile_in = cp.exp(-((Z - 0.15)**2 / 0.03**2 + R**2 / 0.03**2))
fuel_profile_in[outside] = 0
U[:, :, 4] = rho0 * Yf0 * fuel_profile_in

exit_zi = cp.searchsorted(z, 1.5, side='left').item()
exit_ri = int(nr * (0.03 / R_max))

# Simulation loop
progress_interval = max(1, n_steps // 100)

with tqdm(total=n_steps, desc="\n", ncols=80) as pbar:
    time_ = []
    thrust_ = []

    for n in range(n_steps):
        T, p, R_gas_mix = compute_thermo_properties(U)
        R_safe = R[:, :, None] + 1e-10
        Fz, Fr = compute_fluxes(U, p)
        Fz_visc, Fr_visc = viscous_fluxes(U, T)
        dFz_dz = cp.gradient(Fz, dz, axis=0)
        dFr_dr = cp.gradient(Fr, dr, axis=1)
        dFz_visc_dz = cp.gradient(Fz_visc, dz, axis=0)
        dFr_visc_dr = cp.gradient(Fr_visc, dr, axis=1)
        Fr_over_r = cp.zeros_like(Fr)
        Fr_over_r[:, 1:, 0] = Fr[:, 1:, 0] / R_safe[:, 1:, 0]
        Fr_over_r[:, 1:, 1] = (Fr[:, 1:, 1] - p[:, 1:]) / R_safe[:, 1:, 0]
        Fr_over_r[:, 1:, 2] = Fr[:, 1:, 2] / R_safe[:, 1:, 0]
        Fr_over_r[:, 1:, 3] = Fr[:, 1:, 3] / R_safe[:, 1:, 0]
        Fr_over_r[:, 1:, 4:7] = Fr[:, 1:, 4:7] / R_safe[:, 1:, :]
        S, Rf = combustion_source(U)
        update = dt * (dFz_dz + dFr_dr + Fr_over_r - dFz_visc_dz - dFr_visc_dr - S)
        U_new = U - update
        U = apply_boundary_conditions(U_new)
        U[reset_vals, :] = U_ambient[reset_vals, :]
        U[reset_vals, 2] = U[reset_vals, 0] * u_wind
        U[reset_vals, 3] = U[reset_vals, 0] * (E0 + (u_wind**2) / 2)
        cfl = compute_cfl(U, dt, dz, dr)
        if cfl > 1:
            sug_dt = dt / cfl
            print(f" Warning: CFL = {cfl:.2f} > 1 at step {n}, reduce dt to {(sug_dt):.2e} or less")
        # Thrust
        p_exit = p[exit_zi, :exit_ri]
        uz_exit = U[exit_zi, :exit_ri, 2] / U[exit_zi, :exit_ri, 0]
        exit_area = cp.pi * 0.03**2
        thrust_cp = exit_area * ((uz_exit * U[exit_zi, :exit_ri, 2]).mean() + (p_exit.mean() - p0))
        thrust = thrust_cp.item()
        thrust_.append(thrust)
        time_.append(n * dt)
        if n % 50 == 0:
            if cp.any(~cp.isfinite(T)):
                print(f"Invalid temperature at step {n}")
                break
            Yf = U[:, :, 4] / (U[:, :, 0] + 1e-10)
            Yo = U[:, :, 5] / (U[:, :, 0] + 1e-10)
            rho = U[:, :, 0]
            # Visualization
            X = cp.concatenate([Z, Z], axis=1)
            Y = cp.concatenate([-R[:, ::-1], R], axis=1)
            T_mirror = cp.concatenate([T[:, ::-1], T], axis=1)
            p_mirror = cp.concatenate([p[:, ::-1], p], axis=1)
            rho_mirror = cp.concatenate([rho[:, ::-1], rho], axis=1)
            # Transfer to CPU for plotting
            X_cpu = X.get()
            Y_cpu = Y.get()
            T_mirror_cpu = T_mirror.get()
            p_mirror_cpu = p_mirror.get()
            rho_mirror_cpu = rho_mirror.get()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(64, 6), sharey=True)
            c1 = ax1.contourf(X_cpu, Y_cpu, T_mirror_cpu, levels=20, vmin=0, vmax=5000, cmap='inferno')
            plt.colorbar(c1, ax=ax1, label='Temperature (K)')
            ax1.plot(z.get(), r_wall.get(), 'w-', linewidth=2)
            ax1.plot(z.get(), -r_wall.get(), 'w-', linewidth=2)
            ax1.plot(z_inlet.get(), r_inlet.get(), 'w-', linewidth=2)
            ax1.set_xlabel('z (m)')
            ax1.set_ylabel('r (m)')
            ax1.set_title(f'Temperature at t={n*dt:.9f} s')
            ax1.set_aspect('equal')
            ax1.set_ylim(-R_max, R_max)
            ax1.set_xlim(z_start, L)
            c2 = ax2.contourf(X_cpu, Y_cpu, p_mirror_cpu, levels=20, vmin=0.1*p0, vmax=50*p0, cmap='viridis')
            plt.colorbar(c2, ax=ax2, label='Pressure (Pa)')
            ax2.plot(z.get(), r_wall.get(), 'w-', linewidth=2)
            ax2.plot(z.get(), -r_wall.get(), 'w-', linewidth=2)
            ax2.plot(z_inlet.get(), r_inlet.get(), 'w-', linewidth=2)
            ax2.set_xlabel('z (m)')
            ax2.set_title(f'Pressure at t={n*dt:.9f} s')
            ax2.set_aspect('equal')
            ax2.set_xlim(z_start, L)
            c3 = ax3.contourf(X_cpu, Y_cpu, rho_mirror_cpu, levels=20, vmin=0.1*rho0, vmax=50*rho0, cmap='Greys_r')
            plt.colorbar(c3, ax=ax3, label='Density (kg/m³)')
            ax3.plot(z.get(), r_wall.get(), 'w-', linewidth=2)
            ax3.plot(z.get(), -r_wall.get(), 'w-', linewidth=2)
            ax3.plot(z_inlet.get(), r_inlet.get(), 'w-', linewidth=2)
            ax3.set_xlabel('z (m)')
            ax3.set_title(f'Density at t={n*dt:.9f} s')
            ax3.set_aspect('equal')
            ax3.set_xlim(z_start, L)
            plt.tight_layout()
            plt.savefig(f'frames/frame_t={n*dt:09f}.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            if n % 200 == 0 and n != 0:
                plt.plot(time_, thrust_)
                plt.xlabel('Time (s)')
                plt.ylabel('Thrust (N)')
                plt.savefig(f'thrust_frames/thrust_t={n*dt:09f}.png', dpi=100, bbox_inches='tight')
                print("\n Thrust frame saved")
            print("\n Frame saved")
        pbar.update(1)
        if cp.any(~cp.isfinite(U)):
            print(f"NaN/inf detected at step {n}")
            break

# Create GIF
with imageio.get_writer('simulation.gif', mode='I', duration=0.5) as writer:
    for i in range(0, n_steps, 50):
        filename = f'frames/frame_t={i*dt:09f}.png'
        if os.path.exists(filename):
            image = imageio.imread(filename)
            writer.append_data(image)

# Clean up frames
for i in range(0, n_steps, 50):
    filename = f'frames/frame_t={i*dt:09f}.png'
    if os.path.exists(filename):
        os.remove(filename)
print("GIF created: simulation.gif")