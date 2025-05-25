import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from tqdm import tqdm
##########import cupy as cp # use when finished to use gpu on certain operations

# Create frames directory
os.makedirs('frames', exist_ok=True)
os.makedirs('thrust_frames', exist_ok=True)

# Constants
L = np.float64(2.25)  # Length of domain
z_start = np.float64(-0.075)  # Start of z-axis
R_max = np.float64(0.075)  # Maximum radial extent
nz = 630 #450 #2250 #int(2325*1.6)   # Number of grid points in z
nr = 21 #15 #75 #int(75*1.6)    # Number of grid points in r
dz = np.float64((L - z_start) / (nz - 1))  # Grid spacing in z
dr = np.float64(R_max / (nr - 1))  # Grid spacing in r
z = np.linspace(z_start, L, nz, dtype=np.float64)
r = np.linspace(0, R_max, nr, dtype=np.float64)
Z, R = np.meshgrid(z, r, indexing='ij')
r_wall_inlet = 0.05 # free end of petal valve (m)

j_wall = np.searchsorted(r, r_wall_inlet, side='left') # Index of the inlet in the r array
u_wind = np.float64(20.0) # Constant z-wind velocity (m/s)
r_exhaust = np.float64(0.03) # Exhaust exit radius (m)

# Physical constants
gamma = np.float64(1.4)  # Specific heat ratio
R_gas = np.float64(287)  # Gas constant (J/kg·K)
mu = np.float64(1.8e-5)  # Dynamic viscosity (kg/m·s)
k_fluid = np.float64(0.026)  # Thermal conductivity (W/m·K)
D_i = np.float64(2e-5)  # Diffusion coefficient (m²/s)
cv = R_gas / (gamma - 1)  # Specific heat at constant volume
cp = gamma * cv  # Specific heat at constant pressure
qr = np.float64(46.3e6)  # Heat of reaction (J/kg)
A = np.float64(4e12)  # Pre-exponential factor (1/s)
Ea = np.float64(1.22e8)  # Activation energy (J/kmol)
R_u = np.float64(8314)  # Universal gas constant (J/kmol·K)
M_fuel = np.float64(44)  # Molar mass of fuel (kg/kmol)
M_O2 = np.float64(32)  # Molar mass of oxygen (kg/kmol)
X_LFL = np.float64(0.0237)
X_UFL = np.float64(0.095)

Yf_LFL = X_LFL*M_fuel/(X_LFL*M_fuel + (1-X_LFL)*M_O2)#######
Yf_UFL = X_UFL*M_fuel/(X_UFL*M_fuel + (1-X_UFL)*M_O2)# Initial conditions (sea-level air)

rho0 = np.float64(1.225)  # Density (kg/m³)
T0 = np.float64(300)  # Temperature (K)
p0 = rho0 * R_gas * T0  # Pressure (Pa)
E0 = cv * T0  # Specific internal energy (J/kg)
Yf0 = np.float64(0.06)  # Fuel mass fraction inside
Yo0 = np.float64(0.23)  # Oxygen mass fraction in air
Yp0 = np.float64(0)  # Product mass fraction

# Time parameters
dt = np.float64(1e-8)  # Time step (s)
t_max = np.float64(0.02)  # Total simulation time (s)
n_steps = int(t_max / dt)  # Number of time steps

# Pulsejet geometry (walls from z=0 to z=1.5)
def pulsejet_wall(z):
    r_wall = np.full_like(z, np.inf, dtype=np.float64)
    for i, zi in enumerate(z):
        if 0 <= zi <= 0.5:
            r_wall[i] = 0.05  # Combustion chamber radius
        elif 0.5 < zi <= 0.7:
            r_wall[i] = 0.05 - (zi - 0.5) * (0.05 - 0.03) / 0.2  # Tapered section
        elif 0.7 < zi <= 1.5:
            r_wall[i] = 0.03  # Exhaust radius
    return r_wall

i_valve = np.searchsorted(z, 0, side='left') # z = 0

# Wall mask (including 'inlet' at z=0 as a solid wall)
r_wall = pulsejet_wall(z)
r_wall_arr = r_wall[:, None]
wall_mask = np.zeros((nz, nr), dtype=bool)
for i in range(nz):
    if 0 <= z[i] <= 1.5:
        j_wall_i = np.searchsorted(r, r_wall[i], side='left')
        if j_wall_i < nr:
            wall_mask[i, j_wall_i] = True  # Mark only the grid point closest to r_wall(z[i])

wall_mask[i_valve, :j_wall+1] = True
chamber_idx = (Z > 0) & (Z < 0.5) & (R < r_wall_arr)
r_inlet = np.linspace(-r_wall_inlet, r_wall_inlet, 100)

z_inlet = np.zeros_like(r_inlet)

# Functions
def compute_thermo_properties(U):
    """
    Compute temperature, pressure, and mixture-specific gas constant.
    Returns: T (K), p (Pa), R_gas_mix (J/kg·K)
    """
    rho = U[:, :, 0]
    E = U[:, :, 3] / (rho + 1e-10)
    ur = U[:, :, 1] / (rho + 1e-10)
    uz = U[:, :, 2] / (rho + 1e-10)
    Yf = U[:, :, 4] / (rho + 1e-10)
    Yo = U[:, :, 5] / (rho + 1e-10)
    Yp = U[:, :, 6] / (rho + 1e-10)
    # Mixture molar mass (products approximated as CO2/H2O mix)
    M_prod = np.float64(28.0)  # kg/kmol
    R_gas_mix = R_u * np.clip((Yf / M_fuel + Yo / M_O2 + Yp / M_prod), 0, None)
    # Specific heat (placeholder, assuming constant gamma)
    cv_mix = R_gas_mix / (gamma - 1)
    # Temperature: T = (E - 0.5 * (ur^2 + uz^2)) / cv
    T = (E - 0.5 * (ur**2 + uz**2)) / cv_mix
    T = np.clip(T, 1.0, None, dtype=np.float64)
    # Pressure: p = rho * R_gas_mix * T
    p = rho * R_gas_mix * T
    return T, p, R_gas_mix

def combustion_source(U):
    """
    Mechanism for combustion
    """
    rho = U[:, :, 0]
    Yf  = U[:, :, 4]/(rho + 1e-10)
    Yo  = U[:, :, 5]/(rho + 1e-10)
    T, p, R_gas_mix = compute_thermo_properties(U)

    # Ignition & flammability requirements
    T_ign = np.clip(800 - 20 * (p / p0 - 1), 600, None)
    can_burn = (T >= T_ign) & (Yf >= Yf_LFL) & (Yf <= Yf_UFL)

    # Molar concentrations
    C_f = rho * Yf / M_fuel
    C_o2 = rho * Yo / M_O2

    # Arrhenius rate
    k_arr  = A * np.exp(-Ea / (R_u * T))
    Rf_mol = np.where(can_burn, k_arr * C_f * C_o2, 0.0)

    # Molar stoich 
    Ro_mol = 5 * Rf_mol

    # Mass rates (Rf, Ro negative; Rp positive)
    Rf_mass = -Rf_mol * M_fuel
    Ro_mass = -Ro_mol * M_O2
    Rp_mass = - (Rf_mass + Ro_mass)

    # Source array
    S = np.zeros_like(U)
    S[:, :, 4] = Rf_mass # fuel
    S[:, :, 5] = Ro_mass # O2
    S[:, :, 6] = Rp_mass # products

    # Energy = fuel‐mol * M_fuel * qr
    S[:, :, 3] = -Rf_mass * qr

    # No reaction at walls
    S[wall_mask, :] = 0

    return S, Rf_mass

def compute_fluxes(U, p):
    rho = U[:, :, 0]
    ur = U[:, :, 1] / (rho + 1e-10)
    uz = U[:, :, 2] / (rho + 1e-10)
    Fz = np.zeros_like(U, dtype=np.float64)
    Fr = np.zeros_like(U, dtype=np.float64)
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
    dur_dz = np.zeros_like(ur)
    dur_dr = np.zeros_like(ur)
    duz_dz = np.zeros_like(uz)
    duz_dr = np.zeros_like(uz)
    dT_dz = np.zeros_like(T)
    dT_dr = np.zeros_like(T)

    # Custom gradients to handle walls
    for i in range(nz):
        for j in range(nr):
            if wall_mask[i, j]:
                continue  # Skip wall points
            # z-direction gradients
            if i > 0 and i < nz-1 and not wall_mask[i-1, j] and not wall_mask[i+1, j]:
                dur_dz[i, j] = (ur[i+1, j] - ur[i-1, j]) / (2 * dz)
                duz_dz[i, j] = (uz[i+1, j] - uz[i-1, j]) / (2 * dz)
                dT_dz[i, j] = (T[i+1, j] - T[i-1, j]) / (2 * dz)
            elif i == 0 and not wall_mask[i+1, j]:
                dur_dz[i, j] = (ur[i+1, j] - ur[i, j]) / dz
                duz_dz[i, j] = (uz[i+1, j] - uz[i, j]) / dz
                dT_dz[i, j] = (T[i+1, j] - T[i, j]) / dz
            elif i == nz-1 and not wall_mask[i-1, j]:
                dur_dz[i, j] = (ur[i, j] - ur[i-1, j]) / dz
                duz_dz[i, j] = (uz[i, j] - uz[i-1, j]) / dz
                dT_dz[i, j] = (T[i, j] - T[i-1, j]) / dz
            # r-direction gradients
            if j > 0 and j < nr-1 and not wall_mask[i, j-1] and not wall_mask[i, j+1]:
                dur_dr[i, j] = (ur[i, j+1] - ur[i, j-1]) / (2 * dr)
                duz_dr[i, j] = (uz[i, j+1] - uz[i, j-1]) / (2 * dr)
                dT_dr[i, j] = (T[i, j+1] - T[i, j-1]) / (2 * dr)
            elif j == 0 and not wall_mask[i, j+1]:
                dur_dr[i, j] = (ur[i, j+1] - ur[i, j]) / dr
                duz_dr[i, j] = (uz[i, j+1] - uz[i, j]) / dr
                dT_dr[i, j] = (T[i, j+1] - T[i, j]) / dr
            elif j == nr-1 and not wall_mask[i, j-1]:
                dur_dr[i, j] = (ur[i, j] - ur[i, j-1]) / dr
                duz_dr[i, j] = (uz[i, j] - uz[i, j-1]) / dr
                dT_dr[i, j] = (T[i, j] - T[i, j-1]) / dr

    R_safe = R + 1e-10
    tau_rz = mu * (dur_dz + duz_dr)
    div_v = dur_dr + ur / R_safe + duz_dz
    tau_zz = mu * (2 * duz_dz - 2/3 * div_v)
    tau_rr = mu * (2 * dur_dr - 2/3 * div_v)
    Fz_visc = np.zeros_like(U, dtype=np.float64)
    Fr_visc = np.zeros_like(U, dtype=np.float64)
    Fz_visc[:, :, 1] = tau_rz
    Fz_visc[:, :, 2] = tau_zz
    Fr_visc[:, :, 1] = tau_rr
    Fr_visc[:, :, 2] = tau_rz
    Fz_visc[:, :, 3] = tau_zz * uz + tau_rz * ur - k_fluid * dT_dz
    Fr_visc[:, :, 3] = tau_rz * uz + tau_rr * ur - k_fluid * dT_dr
    return Fz_visc, Fr_visc

E_max = rho0 * cv * 5000.0

def apply_boundary_conditions(U):
    T, p, R_gas_mix = compute_thermo_properties(U)
    rho = U[:, :, 0]
    ur = U[:, :, 1] / (rho + 1e-10)
    uz = U[:, :, 2] / (rho + 1e-10)

    ############################# issues here ###################################################### allwoing properties through????
    # Inlet: no-slip wall 
    U[i_valve, :, 1:3] = 0  # Zero radial and axial velocities
    U[i_valve, :, 0] = U[i_valve-1, :, 0]  # Density from adjacent point
    U[i_valve, :, 3] = U[i_valve-1, :, 3]  # Energy from adjacent point
    U[i_valve, :, 4:7] = U[i_valve-1, :, 4:7]  # Species from adjacent point
    ################################################################################################

    # Axis (r=0): Symmetry
    U[:, 0, 1] = 0
    for i in [0, 2, 3, 4, 5, 6]:
        U[:, 0, i] = U[:, 1, i]

    # +Z boundary 
    U[-1, :, :] = U[-2, :, :] # Outflow

    # -Z boundary
    U[0, :, :] = U[1, :, :] # Outflow
    # +R boundary
    U[:, -1, :] = U[:, -2, :] # Outflow

    # Clip state variables
    U[:, :, 0] = np.clip(U[:, :, 0], 1e-6, None, dtype=np.float64)
    U[:, :, 1] = np.clip(U[:, :, 1], -1e3, 1e3, dtype=np.float64) #############
    U[:, :, 2] = np.clip(U[:, :, 2], -1e3, 1e3, dtype=np.float64) ############# these might add inaccuracies in extereme cases
    U[:, :, 3] = np.clip(U[:, :, 3], 0, E_max, dtype=np.float64)  #############

    species_sum = U[:, :, 4] + U[:, :, 5] + U[:, :, 6]
    zero_species = species_sum < 1e-10
    U[zero_species, 4] = 0
    U[zero_species, 5] = U[zero_species, 0] * Yo0 # Fix for issue of 0 species causing divide by 0 error, could add inaccuracies but fixes the error
    U[zero_species, 6] = 0
    U[:, :, 4:] = np.clip(U[:, :, 4:], 0, U[:, :, 0][:, :, None], dtype=np.float64)
    U = np.where(np.isfinite(U), U, np.array([rho0, 0, 0, rho0 * E0, 0, rho0 * Yo0, 0])[None, None, :])
    return U

def compute_cfl(U, dt, dz, dr):
    rho = U[:, :, 0]
    ur = U[:, :, 1] / (rho + 1e-10)
    uz = U[:, :, 2] / (rho + 1e-10)
    T, p, R_gas_mix = compute_thermo_properties(U)
    c = np.sqrt(gamma * R_gas_mix * T) # Speed of sound
    max_speed_z = np.max(np.abs(uz) + c)
    max_speed_r = np.max(np.abs(ur) + c)
    cfl_z = dt * max_speed_z / dz
    cfl_r = dt * max_speed_r / dr
    return max(cfl_z, cfl_r)

# Initialization
U_ambient = np.zeros((nz, nr, 7), dtype=np.float64) # [rho, rho*ur, rho*uz, rho*E, rho*Yf, rho*Yo, rho*Yp]
U_ambient[:, :, 0] = rho0 # Density
U_ambient[:, :, 1] = 0 # rho * u_z
U_ambient[:, :, 2] = 0 # rho * u_r
U_ambient[:, :, 3] = rho0 * E0 # Total energy
U_ambient[:, :, 4] = 0 # Fuel
U_ambient[:, :, 5] = rho0 * Yo0 # Oxygen
U_ambient[:, :, 6] = 0 # Products

U = U_ambient

#inside = (R < r_wall_arr) & (Z > 0) & (Z < 1.5)
outside = (R >= r_wall_arr) & (Z >= 0) 
reset_vals = ((R >= r_wall_arr) & (Z <= 1.5) & (Z >= 0)) | ((Z >= z_start) & (Z <= 0) & (R <= 0.075))

spark_profile = np.exp(-((Z - 0.15)**2 / 0.02**2 + R**2 / 0.02**2))  # Smooth spark
spark_profile[outside] = 0
U[:, :, 3] = rho0 * cv * (300 + 1200 * spark_profile)  # Peak T ≈ 1500 K
#U[:, :, 0] = rho0 * (1 + 0.3 * spark_profile)  # Peak rho ≈ 1.3 * rho0

fuel_profile_in = np.exp(-((Z - 0.15)**2 / 0.03**2 + R**2 / 0.03**2))  # Initial fuel inside pulsejet
fuel_profile_in[outside] = 0
U[:, :, 4] = rho0 * Yf0 * fuel_profile_in

exit_zi = np.searchsorted(z, 1.5, side='left')
exit_ri = int(nr * (0.03 / R_max))# Simulation loop
progress_interval = max(1, n_steps // 100)

with tqdm(total=n_steps, desc="\n", ncols=80) as pbar:
    time_ = []
    thrust_ = []

    for n in range(n_steps):
        T, p, R_gas_mix = compute_thermo_properties(U)

        R_safe = R[:, :, None] + 1e-10
        Fz, Fr = compute_fluxes(U, p)
        Fz_visc, Fr_visc = viscous_fluxes(U, T)
        dFz_dz = np.gradient(Fz, dz, axis=0)
        dFr_dr = np.gradient(Fr, dr, axis=1)
        dFz_visc_dz = np.gradient(Fz_visc, dz, axis=0)
        dFr_visc_dr = np.gradient(Fr_visc, dr, axis=1)
        Fr_over_r = np.zeros_like(Fr, dtype=np.float64)
    
        # Compute Fr_over_r for r > 0
        Fr_over_r[:, 1:, 0] = Fr[:, 1:, 0] / R_safe[:, 1:, 0] # rho * ur / r
        Fr_over_r[:, 1:, 1] = (Fr[:, 1:, 1] - p[:, 1:]) / R_safe[:, 1:, 0] # rho * ur * ur / r
        Fr_over_r[:, 1:, 2] = Fr[:, 1:, 2] / R_safe[:, 1:, 0] # rho * ur * uz / r
        Fr_over_r[:, 1:, 3] = Fr[:, 1:, 3] / R_safe[:, 1:, 0] # (rho * E + p) * ur / r
        Fr_over_r[:, 1:, 4:7] = Fr[:, 1:, 4:7] / R_safe[:, 1:, :] # rho * Y_i * ur / r

        # At r=0, set Fr_over_r to zero for ur-dependent terms
        Fr_over_r[:, 0, :] = 0 # Axial symmetry

        S, Rf = combustion_source(U)

        update = dt * (dFz_dz + dFr_dr + Fr_over_r - dFz_visc_dz - dFr_visc_dr - S)
        
        U_new = U - update

        U = apply_boundary_conditions(U_new)

        U[reset_vals, :] = U_ambient[reset_vals, :]
        U[reset_vals, 2] = U[reset_vals, 0] * u_wind # Set rho*u_z = rho0 * u_wind
        U[reset_vals, 3] = U[reset_vals, 0] * (E0 + (u_wind**2) / 2)
        

        cfl = compute_cfl(U, dt, dz, dr)
        if cfl > 1:
            sug_dt = dt / cfl
            print(f" Warning: CFL = {cfl:.2f} > 1 at step {n}, reduce dt to {(sug_dt):.2e} or less")

        # Thrust
        p_exit = p[exit_zi, :exit_ri]
        uz_exit = U[exit_zi, :exit_ri , 2] / U[exit_zi, :exit_ri , 0]
        exit_area = np.pi * 0.03**2
        thrust = exit_area * ((uz_exit * U[exit_zi, :exit_ri , 2]).mean() + (p_exit.mean() - p0))
        thrust_.append(thrust)
        time_.append(n*dt)
    
        if n % 50 == 0:
            if np.any(~np.isfinite(T)):
                print(f"Invalid temperature at step {n}")
                break
        
            Yf = U[:, :, 4] / (U[:, :, 0] + 1e-10)
            Yo = U[:, :, 5] / (U[:, :, 0] + 1e-10)
            rho = U[:, :, 0]
        
            # Chamber metrics
            #T_chamber = T[chamber_idx].mean() if np.any(chamber_idx) else T[1, :].mean()
            #Yf_chamber = Yf[chamber_idx].mean() if np.any(chamber_idx) else Yf[1, :].mean()
            #Yo_chamber = Yo[chamber_idx].mean() if np.any(chamber_idx) else Yo[1, :].mean()
            #p_int = p[chamber_idx].mean() if np.any(chamber_idx) else p[1, :].mean()
            #p_int = p[chamber_idx].mean() if np.any(chamber_idx) else p[1, :].mean()
        
            # Visualization
            X = np.concatenate([Z, Z], axis=1)
            Y = np.concatenate([-R[:, ::-1], R], axis=1)
            T_mirror = np.concatenate([T[:, ::-1], T], axis=1)
            p_mirror = np.concatenate([p[:, ::-1], p], axis=1)
            rho_mirror = np.concatenate([rho[:, ::-1], rho], axis=1)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(64, 6), sharey=True)
        
            c1 = ax1.contourf(X, Y, T_mirror, levels=20, vmin=0, vmax=5000, cmap='inferno')
            plt.colorbar(c1, ax=ax1, label='Temperature (K)')
            ax1.plot(z, r_wall, 'w-', linewidth=2) # Jet walls
            ax1.plot(z, -r_wall, 'w-', linewidth=2)
            ax1.plot(z_inlet, r_inlet, 'w-', linewidth=2) # Inlet wall
            ax1.set_xlabel('z (m)')
            ax1.set_ylabel('r (m)')
            ax1.set_title(f'Temperature at t={n*dt:.9f} s')
            ax1.set_aspect('equal')
            ax1.set_ylim(-R_max, R_max)
            ax1.set_xlim(z_start, L)

            c2 = ax2.contourf(X, Y, p_mirror, levels=20, vmin=0.1*p0, vmax=50*p0, cmap='viridis')
            plt.colorbar(c2, ax=ax2, label='Pressure (Pa)')
            ax2.plot(z, r_wall, 'w-', linewidth=2)
            ax2.plot(z, -r_wall, 'w-', linewidth=2)
            ax2.plot(z_inlet, r_inlet, 'w-', linewidth=2)
            ax2.set_xlabel('z (m)')
            ax2.set_title(f'Pressure at t={n*dt:.9f} s')
            ax2.set_aspect('equal')
            ax2.set_xlim(z_start, L)

            c3 = ax3.contourf(X, Y, rho_mirror, levels=20, vmin=0.1*rho0, vmax=50*rho0, cmap='Greys_r')
            plt.colorbar(c3, ax=ax3, label='Density (kg/m³)')
            ax3.plot(z, r_wall, 'w-', linewidth=2)
            ax3.plot(z, -r_wall, 'w-', linewidth=2)
            ax3.plot(z_inlet, r_inlet, 'w-', linewidth=2)
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

        if np.any(~np.isfinite(U)):
            print(f"NaN/inf detected at step {n}")
            break

# Create GIF
with imageio.get_writer('simulation.gif', mode='I', duration=0.5) as writer:
    for i in range(0, n_steps, 50):
        filename = f'frames/frame_t={i*dt:09f}.png'
        if os.path.exists(filename):
            image = imageio.imread(filename)
            writer.append_data(image) # Clean up frames

for i in range(0, n_steps, 50):
    filename = f'frames/frame_t={i*dt:09f}.png'
    if os.path.exists(filename):
        os.remove(filename)
        print("GIF created: simulation.gif")

