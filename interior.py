import numpy as np
from scipy.optimize import brentq

def calc_interior_profile(T0, R_total, M_total, CMF):
    # Constants
    M_Earth = 5.972e24  # Planetary mass in kg (Earth's mass)
    G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
    # Initial conditions at the surface
    P0 = 0  # Surface pressure in Pa
    rho0 = 3226  # Surface density in kg/m^3
    T0  # Surface temperature in K
    g0 = G * M_total / R_total**2  # Surface gravity in m/s^2

    # Integration parameters
    dr = -1000  # Step size in meters (negative for inward integration)
    r_values = np.arange(R_total, 0+dr, dr)  # Radial positions from surface to center
    num_steps = len(r_values)

    # Initialize arrays
    m = np.zeros(num_steps)
    g = np.zeros(num_steps)
    P = np.zeros(num_steps)
    rho = np.zeros(num_steps)
    T = np.zeros(num_steps)
    layer_indices = np.zeros(num_steps, dtype=int)

    # Initial conditions
    m[0] = M_total
    g[0] = g0
    P[0] = P0
    rho[0] = rho0
    T[0] = T0

    # Layer boundaries in GPa (convert to Pa)
    mantle_layer_pressures = [28.3e9, 124e9, 750e9, 1300e9]  # Pressures in Pa
    mantle_layer_names = ['Peridotite', 'Perovskite', 'PPV', 'PostPPV1', 'PostPPV2']
    core_layer_names = ['Liquid Iron', 'Solid Iron']

    # Combine layer names
    layer_names = mantle_layer_names + core_layer_names

    # Material properties for each layer (Example values, replace with actual data)
    layer_properties = {
        'Peridotite': {'rho0': 3226,'K_0': 1.28e11, 'K0_prime': 4.2, 'alpha_0': 2e-5, 'gamma_0': 0.99, 'gamma_1': 2.1, 'gamma_inf': None},
        # 'Perovskite': {'rho0': 4109,'K_0': 2.61e11, 'K0_prime': 4.0, 'alpha_0': 2e-5, 'gamma_0': 1.0, 'gamma_1': 1.4},
        'Perovskite': {'rho0': 4109,'K_0': 2.61e11, 'K0_prime': 4.0, 'alpha_0': 2e-5, 'gamma_0': 1.54, 'gamma_1': 0.8, 'gamma_inf':None},
        # 'PPV': {'rho0': 4260,'K_0': 3.24e11, 'K0_prime': 3.3, 'alpha_0': 2e-5, 'gamma_0': 1.48, 'gamma_1': 1.4},
        'PPV': {'rho0': 4260,'K_0': 3.24e11, 'K0_prime': 3.3, 'alpha_0': 2e-5, 'gamma_0': 1.48, 'gamma_1': 2.7, 'gamma_inf': 0.93}, # Most recent results from Sakai et al.
        # 'PostPPV1': {'rho0': 4417,'K_0': 4.02e11, 'K0_prime': 2.7, 'alpha_0': 2e-5, 'gamma_0': 1.5, 'gamma_1': 1.4},
        # 'PostPPV2': {'rho0': 4579,'K_0': 4.99e11, 'K0_prime': 2.2, 'alpha_0': 2e-5, 'gamma_0': 1.5, 'gamma_1': 1.4},
        'Liquid Iron': {'rho0': 7700,'K_0': 1.25e11, 'K0_prime': 5.5, 'alpha_0': 4e-5, 'gamma_0': 1.6, 'gamma_1': 0.92, 'gamma_inf': None},
        'Solid Iron': {'rho0': 8160,'K_0': 1.65e11, 'K0_prime': 4.9, 'alpha_0': 4e-5, 'gamma_0': 1.6, 'gamma_1': 0.92, 'gamma_inf': None},
    }

    # Function to compute K_T
    def compute_K_T(rho_r, rho0_layer, K_0, K0_prime):
        x = rho0_layer / rho_r
        theta = 1.5 * (K0_prime - 1)
        exponent = theta * (1 - x ** (1 / 3))
        term1 = x ** (-2 / 3)
        term2 = 1 + (1 + theta * x ** (1 / 3)) * (1 - x ** (1 / 3))
        K_T = K_0 * term1 * term2 * np.exp(exponent)
        return K_T

    # Function to compute K_s
    def compute_K_s(K_T, alpha_r, gamma_r, T_r):
        K_s = K_T * (1 + alpha_r * gamma_r * T_r)
        return K_s

    # Function to compute alpha
    def compute_alpha(alpha_0, x):
        return alpha_0 * x ** 3

    # Function to compute gamma
    def compute_gamma(gamma_0, x, gamma_1):
        return gamma_0 * x ** gamma_1

    def compute_gamma2(gamma_inf, gamma_0, x, gamma_1):
        return gamma_inf + (gamma_0 - gamma_inf) * x ** gamma_1

    # Function to compute melting temperature of iron
    def T_m(P_m):
        # P_m in Pa, convert to GPa inside the formula
        P_m_GPa = P_m / 1e9
        return (6500 * (P_m_GPa / 340) ** 0.515) / (1 - np.log(0.87))

    # Initialize variables
    current_layer = 0  # Start with the first mantle layer
    mantle_mass = M_total * (1 - CMF)
    mass_accumulated = 0  # Mass accumulated in mantle layers
    core_started = False
    solid_core_started = False
    boundary = False
    g_boundary = None

    for i in range(1, num_steps):
        r = r_values[i]
        dr_step = r_values[i] - r_values[i - 1]

        # Compute derivatives
        dm_dr = 4 * np.pi * r ** 2 * rho[i - 1]
        m[i] = m[i - 1] + dm_dr * dr_step

        # Update gravity

        #g[i] = min(g[i-1] + (4*np.pi*G*rho[i-1]-2*G*m[i-1]/r**3)*dr_step if r != 0 else 0, (4 / 3) * np.pi * G * rho[i - 1] * r)
        if not core_started:
            g[i] = g[i-1] + (4*np.pi*G*rho[i-1]-2*G*m[i-1]/r**3)*dr_step if r != 0 else 0
            # G * m[i] / r ** 2 
        #else:
            #g[i] = (4 / 3) * np.pi * G * rho[i - 1] * r
        else:
            if g_boundary is None:
                # Store g and r at the core-mantle boundary
                g_boundary = g[i - 1]
                r_boundary = r_values[i - 1]
            # Gravity calculation in the core ensuring continuity
            g[i] = g_boundary +  g_boundary/r_boundary* (r - r_boundary)#(4 / 3) * np.pi * G * rho[i - 1] * (r - r_boundary)
        # Compute dP/dr
        dP_dr = -rho[i - 1] * g[i-1]
        P[i] = P[i - 1] + dP_dr * dr_step

        mass_accumulated -= dm_dr * dr_step

        # Check for phase transitions in the mantle
        if not core_started:
            # Check if the mass of mantle layers has reached the mantle mass
            if mass_accumulated >= mantle_mass:
                core_started = True
                old_layer_name = layer_names[current_layer]
                old_props = layer_properties[old_layer_name]
                current_layer = len(mantle_layer_names)  # Move to liquid iron layer
                boundary = True
                boundary_index = i
                CMB_index = i
                print(f"Core started at radius {r / 1e3:.2f} km, depth: {(R_total - r) / 1e3:.2f} km, pressure: {P[i]/1e9:.2f} GPa")
        else:
            # In the core
            if not solid_core_started:
                # Check if temperature falls below melting temperature
                if T[i - 1] <= T_m(P[i - 1]):
                    solid_core_started = True
                    old_layer_name = layer_names[current_layer]
                    old_props = layer_properties[old_layer_name]
                    current_layer = len(mantle_layer_names) + 1  # Move to solid iron layer
                    boundary = True
                    boundary_index = i
                    print(f"Solid core started at radius {r / 1e3:.2f} km")

        # Check for layer transitions based on pressure
        if not core_started:
            # Check mantle layer boundaries
            # if P[i] >= mantle_layer_pressures[3] + 6e6 * T[i - 1]:
            #     if current_layer != 4:
            #         old_layer_name = layer_names[current_layer]
            #         old_props = layer_properties[old_layer_name]
            #         current_layer = 4  # 'PostPPV2'
            #         boundary = True
            #         boundary_index = i
            #         print(f"Entering layer: PostPPV2 at radius {r/1e3:.2f} km.")
            # elif P[i] >= mantle_layer_pressures[2] - 10e6 * T[i - 1]:
            #     if current_layer != 3:
            #         old_layer_name = layer_names[current_layer]
            #         old_props = layer_properties[old_layer_name]
            #         current_layer = 3  # 'PostPPV1'
            #         boundary = True
            #         boundary_index = i
            #         print(f"Entering layer: PostPPV1 at radius {r/1e3:.2f} km.")
            if P[i] >= mantle_layer_pressures[1] + 8e6 * (T[i - 1] - 2500):
                if current_layer != 2:
                    old_layer_name = layer_names[current_layer]
                    old_props = layer_properties[old_layer_name]
                    current_layer = 2  # 'PPV'
                    boundary = True
                    boundary_index = i
                    print(f"Entering layer: PPV at radius {r/1e3:.2f} km, depth: {(R_total - r) / 1e3:.2f} km, pressure: {P[i]/1e9:.2f} GPa")
            elif P[i] >= mantle_layer_pressures[0] - 2.8e6 * T[i - 1]:
                if current_layer != 1:
                    old_layer_name = layer_names[current_layer]
                    old_props = layer_properties[old_layer_name]
                    current_layer = 1  # 'Perovskite'
                    boundary = True
                    boundary_index = i
                    print(f"Entering layer: Perovskite at radius {r/1e3:.2f} km, depth: {(R_total - r) / 1e3:.2f} km, pressure: {P[i]/1e9:.2f} GPa")

        # Get current layer properties
        layer_name = layer_names[current_layer]
        props = layer_properties[layer_name]
        K_0 = props['K_0']
        K0_prime = props['K0_prime']
        alpha_0 = props['alpha_0']
        gamma_0 = props['gamma_0']
        gamma_1 = props['gamma_1']
        rho0_layer = props['rho0']

        if boundary:
            # Compute K_T_old
            K_T_old = compute_K_T(rho[i - 1], old_props['rho0'], old_props['K_0'], old_props['K0_prime'])

            # Define function to find rho_new
            def KT_difference(rho_new):
                K_T_new = compute_K_T(rho_new, rho0_layer, K_0, K0_prime)
                return K_T_new - K_T_old

            # Set reasonable bounds for rho_new
            rho_min = 0.5 * rho[i - 1]
            rho_max = 5 * rho[i - 1]

            # Solve for rho_new
            try:
                rho_new = brentq(KT_difference, rho_min, rho_max)
            except ValueError:
                # If brentq fails, use rho0_layer as default
                rho_new = rho0_layer
                print(f"Warning: Could not solve for rho_new at boundary, using rho0_layer")

            # Set rho at boundary to rho_new
            rho[i - 1] = rho_new

            # Now proceed with calculations using rho_new
            x_r = rho0_layer / rho[i - 1]
            theta = 1.5 * (K0_prime - 1)
            # Compute K_T and K_s with new rho
            K_T = compute_K_T(rho[i - 1], rho0_layer, K_0, K0_prime)
            alpha_r = compute_alpha(alpha_0, x_r)
            if props['gamma_inf'] is not None:
                gamma_inf = props['gamma_inf']
                gamma_r = compute_gamma2(gamma_inf, gamma_0, x_r, gamma_1)
            else:
                gamma_r = compute_gamma(gamma_0, x_r, gamma_1)
            K_s = compute_K_s(K_T, alpha_r, gamma_r, T[i - 1])

            # Compute dρ/dr
            drho_dr = -rho[i - 1] ** 2 * g[i-1] / K_s
            rho[i] = rho[i - 1] + drho_dr * dr_step

            # Compute dT/dr
            dT_dr = -rho[i-1] * g[i-1] * gamma_r * T[i - 1] / K_s
            #print('rho, g, gamma_r, T, K_s, dT_dr, drho_dr, dr_step,Tm,dT_dr * dr_step', rho[i], g[i], gamma_r, T[i - 1], K_s, dT_dr, drho_dr, dr_step,T_m(P[i]),dT_dr * dr_step)
            #T[i] = T[i - 1] + dT_dr * dr_step
            if not core_started:
                T[i] = T[i - 1] + dT_dr * dr_step

            elif not solid_core_started:
                T[i] = T[i - 1] + dT_dr * dr_step + 1400*(M_total/M_Earth)**0.75
            else:
                T[i] = T[i] = T[i - 1] + dT_dr * dr_step
            boundary = False  # Reset boundary flag

        else:
            # Compute x(r)
            x_r = rho0_layer / rho[i - 1]
            theta = 1.5 * (K0_prime - 1)

            # Compute K_T and K_s
            K_T = compute_K_T(rho[i - 1], rho0_layer, K_0, K0_prime)
            alpha_r = compute_alpha(alpha_0, x_r)
            if props['gamma_inf'] is not None:
                gamma_inf = props['gamma_inf']
                gamma_r = compute_gamma2(gamma_inf, gamma_0, x_r, gamma_1)
            else:
                gamma_r = compute_gamma(gamma_0, x_r, gamma_1)
            K_s = compute_K_s(K_T, alpha_r, gamma_r, T[i - 1])

            # Compute dρ/dr
            drho_dr = -rho[i - 1] ** 2 * g[i] / K_s
            rho[i] = rho[i - 1] + drho_dr * dr_step

            # Compute dT/dr
            dT_dr = -rho[i] * g[i] * gamma_r * T[i - 1] / K_s
            T[i] = T[i - 1] + dT_dr * dr_step

        boundary = False  # Reset boundary flag
        #print(f"{layer_name}, Radius: {r / 1e3:.2f} km, Mass: {m[i] / 1e24:.2f} 1e24 kg, pressure: {P[i] / 1e9:.2f} GPa, density: {rho[i]:.2f} kg/m^3, temperature: {T[i]:.2f} K, T_m: {T_m(P[i]):.2f} K")
        # Prevent negative densities and pressures
        if rho[i] <= 0:
            print(f"Density became non-positive at radius {r/1e3:.2f} km.")
            rho = rho[:i]
            m = m[:i]
            g = g[:i]
            P = P[:i]
            T = T[:i]
            r_values = r_values[:i]
            layer_indices = layer_indices[:i]
            break

        if P[i] <= 0:
            P[i] = 0
            print(f"Pressure reached zero at radius {r/1e3:.2f} km.")
            rho = rho[:i + 1]
            m = m[:i + 1]
            g = g[:i + 1]
            P = P[:i + 1]
            T = T[:i + 1]
            r_values = r_values[:i + 1]
            layer_indices = layer_indices[:i + 1]
            break

        # Record the current layer index
        layer_indices[i] = current_layer

    depth = (r_values[0] - r_values) / 1e3

    final_profile = np.column_stack((depth[1:CMB_index-1], rho[1:CMB_index-1] / 1e3, P[1:CMB_index-1] / 1e9, T[1:CMB_index-1]))[::-1]
    R_core = R_total/1e3 - depth[CMB_index-2]
    return final_profile[:,0], final_profile[:,1], final_profile[:,2], final_profile[:,3], R_core/6371.0

if __name__ == "__main__":
    calc_interior_profile(
        T0=1600,  # Surface temperature in K
        R_total=1.574*6371e3,  # Total radius of the planet in meters
        M_total=5.69*5.972e24,  # Total mass of the planet in kg
        CMF=0.406
    )
