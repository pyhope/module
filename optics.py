import numpy as np

h = 6.62607015e-34 # Planck constant in J s
h_ev = 4.135667696e-15 # Planck constant in eV s
c = 299792458 # Speed of light in m/s
k_B = 1.380649e-23 # Boltzmann constant in J/K
epsilon_0 = 8.854187817e-12 # Vacuum permittivity in F/m
stefan_boltzmann = 5.67e-8

def kkr_single(de, eps_imag, cshift=1e-6):
    eps_imag = np.array(eps_imag)
    nedos = eps_imag.shape[0]
    cshift = complex(0, cshift)
    w_i = np.arange(0, nedos*de, de, dtype=np.complex128)
    def integration_element(w_r):
        factor = w_i / (w_i**2 - w_r**2 + cshift)
        total = np.sum(eps_imag * factor)
        return total * (2/np.pi) * de + 1
    return np.real([integration_element(w_r) for w_r in w_i])

def rosseland_mean(x, nu, T):
    def dB_dT(nu, T):
        return (2 * h * nu**3 / c**2) * (h * nu / (k_B * T**2)) * (np.exp(h * nu / (k_B * T)) / (np.exp(h * nu / (k_B * T)) - 1)**2)
    numerator = np.trapz(1 / x * dB_dT(nu, T), nu)
    denominator = np.trapz(dB_dT(nu, T), nu)
    return 1 / (numerator / denominator)

def calc_optics(filename = 'sigma.dat', T = 4000, isvasp=False):
    data = np.loadtxt(filename, skiprows=1, unpack=True)
    energy = data[0]
    if isvasp:
        sigma = data[1] * 1e6
    else:
        sigma = data[1]
    omega_in_hz = 2 * np.pi * energy / h_ev
    eps_imag = sigma / (omega_in_hz * epsilon_0)
    eps_real = kkr_single(energy[1] - energy[0], eps_imag)

    eps_mod = np.sqrt(eps_real**2 + eps_imag**2)
    k = np.sqrt((eps_mod - eps_real) / 2)
    n = np.sqrt((eps_mod + eps_real) / 2)

    alpha = 4 * np.pi * k / c * energy / h_ev
    k_rad = 16 * n**2 * stefan_boltzmann * T**3 / (3 * alpha)

    n_r = rosseland_mean(n, energy / h_ev, T)
    alpha_r = rosseland_mean(alpha, energy / h_ev, T)
    k_rad_r = 16 * n_r**2 * stefan_boltzmann * T**3 / (3 * alpha_r)

    return {
        'energy': energy,
        'eps_real': eps_real,
        'eps_imag': eps_imag,
        'k': k,
        'n': n,
        'alpha': alpha,
        'k_rad': k_rad,
        'n_r': n_r,
        'alpha_r': alpha_r,
        'k_rad_r': k_rad_r
    }
