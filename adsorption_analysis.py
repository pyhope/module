import numpy as np
from matplotlib import pyplot as plt
import my_pyplot as mpt
from numpy import trapz
import pickle as pkl

def analyze(input_filename, output_filename, show=False, print_log=False, save_log=False, save_plot=True):
    with open(input_filename + '.pkl', 'rb') as file:
        average, Counts, System = pkl.load(file)

    Means = {'Mg': {}, 'O': {}, 'Fe': {}, 'W': {}}
    for ele in ['Mg', 'O', 'Fe', 'W']:
        for sys in ['s', 'l', 'i']:
            Means[ele][sys] = np.mean(Counts[ele][sys])

    n_atom_in_l = Means['Mg']['l'] + Means['O']['l'] + Means['Fe']['l'] + Means['W']['l']
    n_atom_in_i = Means['Mg']['i'] + Means['O']['i'] + Means['Fe']['i'] + Means['W']['i']
    n_atom_in_s = Means['Mg']['s'] + Means['O']['s'] + Means['Fe']['s'] + Means['W']['s']
    proximity, Mg, O, Fe, W, density = average
    x_W_in_Fe = Means['W']['l'] / n_atom_in_l
    x_W_in_MgO = Means['W']['s'] / n_atom_in_s
    x_Fe_in_Fe = Means['Fe']['l'] / n_atom_in_l
    reference = Fe * x_W_in_Fe / x_Fe_in_Fe
    threshold1 = x_W_in_MgO
    threshold2 = x_W_in_Fe

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylabel('Atomic fraction (%)')
    ax.set_xlabel('Proximity (Å)')
    ax.plot(proximity, reference * 100, color='C0', linestyle='-', label='Fe * %.5f' % (x_W_in_Fe))
    ax.plot(proximity, W * 100, color='C3', linestyle='-', label='W')
    ax.axhline(y=0, color='black', linestyle='--')
    ax.axhline(y=x_W_in_Fe * 100, color='black', linestyle='--')
    ax.axvline(x=0, color='black', linestyle='--')
    
    i_1 = np.argmax(proximity > 1)
    i_2 = np.argmax(proximity > 2)

    if W[i_1] > threshold2 and W[i_2] > threshold2:
        prefactor = 0.1
        mode = 1
    else:
        mode = 2
    
    print(input_filename, mode)
    
    if mode == 1:
        left_i_1 = 0
        left_i_1_max = np.argmax(proximity > 5)
        while W[left_i_1 + 10] < W[left_i_1]:
            left_i_1 = left_i_1 + 10
            if left_i_1 >= left_i_1_max:
                break
        left_i_2 = np.argmax(proximity > 0)
        for i in range(len(proximity)):
            if (threshold2 - reference[i] < prefactor * threshold2 and W[i] < threshold2) or proximity[i] > 12:
                right_i_1 = i
                break
        x1, Mg1, O1, Fe1, W1, density_ad1 = proximity[left_i_1:left_i_2], Mg[left_i_1:left_i_2], O[left_i_1:left_i_2], Fe[left_i_1:left_i_2], W[left_i_1:left_i_2], density[left_i_1:left_i_2]
        x2, Mg2, O2, Fe2, W2, density_ad2 = proximity[left_i_2:right_i_1], Mg[left_i_2:right_i_1], O[left_i_2:right_i_1], Fe[left_i_2:right_i_1], W[left_i_2:right_i_1], density[left_i_2:right_i_1]
        ax.fill_between(x1, W1  * 100, threshold1  * 100, color='C3', alpha=0.2)
        ax.fill_between(x2, W2  * 100, threshold2  * 100, color='C3', alpha=0.2)

        integrand1 = density_ad1 * 1e-24 * (W1 - threshold1) / (Mg1 * 24.305 + O1 * 15.999 + Fe1 * 55.845 + W1 * 183.84)
        n_ex1 = trapz(integrand1, x1)* 1e20
        integrand2 = density_ad2 * 1e-24 * (W2 - threshold2) / (Mg2 * 24.305 + O2 * 15.999 + Fe2 * 55.845 + W2 * 183.84)
        n_ex2 = trapz(integrand2, x2)* 1e20
        n_ex = n_ex1 + n_ex2
    
    elif mode == 2:
        for i in range(len(proximity)):
            if W[i] > reference[i] and proximity[i] > -5:
                left_i = i
                break
        for i in range(len(proximity)):
            if (i > left_i and W[i] < reference[i]) or proximity[i] > 12:
                right_i = i
                break
        x2, Mg2, O2, Fe2, W2, density_ad2, reference2 = proximity[left_i:right_i], Mg[left_i:right_i], O[left_i:right_i], Fe[left_i:right_i], W[left_i:right_i], density[left_i:right_i], reference[left_i:right_i]
        ax.fill_between(x2, W2  * 100, reference2  * 100, color='C3', alpha=0.2)
        integrand2 = density_ad2 * 1e-24 * (W2 - reference2) / (Mg2 * 24.305 + O2 * 15.999 + Fe2 * 55.845 + W2 * 183.84)
        n_ex = trapz(integrand2, x2)* 1e20

    mpt.minor(ax)
    mpt.legend(ax)
    if save_plot:
        mpt.savepdf(output_filename)
    if show:
        plt.show()
    
    C_W_Fe_m = Means['W']['l'] / (Means['Mg']['l'] * 24.305 + Means['O']['l'] * 15.999 + Means['Fe']['l'] * 55.845 + Means['W']['l'] * 183.84)
    C_W_MgO_m = Means['W']['s'] / (Means['Mg']['s'] * 24.305 + Means['O']['s'] * 15.999 + Means['Fe']['s'] * 55.845 + Means['W']['s'] * 183.84)
    C_W_Fe = C_W_Fe_m * density[-1] * 1e3
    C_W_MgO = C_W_MgO_m * density[-1] * 1e3

    C_W = System['W']/ 6.022e23 / (System['V'] * 1e-27)

    # Partition coefficient
    # Simulation
    lw = System['lw']
    x_inter = proximity[(proximity > - lw/2) & (proximity < lw/2)]
    density_inter = density[(proximity > - lw/2) & (proximity < lw/2)]
    mean_density_inter = 1/lw * trapz(density_inter, x_inter) * 1e6 # units: g/m^3
    mass_inter = (Means['Mg']['i'] * 24.305 + Means['O']['i'] * 15.999 + Means['Fe']['i'] * 55.845 + Means['W']['i'] * 183.84) / 6.022e23 # units: g
    area = mass_inter / (mean_density_inter * lw * 1e-10) # units: m^2

    N_ex_ads_W = n_ex * area * 6.022e23
    N_W_MgO = Means['W']['s']
    N_W_Fe = Means['W']['l']
    N_W_interface = Means['W']['i']

    W_frac_liq = N_W_Fe / n_atom_in_l
    W_frac_sol = (N_ex_ads_W + N_W_MgO) / ((n_atom_in_s + 0.5*n_atom_in_i) / 2)
    D1 = W_frac_liq / W_frac_sol

    W_frac_sol = (N_W_interface + N_W_MgO) / ((n_atom_in_s + 0.5*n_atom_in_i) / 2)
    D2 = W_frac_liq / W_frac_sol

    # Earth
    r = 2e-9 # units: m
    mass_Fe_over_MgO = 55.845 / 40.3044
    rho_MgO_over_Fe_core = 5.4 / 11.3
    D3 = r / 3 * mass_Fe_over_MgO * rho_MgO_over_Fe_core * (C_W_Fe * 1e3 / n_ex)

    if print_log:
        print('*' * 80)
        print('Filename:', output_filename)
        print('Surface excess adsorption:', n_ex  * 1e6, 'μmol m^-2')
        print('Partition coefficient (Simulation, method1):', D1)
        print('Partition coefficient (Simulation, method2):', D2)
        print('Partition coefficient (Earth):', D3)
        print('Counts of W in MgO:', N_W_MgO)
        print('Counts of W in Fe:', N_W_Fe)
        print('Counts of W in interface:', N_W_interface)
        print('W concentration in bulk system:', C_W, 'mol/L')
        print('W concentration in Fe:', C_W_Fe, 'mol/L;', C_W_Fe_m, 'mol/g')
        print('W concentration in MgO:', C_W_MgO, 'mol/L;', C_W_MgO_m, 'mol/g')

        print('*' * 80)

    if save_log:
        with open(output_filename + '_log.txt', 'w') as file:
            file.write('*' * 80 + '\n')
            file.write('Filename: %s\n' % output_filename)
            file.write('Surface excess adsorption: %f μmol m^-2\n' % (n_ex * 1e6))
            file.write('Partition coefficient (Simulation, method1): %f\n' % D1)
            file.write('Partition coefficient (Simulation, method2): %f\n' % D2)
            file.write('Partition coefficient (Earth): %f\n' % D3)
            file.write('Counts of W in MgO: %f\n' % N_W_MgO)
            file.write('Counts of W in Fe: %f\n' % N_W_Fe)
            file.write('Counts of W in interface: %f\n' % N_W_interface)
            file.write('W concentration in bulk system: %f mol/L\n' % C_W)
            file.write('W concentration in Fe: %f mol/L; %f mol/g\n' % (C_W_Fe, C_W_Fe_m))
            file.write('W concentration in MgO: %f mol/L; %f mol/g\n' % (C_W_MgO, C_W_MgO_m))
            file.write('*' * 80 + '\n')

    return n_ex, D1, D2, D3, N_W_MgO, N_W_Fe, N_W_interface, C_W, C_W_Fe, C_W_Fe_m, C_W_MgO, C_W_MgO_m
