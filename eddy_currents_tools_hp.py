import os
import math
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import mpmath # for higher precision calculations
from functools import lru_cache    

mpmath.mp.dps = 50  # Adjust precision as needed

# global constants
c0 = 2.99792458e10 # speed of light [cm/s]
Sm_m = 8.98755224e9 # electrical conductivity units: 1 CGS (CGSM) = 1.11265e-10 Siemens/m;  1 Siemens/m = 8.98755224e9 CGS
epsilon = 1.0 #electric permeability
mu = 1.0 #magnetic permeability
sigma_core = 8.98e11 #conductivity at the core-mantle boundary # Peyronneau
#sigma_pl = 9e6 #conductivity at the upper boundary of the mantle # Peuronneau           
AU = 1.4959787e13 # AU in cm
RSun = 6.957e10 # Solar Radius in cm
MSun = 1.98892e33 # Solar mass in g
Rearth = 6.371e8  # Earth radius in cm    
G = 6.674e-8 # Newton's gravitational constant, CGS

@lru_cache(maxsize=None)
def besselj_cached(n, x):
    return mpmath.besselj(n, x)

@lru_cache(maxsize=None)
def Jder_plus_hp_cached(a):
    if a == 0:
        # Handle special case where a is exactly 0
        J_derivative_plus = mpmath.mpc(0)
    else:
        # General case
        J_derivative_plus = (a)**0.5 * besselj_cached(1/2, a) - (a)**-0.5 * besselj_cached(3/2, a)
    return J_derivative_plus

@lru_cache(maxsize=None)
def Jder_minus_hp_cached(a):
    if a == 0j:
        J_derivative_minus =  mpmath.mpc(0)
    else:
        J_derivative_minus = -(a)**(0.5)*besselj_cached(-1/2,a) + (a)**(-0.5)*besselj_cached(-3/2, a)   # NB: there is a wrong sign in K17 in this equation (the second "+" is shown as a "-")    
    return J_derivative_minus

def read_profile(conductivity):
    curr_path = os.getcwd() # save the path 

    os.chdir('InteriorProfiles/') # go to the directory where all profiles are    
    line = conductivity
    profiles = np.loadtxt(line)
    profiles = np.flipud(profiles)         # flip the array, so that the beginning is the core and not the surface       
    depth = profiles[:,0]                  # depth in Rearth
    density = profiles[:,1]                # density in g/cm3
    pressure = profiles[:,2]               # Pressure in GPa
    mantle_temperature = profiles[:,3]     # Temperature in K       
    conductivity = profiles[:,4]           # Electrical conductivity in log10 S/m       
    indices = np.where(conductivity > -6.0)# conductivity of dry rocks
    depth = depth[indices]                 # depth in Rearth
    density = density[indices]             # density in g/cm3
    pressure = pressure[indices]           # Pressure in GPa
    mantle_temperature = mantle_temperature[indices]     # Temperature in K
    conductivity = conductivity[indices]   # Electrical conductivity in log10 S/m
    sigma_r = Sm_m*10**(conductivity)      # Now not in ln Sm/m, but in CGS              
    r_ax = depth*Rearth # Now r_ax is taken from Lena's fiels       
    mantle_temperature = 0.0               # for compatibility         

    os.chdir(curr_path) # return back to the main directory
    
    return r_ax, sigma_r, density, pressure, mantle_temperature

def magnetic_field(Bdip,Rorb,Rst,inclination,field):
    # magnetic field strength at the orbital location
    # (in case of pure dipole field and inclined motion)
    # after PhD thesis by Colin Johstone
    # arguments: Bdip - global dipole field of the star (in Gauss)
    # Rorb - orbital distance of a planet (cm)
    # Rst - stellar radius (cm)
    # inclination - inclination of the dipole field with repect to the stellar rotational axis (degree)
    # field - an interger value of 2 (the field declines as r^-2; the case of MS stars) or 3 (the field declines as r^-3;
    # the case of white dwarfs, which likely have no winds which could "streth" the magnetic field
    # returns: deltaB - amplitude of the magnetic field change in Gauss
    
    if field==2: #for MS stars, use formulas from Colin's PhD thesis (magnetic field ~ r^-2 outside the source surface)
       if (inclination > 89.0):
         inclination = 89.0
       theta_dip = inclination * 3.142 / 180.0 # dipole tile angle in radians
    
       Rss = 2.5 * Rst # get the source surface assuming it is 2.5 Rstar (solar value)
    
    
       # GET b10 AND b11 FROM THE FIELD STRENGTH AND THE TILT ANGLE
    
       # get the ratio b11 / b10
       b11b10 = - 0.5 * math.tan(theta_dip) 
    
       # now assume a starting value of b10 0f Bdip and increase it slowly until max[B(Rstar,theta)] = Bdip
        
       b10 = Bdip
    
       db10 = Bdip/10.0 # initial change in db10
    
       b11 = b10 * b11b10 # initial b11
       Bpole = b10 * math.cos(theta_dip) - 2.0 * b11 * math.sin(theta_dip) # initial Bpole
       if (Bpole > Bdip):
         db10 = - db10 # i.e. if the initial polar field strength is above the wanted value, should start by making it smaller
    
       nSteps = 1000
       for i in range(0,nSteps):
      
         # save previous Bpole
         BpoleOld = Bpole 
      
         # get b11
         b11 = b10 * b11b10
      
         # now get field strength at pole
         Bpole = b10 * math.cos(theta_dip) - 2.0 * b11 * math.sin(theta_dip)
      
         # compare with previous and expected to see if db10 should be changed
         if ((Bpole-Bdip)*(BpoleOld-Bdip) < 0.0): # this tests if we have passed the desired value
            db10 = - db10 / 2.0
      
         # now change b10
         b10 = b10 + db10
      
    
       print ("Bpole , Bdip , b10 , b11",Bpole , Bdip , b10 , b11)
    
       # NOW GET THE TOTAL CHANGE IN B OVER AN ORBIT AT THE ORBITAL DISTANCE
    
       f1Rss = ( 3.0*(Rss/Rst)**(-3.0) ) / ((Rss/Rst)**(-3.0) + 2.0)
       
       #print (f1Rss)
       #print ((Rorb / Rss)**(-2.0))
    
       deltaB = 4.0 * abs(b11) * f1Rss * (Rorb / Rss)**(-2.0)
       
    if field==3: #for white dwarfs, use a simple dipole-with-no-wind approximation, r^-3
       Borb = Bdip*(Rst/Rorb)**field
       inclination = np.radians(inclination) # degree -> rad
       deltaB = Borb*np.sin(inclination)/2 # maximal dipole change polar/equatorial for 90 deg. 

    return deltaB

def sigma_interpolation(r_ax, sigma_r, density, threshold=5e-4, factor=None):
    mean_diff = np.mean(np.abs(np.diff(sigma_r)/sigma_r[:-1]))
    if mean_diff > threshold:
        print(f'The resolution of conductivity profile is too low ({mean_diff:.6f}). Start interpolation ...')
        factor = int(mean_diff/threshold) if factor is None else factor
        sigma_interp = interp1d(r_ax, sigma_r, kind='linear')
        density_interp = interp1d(r_ax, density, kind='linear')
        r_ax_interp = np.linspace(r_ax[0], r_ax[-1], factor*(len(r_ax)-1)+1)
        sigma_r = sigma_interp(r_ax_interp)
        density = density_interp(r_ax_interp)
        r_ax = r_ax_interp
        mean_diff = np.mean(np.abs(np.diff(sigma_r)/sigma_r[:-1]))
        print(f'Interpolation done with a factor of {factor}. New resolution: {mean_diff}')
    else:
        print(f'The resolution of conductivity profile is good ({mean_diff:.6f}). No interpolation needed.')
    return r_ax, sigma_r, density

def sigma_smoothing(sigma_r, threshold=50, s=5):
    mean_diff = np.mean(np.abs(np.diff(sigma_r)/sigma_r[:-1]))
    max_diff = np.max(np.abs(np.diff(sigma_r)/sigma_r[:-1]))
    if max_diff > threshold * mean_diff:
        print(f'Local conductivity jump found (max diff: {max_diff:.6f}; mean diff: {mean_diff:.6f}). Start smoothing ...')
        sigma_r = gaussian_filter1d(sigma_r, sigma=s, mode='nearest')
        max_diff = np.max(np.abs(np.diff(sigma_r)/sigma_r[:-1]))
        print(f'Smoothing done. New max diff: {max_diff:.6f}')
    else:
        print(f'No local conductivity jump found (max diff: {max_diff:.6f}; mean diff: {mean_diff:.6f}). No smoothing needed.')
    return sigma_r

def calculate_constants_cutoff_Parkinson(r_ax, Borb, omega, mu, sigma_r, n_skin_depth=7.0, Rj_boundary=0.0-1.0j, cutoff_index=None):
    # according to Parkinson, Introduction to Geomagnetism, 1983, Scottish academic press, pages 313-315
    # see also equations in Methods by Kislyakova et al. 2017 NatAstron (after Parkinson, but in CGS)
    # the correct version for a sphere with non-uniform conductivity  
    # This version uses higher precision calculations to fix the overflow errors for Yihang Peng's calculations  

    cons_start = time.time()
    c0 = 2.99792458e10 # speed of light [cm/s]
    epsilon = 1.0
    mu = 1.0
    R_planet = r_ax[-1]
    depth_from_surface = R_planet - r_ax

    if cutoff_index is None:
        # Calculate skin depths for each layer
        skin_depths = np.sqrt(2.0 / (mu * 4 * np.pi * 1e-7 * omega * sigma_r / Sm_m)) * 1e2

        for i in reversed(range(len(r_ax))):
            if depth_from_surface[i] > n_skin_depth * skin_depths[i]:
                cutoff_index = i
                break
        
        if cutoff_index is None:
            cutoff_index = 0  # In case skin depth covers entire profile

        print(f"Skin depth cutoff for {n_skin_depth} skin depths. Critical skin depth: {skin_depths[cutoff_index]/1e5:.1f} km); conductivity: {sigma_r[cutoff_index] / Sm_m:.1f} S/m")
        print(f"Cutoff at r[{cutoff_index}] = {r_ax[cutoff_index]/1e5:.1f} km, depth = {depth_from_surface[cutoff_index]/1e5:.1f} km", flush=True)
    else:
        # Use provided cutoff index
        print(f"Using provided cutoff index: {cutoff_index}")
        print(f"Cutoff at r[{cutoff_index}] = {r_ax[cutoff_index]/1e5:.1f} km, depth = {depth_from_surface[cutoff_index]/1e5:.1f} km", flush=True)

    Be = mpmath.mpf(Borb)  # Convert to high precision
    sigma = [mpmath.mpf(s) for s in sigma_r]

    c0 = mpmath.mpf(c0)  # Speed of light in cm/s (CGS)
    mu = mpmath.mpf(mu)  # magnetic permeability
    epsilon = mpmath.mpf(epsilon)  # electric permeability
    omega = mpmath.mpf(omega)  # Convert omega to high precision
    mu0 = mpmath.mpf(4 * mpmath.pi * 1e-7)  # Permeability of vacuum in SI

    r = [mpmath.mpf(i) for i in r_ax] # agrees with the low-res version
 
    nrad = len(r)
    Cj_cut = [mpmath.mpc(0) for _ in range(nrad)]
    Dj_cut = [mpmath.mpc(0) for _ in range(nrad)]
    Rj_cut = [mpmath.mpc(0) for _ in range(nrad)] # for recursive calculation of Cj and Dj
    kj_cut = [mpmath.mpc(0) for _ in range(nrad)]

    alphaj = [mpmath.mpc(0) for _ in range(nrad)]
    betaj = [mpmath.mpc(0) for _ in range(nrad)]
    gammaj = [mpmath.mpc(0) for _ in range(nrad)]
    deltaj = [mpmath.mpc(0) for _ in range(nrad)]
    epsilonj = [mpmath.mpc(0) for _ in range(nrad)]

    for i in range(nrad):
        # agrees with the low-precision version up to some mismatch in machine precision
        kj_cut[i] = mpmath.sqrt(-1j * 4 * mpmath.pi * omega * mu * sigma[i] / c0**2) 

    print('Total number of layers:', nrad, flush=True)
    #below we calculate the initial value at 0 (core-mantle boundary, CMB) separately            
    for i in range(nrad):
        if i % int(nrad/10) == 0:
            print('Layer:', i, flush=True)
        alphaj[i] = (
            Jder_minus_hp_cached(r[i] * kj_cut[i]) * besselj_cached(3/2, r[i] * kj_cut[i]) -
            besselj_cached(-3/2, r[i] * kj_cut[i]) * Jder_plus_hp_cached(r[i] * kj_cut[i])
        )
        betaj[i] = (
            (kj_cut[i] / kj_cut[i-1])**0.5 * besselj_cached(3/2, r[i] * kj_cut[i-1]) * Jder_minus_hp_cached(r[i] * kj_cut[i]) -
            Jder_plus_hp_cached(r[i] * kj_cut[i-1]) * besselj_cached(-3/2, r[i] * kj_cut[i])
        )
        gammaj[i] = (
            (kj_cut[i] / kj_cut[i-1])**0.5 * besselj_cached(-3/2, r[i] * kj_cut[i-1]) * Jder_minus_hp_cached(r[i] * kj_cut[i]) -
            Jder_minus_hp_cached(r[i] * kj_cut[i-1]) * besselj_cached(-3/2, r[i] * kj_cut[i])
        )
        deltaj[i] = (
            Jder_plus_hp_cached(r[i] * kj_cut[i-1]) * besselj_cached(3/2, r[i] * kj_cut[i]) -
            (kj_cut[i] / kj_cut[i-1])**0.5 * besselj_cached(3/2, r[i] * kj_cut[i-1]) * Jder_plus_hp_cached(r[i] * kj_cut[i])
        )
        epsilonj[i] = (
            Jder_minus_hp_cached(r[i] * kj_cut[i-1]) * besselj_cached(3/2, r[i] * kj_cut[i]) -
            (kj_cut[i] / kj_cut[i-1])**0.5 * besselj_cached(-3/2, r[i] * kj_cut[i-1]) * Jder_plus_hp_cached(r[i] * kj_cut[i])
        )

    # Recursion for Rj_cut
    Rj_cut[cutoff_index] = Rj_boundary
    print('Boundary condition N1:', Rj_cut[cutoff_index])

    for i in range(cutoff_index + 1, nrad):
        Rj_cut[i] = (
            (Rj_cut[i-1] * betaj[i] + gammaj[i]) /
            (Rj_cut[i-1] * deltaj[i] + epsilonj[i])
        )

    # Surface boundary conditions
    betaj0 = (
        (r[-1] * kj_cut[-1])**0.5 * Jder_plus_hp_cached(r[-1] * kj_cut[-1]) -
        2 * besselj_cached(3/2, r[-1] * kj_cut[-1])
    )
    gammaj0 = (
        (r[-1] * kj_cut[-1])**0.5 * Jder_minus_hp_cached(r[-1] * kj_cut[-1]) -
        2 * besselj_cached(-3/2, r[-1] * kj_cut[-1])
    )
    deltaj0 = (
        2 * ((r[-1] * kj_cut[-1])**0.5 * Jder_plus_hp_cached(r[-1] * kj_cut[-1]) +
             besselj_cached(3/2, r[-1] * kj_cut[-1]))
    )
    epsilonj0 = (
        2 * ((r[-1] * kj_cut[-1])**0.5 * Jder_minus_hp_cached(r[-1] * kj_cut[-1]) +
             besselj_cached(-3/2, r[-1] * kj_cut[-1]))
    )
    Bi = Be * (Rj_cut[-1] * betaj0 + gammaj0) / (Rj_cut[-1] * deltaj0 + epsilonj0)

    # Calculate Cj and Dj recursively
    Cj_cut[-1] = 0.5 * (2 * Bi - Be) * (r[-1] * kj_cut[-1])**0.5 * r[-1] * (
        besselj_cached(3/2, r[-1] * kj_cut[-1]) +
        besselj_cached(-3/2, r[-1] * kj_cut[-1]) / Rj_cut[-1]
    )**(-1)
    Dj_cut[-1] = Cj_cut[-1] / Rj_cut[-1]

    for i in range(nrad-2, cutoff_index - 1, -1):
        Cj_cut[i] = alphaj[i] * Cj_cut[i+1] / (betaj[i] + gammaj[i] / Rj_cut[i+1])
        Dj_cut[i] = Cj_cut[i] / Rj_cut[i]
    
    #this is just a set of longer arrays, which will contain zeros in the upper part, where the equations are not applicable
    # now it is superfluous because the conductivity profile is actually cut by a different function, so there is no need to
    # check the equation applicability in this function; but I'm too lazy to remove it now. Maybe later. Doesn't lead to any errors anyway
    nrad0 = len(r_ax)
    Cj = [mpmath.mpc(0) for _ in range(nrad0)] 
    Dj = [mpmath.mpc(0) for _ in range(nrad0)] 
    Rj = [mpmath.mpc(0) for _ in range(nrad0)] 
    kj = [mpmath.mpc(0) for _ in range(nrad0)] 
 
    for i in range(nrad):  # the upper part, where the equations are not applicable, stays equal zero
        Cj[i] = Cj_cut[i]
        Dj[i] = Dj_cut[i]
        Rj[i] = Rj_cut[i]
        kj[i] = kj_cut[i]
        #print ('{:.20e}'.format(float(abs(Cj[i]))))
        
    # we also shift the arrays by one point to avoid jumps in the magnetic field and heating profiles
    # the jumps arise due to "delay" by one point in Cj and Dj (recursive formula leads to that)
    # is there a better solution than just a shift? No - this is the best way of solving it, all other solutions are artificial
    for i in range(1,nrad):  
        Cj[i-1] = Cj[i]
        Dj[i-1] = Dj[i]    
    
    # the values returned by this function look similar to what calculate_constants2 returns 
    # up to machine precision (the values for the constants are different only many digits after the dot)

    print('Constants calculation time:', time.time() - cons_start, 's', flush=True)
    return Cj, Dj, kj, Bi, Rj

def calculate_constants_rcmb(r_ax, Borb, omega, mu, sigma_r, CMB_ratio=2.0, Rj_boundary=None):
    
    # according to Parkinson, Introduction to Geomagnetism, 1983, Scottish academic press, pages 313-315
    # see also equations in Methods by Kislyakova et al. 2017 NatAstron (after Parkinson, but in CGS)
    # the correct version for a sphere with non-uniform conductivity  
    # This version uses higher precision calculations to fix the overflow errors for Yihang Peng's calculations  

    cons_start = time.time()
    c0 = 2.99792458e10 # speed of light [cm/s]
    epsilon = 1.0
    mu = 1.0

    Be = mpmath.mpf(Borb)  # Convert to high precision
    sigma = [mpmath.mpf(s) for s in sigma_r]
    sigma_core = sigma[0] * CMB_ratio

    c0 = mpmath.mpf(c0)  # Speed of light in cm/s (CGS)
    mu = mpmath.mpf(mu)  # magnetic permeability
    epsilon = mpmath.mpf(epsilon)  # electric permeability
    omega = mpmath.mpf(omega)  # Convert omega to high precision
    mu0 = mpmath.mpf(4 * mpmath.pi * 1e-7)  # Permeability of vacuum in SI


    kj_core = mpmath.sqrt(-1j * 4 * mpmath.pi * omega * mu * sigma_core / c0**2)
    
    tmp1 = [4 * mpmath.pi * s for s in sigma]
    tmp2 = epsilon * omega 

    # Determine applicable regions
    equations_applicable = [i for i, t1 in enumerate(tmp1) if t1 > 10 * tmp2]
    equations_applicable = np.asarray(equations_applicable)

    r = [mpmath.mpf(r_ax[i]) for i in equations_applicable] # agrees with the low-res version
 
    nrad = len(r)
    Cj_cut = [mpmath.mpc(0) for _ in range(nrad)]
    Dj_cut = [mpmath.mpc(0) for _ in range(nrad)]
    Rj_cut = [mpmath.mpc(0) for _ in range(nrad)] # for recursive calculation of Cj and Dj
    kj_cut = [mpmath.mpc(0) for _ in range(nrad)]

    alphaj = [mpmath.mpc(0) for _ in range(nrad)]
    betaj = [mpmath.mpc(0) for _ in range(nrad)]
    gammaj = [mpmath.mpc(0) for _ in range(nrad)]
    deltaj = [mpmath.mpc(0) for _ in range(nrad)]
    epsilonj = [mpmath.mpc(0) for _ in range(nrad)]

    for i in range(nrad):
        # agrees with the low-precision version up to some mismatch in machine precision
        kj_cut[i] = mpmath.sqrt(-1j * 4 * mpmath.pi * omega * mu * sigma[i] / c0**2) 
        #print ('{:.20e}'.format(float(abs(kj_cut[i]))))
    #for i in range(nrad): print (abs(kj_cut[i]))

    #below we calculate the initial value at 0 (core-mantle boundary, CMB) separately            
    for i in range(nrad):
        #print ("1 ", i)
        if i == 0:
            betaj[i] = (
                (kj_cut[i] / kj_core)**0.5 * besselj_cached(3/2, r[i] * kj_core) * Jder_minus_hp_cached(r[i] * kj_cut[i]) -
                Jder_plus_hp_cached(r[i] * kj_core) * besselj_cached(-3/2, r[i] * kj_cut[i])
            )
            deltaj[i] = (
                Jder_plus_hp_cached(r[i] * kj_core) * besselj_cached(3/2, r[i] * kj_cut[i]) -
                (kj_cut[i] / kj_core)**0.5 * besselj_cached(3/2, r[i] * kj_core) * Jder_plus_hp_cached(r[i] * kj_cut[i])
            )
        else:
            alphaj[i] = (
                Jder_minus_hp_cached(r[i] * kj_cut[i]) * besselj_cached(3/2, r[i] * kj_cut[i]) -
                besselj_cached(-3/2, r[i] * kj_cut[i]) * Jder_plus_hp_cached(r[i] * kj_cut[i])
            )
            betaj[i] = (
                (kj_cut[i] / kj_cut[i-1])**0.5 * besselj_cached(3/2, r[i] * kj_cut[i-1]) * Jder_minus_hp_cached(r[i] * kj_cut[i]) -
                Jder_plus_hp_cached(r[i] * kj_cut[i-1]) * besselj_cached(-3/2, r[i] * kj_cut[i])
            )
            gammaj[i] = (
                (kj_cut[i] / kj_cut[i-1])**0.5 * besselj_cached(-3/2, r[i] * kj_cut[i-1]) * Jder_minus_hp_cached(r[i] * kj_cut[i]) -
                Jder_minus_hp_cached(r[i] * kj_cut[i-1]) * besselj_cached(-3/2, r[i] * kj_cut[i])
            )
            deltaj[i] = (
                Jder_plus_hp_cached(r[i] * kj_cut[i-1]) * besselj_cached(3/2, r[i] * kj_cut[i]) -
                (kj_cut[i] / kj_cut[i-1])**0.5 * besselj_cached(3/2, r[i] * kj_cut[i-1]) * Jder_plus_hp_cached(r[i] * kj_cut[i])
            )
            epsilonj[i] = (
                Jder_minus_hp_cached(r[i] * kj_cut[i-1]) * besselj_cached(3/2, r[i] * kj_cut[i]) -
                (kj_cut[i] / kj_cut[i-1])**0.5 * besselj_cached(-3/2, r[i] * kj_cut[i-1]) * Jder_plus_hp_cached(r[i] * kj_cut[i])
            )

    # Recursion for Rj_cut
    if Rj_boundary is not None:
        Rj_cut[0] = Rj_boundary
    else:
        Rj_cut[0] = betaj[0] / deltaj[0]
    print('Boundary condition at core:', Rj_cut[0])
    for i in range(1, nrad):
        Rj_cut[i] = (
            (Rj_cut[i-1] * betaj[i] + gammaj[i]) /
            (Rj_cut[i-1] * deltaj[i] + epsilonj[i])
        )

    # Surface boundary conditions
    betaj0 = (
        (r[-1] * kj_cut[-1])**0.5 * Jder_plus_hp_cached(r[-1] * kj_cut[-1]) -
        2 * besselj_cached(3/2, r[-1] * kj_cut[-1])
    )
    gammaj0 = (
        (r[-1] * kj_cut[-1])**0.5 * Jder_minus_hp_cached(r[-1] * kj_cut[-1]) -
        2 * besselj_cached(-3/2, r[-1] * kj_cut[-1])
    )
    deltaj0 = (
        2 * ((r[-1] * kj_cut[-1])**0.5 * Jder_plus_hp_cached(r[-1] * kj_cut[-1]) +
             besselj_cached(3/2, r[-1] * kj_cut[-1]))
    )
    epsilonj0 = (
        2 * ((r[-1] * kj_cut[-1])**0.5 * Jder_minus_hp_cached(r[-1] * kj_cut[-1]) +
             besselj_cached(-3/2, r[-1] * kj_cut[-1]))
    )
    Bi = Be * (Rj_cut[-1] * betaj0 + gammaj0) / (Rj_cut[-1] * deltaj0 + epsilonj0)

    # Calculate Cj and Dj recursively
    Cj_cut[-1] = 0.5 * (2 * Bi - Be) * (r[-1] * kj_cut[-1])**0.5 * r[-1] * (
        besselj_cached(3/2, r[-1] * kj_cut[-1]) +
        besselj_cached(-3/2, r[-1] * kj_cut[-1]) / Rj_cut[-1]
    )**(-1)
    Dj_cut[-1] = Cj_cut[-1] / Rj_cut[-1]

    for i in range(nrad-2, -1, -1):
        Cj_cut[i] = alphaj[i] * Cj_cut[i+1] / (betaj[i] + gammaj[i] / Rj_cut[i+1])
        Dj_cut[i] = Cj_cut[i] / Rj_cut[i]
    
    #this is just a set of longer arrays, which will contain zeros in the upper part, where the equations are not applicable
    # now it is superfluous because the conductivity profile is actually cut by a different function, so there is no need to
    # check the equation applicability in this function; but I'm too lazy to remove it now. Maybe later. Doesn't lead to any errors anyway
    nrad0 = len(r_ax)
    Cj = [mpmath.mpc(0) for _ in range(nrad0)] 
    Dj = [mpmath.mpc(0) for _ in range(nrad0)] 
    Rj = [mpmath.mpc(0) for _ in range(nrad0)] 
    kj = [mpmath.mpc(0) for _ in range(nrad0)] 
 
    for i in range(nrad):  # the upper part, where the equations are not applicable, stays equal zero
        Cj[i] = Cj_cut[i]
        Dj[i] = Dj_cut[i]
        Rj[i] = Rj_cut[i]
        kj[i] = kj_cut[i]
        #print ('{:.20e}'.format(float(abs(Cj[i]))))
        
    # we also shift the arrays by one point to avoid jumps in the magnetic field and heating profiles
    # the jumps arise due to "delay" by one point in Cj and Dj (recursive formula leads to that)
    # is there a better solution than just a shift? No - this is the best way of solving it, all other solutions are artificial
    for i in range(1,nrad):  
        Cj[i-1] = Cj[i]
        Dj[i-1] = Dj[i]    
    
    # the values returned by this function look similar to what calculate_constants2 returns 
    # up to machine precision (the values for the constants are different only many digits after the dot)

    print('Constants calculation time:', time.time() - cons_start, 's', flush=True)
    return Cj, Dj, kj, Bi, Rj

def energy_release_Parkinson(Cj, Dj, kj, r_ax, sigma_r, nTheta, omega):
    # according to Parkinson, Introduction to Geomagnetism, 1983, Scottish academic press, pages 313-315
    # see also equations in Methods by Kislyakova et al. 2017 NatAstron
    # another useful article is Srivastava, 1966, Theory of the magnetotelluric method  for a spherical conductor
    # this version is for high precision using mpmath to fix overflow errors
    # NB: all constants and all values used in calculations should be in the mpmath format

    energy_start = time.time()

    c0 = 2.99792458e10 # speed of light [cm/s]
    r = np.array(r_ax, dtype=np.float64)
    sigma = np.array(sigma_r, dtype=np.float64)
    kj = np.array(kj, dtype=np.complex128)
    nrad = len(r)
    Theta = np.linspace(0, np.pi, nTheta)
    sin_Theta = np.sin(Theta)
    cos_Theta = np.cos(Theta)

    A_phi = np.zeros((nrad, nTheta), dtype=np.complex128)
    J_phi = np.zeros((nrad, nTheta), dtype=np.complex128)
    Q = np.zeros((nrad, nTheta))

    for i in range(nrad):
        if i % 500 == 0:
            print(i)
        if kj[i] == 0:
            continue
        Fr =(r[i] * kj[i])**(-0.5)*(Cj[i]*besselj_cached((3/2),r[i] * kj[i]) + Dj[i]*besselj_cached((-3/2),r[i] * kj[i]))
        A_phi[i, :] = sin_Theta * Fr
        J_phi[i, :] = c0 * A_phi[i, :] * (kj[i]**2) / (4 * np.pi)
    
    for i in range(1, nrad):
        dV = (2/3)*np.pi*(r[i]**3 - r[i-1]**3) * (cos_Theta[:-1] - cos_Theta[1:])
        J2 = np.abs(J_phi[i,1:])**2
        Q[i, 1:] = 0.5 * J2 * dV / sigma[i]

    print('Energy release calculation time', time.time() - energy_start, 's', flush=True)

    return Q, np.abs(J_phi), Theta

def magnetic_field_Parkinson(kj, r_ax, nTheta, Cj, Dj, sigma_r):

    mag_start = time.time()

    r = np.array(r_ax, dtype=np.float64)
    kj = np.array(kj, dtype=np.complex128)
    Cj = np.array(Cj, dtype=np.complex128)
    Dj = np.array(Dj, dtype=np.complex128)

    nrad = len(r)
    Theta = np.linspace(0, np.pi, nTheta)
    dtheta = Theta[1] - Theta[0]
    sin_Theta = np.sin(Theta)
    cos_Theta = np.cos(Theta)

    Br = np.zeros((nrad, nTheta), dtype=np.complex128)
    Btheta = np.zeros((nrad, nTheta), dtype=np.complex128)
    Bmod = np.zeros((nrad, nTheta), dtype=np.float64)

    print('Total number of layers:', nrad, flush=True)
    for i in range(nrad):
        if i % int(nrad/10) == 0:
            print('Layer:', i, flush=True)
        if kj[i] == 0:
            continue
        rk = r[i] * kj[i]
        rk_sqrt_inv = (rk)**-0.5
        r_inv = 1.0 / r[i]

        Jp = Jder_plus_hp_cached(rk)
        Jm = Jder_minus_hp_cached(rk)
        J32 = besselj_cached(1.5, rk)
        Jm32 = besselj_cached(-1.5, rk)

        fr = rk_sqrt_inv * r_inv * (Cj[i] * J32 + Dj[i] * Jm32)
        fth = -r_inv * (Cj[i] * Jp + Dj[i] * Jm)

        Br[i, :] = 2.0 * fr * cos_Theta
        Btheta[i, :] = fth * sin_Theta

    # Compute |B|
    Bmod = np.sqrt(np.abs(Br)**2 + np.abs(Btheta)**2)

    # Compute ∂Br/∂θ and ∂(rBtheta)/∂r using central finite differences
    dBr_dtheta = np.gradient(Br, dtheta, axis=1)
    rBtheta = r[:, None] * Btheta
    drBtheta_dr = np.gradient(rBtheta, r, axis=0)

    # Compute jphi and Q
    curlB_phi = (drBtheta_dr - dBr_dtheta) / r[:, None]
    J_phi = (c0 / (4 * np.pi)) * curlB_phi
    Q = np.zeros((nrad, nTheta))

    for i in range(1, nrad-2):
        dV = (2/3)*np.pi*(r[i]**3 - r[i-1]**3) * (cos_Theta[:-1] - cos_Theta[1:])
        J2 = np.abs(J_phi[i,1:])**2
        Q[i, 1:] = 0.5 * J2 * dV / sigma_r[i]

    print('Magnetic field calculation time', time.time() - mag_start, 's', flush=True)
    return np.abs(Br), np.abs(Btheta), Bmod, np.abs(J_phi), Q
