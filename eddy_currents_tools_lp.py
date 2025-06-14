import os
import math
import cmath
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy import special as sci_sp

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

def Jder_plus(a):
    # Parker's J^*_m(rk)=(d/dr)((r/k)^{1/2}Jm(rk)), for m=1+1/2 (eq. 386) (derivative)
    if a == 0j: J_derivative_plus = 0j
    else: J_derivative_plus =  (a)**(0.5)*sci_sp.jn(1./2.,a)  - (a)**(-0.5)*sci_sp.jn(3./2.,a)      
    return J_derivative_plus

def Jder_minus(a):
    # Parker's J^*_m(rk)=(d/dr)((r/k)^{1/2}Jm(rk)), for m=-1-1/2 (derivative)
    #print a
    if a == 0j: J_derivative_minus = 0j
    else: J_derivative_minus = -(a)**(0.5)*sci_sp.jn(-1./2.,a) + (a)**(-0.5)*sci_sp.jn(-3./2.,a)   # NB: there is a wrong sign in K17 in this equation (the second "+" is shown as a "-")    
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
    cons_start = time.time()
    Be = Borb
    sigma = sigma_r

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
    
    r = r_ax

    # Preallocate arrays
    Cj_cut = np.zeros(r.shape[0], dtype=complex)
    Dj_cut = np.zeros(r.shape[0], dtype=complex)
    Rj_cut = np.zeros(r.shape[0], dtype=complex)
    kj_cut = np.zeros(r.shape[0], dtype=complex)

    alphaj = np.zeros(r.shape[0], dtype=complex)
    betaj = np.zeros(r.shape[0], dtype=complex)
    gammaj = np.zeros(r.shape[0], dtype=complex)
    deltaj = np.zeros(r.shape[0], dtype=complex)
    epsilonj = np.zeros(r.shape[0], dtype=complex)

    # Compute kj for each layer
    for i in range(r.shape[0]):
        kj_cut[i] = cmath.sqrt(-1j * 4.0 * cmath.pi * omega * mu * sigma[i] / c0**2)

    for i in range(r.shape[0]):
        alphaj[i] = Jder_minus(r[i] * kj_cut[i]) * sci_sp.jn(1.5, r[i] * kj_cut[i]) - sci_sp.jn(-1.5, r[i] * kj_cut[i]) * Jder_plus(r[i] * kj_cut[i])
        betaj[i] = (kj_cut[i] / kj_cut[i - 1])**0.5 * sci_sp.jn(1.5, r[i] * kj_cut[i - 1]) * Jder_minus(r[i] * kj_cut[i]) - Jder_plus(r[i] * kj_cut[i - 1]) * sci_sp.jn(-1.5, r[i] * kj_cut[i])
        gammaj[i] = (kj_cut[i] / kj_cut[i - 1])**0.5 * sci_sp.jn(-1.5, r[i] * kj_cut[i - 1]) * Jder_minus(r[i] * kj_cut[i]) - Jder_minus(r[i] * kj_cut[i - 1]) * sci_sp.jn(-1.5, r[i] * kj_cut[i])
        deltaj[i] = Jder_plus(r[i] * kj_cut[i - 1]) * sci_sp.jn(1.5, r[i] * kj_cut[i]) - (kj_cut[i] / kj_cut[i - 1])**0.5 * sci_sp.jn(1.5, r[i] * kj_cut[i - 1]) * Jder_plus(r[i] * kj_cut[i])
        epsilonj[i] = Jder_minus(r[i] * kj_cut[i - 1]) * sci_sp.jn(1.5, r[i] * kj_cut[i]) - (kj_cut[i] / kj_cut[i - 1])**0.5 * sci_sp.jn(-1.5, r[i] * kj_cut[i - 1]) * Jder_plus(r[i] * kj_cut[i])

    # Set artificial boundary at cutoff_index
    Rj_cut[cutoff_index] = Rj_boundary  # Perfect reflector
    print('Boundary condition N1:', Rj_cut[cutoff_index])

    # Recursively compute Rj from cutoff_index upward
    for i in range(cutoff_index + 1, r.shape[0]):
        Rj_cut[i] = (Rj_cut[i - 1] * betaj[i] + gammaj[i]) / (Rj_cut[i - 1] * deltaj[i] + epsilonj[i])

    # Surface boundary condition
    r_surface = r[-1]
    kj_surface = kj_cut[-1]
    betaj0 = (r_surface * kj_surface)**0.5 * Jder_plus(r_surface * kj_surface) - 2.0 * sci_sp.jn(1.5, r_surface * kj_surface)
    gammaj0 = (r_surface * kj_surface)**0.5 * Jder_minus(r_surface * kj_surface) - 2.0 * sci_sp.jn(-1.5, r_surface * kj_surface)
    deltaj0 = 2.0 * ((r_surface * kj_surface)**0.5 * Jder_plus(r_surface * kj_surface) + sci_sp.jn(1.5, r_surface * kj_surface))
    epsilonj0 = 2.0 * ((r_surface * kj_surface)**0.5 * Jder_minus(r_surface * kj_surface) + sci_sp.jn(-1.5, r_surface * kj_surface))

    Bi = Be * (Rj_cut[-1] * betaj0 + gammaj0) / (Rj_cut[-1] * deltaj0 + epsilonj0)

    # Compute Cj and Dj from surface downward
    Cj_cut[-1] = 0.5 * (2.0 * Bi - Be) * (r_surface * kj_surface)**0.5 * r_surface / (
        sci_sp.jn(1.5, r_surface * kj_surface) + sci_sp.jn(-1.5, r_surface * kj_surface) / Rj_cut[-1]
    )
    Dj_cut[-1] = Cj_cut[-1] / Rj_cut[-1]

    for i in range(r.shape[0] - 2, cutoff_index - 1, -1):
        Cj_cut[i] = alphaj[i] * Cj_cut[i + 1] / (betaj[i] + gammaj[i] / Rj_cut[i + 1])
        Dj_cut[i] = Cj_cut[i] / Rj_cut[i]

    # Final arrays
    Cj = np.zeros(r_ax.shape[0], dtype=complex)
    Dj = np.zeros(r_ax.shape[0], dtype=complex)
    Rj = np.zeros(r_ax.shape[0], dtype=complex)
    kj = np.zeros(r_ax.shape[0], dtype=complex)

    for i in range(r.shape[0]):
        Cj[i] = Cj_cut[i]
        Dj[i] = Dj_cut[i]
        Rj[i] = Rj_cut[i]
        kj[i] = kj_cut[i]

    # Shift arrays to remove 1-layer delay
    for i in range(1, r.shape[0]):
        Cj[i - 1] = Cj[i]
        Dj[i - 1] = Dj[i]

    print('Constants calculation time:', time.time() - cons_start, 's', flush=True)
    return Cj, Dj, kj, Bi, Rj

def energy_release_Parkinson(Cj, Dj, kj, r_ax, sigma_r, nTheta, omega):
    # according to Parkinson, Introduction to Geomagnetism, 1983, Scottish academic press, pages 313-315
    # see also equations in Methods by Kislyakova et al. 2017 NatAstron
    # another useful article is Srivastava, 1966, Theory of the magnetotelluric method  for a spherical conductor
    
    energy_start = time.time()
    Theta = np.linspace(0.0, math.pi, nTheta) # Theta axis
    J_phi = np.zeros((r_ax.shape[0], nTheta), dtype=np.complex128)    
    J_phi_dens = np.zeros((r_ax.shape[0], nTheta), dtype=float)    
    A_phi = np.zeros_like(J_phi) # vector potential
    sin_theta = np.sin(Theta)
    
    r = r_ax#/1e2  # cm -> m
    sigma = sigma_r#/Sm_m # CGS -> SI

    for i in range(0,r.shape[0]):
        if i % int(r.shape[0]/10) == 0:
            print('Layer:', i, flush=True)
        if kj[i] == 0j: 
            continue
        else:
            F_r = (r[i]*kj[i])**(-0.5)*(Cj[i]*sci_sp.jn(3./2.,r[i]*kj[i]) + Dj[i]*sci_sp.jn(-3./2.,r[i]*kj[i]))  # CORRECT
            A_phi[i,:] = sin_theta*F_r              # vector potential for the external spherical harmonics of the first order (n=1)
            J_phi[i,:] = c0*A_phi[i,:]*kj[i]**2/(4.*np.pi) # ----> here we find current based on the vector potential; the results is the same as when calculated with the electric field

    # now we calculate energy release inside a layer as: Q = 1/2 int |E|^2 sigma dV 
    Q = np.zeros((r.shape[0], nTheta), dtype=float)    
    for i in range (1,r.shape[0]):
      for j in range(1,nTheta):
        #volume of the layer
        dV = 2./3.*np.pi*(r[i]**3-r[i-1]**3)*(np.cos(Theta[j-1])-np.cos(Theta[j]))

        if (dV < 0): print('Negative volume of a shell!')     
        J_phi_squared_av = (J_phi[i,j].real**2. + J_phi[i,j].imag**2.) 
        J_phi_dens[i,j] = np.sqrt(J_phi_squared_av)                

        # energy release                       
        Q[i,j] = 0.5*J_phi_squared_av*dV/sigma[i]  # now we have here a dependence on theta, in case we decide to use it. NB: 0.5 comes from <e^iwt>, i.e., from averaging over time

    print('Energy release calculation time', time.time() - energy_start, 's', flush=True)
    return Q, np.abs(J_phi_dens), Theta

def magnetic_field_Parkinson(kj, r_ax, nTheta, Cj, Dj, sigma_r):
    # magnetic field inside a sphere with layers with different conductivity. 
    # after Parkinson, 1983, Edinburgh, chapter 5, p. 311-315
    # The function calculates the magnetic field inside the jth layer. 
    # The constants Cj and Dj are calculated in a different function
    # arguments:
    # nTheta - number of steps in Theta coordinate
    # kj - the complex wave number vs radius
    # Cj, Dj, Rj - complex constants vs radius
    # returns: two components of the magnetic field, Br and Btheta, and module of the field Bmod  
    # this version is only for the uniform external field

    mag_start = time.time()
    r = r_ax#/1e2  # cm -> m; CGS ->SI
    Br = np.zeros((r.shape[0], nTheta), dtype=np.complex128)
    Btheta = np.zeros((r.shape[0], nTheta), dtype=np.complex128)
    Bmod = np.zeros((r.shape[0], nTheta), dtype=float)
    Theta = np.linspace(0.0, math.pi, nTheta) # Theta axis
    dtheta = Theta[1] - Theta[0]
    sin_theta = np.sin(Theta)
    cos_theta = np.cos(Theta)

    print('Total number of layers:', r.shape[0], flush=True)
    for i in range(0,r.shape[0]):
        if i % int(r.shape[0]/10) == 0:
            print('Layer:', i, flush=True)
        if (kj[i] == 0j):
            continue
        else:
            F_r = (r[i]*kj[i])**(-0.5)*(Cj[i]*sci_sp.jn(3./2.,r[i]*kj[i]) + Dj[i]*sci_sp.jn(-3./2.,r[i]*kj[i]))  # CORRECT
            Br[i,:] = (1/r[i]) * F_r * 2 * cos_theta  
            Btheta[i,:] = -r[i]**(-1)*(Cj[i]*Jder_plus(r[i]*kj[i]) + Dj[i]*Jder_minus(r[i]*kj[i]))*sin_theta

    # Compute |B|
    Bmod = np.sqrt(np.abs(Br)**2 + np.abs(Btheta)**2)

    # Compute ∂Br/∂θ and ∂(rBtheta)/∂r using central finite differences
    dBr_dtheta = np.gradient(Br, dtheta, axis=1)
    rBtheta = r[:, None] * Btheta
    drBtheta_dr = np.gradient(rBtheta, r, axis=0)

    # Compute jphi and Q
    curlB_phi = (dBr_dtheta - drBtheta_dr) / r[:, None]
    J_phi = (c0 / (4 * np.pi)) * curlB_phi

    # now we calculate energy release inside a layer as: Q = 1/2 int |E|^2 sigma dV 
    Q = np.zeros((r.shape[0], nTheta), dtype=float)    
    for i in range (1,r.shape[0]-2):
      for j in range(1,nTheta):
        #volume of the layer
        dV = 2./3.*np.pi*(r[i]**3-r[i-1]**3)*(np.cos(Theta[j-1])-np.cos(Theta[j]))

        if (dV < 0): print('Negative volume of a shell!')     
        J_phi_squared_av = np.abs(J_phi[i,j])**2

        # energy release                       
        Q[i,j] = 0.5*J_phi_squared_av*dV/sigma_r[i]  # now we have here a dependence on theta, in case we decide to use it. NB: 0.5 comes from <e^iwt>, i.e., from averaging over time

    print('Magnetic field calculation time', time.time() - mag_start, 's', flush=True)
    return np.abs(Br), np.abs(Btheta), Bmod, np.abs(J_phi), Q
