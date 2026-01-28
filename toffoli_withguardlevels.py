#  Quandary's python interface functions are defined in /path/to/quandary/quandary.py. Import them here. 
from quandary import * 
import numpy as np 

## Two qubit test case: CNOT gate, two levels each, 1 or 2 guard levels, dipole-dipole coupling 5MHz ##
Ne = [2, 2, 2]
Ng = [2, 2, 2] # [1, 1]

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.8, 5.0, 5.2] 
selfkerr = [0.15, 0.15, 0.15]

# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [0.005, 0.0, 0.005]  # Dipole-Dipole coupling of qubit 0<->1

# Frequency of rotations for computational frame [GHz] per oscillator
favg = freq01[1] #sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# Set the pulse duration (ns)
T = 300.0

# Set up the CNOT target gate
unitary = np.identity(8)
unitary[6,6] = 0.0
unitary[7,7] = 0.0
unitary[6,7] = 1.0
unitary[7,6] = 1.0
# print("Target gate: ", unitary)

# Flag for printing out more information to screen
verbose = True

# For reproducability: Random number generator seed
rand_seed=1234

# Piecewise constant B-spline
# spline_order = 0 
# spline_knot_spacing = 1.0  # [ns] Width of the constant control segments (dist. between basis functions)

# Piecewise quadratic B-spline
spline_order = 2
spline_knot_spacing = 10.0 # [ns] Distance between basis functions

# In order get less noisy control functions, activate the penalty term for variation of the control parameters
# gamma_variation = 1.0

# Optionally: let controls functions start and end near zero
control_enforce_BC = False

# Max # optimization iterations
maxiter = 500

# Amplitude of randomized initial control vector
initctrl = 10.0e-3
maxctrl = 30.0e-3

cw_amp_thres = 5e-3 # the resonant growth rate must be larger than this number
cw_prox_thres = 0.5*selfkerr[0] # no two carrier frequencies can be closer than this number
gamma_leakage = 0.1

# Set up the Quandary configuration for this test case. Make sure to pass all of the above to the corresponding fields, compare help(Quandary)!
qmodel = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, Jkl=Jkl, rotfreq=rotfreq, T=T,targetgate=unitary, verbose=verbose, rand_seed=rand_seed, initctrl=initctrl, cw_prox_thres=cw_prox_thres, cw_amp_thres=cw_amp_thres, spline_knot_spacing=spline_knot_spacing, control_enforce_BC=control_enforce_BC, gamma_leakage=gamma_leakage, maxiter=maxiter) 
#gamma_variation=gamma_variation,

# Turn off verbosity after the carrier frequencies have been reported
qmodel.verbose = False

for q in range(len(qmodel.carrier_frequency)):
    print(f"Site {q} Carrier freq: {qmodel.carrier_frequency[q][:]}")

datadir="./Toffoli_run_dir"
# Optimize the control vector
t, pt, qt, infidelity, expectedEnergy, population = qmodel.optimize(datadir=datadir)

# Plot the control pulse and expected energy level evolution
if True:
    plot_pulse(qmodel.Ne, t, pt, qt)
    plot_expectedEnergy(qmodel.Ne, t, expectedEnergy) 
    plot_population(qmodel.Ne, t, population)

