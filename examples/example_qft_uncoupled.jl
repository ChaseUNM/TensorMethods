using ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles, Plots
using TensorMethods


#Simulate quantum evolution for N (up to 18) uncoupled qubits, with control pulses obtained from Quandary

#Set because otherwise ITensors gives long warning messages
ITensors.set_warn_order!(100)


#Conditions determine whether a certain algorithm is solving the test problem.
TDVP = true
BUG_Tucker = false
plot_spline = false

#Set up # of systems and respective energy levels 
nqubits = 10
nlevels = fill(2, nqubits)
sites = qudit_siteinds(nqubits, nlevels)


#Set up transitional frequencies
freq01_all = zeros(nqubits)  # [GHz] 0-1 frequency
for i in 0:nqubits - 1
    block = fld(i,4)
    k = i % 4
    if block % 2 == 0 # up block
        f = 5.18 - 0.06*k
    else # down block shifted by half spacing to avoid collisions
        f = 5.18 - 0.06*(3 - k) + 0.03
    end 
    freq01_all[i + 1] = f
end
freq01_all = freq01_all*2pi

# Rotational frequency: average of all qubit frequencies
favg = sum(freq01_all)/length(freq01_all)
t0 = 0.0
T = 40.0

#Set other Hamiltonian parameters
Jkl_coupling_strength = 0*2pi # 5e-3  	# [GHz] Coupling strength for qubit CHAIN topology
Jkl = zeros(nqubits, nqubits)
for i in 1:nqubits
	for j in i+1:nqubits
		if j == i+1
			Jkl[i,j] = Jkl_coupling_strength
        end
    end
end 

self_kerr = zeros(nqubits)*2pi 
zz = zeros(nqubits, nqubits)*2pi

carrier_freq = [[freq01_all[iq] - favg] for iq in 1:nqubits] 
rotfreq = favg*ones(nqubits)

dT = 0.01
steps = Int64((T - t0)/dT)
splines = 6


#Import spline parameters
datafile = joinpath(@__DIR__, "spline_params", "params_$nqubits.dat")
pcof = vec(readdlm(datafile))

bc_params = bcparams(T, splines, carrier_freq, pcof)


#Set SVD truncation cutoff
cutoff = 1E-3
q_state = fill(0, nqubits)

#Create target vector and target_mps
target_vec = fill(1/sqrt(2)^nqubits, 2^nqubits)
target_mps = equal_separable(sites)

#Set up initial condition as separable state (every qubit is in 0 state)
if TDVP == true
    init_mps = init_separable(sites, q_state)
    init_vec = vectorize_mps(init_mps)
    H_d = drift_MPO(nqubits, sites, freq01_all, rotfreq, self_kerr, zz, Jkl)
    ans_tdvp, link_tdvp, magnet_tdvp, energy_tdvp, cutoff_error = tdvp2(H_d, init_mps, t0, T, steps, bc_params; cutoff = cutoff^2, strang = false, magnet = false, energy = false)
    tdvp_fidelity = abs2(inner(conj(ans_tdvp), target_mps))
    println("TDVP fidelity: $tdvp_fidelity")
end 

if BUG_Tucker == true 
    init_core, init_factors = tucker_separable(q_state)
    init_array = Multi_TTM_recursive(init_core, init_factors)
    init_vec = vec(permutedims(init_array, reverse(1:ndims(init_array))))

    H_s_ops = H_sys_rot(nqubits, nlevels, freq01_all, rotfreq, self_kerr, Jkl, zz)

    ans_core, ans_factors, state_bug_tucker, energy_bug_tucker, link_bug_tucker = bug_integrator_mat_ra(H_s_ops, bc_params, init_core, init_factors, t0, T, steps; cutoff = cutoff^2, state = false, energy = false)
    bug_array = Multi_TTM_recursive(ans_core, ans_factors)
    bug_tucker_vec = vec(permutedims(bug_array, reverse(1:ndims(bug_array))))
    bug_tucker_fidelity = abs2(bug_tucker_vec'*target_vec)
    println("BUG Tucker fidelity: $bug_tucker_fidelity")
end

#Plots control pulses for specific qubit
if plot_spline == true 
    qubit = 1
    time_range = LinRange(t0, T, steps)
    p_eval = zeros(length(time_range))
    q_eval = zeros(length(time_range))
    for j = 1:steps 
        p_eval[j] = bcarrier2(time_range[j], bc_params, 2*(qubit - 1))*(500/pi)
        q_eval[j] = bcarrier2(time_range[j], bc_params, 2*(qubit - 1) + 1)*(500/pi)
    end

    pulse_plot = plot(time_range, [p_eval, q_eval], labels = ["p(t)" "q(t)"], xlabel = "time(ns)", ylabel = "MHz")
end