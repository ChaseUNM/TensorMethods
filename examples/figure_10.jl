# Increase ITensor warning threshold for tensor order
ITensors.set_warn_order!(100)

################################################################################
# Generate data and plots for blue lines in Figure 10 (arxiv 2603.13990)
################################################################################

# -------------------------------------------------------------------
# Flags to determine which algorithm(s) are run for this test problem
# -------------------------------------------------------------------
TDVP = true
BUG_MPS = true
BUG_Tucker = false
plot_spline = false

# -------------------------------------------------------------------
# System size and local Hilbert space dimensions
# -------------------------------------------------------------------

# Number of qubits in the chain
nqubits = 11

# Local Hilbert space dimension at each site
nlevels = fill(2, nqubits)

# Create qudit site indices
sites = qudit_siteinds(nqubits, nlevels)

# -------------------------------------------------------------------
# Set up qubit transition frequencies
# -------------------------------------------------------------------

# Bare 0-1 transition frequencies [GHz]
freq01_all = zeros(nqubits)

for i in 0:nqubits - 1
    # Determine which block of 4 qubits this site belongs to
    block = fld(i, 4)

    # Position within the current block
    k = i % 4

    if block % 2 == 0
        # "Up" block: descending frequency ladder
        f = 5.18 - 0.06 * k
    else
        # "Down" block: shifted to reduce frequency collisions
        f = 5.18 - 0.06 * (3 - k) + 0.03
    end 

    freq01_all[i + 1] = f
end

# Convert frequencies to angular frequency units
freq01_all = freq01_all * 2pi

# -------------------------------------------------------------------
# Time interval and rotating frame setup
# -------------------------------------------------------------------

# Rotating-frame frequency: average over all qubit frequencies
favg = sum(freq01_all) / length(freq01_all)

# Initial and final time [ns]
t0 = 0.0
T = 40.0

# -------------------------------------------------------------------
# Hamiltonian coupling parameters
# -------------------------------------------------------------------

# Nearest-neighbor coupling strength [GHz] converted to angular frequency
Jkl_coupling_strength = 0.0 * 2pi

# Coupling matrix for chain topology
Jkl = zeros(nqubits, nqubits)

for i in 1:nqubits
    for j in i+1:nqubits
        if j == i + 1
            Jkl[i, j] = Jkl_coupling_strength
        end
    end
end 

# Self-Kerr nonlinearities (set to zero here)
self_kerr = zeros(nqubits) * 2pi 

# ZZ interaction matrix (set to zero here)
zz = zeros(nqubits, nqubits) * 2pi

# -------------------------------------------------------------------
# Carrier frequencies and rotating frame frequencies
# -------------------------------------------------------------------

# Each qubit gets a single carrier frequency relative to rotating frame
carrier_freq = [[freq01_all[iq] - favg] for iq in 1:nqubits] 

# Rotating-frame frequencies for all qubits
rotfreq = favg * ones(nqubits)

# -------------------------------------------------------------------
# Time discretization and pulse parameterization
# -------------------------------------------------------------------

# Time step size [ns]
dT = 0.01

# Number of time steps
steps = Int64((T - t0) / dT)

# Number of spline segments for pulse parameterization
splines = 6

# -------------------------------------------------------------------
# Import optimized spline pulse parameters
# -------------------------------------------------------------------

# File containing spline coefficients for this nqubits configuration
datafile = joinpath(@__DIR__, "spline_params", "params_$nqubits.dat")

# Load spline coefficients
pcof = vec(readdlm(datafile))

# Build pulse parameter object
bc_params = bcparams(T, splines, carrier_freq, pcof)

# -------------------------------------------------------------------
# Initial state and target state
# -------------------------------------------------------------------

# SVD truncation cutoff
cutoff = 1E-10

# Initial computational basis state |0,0,...,0⟩
q_state = fill(0, nqubits)

# Target state: equal superposition state
target_vec = fill(1 / sqrt(2)^nqubits, 2^nqubits)

# Same target state represented as an MPS
target_mps = equal_separable(sites)

# -------------------------------------------------------------------
# Build initial condition and Hamiltonian for MPS-based methods
# -------------------------------------------------------------------

if TDVP == true || BUG_MPS == true
    # Initial separable MPS state
    init_mps = init_separable(sites, q_state)

    # Convert initial MPS to full vector form
    init_vec = vectorize_mps(init_mps)

    # Construct drift Hamiltonian MPO
    H_d = drift_MPO(nqubits, sites, freq01_all, rotfreq, self_kerr, zz, Jkl)
end 

# -------------------------------------------------------------------
# Run BUG-Tucker method (if enabled)
# -------------------------------------------------------------------

if BUG_Tucker == true 
    # Construct separable Tucker tensor initial condition
    init_core, init_factors = tucker_separable(q_state)

    # Reconstruct dense tensor from Tucker form
    init_array = Multi_TTM_recursive(init_core, init_factors)

    # Convert dense tensor to vector form
    init_vec = vec(permutedims(init_array, reverse(1:ndims(init_array))))

    # Construct Hamiltonian in operator/tensor form
    H_s_ops = H_sys_rot(nqubits, nlevels, freq01_all, rotfreq, self_kerr, Jkl, zz)

    t = @elapsed begin 
        # Run BUG-Tucker time evolution
        ans_core, ans_factors, state_bug_tucker, energy_bug_tucker, link_bug_tucker =
            bug_integrator_mat_ra(
                H_s_ops,
                bc_params,
                init_core,
                init_factors,
                t0,
                T,
                steps;
                cutoff = cutoff^2,
                state = false,
                energy = false
            )
    end

    # Reconstruct final dense tensor
    bug_array = Multi_TTM_recursive(ans_core, ans_factors)

    # Convert final tensor to vector form
    bug_tucker_vec = vec(permutedims(bug_array, reverse(1:ndims(bug_array))))

    # Compute fidelity with target state
    bug_tucker_fidelity = abs2(bug_tucker_vec' * target_vec)
    println("BUG Tucker time: $t seconds")
    println("BUG Tucker infidelity: $(1 - bug_tucker_fidelity)")
end

# -------------------------------------------------------------------
# Run TDVP2 method (if enabled)
# -------------------------------------------------------------------

if TDVP == true 
    t = @elapsed begin 
        # Run TDVP2 time evolution
        ans_tdvp, link_tdvp, magnet_tdvp, energy_tdvp, cutoff_error =
            tdvp2(
                H_d,
                init_mps,
                t0,
                T,
                steps,
                bc_params;
                cutoff = cutoff^2,
                strang = false,
                magnet = false,
                energy = false
            )
    end

    # Compute fidelity with target state
    tdvp_fidelity = abs2(inner(conj(ans_tdvp), target_mps))

    println("TDVP time: $t seconds")
    println("TDVP infidelity: $(1 - tdvp_fidelity)")
end

# -------------------------------------------------------------------
# Run BUG-MPS method (if enabled)
# -------------------------------------------------------------------

if BUG_MPS == true 
    t = @elapsed begin 
        # Run BUG-MPS time evolution
        ans_bug_mps, link_bug_mps, magnet_bug_mps, energy_bug_mps =
            mps_bug(
                H_d,
                bc_params,
                init_mps,
                t0,
                T,
                steps;
                cutoff = cutoff^2,
                magnet = false,
                energy = false
            )
    end

    # Compute fidelity with target state
    bug_mps_fidelity = abs2(inner(conj(ans_bug_mps), target_mps))
    println("BUG MPS time: $t seconds")
    println("BUG MPS infidelity: $(1 - bug_mps_fidelity)")
end

# -------------------------------------------------------------------
# Optional: plot control pulse for a chosen qubit
# -------------------------------------------------------------------

if plot_spline == true 
    # Choose qubit whose control pulse is plotted
    qubit = 1

    # Time grid for evaluating pulse
    time_range = LinRange(t0, T, steps)

    # Arrays for in-phase and quadrature pulse components
    p_eval = zeros(length(time_range))
    q_eval = zeros(length(time_range))

    # Evaluate pulse over time
    for j = 1:steps 
        p_eval[j] = bcarrier2(time_range[j], bc_params, 2 * (qubit - 1)) * (500 / pi)
        q_eval[j] = bcarrier2(time_range[j], bc_params, 2 * (qubit - 1) + 1) * (500 / pi)
    end

    # Plot pulse components
    pulse_plot = plot(
        time_range,
        [p_eval, q_eval],
        labels = ["p(t)" "q(t)"],
        xlabel = "time(ns)",
        ylabel = "MHz"
    )
end