using ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles, Plots, BenchmarkTools
using TensorMethods

################################################################################
# Generate data and plots for Figure 9, N = 10 case (arxiv 2603.13990)
################################################################################

# Toggle which plots / methods to include
plot_pulse = false
plot_tdvp = true

# renamed original `plot_bug` to `plot_bug_tucker` and add `plot_bug_mps`
plot_bug_tucker = false
plot_bug_mps = true

# -------------------------
# System setup
# -------------------------

# Number of qudits / qubits
N = 10

# Local Hilbert space dimension for each subsystem
nlevels = fill(2, N)

# Standard qubit site indices (currently unused)
# sites = siteinds("Qubit", N)

# Build qudit site indices with specified local dimensions
sites = qudit_siteinds(N, nlevels)

# Bare 0-1 transition frequencies for each qubit [GHz], converted to angular frequency
freq01_all = [5.18, 5.12, 5.06, 5.0, 4.94, 4.88, 4.82, 4.76, 4.7, 4.74] .* 2pi

# Self-Kerr nonlinearities (set to zero here)
self_kerr = zeros(N)

# ZZ coupling matrix (set to zero here)
zz = zeros(N, N)

# Nearest-neighbor coupling strength
J = 5E-3 * 2pi

# Coupling matrix Jkl
Jkl = zeros(N, N)
for i = 2:N 
    Jkl[i - 1, i] = J 
end

# -------------------------
# Time evolution parameters
# -------------------------

t0 = 0.0
T = 40.0 				# [ns] Pulse duration

# Number of spline segments used for control pulse parameterization
splines = 6

# Number of time steps used in evolution
steps = 2620 * 4

# Reverse frequency ordering and truncate to N qubits
freq01 = reverse(freq01_all)[1:N]

# Average frequency used for rotating frame
favg = sum(freq01) / N 

# Rotating frame frequencies (all set equal to the average)
rot_freq = ones(N) .* favg 

# -------------------------
# Load pulse parameters
# -------------------------

# Path to optimized pulse spline coefficients
datafile = joinpath(@__DIR__, "spline_params", "params_10_coupled.dat")

# Read spline coefficient data from file
pcof = vec(readdlm(datafile))

# -------------------------
# Carrier frequency setup
# -------------------------

# Each qubit has a list of carrier frequencies used in the pulse expansion
carrier_frequency_list = Vector{Vector{Float64}}(undef, N)

carrier_frequency_list[1] = [-0.17999999999999972, -0.21999999999999975] .* 2pi
carrier_frequency_list[2] = [-0.17999999999999972, -0.21999999999999975, -0.16000000000000014] .* 2pi
carrier_frequency_list[3] = [-0.21999999999999975, -0.16000000000000014, -0.09999999999999964] .* 2pi
carrier_frequency_list[4] = [-0.16000000000000014, -0.09999999999999964, -0.040000000000000036] .* 2pi
carrier_frequency_list[5] = [-0.09999999999999964, -0.040000000000000036, 0.020000000000000462] .* 2pi
carrier_frequency_list[6] = [-0.040000000000000036, 0.020000000000000462, 0.08000000000000007] .* 2pi
carrier_frequency_list[7] = [0.020000000000000462, 0.08000000000000007, 0.13999999999999968] .* 2pi
carrier_frequency_list[8] = [0.08000000000000007, 0.13999999999999968, 0.20000000000000018] .* 2pi 
carrier_frequency_list[9] = [0.13999999999999968, 0.20000000000000018, 0.2599999999999998] .* 2pi 
carrier_frequency_list[10] = [0.20000000000000018, 0.2599999999999998] .* 2pi

# Build pulse boundary/carrier parameter object
bc_params = bcparams(T, splines, carrier_frequency_list, pcof)

# -------------------------
# Initial state and Hamiltonians
# -------------------------

# Initial product state |0,0,...,0⟩
q_state = fill(0, N)

# Construct initial MPS
init_MPS = init_separable(sites, q_state)

# Convert initial MPS to full vector form
init_vec = vectorize_mps(init_MPS)

# Reshape full vector into dense tensor form
init_ten = reshape(init_vec, fill(2, N)...)

# Compute Tucker decomposition of the initial tensor
init_core, init_factors = tucker(init_ten; cutoff = 0.0)

# Construct drift Hamiltonian as MPO
H_s = drift_MPO(N, sites, freq01, rot_freq, self_kerr, zz, Jkl)

# Construct system Hamiltonian in operator/tensor form
H_s_ops = H_sys_rot(N, nlevels, freq01, rot_freq, self_kerr, Jkl, zz)

# ============================================================
# Reference TDVP run
# ============================================================

# Run TDVP2 once with a fixed cutoff to compare against Quandary
ans_mps, link_history, _, _, _ = tdvp2(
    H_s,
    init_MPS,
    t0,
    T,
    Int64(steps / 2),
    bc_params;
    cutoff = 1E-5,
    strang = true
)

# Convert final TDVP solution to full vector form
tdvp_ans = vectorize_mps(ans_mps; order = "reverse")

# -------------------------
# Load Quandary reference data
# -------------------------

# Real part of reference state from Quandary which is only the last time-step due to the full state trajectory being too large to save
datafile = joinpath(@__DIR__, "quandary_data")
rho_re = readdlm(joinpath(datafile, "quandary_re_coupled_$N.dat"))

# Imaginary part of reference state from Quandary
rho_im = readdlm(joinpath(datafile, "quandary_im_coupled_$N.dat"))

# Construct final complex reference state from Quandary output
quandary_ans = rho_re + im * rho_im

# Compare TDVP solution to Quandary
println("Error between quandary and tdvp: ", norm(tdvp_ans - quandary_ans))

# Print final bond dimensions of TDVP solution
println("bond dim at end: ", linkdims(ans_mps))

# ============================================================
# Cutoff sweep setup
# ============================================================

# Number of cutoff values to test
pts = 20

# Generate exponents from -35 to -3
pts_range = LinRange(-35, -3, pts)

# Generate cutoff values
cutoff_list = 2.0 .^ pts_range

# Error storage
err_list_tdvp = zeros(pts)
err_list_bug_tucker = zeros(pts)
err_list_bug_mps = zeros(pts)

# Bond dimension / rank history storage
bd_list_tdvp = []
bd_list_bug_tucker = []
bd_list_bug_mps = []

# Store final solutions if desired
ans_tdvp = []
ans_bug_tucker = []
ans_bug_mps = []

# CPU timing storage
time_tdvp_cpu = zeros(pts)
time_bug_tucker_cpu = zeros(pts)
time_bug_mps_cpu = zeros(pts)

# Reinitialize initial state
q_state = fill(0, N)
init_MPS = init_separable(sites, q_state)

# Rebuild drift Hamiltonian MPO
H_s = drift_MPO(N, sites, freq01, rot_freq, self_kerr, zz, Jkl)

# ============================================================
# Sweep over truncation cutoffs
# ============================================================

for i in 1:pts 
    
    println("Cutoff: ", cutoff_list[i]^2)

    # -------------------------
    # TDVP2
    # -------------------------
    if plot_tdvp == true 
        # Benchmark TDVP runtime
        time_tdvp_cpu[i] = @elapsed begin 
            ans_mps, link_history, _, _, _ = tdvp2(
                H_s,
                init_MPS,
                t0,
                T,
                Int(steps/2),
                bc_params;
                cutoff = cutoff_list[i]^2
            )
        end 

        # Store final TDVP MPS
        push!(ans_tdvp, ans_mps)

        # Convert TDVP solution to full vector form
        tdvp_sol = vectorize_mps(ans_mps; order = "reverse")

        # Compute error relative to Quandary reference
        err_list_tdvp[i] = norm(tdvp_sol - quandary_ans)

        # Store bond dimension history
        push!(bd_list_tdvp, link_history)
    end

    # -------------------------
    # BUG-Tucker
    # -------------------------
    if plot_bug_tucker == true 
        # Benchmark BUG runtime (Tucker variant)
        time_bug_tucker_cpu[i] = @elapsed begin 
            ans_core, ans_factors, _, _, bd = bug_integrator_mat_ra(
                H_s_ops,
                bc_params,
                init_core,
                init_factors,
                t0,
                T,
                steps;
                cutoff = cutoff_list[i]^2
            )
        end

        # Store final Tucker solution
        push!(ans_bug_tucker, [ans_core, ans_factors])

        # Reconstruct dense tensor from Tucker form
        bug_array = Multi_TTM_recursive(ans_core, ans_factors)

        # Convert tensor to full vector
        bug_ans = vec(permutedims(bug_array, reverse(1:ndims(bug_array))))

        # Compute error relative to Quandary reference
        err_list_bug_tucker[i] = norm(bug_ans - quandary_ans)

        # Store Tucker rank history
        push!(bd_list_bug_tucker, bd)
    end

    # -------------------------
    # BUG-MPS (mps_bug)
    # -------------------------
    if plot_bug_mps == true
        # Benchmark BUG runtime (MPS variant)
        time_bug_mps_cpu[i] = @elapsed begin 
            # assumed mps_bug returns an MPS solution and bond history similar to tdvp2
            ans_mps_bug, bd_mps, _, _ = mps_bug(
                H_s,
                bc_params, 
                init_MPS,
                t0,
                T,
                steps;
                cutoff = cutoff_list[i]^2
            )
        end

        # Store final MPS-BUG solution
        push!(ans_bug_mps, ans_mps_bug)

        # Convert MPS-BUG solution to full vector
        bug_mps_vec = vectorize_mps(ans_mps_bug; order = "reverse")

        # Compute error relative to Quandary reference
        err_list_bug_mps[i] = norm(bug_mps_vec - quandary_ans)

        # Store bond history for MPS-BUG
        push!(bd_list_bug_mps, bd_mps)
    end
end

# ============================================================
# Compute storage costs
# ============================================================

# Compute storage costs conditionally to avoid indexing empty lists
if plot_tdvp
    entries_list_tdvp = [count_MPS(bd_list_tdvp[i][end, :], nlevels) for i in 1:pts]
else
    entries_list_tdvp = Int[]  # empty placeholder
end

if plot_bug_tucker
    entries_list_bug_tucker = [count_tucker(bd_list_bug_tucker[i][end, :], nlevels) for i in 1:pts]
else
    entries_list_bug_tucker = Int[]
end

if plot_bug_mps
    entries_list_bug_mps = [count_MPS(bd_list_bug_mps[i][end, :], nlevels) for i in 1:pts]
else
    entries_list_bug_mps = Int[]
end

# ============================================================
# Plot error and storage results
# ============================================================

# Build lists for error plotting depending on enabled methods
error_traces = []
labels_err = String[]
if plot_tdvp; push!(error_traces, err_list_tdvp); push!(labels_err, "TDVP2 Err"); end
if plot_bug_tucker; push!(error_traces, err_list_bug_tucker); push!(labels_err, "BUG Tucker Err"); end
if plot_bug_mps; push!(error_traces, err_list_bug_mps); push!(labels_err, "BUG MPS Err"); end

if !isempty(error_traces)
    cutoff_plot = plot(
        cutoff_list,
        error_traces,
        label = reshape(labels_err, 1, :),
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        dpi = 250,
        yticks = [10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xlabel = "SVD Truncation Parameter(ε)"
    )
end

# Build lists for storage plotting depending on enabled methods
storage_traces = []
labels_store = String[]
if plot_tdvp; push!(storage_traces, entries_list_tdvp); push!(labels_store, "MPS Storage (TDVP)"); end
if plot_bug_tucker; push!(storage_traces, entries_list_bug_tucker); push!(labels_store, "Tucker Storage (BUG)"); end
if plot_bug_mps; push!(storage_traces, entries_list_bug_mps); push!(labels_store, "MPS Storage (BUG-MPS)"); end

if !isempty(storage_traces)
    entries_plot = plot(
        cutoff_list,
        storage_traces,
        label = labels_store,
        dpi = 250,
        xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xlabel = "SVD Truncation Parameter(ε)",
        xscale = :log10
    )

    # Add dense vector storage reference line
    plot!(cutoff_list, fill(2^N, pts), label = "Vector storage", linestyle = :dash)
end

# ============================================================
# Optional pulse visualization
# ============================================================

if plot_pulse == true
    # Choose which pulse channel to plot
    pulse = 1

    # Time grid for pulse evaluation
    time_range = LinRange(0, T, steps)

    # Arrays for in-phase and quadrature pulse components
    p_eval = zeros(length(time_range))
    q_eval = zeros(length(time_range))

    # Evaluate pulse over time
    for j = 1:steps 
        p_eval[j] = bcarrier2(time_range[j], bc_params, 2*(pulse - 1)) * (500 / pi)
        q_eval[j] = bcarrier2(time_range[j], bc_params, 2*(pulse - 1) + 1) * (500 / pi)
    end

    # Plot pulse envelope
    pulse_plot = plot(
        time_range,
        [p_eval, q_eval],
        labels = ["p(t)" "q(t)"],
        xlabel = "time(ns)",
        ylabel = "MHz"
    )
end
