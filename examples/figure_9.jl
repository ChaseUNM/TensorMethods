using ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles, Plots, BenchmarkTools
using TensorMethods

################################################################################
# Generate data and plots for Figure 9, N = 10 case (arxiv 2603.13990)
################################################################################

# Toggle which plots / methods to include
plot_pulse = false
plot_tdvp = true 
plot_bug = true

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

# Real part of reference state from Quandary
rho_re = readdlm("../Rho_data/rho_Re.iinit0000_10_coupled.dat")

# Imaginary part of reference state from Quandary
rho_im = readdlm("../Rho_data/rho_Im.iinit0000_10_coupled.dat")

# Construct final complex reference state from Quandary output
quandary_ans = rho_re[end, 2:end] + im * rho_im[end, 2:end]

# Compare TDVP solution to Quandary
println("Error between quandary and tdvp: ", norm(tdvp_ans - quandary_ans))

# Print final bond dimensions of TDVP solution
println("bond dim at end: ", linkdims(ans_mps))

# ============================================================
# Cutoff sweep setup
# ============================================================

# Number of cutoff values to test
pts = 10

# Generate exponents from -35 to -3
pts_range = LinRange(-35, -3, pts)

# Generate cutoff values
cutoff_list = 2.0 .^ pts_range

# Error storage
err_list_tdvp = zeros(pts)
err_list_bug = zeros(pts)

# Bond dimension / rank history storage
bd_list_tdvp = []
bd_list_bug = []

# Store final solutions if desired
ans_tdvp = []
ans_bug = []

# CPU timing storage
time_tdvp_cpu = zeros(pts)
time_bug_cpu = zeros(pts)

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
        time_tdvp_cpu[i] = @belapsed begin 
            ans_mps, link_history, _, _, _ = tdvp2(
                H_s,
                init_MPS,
                t0,
                T,
                steps,
                bc_params;
                cutoff = cutoff_list[i]^2,
                strang = false
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
    if plot_bug == true 
        # Benchmark BUG runtime
        time_bug_cpu[i] = @belapsed begin 
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
        push!(ans_bug, [ans_core, ans_factors])

        # Reconstruct dense tensor from Tucker form
        bug_array = Multi_TTM_recursive(ans_core, ans_factors)

        # Convert tensor to full vector
        bug_ans = vec(permutedims(bug_array, reverse(1:ndims(bug_array))))

        # Compute error relative to Quandary reference
        err_list_bug[i] = norm(bug_ans - quandary_ans)

        # Store Tucker rank history
        push!(bd_list_bug, bd)
    end
end

# ============================================================
# Compute storage costs
# ============================================================

# Count final MPS storage cost
entries_list_tdvp = [count_MPS(bd_list_tdvp[i][end, :], nlevels) for i in 1:pts]

# Count final Tucker storage cost
entries_list_bug = [count_tucker(bd_list_bug[i][end, :], nlevels) for i in 1:pts]

# ============================================================
# Plot error and storage results
# ============================================================

if plot_tdvp == true & plot_bug == true 
    # Plot TDVP and BUG error vs cutoff
    cutoff_plot = plot(
        cutoff_list,
        [err_list_tdvp, err_list_bug],
        label = ["TDVP2 Err" "BUG Err"],
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        dpi = 250,
        yticks = [10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xlabel = "SVD Truncation Parameter(ε)"
    )

    # Plot TDVP and BUG storage cost vs cutoff
    entries_plot = plot(
        cutoff_list,
        [entries_list_tdvp, entries_list_bug],
        label = ["MPS Storage" "Tucker Tensor Storage"],
        dpi = 250,
        xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xlabel = "SVD Truncation Parameter(ε)",
        xscale = :log10
    )

    # Add dense vector storage reference line
    plot!(cutoff_list, fill(2^N, pts), label = "Vector storage", linestyle = :dash)
    
elseif plot_tdvp == false 
    # Plot BUG-only error
    cutoff_plot_bug = plot(
        cutoff_list,
        err_list_bug,
        label = "BUG Err",
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        dpi = 250,
        yticks = [10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xlabel = "SVD Truncation Parameter(ε)"
    )

    # Plot TDVP storage only (note: this branch may not match intended logic)
    entries_plot = plot(
        cutoff_list,
        entries_list_tdvp,
        label = "Tensor-train storage",
        dpi = 250,
        xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xlabel = "SVD Truncation Parameter(ε)",
        xscale = :log10
    )

elseif plot_bug == false 
    # Plot TDVP-only error
    cutoff_plot_tdvp = plot(
        cutoff_list,
        err_list_tdvp,
        label = "TDVP2 Err",
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        dpi = 250,
        yticks = [10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1],
        xlabel = "SVD Truncation Parameter(ε)"
    )

    # Optional reference slope line
    # plot!(cutoff_list, 10^3*cutoff_list.^1.06, label = L"O(\epsilon^{1.06})")

    # Plot BUG/Tucker storage only
    entries_plot = plot(
        cutoff_list,
        entries_list_bug,
        label = "Tucker-tensor storage",
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