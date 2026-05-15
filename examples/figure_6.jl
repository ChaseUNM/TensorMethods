using Revise
using ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles, Plots, CPUTime, LaTeXStrings
using TensorMethods


####################################################################################
# Generate data for Figure 6 (arxiv 2603.13990)
####################################################################################


# ============================================================
# Cutoff Study:
# Compare TDVP2, BUG-Tucker, and BUG-MPS as the truncation
# tolerance changes while keeping the time discretization fixed.
# ============================================================

# -------------------------
# Toggle which methods to run
# -------------------------
run_tdvp = true
run_bug_tucker = false
run_bug_mps = true

# -------------------------
# Problem setup
# -------------------------

# Number of qubits
N = 10

# Create qubit site indices and local dimensions
sites = siteinds("Qubit", N)
N_levels = fill(2, N)

# Define initial product state |0,1,1,...,1⟩
q_state = Int64.(fill(1, N))
q_state[1] = 0
# q_state[N] = 0

# Construct initial MPS
init_MPS = init_separable(sites, q_state)

# Alternative random initial state (optional)
# init_MPS = random_mps(sites, linkdims = 16)

# Convert state values {0,1} to array indices {1,2}
q_state_arr = q_state .+ 1

# Build equivalent dense tensor initial condition
init_ten = zeros(ComplexF64, N_levels...)
init_ten[q_state_arr...] = 1.0 + 0.0im

# Compute Tucker decomposition of the initial tensor
init_core, init_factors = tucker(init_ten; cutoff = 0.0)

# Vectorize initial MPS for comparison to exact solution
init_vec = vectorize_mps(init_MPS; order = "reverse")

# -------------------------
# Time evolution parameters
# -------------------------
t0 = 0.0
T = 10.0

# Hamiltonian parameters
J = 1.0
g = 0.5

# Construct Hamiltonian in different representations
H_mpo = xxx_mpo_scaled(N, sites, J, g)     # MPO form
H_mat = xxx_scaled(N, J, g)                # Dense matrix form
H_ops_xxx = ops_xxx_scaled(N, J, g)        # Operator/tensor form

# -------------------------
# Cutoff sweep setup
# -------------------------

# Number of cutoff values to test
pts = 10

# Sweep cutoff values from 2^-35 to 2^-6
cutoff_exponents = LinRange(-35, -6, pts)
cutoff_list = 2.0 .^ cutoff_exponents

# -------------------------
# Time-step study setup
# -------------------------

# Use these exponents to define number of time steps: steps = 2^k
# Example: k = 8 means steps = 256
step_exponents = [8, 9]   # Change to [8,9,10,11,12] if desired
steps_list = Int.(2 .^ step_exponents)

# Labels for plots
step_labels = [L"2^{%$k} steps" for k in step_exponents]

# -------------------------
# Storage for results
# -------------------------

# Error histories for each method
cutoff_err_tdvp_list = []
cutoff_err_bug_tucker_list = []
cutoff_err_bug_mps_list = []

# Bond dimension / rank histories for each method
bd_list_tdvp2 = []
bd_list_bug_tucker = []
bd_list_bug_mps = []

# -------------------------
# Exact solution
# -------------------------

# Compute exact dense solution for comparison
true_sol = exp(-im * (T - t0) * H_mat) * init_vec

# ============================================================
# Main loop over step counts
# ============================================================

for steps in steps_list
    println("================================================")
    println("Running cutoff sweep with $steps time steps")
    println("================================================")

    # Error arrays for this step count
    cutoff_err_tdvp = zeros(pts)
    cutoff_err_bug_tucker = zeros(pts)
    cutoff_err_bug_mps = zeros(pts)

    # Bond dimension / rank history for this step count
    bd_tdvp = []
    bd_bug_tucker = []
    bd_bug_mps = []

    # --------------------------------------------------------
    # Loop over cutoff values
    # --------------------------------------------------------
    for j in 1:pts
        cutoff = cutoff_list[j]^2
        println("Cutoff: $cutoff")

        # -------------------------
        # TDVP2
        # -------------------------
        if run_tdvp
            init_MPS_copy = deepcopy(init_MPS)

            ans_mps, bd_history_tdvp, magnet_history, energy_history, trunc_err =
                tdvp2_constant(H_mpo, init_MPS_copy, t0, T, Int64(steps ÷ 2); cutoff = cutoff)

            tdvp_vec = vectorize_mps(ans_mps; order = "reverse")
            cutoff_err_tdvp[j] = norm(tdvp_vec - true_sol)

            push!(bd_tdvp, bd_history_tdvp)
        end

        # -------------------------
        # BUG-Tucker
        # -------------------------
        if run_bug_tucker
            ans_core, ans_factors, state, nrg, bd_history_bug_tucker =
                bug_integrator_mat_ra(H_ops_xxx, init_core, init_factors, t0, T, steps; cutoff = cutoff)

            bug_array = Multi_TTM_recursive(ans_core, ans_factors)
            bug_vec = vec(permutedims(bug_array, reverse(1:ndims(bug_array))))

            cutoff_err_bug_tucker[j] = norm(bug_vec - true_sol)

            push!(bd_bug_tucker, bd_history_bug_tucker)
        end

        # -------------------------
        # BUG-MPS
        # -------------------------
        if run_bug_mps
            init_MPS_copy = deepcopy(init_MPS)

            ans_bug_mps, bd_history_bug_mps, _, _ =
                mps_bug_constant(H_mpo, init_MPS_copy, t0, T, steps; cutoff = cutoff)

            bug_mps_vec = vectorize_mps(ans_bug_mps; order = "reverse")
            cutoff_err_bug_mps[j] = norm(bug_mps_vec - true_sol)

            push!(bd_bug_mps, bd_history_bug_mps)
        end
    end

    # --------------------------------------------------------
    # Store results for this step count
    # --------------------------------------------------------
    if run_tdvp
        push!(cutoff_err_tdvp_list, cutoff_err_tdvp)
        push!(bd_list_tdvp2, bd_tdvp)
    end

    if run_bug_tucker
        push!(cutoff_err_bug_tucker_list, cutoff_err_bug_tucker)
        push!(bd_list_bug_tucker, bd_bug_tucker)
    end

    if run_bug_mps
        push!(cutoff_err_bug_mps_list, cutoff_err_bug_mps)
        push!(bd_list_bug_mps, bd_bug_mps)
    end
end

# ============================================================
# Post-processing: storage counts
# ============================================================

if run_tdvp
    entries_list_tdvp = [
        [count_MPS(bd_list_tdvp2[i][j][end, :], N_levels) for j in 1:pts]
        for i in 1:length(steps_list)
    ]
end

if run_bug_tucker
    entries_list_bug_tucker = [
        [count_tucker(bd_list_bug_tucker[i][j][end, :], N_levels) for j in 1:pts]
        for i in 1:length(steps_list)
    ]
end

if run_bug_mps
    entries_list_bug_mps = [
        [count_MPS(bd_list_bug_mps[i][j][end, :], N_levels) for j in 1:pts]
        for i in 1:length(steps_list)
    ]
end

# ============================================================
# Individual method plots
# ============================================================

if run_tdvp
    entries_plot_tdvp = plot(
        cutoff_list,
        entries_list_tdvp,
        labels = permutedims(step_labels),
        xlabel = "SVD Truncation Error (ε)",
        ylabel = "Total tensor-train entries",
        xscale = :log10,
        dpi = 250,
        title = "TDVP2 Storage vs Cutoff"
    )

    cutoff_err_plot_tdvp = plot(
        cutoff_list,
        cutoff_err_tdvp_list,
        labels = permutedims(step_labels),
        xlabel = "SVD Truncation Error (ε)",
        ylabel = "Final State Error",
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        dpi = 250,
        title = "TDVP2 Error vs Cutoff"
    )
end

if run_bug_tucker
    entries_plot_bug_tucker = plot(
        cutoff_list,
        entries_list_bug_tucker,
        labels = permutedims(step_labels),
        xlabel = "SVD Truncation Error (ε)",
        ylabel = "Total Tucker entries",
        xscale = :log10,
        dpi = 250,
        title = "BUG-Tucker Storage vs Cutoff"
    )

    cutoff_err_plot_bug_tucker = plot(
        cutoff_list,
        cutoff_err_bug_tucker_list,
        labels = permutedims(step_labels),
        xlabel = "SVD Truncation Error (ε)",
        ylabel = "Final State Error",
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        dpi = 250,
        title = "BUG-Tucker Error vs Cutoff"
    )
end

if run_bug_mps
    entries_plot_bug_mps = plot(
        cutoff_list,
        entries_list_bug_mps,
        labels = permutedims(step_labels),
        xlabel = "SVD Truncation Error (ε)",
        ylabel = "Total tensor-train entries",
        xscale = :log10,
        dpi = 250,
        title = "MPS-BUG Storage vs Cutoff (1-site-truncation)"
    )

    cutoff_err_plot_bug_mps = plot(
        cutoff_list,
        cutoff_err_bug_mps_list,
        labels = permutedims(step_labels),
        xlabel = "SVD Truncation Error (ε)",
        ylabel = "Final State Error",
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        legend_columns = 2,
        dpi = 250,
        title = "MPS-BUG Error vs Cutoff (1-site-truncation)"
    )
end

# ============================================================
# Combined comparison plots
# ============================================================

# TDVP2 vs MPS-BUG error comparison
if run_tdvp && run_bug_mps
    cutoff_err_plot = plot(
        cutoff_list,
        cutoff_err_tdvp_list,
        labels = "TDVP2",
        xlabel = "SVD Truncation Error (ε)",
        ylabel = "Final State Error",
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        legendfontsize = 7,
        dpi = 250,
        title = "TDVP2 vs MPS-BUG Error"
    )

    plot!(
        cutoff_list,
        cutoff_err_bug_mps_list,
        label = "MPS-BUG",
        linestyle = :dash
    )

    # TDVP2 vs MPS-BUG storage comparison
    entries_list_plot = plot(
        cutoff_list,
        entries_list_tdvp,
        labels = "TDVP2",
        xlabel = "SVD Truncation Error (ε)",
        ylabel = "Total tensor-train entries",
        xscale = :log10,
        dpi = 250,
        legend = :topleft,
        legendfontsize = 7,
        title = "TDVP2 vs MPS-BUG Storage"
    )

    plot!(
        cutoff_list,
        entries_list_bug_mps,
        label = "MPS-BUG",
        linestyle = :dash
    )
end
