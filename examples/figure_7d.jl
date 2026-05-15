using ITensors, ITensorMPS, LinearAlgebra, LaTeXStrings, Plots
using TensorMethods


################################################################################
# Generate data and plot for Figure 7(d) (arxiv 2603.13990)
################################################################################

# Number of qubits in the spin chain
N = 10

# Optional ITensors warning control (currently unused)
# ITensors.warn_order!(100)

# Create ITensor site indices for N qubits
sites = siteinds("Qubit", N)

# Define initial product state |0,1,1,...,1⟩
q_state = Int64.(fill(1, N))
q_state[1] = 0

# Construct initial MPS from the product state
init_MPS = init_separable(sites, q_state)
init_MPS_copy = init_separable(sites, q_state)  # Optional copy for baseline (currently unused)
# -------------------------
# Time evolution parameters
# -------------------------
t0 = 0.0
T = 5.0

# Number of cutoff values to test
pts = 25

# Generate array of SVD truncation cutoffs from 10^-7 to 10^-2

cutoff_arr = 10 .^ LinRange(-15, -2, pts)

# -------------------------
# Storage for results
# -------------------------

# Store final MPS solutions for each cutoff
ans_arr = []
ans_arr_mps_bug = []

# Store final bond dimensions for each cutoff
bd_arr = []
bd_arr_mps_bug = []

# Store final magnetization vectors for each cutoff
magnet_arr = []
magnet_arr_mps_bug = []


# Store differences between consecutive solutions
diff_arr = zeros(length(cutoff_arr) - 1)

# Number of time steps used in the evolution
steps = 500

# Hamiltonian parameters
J = 1.0
g = 0.5

# Construct Hamiltonian MPO
H_mpo = xxx_mpo_scaled(N, sites, J, g)


# if N is small enough, can compute true final state with very small cutoff and use as baseline for difference calculations
if N <= 10
    H = xxx_scaled(N, J, g)
    vec_final = exp(-im * H * T) * vectorize_mps(init_MPS)
end


# Optional baseline calculation (currently commented out)
# base_svd_cutoff = 1/sqrt(10)
# base_svd_cutoff_squared = base_svd_cutoff^2
# base_mps, base_bd, _, _ = mps_bug_constant(H_mpo, init_MPS, t0, T, steps; cutoff = base_svd_cutoff_squared)

# ============================================================
# Run time evolution for each truncation cutoff
# ============================================================

for i in 1:length(cutoff_arr)
    # Square the SVD cutoff before passing it to the solver
    svd_cutoff_squared = cutoff_arr[i]^2

    # Optional next cutoff (currently unused)
    # svd_cutoff_squared2 = cutoff_arr[i + 1]^2

    println("svd_cutoff_squared $i: ", svd_cutoff_squared)

    # Run TDVP2 evolution and record:
    # - final MPS state
    # - bond dimension history
    # - magnetization history
    ans_mps, bd, magnet, _ = tdvp2_constant(
        H_mpo,
        init_MPS,
        t0,
        T,
        Int(steps/2);
        cutoff = svd_cutoff_squared,
        magnet = true,
        verbose = false
    )

    # Run BUG MPS evolution and record:
    # - final MPS state
    # - bond dimension history
    # - magnetization history
    ans_mps_bug, bd_bug, magnet_bug, _ = mps_bug_constant(H_mpo, init_MPS_copy, t0, T, steps; cutoff = svd_cutoff_squared, magnet = true, verbose = false)

    # Store the final MPS state for this cutoff
    push!(ans_arr, ans_mps)
    push!(ans_arr_mps_bug, ans_mps_bug)
    # Store the final bond dimension vector
    push!(bd_arr, bd[end, :])
    push!(bd_arr_mps_bug, bd_bug[end, :])
    # Store the final magnetization vector
    push!(magnet_arr, magnet[end, :])
    push!(magnet_arr_mps_bug, magnet_bug[end, :])
    # Optional baseline update (currently unused)
    # base_mps = ans_mps
end

# ============================================================
# Compute differences between consecutive cutoff runs
# ============================================================

# Difference between full final MPS states at neighboring cutoff values
diff_arr = [norm(ans_arr[n] - ans_arr[n+1]) for n in 1:length(ans_arr)-1]
diff_arr_mps_bug = [norm(ans_arr_mps_bug[n] - ans_arr_mps_bug[n+1]) for n in 1:length(ans_arr_mps_bug)-1]

# Difference between final magnetization vectors at neighboring cutoff values
diff_arr_magnet = [norm(magnet_arr[n] - magnet_arr[n+1]) for n in 1:length(magnet_arr)-1]
diff_arr_magnet_mps_bug = [norm(magnet_arr_mps_bug[n] - magnet_arr_mps_bug[n+1]) for n in 1:length(magnet_arr_mps_bug)-1]

# also plot difference in final MPS states and magnetization for MPS compared to TDVP 
diff_arr_mps_bug_tdvp = [norm(ans_arr[n] - ans_arr_mps_bug[n]) for n in 1:length(ans_arr)]
diff_arr_magnet_mps_bug_tdvp = [norm(magnet_arr[n] - magnet_arr_mps_bug[n]) for n in 1:length(magnet_arr)]

# ============================================================
# Plot: difference in full state vs cutoff
# ============================================================


diff_plot = plot(
    cutoff_arr[1:length(cutoff_arr) - 1],
    diff_arr,
    yscale = :log10,
    xscale = :log10,
    xlabel = "εₙ (SVD Truncation Parameter)",
    label = "||y(εₙ) - y(εₙ₊₁)||",
    title = "Ising Model with N = $N qubits",
    yticks = [10^-1, 10^-3, 10^-5, 10^-7, 10^-9, 10^-11, 10^-13],
    xticks = [10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9, 10^-10],
    dpi = 250,
    legend = :topleft,
    xlims = (10^-8, 10^-3.5),
    ylims = (10^-7, 10^-2)
)

# ============================================================
# Plot: difference in magnetization vs cutoff
# ============================================================

diff_plot_magnet = plot(
    cutoff_arr[1:length(cutoff_arr) - 1],
    diff_arr_magnet,
    yscale = :log10,
    xscale = :log10,
    xlabel = "εₙ (SVD Truncation Parameter)",
    label = "||m(εₙ) - m(εₙ₊₁)||",
    title = "Ising Model with N = $N qubits",
    yticks = [10^-1, 10^-3, 10^-5, 10^-7, 10^-9, 10^-11, 10^-13],
    xticks = [10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9, 10^-10],
    dpi = 250,
    legend = :topleft,
    xlims = (10^-7, 10^-2),
    ylims = (10^-7, 10^1),
    legendfontsize = 14
)

# Add a reference power-law scaling line to the magnetization difference plot
plot!(
    cutoff_arr[1:length(cutoff_arr) - 1],
    10^(1.7) .* (cutoff_arr[1:length(cutoff_arr) - 1].^1.25),
    label = L"ε^p, \quad p = 1.25",
    linestyle = :dash
)

# ============================================================
# Plot: difference in MPS-BUG vs TDVP and magnetization difference
# ============================================================
diff_plot_mps_bug_tdvp = plot(
    cutoff_arr,
    diff_arr_mps_bug_tdvp,
    yscale = :log10,
    xscale = :log10,
    xlabel = "εₙ (SVD Truncation Parameter)",
    label = L"\|\psi_{\mathrm{BUG}}(ε_n) - \psi_{\mathrm{TDVP2}}(ε_n)\|",
    title = "Ising Model with N = $N qubits",
    yticks = [10^-1, 10^-3, 10^-5, 10^-7, 10^-9, 10^-11, 10^-13],
    xticks = [10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9, 10^-10, 10^-11, 10^-12, 10^-13, 10^-14, 10^-15],
    dpi = 250,
    legend = :outertop,
    legend_columns = 3, 
    xlims = (10^-15, 10^-2),
    ylims = (10^-13, 10^1),
    legendfontsize = 6
)

plot!(cutoff_arr, diff_arr_magnet_mps_bug_tdvp, label = L"|m_{\mathrm{BUG}}(ε_n) - m_{\mathrm{TDVP2}}(ε_n)|")
plot!(cutoff_arr[1:length(cutoff_arr) - 1], diff_arr, label = L"\|\psi_{\mathrm{TDVP2}}(ε_n) - \psi_{\mathrm{TDVP2}}(ε_{n + 1})\|")
plot!(cutoff_arr[1:length(cutoff_arr) - 1], diff_arr_magnet, label = L"|m_{\mathrm{TDVP2}}(ε_n) - m_{\mathrm{TDVP2}}(ε_{n + 1})|")
plot!(cutoff_arr[1:length(cutoff_arr) - 1], diff_arr_mps_bug, label = L"\|\psi_{\mathrm{BUG}}(ε_n) - \psi_{\mathrm{BUG}}(ε_{n + 1})\|")
plot!(cutoff_arr[1:length(cutoff_arr) - 1], diff_arr_magnet_mps_bug, label = L"|m_{\mathrm{BUG}}(ε_n) - m_{\mathrm{BUG}}(ε_{n + 1})|")

if N <= 10
     # If we have a true final state, also compute differences to the true state
    diff_arr_true = [norm(vectorize_mps(ans_arr[n]) - vec_final) for n in 1:length(ans_arr)]
    diff_arr_mps_bug_true = [norm(vectorize_mps(ans_arr_mps_bug[n]) - vec_final) for n in 1:length(ans_arr_mps_bug)]

    # Add these differences to the plot
    plot!(
        cutoff_arr,
        diff_arr_true,
        label = L"\|\psi_{\mathrm{TDVP2}}(ε_n) - \psi_{\mathrm{exact}}(ε_n)\|",
        linestyle = :dash
    )
    plot!(
        cutoff_arr,
        diff_arr_mps_bug_true,
        label = L"\|\psi_{\mathrm{BUG}}(ε_n) - \psi_{\mathrm{exact}}(ε_n)\|",
        linestyle = :dash
    )
end
