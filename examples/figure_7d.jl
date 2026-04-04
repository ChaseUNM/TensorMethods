using ITensors, ITensorMPS, LinearAlgebra, LaTeXStrings, Plots
using TensorMethods


################################################################################
# Generate data and plot for Figure 7(d) (arxiv 2603.13990)
################################################################################

# Number of qubits in the spin chain
N = 20

# Optional ITensors warning control (currently unused)
# ITensors.warn_order!(100)

# Create ITensor site indices for N qubits
sites = siteinds("Qubit", N)

# Define initial product state |0,1,1,...,1⟩
q_state = Int64.(fill(1, N))
q_state[1] = 0

# Construct initial MPS from the product state
init_MPS = init_separable(sites, q_state)

# -------------------------
# Time evolution parameters
# -------------------------
t0 = 0.0
T = 5.0

# Number of cutoff values to test
pts = 21

# Generate array of SVD truncation cutoffs from 10^-7 to 10^-2
cutoff_arr = 10 .^ LinRange(-7, -2, pts)

# -------------------------
# Storage for results
# -------------------------

# Store final MPS solutions for each cutoff
ans_arr = []

# Store final bond dimensions for each cutoff
bd_arr = []

# Store final magnetization vectors for each cutoff
magnet_arr = []

# Store differences between consecutive solutions
diff_arr = zeros(length(cutoff_arr) - 1)

# Number of time steps used in the evolution
steps = 85

# Hamiltonian parameters
J = 1.0
g = 0.5

# Construct Hamiltonian MPO
H_mpo = xxx_mpo_scaled(N, sites, J, g)

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

    # Optional MPS-BUG evolution (currently commented out)
    # ans_mps, bd, _, _ = mps_bug_constant(H_mpo, init_MPS, t0, T, steps; cutoff = svd_cutoff_squared)

    # Run TDVP2 evolution and record:
    # - final MPS state
    # - bond dimension history
    # - magnetization history
    ans_mps, bd, magnet, _ = tdvp2_constant(
        H_mpo,
        init_MPS,
        t0,
        T,
        steps;
        cutoff = svd_cutoff_squared,
        magnet = true,
        verbose = false
    )

    # Store the final MPS state for this cutoff
    push!(ans_arr, ans_mps)

    # Store the final bond dimension vector
    push!(bd_arr, bd[end, :])

    # Store the final magnetization vector
    push!(magnet_arr, magnet[end, :])

    # Optional baseline update (currently unused)
    # base_mps = ans_mps
end

# ============================================================
# Compute differences between consecutive cutoff runs
# ============================================================

# Difference between full final MPS states at neighboring cutoff values
diff_arr = [norm(ans_arr[n] - ans_arr[n+1]) for n in 1:length(ans_arr)-1]

# Difference between final magnetization vectors at neighboring cutoff values
diff_arr_magnet = [norm(magnet_arr[n] - magnet_arr[n+1]) for n in 1:length(magnet_arr)-1]

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