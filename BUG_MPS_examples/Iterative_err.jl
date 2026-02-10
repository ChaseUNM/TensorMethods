using ITensors, ITensorMPS, LinearAlgebra, LaTeXStrings, Plots
using TensorMethods

N = 30
# ITensors.warn_order!(100)
sites = siteinds("Qubit", N)
q_state = Int64.(fill(1, N))
q_state[1] = 0

init_MPS = init_separable(sites, q_state)

t0 = 0.0
T = 1.0


cutoff_arr = 10 .^ -LinRange(5,10,30)
ans_arr = []
bd_arr = []
diff_arr = zeros(length(cutoff_arr) - 1)

steps = 2^6

J = 0.2
g = 1.0

H_mpo = xxx_mpo_scaled(N, sites, J, g)

base_svd_cutoff = 1/sqrt(10)
base_svd_cutoff_squared = base_svd_cutoff^2
base_mps, base_bd, _, _ = mps_bug_constant(H_mpo, init_MPS, t0, T, steps; cutoff = base_svd_cutoff_squared)

for i in 1:length(cutoff_arr)-1 
    svd_cutoff_squared = cutoff_arr[i]^2
    # svd_cutoff_squared2 = cutoff_arr[i + 1]^2
    println("svd_cutoff_squared $i: ", svd_cutoff_squared)
    ans_mps, bd, _, _ = mps_bug_constant(H_mpo, init_MPS, t0, T, steps; cutoff = svd_cutoff_squared)
    push!(ans_arr, ans_mps)
    push!(bd_arr, bd[end,:])
    # base_mps = ans_mps
end

diff_arr = [norm(ans_arr[n] - ans_arr[n+1]) for n in 1:length(ans_arr)-1]
diff_plot = plot(collect(1:length(diff_arr)), diff_arr, yscale =:log10, xlabel = "n", label = "||y(εₙ) - y(εₙ₊₁)||", title = "XXX Heisenberg, N = $N qubits, J = $J, g = $g", yticks = [10^-1, 10^-3, 10^-5, 10^-7, 10^-9, 10^-11, 10^-13], xticks = [5, 10, 15, 20, 25, 30], dpi = 250)
plot!(collect(1:length(diff_arr)), cutoff_arr[1:length(diff_arr)], label = "εₙ (SVD truncation parameter)")
savefig("difference_plot_N$N.png")