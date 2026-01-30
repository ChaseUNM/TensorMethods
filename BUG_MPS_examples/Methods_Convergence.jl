using Revise
using ITensors, ITensorMPS, LinearAlgebra, LaTeXStrings, Plots
using TensorMethods

N = 10

sites = siteinds("Qubit", N)
q_state = Int64.(fill(1, N))
q_state[1] = 0

init_MPS = init_separable(sites, q_state)
init_ten = zeros(ComplexF64, fill(2, N)...)
q_state_arr = q_state .+ 1
init_ten[q_state_arr...] = 1.0 + 0.0*im
init_core, init_factors = tucker(init_ten; cutoff = 0.0)

init_vec = vectorize_mps(init_MPS; order = "reverse")

t0 = 0.0
T = 1.0
step_list = 2 .^collect(5:6)
err_list_tdvp = zeros(length(step_list))
err_list_bug_tucker = zeros(length(step_list))
err_list_bug_mps = zeros(length(step_list))

J = 1.0
g = 1.0

H_mpo = xxx_mpo_scaled(N, sites, J, g)
H_mat = xxx_scaled(N, J, g)
H_ops_xxx = ops_xxx_scaled(N, J, g)

# true_sol, true_magnet, true_energy = exp_solver(H_mat, init_vec, N, t0, T, steps)
true_sol = exp(-im*H_mat*(T - t0))*init_vec
for i in 1:length(step_list)
    println("Steps: $(step_list[i])")
    
    ans_mps, bd_history, magnet_history, energy_history = tdvp2_constant(H_mpo, init_MPS, t0, T, Int64(step_list[i]/2); cutoff = 0.0, magnet = false, energy = false, verbose = false, strang = true)
    # println(linkdims(ans_mps))
    ans_core, ans_factors, state, nrg, bd = bug_integrator_mat_ra(H_ops_xxx, init_core, init_factors, t0, T, step_list[i])
    ans_mps_bug, _, _, _ = mps_bug_constant(H_mpo, init_MPS, t0, T, step_list[i], 5)
    tdvp_vec = vectorize_mps(ans_mps; order = "reverse")
    bug_array = Multi_TTM_recursive(ans_core, ans_factors)
    bug_tucker_vec = vec(permutedims(bug_array, reverse(1:ndims(bug_array))))
    bug_mps_vec = vectorize_mps(ans_mps_bug; order = "reverse")
    
    err_list_tdvp[i] = norm(true_sol - tdvp_vec)
    err_list_bug_tucker[i] = norm(true_sol - bug_tucker_vec)
    err_list_bug_mps[i] = norm(true_sol - bug_mps_vec)
    # ortho_properties(ans_mps_bug)
end
h_list = (T - t0) ./step_list
err_plot = plot(h_list, [err_list_tdvp, err_list_bug_tucker, err_list_bug_mps], label = ["TDVP2 Error" "BUG Tucker Error" "BUG MPS Error"], xlabel = L"\Delta t (timestep)", title = "Final Error of TDVP2, BUG", dpi = 250)
plot!(h_list, h_list.^2, label = L"O(\Delta t^2)", xscale =:log10, yscale =:log10, legend =:topleft, linestyle =:dash)
# savefig(err_plot, "convergence_plot.png")