using LinearAlgebra, ITensors, ITensorMPS, Plots, BenchmarkTools, Printf
using TensorMethods

#Get error of nice example with small amount of truncation

#Set up hamiltonians
J = 1.0
g = 1.0
h_j = ones(N)
h_p = zeros(N)
H_mpo = xxx_mpo_scaled(N, sites, J, g)
H_mat = xxx_scaled(N, J, g)
H_ops_xxx = ops_xxx_scaled(N, J, g)

#Set up total number of qubits
N_list = collect(3:10)
J = 1.0
g = 0.0


t0 = 0.0
T = 10.0
steps = 1000
err_list_qubits_tdvp = zeros(length(N_list))
err_list_qubits_bug = zeros(length(N_list))
cutoff = 1E-10
for i in 1:length(N_list)
    println("$(N_list[i]) qubits")
    sites = siteinds("Qubit", N_list[i])
    # q_state = Int64.(fill(1, N_list[i]))
    # q_state[1] = 0
    # init_MPS = init_separable(sites, q_state)
    init_MPS = random_mps(sites)
    init_vec = vectorize_mps(init_MPS; order = "reverse")

    H_mpo = xxx_mpo_scaled(N_list[i], sites, J, g)
    H_mat = xxx_scaled(N_list[i], J, g) 
    true_sol, true_magnet, true_energy = exp_solver(H_mat, init_vec, N_list[i], t0, T, steps)
    ans_mps, bd_history, magnet_history, energy_history = tdvp2_constant(H_mpo, init_MPS, t0, T, Int64(steps/2), cutoff; magnet = true, energy = true, verbose = false)

    # println(bd_history[end,:])
    ans_vec = vectorize_mps(ans_mps; order = "reverse")

    err_list_qubits_tdvp[i] = norm(ans_vec - true_sol)

    
end

err_plot_qubits = plot(N_list, err_list_qubits_tdvp, label = "TDVP2 Error", xlabel = "# of qubits", dpi = 250)