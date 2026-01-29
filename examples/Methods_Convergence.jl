using ITensors, ITensorMPS, LinearAlgebra, LaTeXStrings, Plots
using TensorMethods

#Set up total number of subsystems
N = 10

#Create sites and initial states for both MPS and Tucker-tensor
sites = siteinds("Qubit", N)
q_state = Int64.(fill(1, N))
q_state[1] = 0

init_MPS = init_separable(sites, q_state)
init_ten = zeros(ComplexF64, fill(2, N)...)
q_state_arr = q_state .+ 1
init_ten[q_state_arr...] = 1.0 + 0.0*im
init_core, init_factors = tucker(init_ten; cutoff = 0.0)

init_vec = vectorize_mps(init_MPS; order = "reverse")

#Set initial and final time and determine different number of time steps
t0 = 0.0
T = 1.0
step_list = 2 .^collect(5:6)
#Create empty vectors to store errors in different methods
err_list_tdvp = zeros(length(step_list))
err_list_bug_tucker = zeros(length(step_list))

J = 1.0
g = 1.0

H_mpo = xxx_mpo_scaled(N, sites, J, g)
H_mat = xxx_scaled(N, J, g)
H_ops_xxx = ops_xxx_scaled(N, J, g)

#Get true solution
true_sol = exp(-im*H_mat*(T - t0))*init_vec
for i in 1:length(step_list)
    println("Steps: $(step_list[i])")
    
    #Solve using TDVP2
    ans_mps, bd_history, magnet_history, energy_history = tdvp2_constant(H_mpo, init_MPS, t0, T, Int64(step_list[i]/2); cutoff = 0.0, magnet = false, energy = false, verbose = false, strang = true)
    #Solve using BUG integrator
    ans_core, ans_factors, state, nrg, bd = bug_integrator_mat_ra(H_ops_xxx, init_core, init_factors, t0, T, step_list[i])
    #Convert to vectors to compare to true solution
    tdvp_vec = vectorize_mps(ans_mps; order = "reverse")
    bug_array = Multi_TTM_recursive(ans_core, ans_factors)
    bug_tucker_vec = vec(permutedims(bug_array, reverse(1:ndims(bug_array))))
    
    err_list_tdvp[i] = norm(true_sol - tdvp_vec)
    err_list_bug_tucker[i] = norm(true_sol - bug_tucker_vec)
end

#Plot errors of both TDVP2 and BUG
h_list = (T - t0) ./step_list
err_plot = plot(h_list, [err_list_tdvp, err_list_bug_tucker], label = ["TDVP2 Error" "BUG Tucker Error"], xlabel = L"\Delta t (timestep)", title = "Final Error of TDVP2, BUG", dpi = 250)
plot!(h_list, h_list.^2, label = L"O(\Delta t^2)", xscale =:log10, yscale =:log10, legend =:topleft, linestyle =:dash)
# savefig(err_plot, "convergence_plot.png")