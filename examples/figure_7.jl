using Revise
using LinearAlgebra, ITensors, ITensorMPS, Plots, BenchmarkTools, Printf, JLD2
using TensorMethods

################################################################################
# Generate plot and data for Figure 7(a)-(c) and Figure 8 (arxiv 2603.13990)
################################################################################

#Plot magnetization evolution of 4 different cutoffs

#Create initial state
N = 15
sites = siteinds("Qubit", N)

#Set up Hamiltonian, total time duration and number of time-steps
t0 = 0.0
T = 15.0

J = 1.0
g = 0.5
H_mpo = xxx_mpo_scaled(N, sites, J, g)

# H_ops_xxx = ops_xxx_scaled(N, J, g)

steps = 2^8

N_levels = fill(2, N)
q_state = Int64.(fill(0, N))
q_state[1] = 1
init_MPS = init_separable(sites, q_state)

sites_heatmap = collect(1:N)
#Get true magnetization
if N <= 10
    H_mat = xxx_scaled(N, J, g)
    init_vec = vectorize_mps(init_MPS; order = "reverse")
    true_sol, true_magnet, true_energy = exp_solver(H_mat, init_vec, N, t0, T, steps)
    
    true_heatmap = heatmap(LinRange(t0, T, steps + 1), sites_heatmap, true_magnet', c=:bluesreds, title = "Magnetization with matrix exponentiation", xlabel = "time(s)", ylabel = "Magnetization site", dpi = 250)
    savefig(true_heatmap, joinpath(@__DIR__, "true_magnet_g_$g.png"))
end

# magnet_plots = plot(layout = (4, 1))


#Plot magnetization for 3 different cutoff values, these values were chosen visually but could be changed to anything
pts = 60
cutoff_list = 2 .^ LinRange(-30, -6, pts)

cutoff_vals = [cutoff_list[20], cutoff_list[30], cutoff_list[40]]
for i in 1:length(cutoff_vals) 
    println("Cutoff: $(cutoff_vals[i])")
    str = @sprintf "%.2E" cutoff_vals[i]
    init_mps_copy = copy(init_MPS)
    ans_mps,bd_history,magnet,_,trunc_err = tdvp2_constant(H_mpo, init_mps_copy, t0, T, Int64(steps/2); cutoff = cutoff_vals[i]^2, magnet = true, verbose = false)

    # bug_ans_mps, bd_history, magnet, _ = mps_bug_constant(H_mpo, init_MPS, t0, T, steps; cutoff = cutoff_vals[i]^2, magnet = true, verbose = true)
    
    tdvp_heatmap = heatmap(LinRange(t0, T, steps + 1), sites_heatmap, magnet', c =:bluesreds, title = "Magnetization with TDVP2, ε = $str", dpi = 250)
    savefig(tdvp_heatmap, joinpath(@__DIR__, "TDVP2_N_$(N)_magnet_g_$(g)_cutoff_$(str).png"))
    save_object(joinpath(@__DIR__, "TDVP2_N_$(N)_magnet_g_$(g)_cutoff_$(str).jld2"), magnet)

end