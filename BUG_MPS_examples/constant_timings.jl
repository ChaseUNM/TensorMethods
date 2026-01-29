using LinearAlgebra, ITensors, ITensorMPS, Plots, BenchmarkTools, Printf
using TensorMethods

#Get timings for n = 3:100 qubits 
t0 = 0.0
T = 10.0
steps = 1000
N_min = 3
N_max = 100
N_list = collect(N_min:N_max)
total_entries_list = []
trunc_err_list = []
t_list = zeros(length(N_list))
J = 1.0
g = 0.0

for n in N_list 
    println("$n qubits")
    N_levels = fill(2, n)
    sites = siteinds("Qubit", n)
    H = xxx_mpo_scaled(n, sites, 1.0, 0.0)
    init_MPS = random_mps(sites)
    init_MPS_copy = copy(init_MPS)
    _,bd_history,_,_,trunc_err = tdvp2_constant(H, init_MPS, t0, T, Int64(steps/2);cutoff = 1E-10)
    entries_history = count_MPS_history(bd_history, N_levels)
    push!(total_entries_list, entries_history)
    push!(trunc_err_list, trunc_err)
    t_list[n - n_start + 1] = @belapsed begin 
        _,_,_,_,_= tdvp2_constant($H, $init_MPS_copy, $t0, $T, Int64($steps/2);cutoff = 1E-10)
    end
end

trunc_err_avg = zeros(length(N_list))
total_entries_end = zeros(length(N_list))
for i = 1:length(N_list)
    trunc_err_avg[i] = sum(trunc_err_list[i])/length(trunc_err_list[i])
end
for i = 1:length(N_list)
    total_entries_end[i] = total_entries_list[i][end]
end
# writedlm("t_list", t_list)
# writedlm("total_entries_list", total_entries_list)
# writedlm("trunc_err_list", trunc_err_list)
t_plot = plot(N_list, t_list, xlabel = "# of qubits", ylabel = "Runtime(s)")
trunc_err_plot = plot(N_list, trunc_err_avg, xlabel = "# of qubits", label = "Average truncation error")
total_entries_plot = plot(N_list, total_entries_end, xlabel = "# of qubits", label = "Total # of entries stored")