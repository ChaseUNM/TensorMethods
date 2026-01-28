using LinearAlgebra, ITensors, ITensorMPS, Plots, BenchmarkTools, Printf
using TensorMethods

#Get timings for n = 3:100 qubits 

#Set up step-size, total runtime, minimum qubits and maximum qubits as well as hamiltonian parameters J and g.
t0 = 0.0
T = 10.0
steps = 1000
#Set the smallest and largest number of subsystems
N_min = 3
N_max = 5
N_list = collect(N_min:N_max)
total_entries_list = []
trunc_err_list = []
t_list = zeros(length(N_list))
J = 1.0
g = 0.0

#For each qubit evolve the system with tdvp2, time with @belapsed
for n in N_list 
    println("$n qubits")
    N_levels = fill(2, n)
    sites = siteinds("Qubit", n)
    H = xxx_mpo_scaled(n, sites, 1.0, 0.0)
    init_MPS = random_mps(sites)
    init_MPS_copy = copy(init_MPS)
    #Set sufficiently small cutoff, storing both the bond dimension history and the truncation error to ensure there's no truncation
    #Steps are halved when using Strang splitting, which is the default
    t_list[n - N_min + 1] = @belapsed begin
        _,_,_,_,_= tdvp2_constant($H, $init_MPS_copy, $t0, $T, Int64($steps/2);cutoff = 1E-15^2)
    end

    _,bd_history,_,_,trunc_err= tdvp2_constant(H, init_MPS_copy, t0, T, Int64(steps/2);cutoff = 1E-15^2)
    #Count total number of entries
    entries_history = count_MPS_history(bd_history, N_levels)
    push!(total_entries_list, entries_history)
    push!(trunc_err_list, trunc_err)
    
end

trunc_err_avg = zeros(length(N_list))
total_entries_end = zeros(length(N_list))
for i = 1:length(N_list)
    trunc_err_avg[i] = sum(trunc_err_list[i])/length(trunc_err_list[i])
end
for i = 1:length(N_list)
    total_entries_end[i] = total_entries_list[i][end]
end
#Plot runtime, average truncation error, and total entries
t_plot = plot(N_list, t_list, xlabel = "# of qubits", ylabel = "Runtime(s)")
trunc_err_plot = plot(N_list, trunc_err_avg, xlabel = "# of qubits", label = "Average truncation error")
total_entries_plot = plot(N_list, total_entries_end, xlabel = "# of qubits", label = "Total # of entries stored")