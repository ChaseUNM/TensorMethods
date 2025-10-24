using LinearAlgebra, ITensors, DelimitedFiles, LaTeXStrings

# include("BUG_tucker(8-27).jl")
# include("Tucker_Matrices.jl")
# include("hamiltonian(8-4).jl")
# include("BUG_small.jl")
include("controlH.jl")

N = 2
N_levels = fill(4, N)
sites = qudit_siteinds(N, N_levels)
transition_freq = rand(N)
rot_freq = zeros(N)
self_kerr = rand(N)
dipole = rand(N,N)
zz = rand(N,N)

H_ops = H_sys_rot(N, N_levels, transition_freq, rot_freq,  self_kerr, dipole, zz)
H_mat = H_sys(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz)
H_ten = ITensor(H_mat, sites, sites')

J = 1.0
g = 1.0
ops = xxx_ops(N, J, g)

A = rand(ComplexF64, N_levels...)
A = A/norm(A)
init_vec = vec(A)
core, factors = tucker(A; cutoff = 0.0)
core_ten, factors_ten = tucker_itensor(A, sites; cutoff = 0.0)

t0 = 0.0
T = 10.0
steps = 1000
h = (T - t0)/steps

#Test mat-vec 
storage_vec = zeros(ComplexF64, steps + 1, prod(N_levels))
storage_vec[1,:] .= init_vec
init_vec_copy = copy(init_vec)
sol_op = exp(-im*h*H_mat)
@showprogress 1 "Evolving mat-vec" for i in 1:steps
    ans_vec = sol_op*init_vec_copy 
    storage_vec[i + 1,:] = ans_vec 
    init_vec_copy .= ans_vec
end

core_final,factors_final,state_history,energy_history = bug_integrator_mat(H_ops, core, factors, t0, T, steps)
core_final_ten,factors_final_ten,state_history_ten, energy_history_ten = bug_integrator_itensor(H_ten, core_ten, factors_ten, t0, T, steps, sites)
display(abs2.(storage_vec))
display(abs2.(state_history))
display(abs2.(state_history_ten))
# p1 = plot(LinRange(t0, T, steps + 1), abs2.(storage_vec) .- abs2.(state_history))