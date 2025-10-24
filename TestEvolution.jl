using LinearAlgebra, ITensors, ITensorMPS, Random, LaTeXStrings

Random.seed!(42)
include("BUG_tucker(8-27).jl")
include("Tucker_Matrices.jl")
include("Hamiltonian.jl")
include("BUG_small.jl")
include("hamiltonian(8-4).jl")
include("controlH.jl")

N = 3
sites = siteinds("Qubit", N) 
g = 1.0
J = 1.0
H_mat = xxx(N, J, g)
# a = rand(2, 2)
# b = rand(2, 2)
# c = rand(2, 2)
# H_mat = kron(c, kron(b, a))
H_mat = xxx(N, J, g)
xxx_ops_list = xxx_ops(N, J, g)
total_H_xxx = total_H_itensor(xxx_ops_list, sites)
# abc_ops = ops_ex_3(a, b, c)
# abc_ops_H = total_H_itensor(abc_ops, sites)

# H_ten = ITensor(H_mat, sites, sites')
function ops_xxx_new(N, J, g)
    H_single_site = Vector{Tuple{AbstractMatrix, Int64}}()
    H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    for i in 1:N 
        tup = (-g*J*sx, i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        tup = (-J*sz, i, sz, i + 1)
        push!(H_two_site, tup)
    end
    return vcat(H_single_site, H_two_site)
end
xxx_ops2 = ops_xxx_new(N, J, g)

# H_mat = Matrix(1.0*I, 2^N, 2^N)
ident_ops = [identity_ops(N)]
total_H_ident = total_H_itensor(ident_ops, sites)
H_ten = ITensor(H_mat, sites, sites')

A = rand(ComplexF64, collect(fill(2, N))...)
core_ten, factors_ten = tucker_itensor(A, sites; cutoff = 0.0)
core_arr, factors_arr = tucker(A; cutoff = 0.0)

alloc_tensors = preallocate_itensor(core_ten, factors_ten)

M_list, P_list, Y_list = pre_allocate(core_arr, factors_arr)

h = 0.01

# @btime begin 
# core_1,factors_1 = bug_step_itensor(H_ten, core_ten, factors_ten, h, sites)
# end 



# @btime begin 
# core_2,factors_2 = bug_step_eff_ra(total_H_xxx, core_ten, factors_ten, h, sites, 0.0)
# end 
# @btime begin
# core_3,factors_3 = bug_step_mat_ra(xxx_ops2, core_arr, factors_arr, h, M_list, P_list, Y_list, 0.0)
# end



# ans_1 = reconstruct(core_1, factors_1)
# ans_2 = reconstruct(core_2, factors_2)
# ans_3 = Multi_TTM_recursive(core_3, factors_3)
# println("Norm difference: ", norm(Array(ans_1, inds(ans_1)) - Array(ans_2, inds(ans_2))))
# println("Norm difference 2: ", norm(Array(ans_1, inds(ans_1)) - ans_3))

#Now test evolution with drift Hamiltonian 
N = 3
level_max = 4
N_levels = fill(level_max, N)
sites = qudit_siteinds(N, N_levels)
transition_freq = rand(N)
rot_freq = zeros(N)
self_kerr = rand(N)
# dipole = [0.0 2.0 1.0; 0.0 0.0 4.0; 0.0 0.0 0.0]
dipole = rand(N, N)
# zz = [0.0 0.4 0.5; 0.0 0.0 0.6; 0.0 0.0 0.0]
zz = rand(N, N)
H_ops = quandary_h(N, N_levels, transition_freq, self_kerr, dipole, zz)
H_ops2 = H_sys_rot(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz)
H_ops_ten = quandary_h(N, N_levels, sites, transition_freq, self_kerr, dipole, zz)
H_mat = H_sys(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz)
H_ten = ITensor(H_mat, sites, sites')
A = rand(ComplexF64, N_levels...)
A = A/norm(A)
# A = zeros(ComplexF64, N_levels...)
# A[rand(big.(1:level_max), N)...] = 1.0 + 0.0*im
A_vec = vec(A)
core_ten, factors_ten = tucker_itensor(A, sites; cutoff = 0.0)
core_arr, factors_arr = tucker(A; cutoff = 0.1)
# core_arr, factors_arr = tucker(A; target_rank = fill(1, N))

# core_1, factors_1 = bug_step_itensor(H_ten, core_ten, factors_ten, h, sites)
# core_2, factors_2 = bug_step_mat(H_ops, core_arr, factors_arr, h)

# ans_1 = reconstruct(core_1, factors_1)
# ans_2 = Multi_TTM_recursive(core_2, factors_2)
# ans_3 = exp(-im*H_mat*h)*A_vec
# println("Norm difference: ", norm(Array(ans_1, inds(ans_1)) - ans_2))
# println("Norm difference: ", norm(vec(ans_2) - ans_3))

#Now test evolution 
#Compare with mat-vec multiplication 
# H_mat = xxx(N, J, g)
# H_ops = ops_xxx_new(N, J, g)

t0 = 0.0
T = 10.0
steps = 1000
energy_history_exp = zeros(steps + 1)
state_history_exp = zeros(ComplexF64, steps + 1, prod(N_levels))
h = (T - t0)/steps
sol_op = exp(-im*H_mat*h)
ans_vec = copy(A_vec)
for j in 1:steps + 1
    energy_history_exp[j] = real(ans_vec'*H_mat*ans_vec)
    state_history_exp[j,:] = ans_vec
    if j == steps + 1
        break
    end
    global ans_vec = sol_op*ans_vec
    # init_vec_copy = exp(-im*H*h)*init_vec_copy
end

# core_final,factors_final,energy_history_tucker, bd_history = bug_integrator_mat(H_ops, core_arr, factors_arr, t0, T, steps)
# core_final,factors_final,energy_history_tucker, bd_history = bug_integrator_midpoint(H_ops, core_arr, factors_arr, t0, T, steps, 0.0)
# core_final,factors_final,energy_history_tucker = bug_integrator_mat(H_ops, core_arr, factors_arr, t0, T, steps)
# ans_tucker = vec(Multi_TTM_recursive(core_final, factors_final))

# println("Final Error: ", norm(ans_vec - ans_tucker))

# energy_plot_diff = plot(LinRange(t0, T, steps + 1), [energy_history_exp, energy_history_tucker, energy_history_exp .- energy_history_tucker], labels = ["Exp" "Tucker" "Difference"])
# bd_plot = plot(LinRange(t0, T, steps + 1), bd_history, labels = ["Qudit 1" "Qudit 2" "Qudit 3"])

#Now test the error for different values of h 
t0 = 0.0
T = 10.0
steps_init = 2^9
state_history_exp = zeros(ComplexF64, steps_init + 1, prod(N_levels))
h = (T - t0)/steps_init 
sol_op = exp(-im*H_mat*h)
ans_vec = copy(A_vec)
for j in 1:steps_init + 1
    state_history_exp[j,:] = ans_vec 
    if j == steps_init + 1 
        break 
    end
    global ans_vec = sol_op*ans_vec
end

core_final,factors_final,state_history, _ = bug_integrator_mat(H_ops, core_arr, factors_arr, t0, T, 1000)
ans_tucker = Multi_TTM_recursive(core_final, factors_final)
println("Error: ", norm(ans_vec - vec(ans_tucker)))


steps_list = 2 .^(collect(10:15))
h_list = (T - t0) ./steps_list
err_list2 = zeros(length(steps_list))
count = 1
for i in 1:length(steps_list)
    _,_,state_history, _ = bug_integrator_mat(H_ops, core_arr, factors_arr, t0, T, steps_list[i])
    # core_final,factors_final,state_history, _ = bug_integrator_midpoint(H_ops, core_arr, factors_arr, t0, T, steps_list[i], 1E-14)
    temp_err = zeros(steps_init + 1)
    # display(state_history)
    for k in 1:steps_init + 1
        # temp_err += abs2(abs2.(state_history[k,:]) - abs2.(state_history_exp[k,:]))
        # display(abs2.(state_history[k,:]))
        # display(abs2.(state_history_exp[2*(k - 1) + 1]))
        err = norm(abs2.(state_history_exp[k,:]) - abs2.(state_history[(2^i)*(k - 1) + 1,:]))
        # display(abs2.(state_history_exp[k,:]) - abs2.(state_history[(2^i)*(k - 1) + 1,:]))
        temp_err[k] = err 
    end
    err_list2[i] = sum(temp_err)/(steps_init + 1)
end

p = plot(h_list, err_list2, label = "Error", xlabel = "h", xscale =:log10, yscale =:log10)
plot!(h_list, h_list.^3, label = L"h^3")