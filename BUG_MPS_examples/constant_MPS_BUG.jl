using Revise
using ITensors, ITensorMPS, LinearAlgebra, LaTeXStrings, Plots, Random
using TensorMethods


Random.seed!(42)
N = 4

sites = siteinds("Qubit", N)
q_state = Int64.(fill(1, N))
q_state[1] = 0

t0 = 0.0
T = 0.1

steps = 2

J = 1.0
g = 1.0

center = N
println("Orthogonality Center: $center")
init_MPS = init_separable(sites, q_state)
init_vec = vectorize_mps(init_MPS; order = "reverse")
init_MPS_bug = orthogonalize!(init_MPS, center)



# init_vec = rand(ComplexF64, 2^N)
# init_vec = init_vec/norm(init_vec)
# init_MPS = MPS(init_vec, sites)
# init_MPS_bug = orthogonalize(init_MPS, center)

H_mpo = xxx_mpo_scaled(N, sites, J, g)
H_mat = xxx_scaled(N, J, g)
# true_sol = exp(-im*H_mat*(T - t0))*init_vec

ans_tdvp, link_dim, _, _, _ = tdvp2_constant(H_mpo, init_MPS, t0, T, steps; strang = false)
ans_mps_bug1, _, _, _ = mps_bug_constant(H_mpo, init_MPS_bug, t0, T, steps, center) 
# ans_mps_bug2, _, _, _ = mps_bug_constant(H_mpo, init_MPS_bug, 0.0, 0.1, 1, center) 
# ans_mps_bug3, _, _, _ = mps_bug_constant(H_mpo, ans_mps_bug2, 0.1, 0.2, 1, center)
# ans_mps_bug4, _, _, _ = mps_bug_constant(H_mpo, init_MPS_bug, t0, T, 1, center)
bug_mps_vec = vectorize_mps(ans_mps_bug1; order = "reverse")
tdvp_vec = vectorize_mps(ans_tdvp; order = "reverse")
# bug3_mps_vec = vectorize_mps(ans_mps_bug3; order = "reverse")
# bug4_mps_vec = vectorize_mps(ans_mps_bug4; order = "reverse")

# println("Error BUG MPS: ", norm(bug_mps_vec - true_sol))
# println("Error BUG MPS 2: ", norm(bug3_mps_vec - true_sol))
# println("Error BUG MPS 3: ", norm(bug4_mps_vec - true_sol))
# println("Error TDVP: ", norm(tdvp_vec - true_sol))
println("Norm of BUG MPS: ", norm(ans_mps_bug1))

print_norm = false

if print_norm == true 
    for i in 1:N 
        println("Norm of site $i")
        println("Initial MPS: ", norm(init_MPS))
        println("BUG MPS: ", norm(ans_mps_bug1[i]))
        println("TDVP: ", norm(ans_tdvp[i]))
    end
end