using ITensors, ITensorMPS, Plots, ProgressMeter
gr()

include("hamiltonian(8-4).jl")
include("BUG_tucker(8-27).jl")
include("tdvp(8-4).jl")
include("Hamiltonian.jl")

#Need to first set up initial conditions as well as the tensors

#Initial conditions can be set up by specifying the state and the respective sites that the qubits are in as follows 
q_state = [0,0,0,0,0,0]
N = length(q_state)
sites = siteinds("Qubit", N)
#Then call functions that convert this into an MPS, Tucker Tensor, or vector 
init_MPS = init_separable(sites, q_state)
init_core, init_factors = tucker_separable(sites, q_state)
init_vec = vec_separable(q_state) #This is in reverse order because the Hamiltonian is defined in reverse order due to the Kronecker products
#Also possible to convert to vector from MPS to vector
vec_mps = vectorize_mps(init_MPS; order = "reverse") #Use "reverse" if you want reverse order indexing and "natural" if you want natural order indexing, "natural" is default but "reverse" works if your definition of the Hamiltonian is given by kronecker products

#Can convert from tucker format to vector format by first reconstructing the tensor using reconstruct 
full_tensor = reconstruct(init_core, init_factors)
#And then calling vec_itensor
vec_tucker = vec_itensor(full_tensor; order = "reverse") #Can either use natural or reverse order here but again, reverse is better if applying a Hamiltonian to it

#Now can create a Hamiltonian 
#There are 2 main Hamiltonians that can be looked at for Heisenberg models
#The xxx hamiltonian, and the hamiltonian from Frank's slides 

#Can create the matrix for xxx using 
J = 1.0
g = 1.0
H_xxx = xxx(N, J, g)
#Can create the MPO version of this using 
H_xxx_mpo = xxx_mpo(N, sites, J, g)
#Can create the tensor version using 
H_xxx_ten = ITensor(H_xxx, reverse(sites), reverse(sites)')

#Can create Frank's Hamiltonian using
J = 0.0
U = 1.0
hj_list = 0*rand(N)
hp_list = zeros(N) #Can always set this to 0 
H_graziani = graziani_H(N, J, U, hj_list, hp_list)

#and now for the MPO 
H_graziani_mpo = graziani_H_MPO(N, sites, J, U, hj_list, hp_list)

#and now for the tensor version 
H_graziani_ten = ITensor(H_graziani, reverse(sites), reverse(sites'))

#Now can actually evolve the state with the hamiltonian 
#there is either the TDVP2 method or the BUG method
#The TDVP2 method works for MPS and the BUG method works for tucker tensor 

#to evolve with TDVP2 for a constant Hamiltonian call tdvp2_constant(MPO, MPS, initial_time, final_time, # of steps, svd cutoff, normalize, verbose)
#the normalize takes in a boolean and normalizes the singular values if normalize = true, otherwise doesn't normalize singular values by default it's set to false


#tdvp2_constant returns the MPS at the final time-step, as well as the state history as a vector in reverse order, and then the bond dimension throughout evolution 
#Here is an example evolution 
t0 = 0.0
T = 10.0
steps = 2000
h = (T - t0)/steps
svd_cutoff = 0.0

M_final, state_history, bd_history = tdvp2_constant(H_xxx_mpo, init_MPS, t0, T, steps, svd_cutoff)

#Can compare M_final to evolution using matrix exponentiation 
mat_vec = exp(-im*H_xxx*(T - t0))*init_vec 
vec_final = vectorize_mps(M_final; order = "reverse")
println("MPS Difference: ", norm(mat_vec - vec_final))

#to evolve with rank adaptive BUG, call bug_integrator_itensor_ra(Hamiltonian tensor, core, factors, initial_time, final_time, # of steps, sites, svd_cutoff)
#bug_integrator_ra returns the core and factors at the final time-step, as well as the state history as a vector in reverse order, and then the bond dimension throughout evolution
final_core, final_factors, state_history_tucker, bd_history_tucker = bug_integrator_itensor_ra(H_xxx_ten, init_core, init_factors, t0, T, steps, sites, svd_cutoff)
#Can reconstruct tensor at final time and the vectorize and compare it to using matrix exponentiation 
final_tensor = reconstruct(final_core, final_factors)
vec_tucker = vec_itensor(final_tensor; order = "reverse")
println("Tucker Difference: ", norm(vec_tucker - mat_vec))

#Now try the magnetization
#magnetization is a different function, for TDVP2 it's tdvp2_constant_magnet(MPO, MPS, initial_time, final_time, #of steps, svd_cutoff)
#for bug it's bug_integrator_itensor_magnet_ra
#both of these functions return the magnetization history at a specific site as opposed to the state history 
magnet_site = 1

_, magnet_history_tdvp, _ = tdvp2_constant_magnet(H_xxx_mpo, init_MPS, t0, T, steps, svd_cutoff, N - magnet_site + 1)
_, _, magnet_history_bug, _ = bug_integrator_itensor_ra_magnet(H_xxx_ten, init_core, init_factors, t0, T, steps,sites, N - magnet_site + 1, svd_cutoff)

#Can also evolve the magnetization with matrix exponentiation 
magnet_history_vec = zeros(steps + 1)
magnet_op = s_op(sz, N - magnet_site + 1, N)
magnet_history_vec[1] = init_vec'*magnet_op*init_vec 
init_vec_copy = ComplexF64.(copy(init_vec))
for i in 1:steps 
    vec_evolve = exp(-im*(H_xxx)*h)*init_vec_copy 
    magnet_history_vec[i + 1] = real(vec_evolve'*magnet_op*vec_evolve)
    init_vec_copy .= vec_evolve 
end

plot(LinRange(t0, T, steps + 1), [magnet_history_tdvp magnet_history_bug magnet_history_vec], ["TDVP2" "BUG" "Mat_vec"])


