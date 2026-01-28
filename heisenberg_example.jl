using ITensors, ITensorMPS, Plots, ProgressMeter
pyplot()

include("hamiltonian(8-4).jl")
include("BUG_tucker(8-27).jl")
include("tdvp(8-4).jl")
include("Hamiltonian.jl")

#Need
ITensors.set_warn_order(20)


#Define Hamiltonian 

#Specify number of sites and create indices for sites
N = 8
sites = siteinds("Qubit", N)
#Set parameters for Hamiltonian
J = 1.0
U = 0.0
hj_list = 0*rand(N)
hp_list = 0*rand(N)

#Create matrix form of Hamiltonian
H = graziani_H(N, J, U, hj_list, hp_list)
#Create MPO for Hamiltonian
H_MPO = graziani_H_MPO(N, sites, J, U, hj_list, hp_list)
#Create tensor form of Hamiltonian for BUG
H_ten = ITensor(H, reverse(sites), reverse(sites)')

#Specify initial state
q_state = [0,0,0,0,1,1,1,1]

#Plots heatmap using TDVP2 method, compares evolution to matrix exponentiation
function magnet_heatmap_TDVP2(t0,T, steps, cutoff)
    N = length(q_state)
    init = init_separable(sites, q_state)
    init_vec = ComplexF64.(vectorize_mps(init, order = "reverse"))
    magnet_storage_exp = zeros(N, steps + 1)
    magnet_storage_mps = zeros(N, steps + 1) 
    h = (T - t0)/steps
    for i in 1:N
        init_vec_copy = copy(init_vec)
        m_mat = s_op(sz, N -i + 1, N)
        magnet_storage_exp[i,1] = real(init_vec'*m_mat*init_vec)
        @showprogress 1 "Matrix Exponentiation" for j in 1:steps 
            evolve_vec = exp(-im*H*h)*init_vec_copy 
            magnet_storage_exp[i,j + 1] = real(evolve_vec'*m_mat*evolve_vec)
            init_vec_copy .= evolve_vec 
        end
        _,magnet_history,_ = tdvp2_constant_magnet(H_MPO, init, t0,T,steps, cutoff, N - i + 1)
        magnet_storage_mps[i,:] = magnet_history
    end
    x = LinRange(t0,T,steps + 1)
    y = collect(1:N)
    h_plot_exp = heatmap(x, y, magnet_storage_exp, c=:bluesreds, ylabel = "Site Index", xlabel = "time", yflip = true; colorrange = (-1,1))
    h_plot_mps = heatmap(x, y, magnet_storage_mps, c=:bluesreds, yflip = true; colorrange = (-1,1))
    h_diff = heatmap(x, y, abs.(magnet_storage_exp - magnet_storage_mps), c=:bluesreds, yflip = true)
    # h_diff = heatmap(abs.(magnet_storage_exp - magnet_storage_mps), yflip = true, c=:bluesreds,clims = (1E-8, 1E-4), colorbar_ticks = log10.([1E-9, 1E-8, 1E-7, 1E-6, 1E-5, 1E-4]))
    # display(magnet_storage_exp - magnet_storage_mps)
    # difference = abs.(x, y, magnet_storage_exp - magnet_storage_mps)
    return h_plot_exp, h_plot_mps, h_diff
end

#Plots heatmap of magnetization on each site using rank-adaptive BUG, compares evolution to matrix exponentiation
function magnet_heatmap_BUG(t0,T, steps, cutoff)
    N = length(q_state)
    core_sep, factors_sep = tucker_separable(sites, q_state)
    init_vec = ComplexF64.(reverse_vec(reconstruct(core_sep, factors_sep)))
    magnet_storage_exp = zeros(N, steps + 1)
    magnet_storage_tucker = zeros(N, steps + 1) 
    h = (T - t0)/steps
    for i in 1:N
        init_vec_copy = copy(init_vec)
        m_mat = s_op(sz, N - i + 1, N)
        magnet_storage_exp[i,1] = real(init_vec'*m_mat*init_vec)
        @showprogress 1 "Evolving Mat-vec" for j in 1:steps 
            evolve_vec = exp(-im*H*h)*init_vec_copy 
            magnet_storage_exp[i,j + 1] = real(evolve_vec'*m_mat*evolve_vec)
            init_vec_copy .= evolve_vec 
        end
        _,_,magnet_history,_ = bug_integrator_itensor_ra_magnet(H_ten, core_sep, factors_sep, t0, T, steps, sites, N - i + 1, cutoff)
        # display(magnet_history)
        magnet_storage_tucker[i,:] = magnet_history
    end
    x = LinRange(t0,T,steps + 1)
    y = collect(1:N)
    h_plot_exp = heatmap(x, y, magnet_storage_exp, c=:bluesreds, ylabel = "Site Index", xlabel = "time", yflip = true; colorrange = (-1,1))
    h_plot_tucker = heatmap(x, y, magnet_storage_tucker, c=:bluesreds, yflip = true; colorrange = (-1,1))
    h_diff = heatmap(x, y, abs.(magnet_storage_exp - magnet_storage_tucker), c=:bluesreds, yflip = true)
    # h_diff = heatmap(abs.(magnet_storage_exp - magnet_storage_mps), yflip = true, c=:bluesreds,clims = (1E-8, 1E-4), colorbar_ticks = log10.([1E-9, 1E-8, 1E-7, 1E-6, 1E-5, 1E-4]))
    # display(magnet_storage_exp - magnet_storage_mps)
    # difference = abs.(x, y, magnet_storage_exp - magnet_storage_mps)
    return h_plot_exp, h_plot_tucker, h_diff
end

t0 = 0.0
T = 10.0
steps = 2000
cutoff = 0.0

h1, h2, h3 = magnet_heatmap_BUG(t0,T,steps, cutoff)

total_plot = plot(h1, h2, h3, layout = (3, 1))