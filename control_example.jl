using LinearAlgebra, ITensors, DelimitedFiles, LaTeXStrings

# include("BUG_tucker(8-27).jl")
# include("Tucker_Matrices.jl")
# include("hamiltonian(8-4).jl")
# include("BUG_small.jl")
include("controlH.jl")

N = 3
N_levels = [4, 4, 4]
sites = qudit_siteinds(N, N_levels)
# transition_freq = [4.80595, 4.8601, 5.1]*(2*pi)
transition_freq = [4.8, 5.0, 5.2]*2*pi
rot_freq = sum(transition_freq)/N*ones(N)
dipole = [0 0.005 0; 0 0 0.005; 0 0 0]*(2*pi)
# dipole = [0 0.005 0.005; 0 0 0.005; 0 0 0]*(2*pi)
# self_kerr = [0.2, 0.2, 0.2]*(2*pi)
self_kerr = [0.15, 0.15, 0.15]*(2*pi)
zz = zeros(N, N)

om = zeros(2,2)
om[1,1] = 0.027532809972830558*2*pi
om[1,2] = -0.027532809972830558*2*pi 
om[2,1] = 0.027532809972830558*2*pi 
om[2,2] = -0.027532809972830558*2*pi
om = zeros(3, 6) 
om[1,:]= [0.17817603064445744, -0.06158197214672839, -0.11659405849772948, -0.23190739860436582, 0.12279645755956273, -0.35582688473631774].*2*pi
om[2,:] = [0.17817603064445744, -0.06158197214672839, -0.11659405849772948, -0.23190739860436582, 0.12279645755956273, -0.35582688473631774].*2*pi
om[3,:] = [0.17817603064445744, -0.06158197214672839, -0.11659405849772948, -0.23190739860436582, 0.12279645755956273, -0.35582688473631774].*2*pi
# om = zeros(3, 3)
# om[1,:] = []
om = Vector{Vector{Float64}}(undef, N)
om[1] = [0.0, -0.20012496]*2*pi
om[2] = [0.20012496, 0.0, -0.20012496]*2*pi
om[3] = [0.20012496, 0.0]*2*pi
t0 = 0.0
t0_copy = 0.0
T = 300.0
steps = 68000
h = (T - t0)/steps

# nsplines = 92
nsplines = 32
# pars = vec(readdlm("params_N3.dat"))
pars = vec(readdlm("params_toffoli.dat"))
bc_params = bcparams2((T - t0),nsplines, om, pars)

H_s = H_sys_rot(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz)
H_s_ten = H_sys_rot_ten(N, N_levels, sites, transition_freq, rot_freq, self_kerr, dipole, zz)
H_s_mat = H_sys(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz)

# init = rand(ComplexF64, N_levels...)
# init = init/norm(init)
init = zeros(ComplexF64, N_levels...)
init[fill(1, N)...] = 1.0 + 0.0*im
init_core, init_factors = tucker(init; cutoff = 0.0)
init_vec = vec(init)
init_core_t, init_factors_t = tucker_itensor(init, sites; cutoff = 0.0)

#Test mat-vec 
# storage_vec = zeros(ComplexF64, steps + 1, prod(N_levels))
# storage_vec[1,:] .= init_vec
# init_vec_copy = copy(init_vec)
# @showprogress 1 "Evolving mat-vec" for i in 1:steps
#     global H_total = H_s_mat .+ H_ctrl(H_s_mat, N, N_levels, bc_params, t0_copy)
#     global ans_vec = exp(-im*H_total*h)*init_vec_copy 
#     storage_vec[i + 1,:] = ans_vec 
#     init_vec_copy .= ans_vec
#     global t0_copy += h 
# end
# display(abs2.(storage_vec[end,:]))
# _,_,state_history, energy_history = bug_integrator_mat(H_s, init_core, init_factors, t0, T, steps)


# core_final,factors_final,state_history, energy_history, bd_history = bug_integrator_mat_ra(H_s, bc_params, init_core, init_factors, t0, T, steps, 0.0)
# _,_,state_history_ten, energy_history_tucker, bd_history = bug_integrator_itensor_ra(H_s_ten, bc_params, init_core_t, init_factors_t, t0, T, steps, sites, cutoff)
# _,_,state_history_ten, energy_history_tucker = bug_integrator_itensor(H_s_ten, bc_params, init_core_t, init_factors_t, t0, T, steps, sites)
# display(abs2.(state_history[end,:]))
# display(abs2.(state_history_ten[end,:]))
# for i in 1:length(H_s)
#     if length(H_s[i]) == 2
#         println("Norm difference: ", norm(H_c1[i][1] - H_c2[i][1]))
#     elseif length(H_s[i]) == 4
#         println("Norm difference: ", norm(H_c1[i][1] - H_c2[i][1]) + norm(H_c1[i][3] - H_c2[i][3]))
#     end
# end
# fidelity = 0.0
# init_conditions = [[1,1,1], [1,1,2], [1,2,1], [1,2,2], [2,1,1], [2,1,2], [2,2,1], [2,2,2]]
# for i in init_conditions
#     init = zeros(ComplexF64, N_levels...)
#     init[i...] = 1.0 + 0.0*im
#     init_core, init_factors = tucker(init; cutoff = 0.0)
#     core_final, factors_final, state_history, energy_history, bd_history = bug_integrator_mat_ra(H_s, bc_params, init_core, init_factors, t0, T, steps, 0.0)
#     if i == [2,2,1]
#         desired = [2, 2, 2]
#     elseif i == [2,2,2]
#         desired = [2,2,1]
#     else 
#         desired = i 
#     end
#     desired_arr = zeros(ComplexF64, N_levels...)
#     desired_arr[desired...] = 1.0 + 0.0*im
#     core_desired, factors_desired = tucker(desired_arr; cutoff = 0.0)
#     global fidelity += abs2(inner(core_final, factors_final, core_desired, factors_desired))
#     println(fidelity)
# end
# fidelity = fidelity/length(init_conditions)
# println("Fidelity: ", fidelity) 
function plot_essential(state_history, states, N_levels)
    steps, _ = size(state_history)
    N_essential = length(states)
    essential_history = zeros(ComplexF64, steps, N_essential)
    for i in 1:N_essential
        index = linear_index_natural(states[i], N_levels)
        essential_history[:,i] .= state_history[:,index]
    end
    return essential_history 
end
        


function toffoli_plot(cutoff)
    fidelity = 0.0
    init_conditions = [[1,1,1], [1,1,2], [1,2,1], [1,2,2], [2,1,1], [2,1,2], [2,2,1], [2,2,2]]
    bd_plots = plot(layout = (4, 2))
    state_plots = plot(layout = (4, 2))
    count = 1
    for i in init_conditions
        init = zeros(ComplexF64, N_levels...)
        init[i...] = 1.0 + 0.0*im
        init_core, init_factors = tucker(init; cutoff = 0.0)
        core_final, factors_final, state_history, energy_history, bd_history = bug_integrator_mat_ra(H_s, bc_params, init_core, init_factors, t0, T, steps, cutoff)
        essential_history = plot_essential(state_history, init_conditions, N_levels)
        if i == [2,2,1]
            desired = [2, 2, 2]
        elseif i == [2,2,2]
            desired = [2,2,1]
        else 
            desired = i 
        end
        desired_arr = zeros(ComplexF64, N_levels...)
        desired_arr[desired...] = 1.0 + 0.0*im
        core_desired, factors_desired = tucker(desired_arr; cutoff = 0.0)

        fidelity += abs2(inner(core_final, factors_final, core_desired, factors_desired))
        # bd_plot = plot(LinRange(t0,T,steps + 1), title = "|$(tuple(i...))>", xlabel = "t", ylabel = "Bond Dimension", bd_history)
        i_new = i .- 1
        if count == 1
            plot!(bd_plots[count], LinRange(t0,T,steps + 1), bd_history,title = "|$(tuple(i_new...))>", titlefontsize = 4, labels = ["Qudit 1" "Qudit 2" "Qudit 3"], xlabel = "t", ylabel = "Bond Dimension", xguidefontsize = 4, yguidefontsize = 4, legendfontsize = 4)
            plot!(state_plots[count], LinRange(t0, T, steps + 1), abs2.(essential_history),title = "|$(tuple(i_new...))>", titlefontsize = 4, labels = [L"|000\rangle" L"|001\rangle" L"|010\rangle" L"|011\rangle" L"|100\rangle" L"|101\rangle" L"|110\rangle" L"|111\rangle"], xlabel = "t", ylabel = "Population", xguidefontsize = 4, yguidefontsize = 4, legend =:top, legend_columns = 8, legend_background_color = RGBA(1,1,1,0.25), legendfontsize = 3)
        else
            plot!(bd_plots[count], LinRange(t0,T,steps + 1), bd_history,title = "|$(tuple(i_new...))>", titlefontsize = 4, legend = false)
            plot!(state_plots[count], LinRange(t0, T, steps + 1), abs2.(essential_history),title = "|$(tuple(i_new...))>", titlefontsize = 4, legendfontsize = 4, legend = false)
        end
        count += 1
    end
    fidelity = fidelity/length(init_conditions)
    return bd_plots, state_plots, fidelity
end 

cut = 1E-15
# bd, state, fidelity = toffoli_plot(cut)
# plot!(bd, suptitle = "Cutoff: $cut, Gate Fidelity: $fidelity", titlefontsize = 6)
# plot!(state, suptitle = "Cutoff: $cut, Gate Fidelity: $fidelity", titlefontsize = 6)
#plot control pulse 
t_range = LinRange(t0,T,steps)

pulse1 = [bcarrier2(t, bc_params, 0) for t in LinRange(t0, T, steps)].*(500/pi)
pulse2 = [bcarrier2(t, bc_params, 1) for t in LinRange(t0, T, steps)].*(500/pi)
pulse3 = [bcarrier2(t, bc_params, 2) for t in LinRange(t0, T, steps)].*(500/pi)
pulse4 = [bcarrier2(t, bc_params, 3) for t in LinRange(t0, T, steps)].*(500/pi)
pulse5 = [bcarrier2(t, bc_params, 4) for t in LinRange(t0, T, steps)].*(500/pi)
pulse6 = [bcarrier2(t, bc_params, 5) for t in LinRange(t0, T, steps)].*(500/pi)
p1 = plot(t_range, [pulse1, pulse2], labels = ["Real" "Imaginary"], title = "Qudit 1", legendfontsize = 6)
p2 = plot(t_range, [pulse3, pulse4], labels = ["Real" "Imaginary"], title = "Qudit 2", legendfontsize = 6)
p3 = plot(t_range, [pulse5, pulse6], labels = ["Real" "Imaginary"], title = "Qudit 3", legendfontsize = 6)
total_plot = plot(p1, p2, p3, layout = (3, 1))
# bd_plot = plot(LinRange(t0,T,steps + 1), title = L"|000\rangle", xlabel = "t", ylabel = "Bond Dimension", bd_history, labels = ["Qudit 1" "Qudit 2" "Qudit 3"])
# savefig(bd_plot, "bd_plot.png")