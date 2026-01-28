using ITensors, ITensorMPS, LinearAlgebra, Random, Plots, LaTeXStrings, Printf
pyplot()
include("hamiltonian(8-4).jl")
include("tdvp(8-4).jl")
Random.seed!(42)


# N = 3 
# sites = siteinds("S=1/2", N)

# init = rand(8)
# init_mps = MPS(init, sites)

# J = 1.0
# U = 1.0 
# hj_list = rand(3)

# H = graziani_H(N, J, U, hj_list)

# H_MPO = graziani_H_MPO(N, sites, J, U, hj_list)

# H_MPO_mat = matrix_form(H_MPO, sites)

# println("Norm difference: ", norm(H - H_MPO_mat))
# display(H)
# display(H_MPO_mat)

# println("Norm difference application: ", norm(H*init - reconstruct_arr_v2(H_MPO*init_mps)))

#Now implement possible initial conditions 
function all_up_spin(N, sites) 
    state = fill("Up", N)
    # display(state)
    M = MPS(sites, state)
    
    return M 
end 

function one_up_spin(N, sites, i)
    state = fill("Dn", N)
    state[i] = "Up"
    M = MPS(sites, state)
    # display(state)
    return M 
end

function neel_state(N, sites)
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    # display(state)
    M = MPS(sites, state)
    return M
end

function domain_state(N, sites, half = "left")
    if N % 2 == 0 
        state_left = fill("Up", Int64(N/2))
        state_right = fill("Dn", Int64(N/2))
    else 
        if half == "left"
            state_left = fill("Up", Int64(ceil(N/2)))
            state_right = fill("Dn", Int64(floor(N/2)))
        else
            state_left = fill("Up", Int64(floor(N/2)))
            state_right = fill("Dn", Int64(ceil(N/2)))
        end
    end
    state = vcat(state_left, state_right)
    M = MPS(sites, state)
    # display(state)
    return M 
end



function domain_state2(N, sites, dim)
    up = [0, 1]
    dn = [1, 0]
    init = [0,1]
    for i in 1:N 
        if i < Int64(N/2)
            init = kron(init, up)
        elseif i > Int64(N/2)
            init = kron(init, dn)
        end
    end
    M = MPS(init, sites; maxdim = dim)
    return M 
end


function get_total_elements(M)
    sum = 0
    site_dim = dim.(siteinds(M))
    link_dim = linkdims(M) 
    N = length(site_dim)
    for i = 1:N
        if i == 1
            sum += site_dim[i]*link_dim[i]
        elseif i == N 
            sum += site_dim[i]*link_dim[i - 1]
        else 
            sum += site_dim[i]*link_dim[i - 1]*link_dim[i]
        end
    end
    return sum 
end

function get_total_elements(site_inds, link_dim)
    sum = 0
    site_dim = dim.(site_inds)
    N = length(site_dim)
    for i = 1:N
        if i == 1
            sum += site_dim[i]*link_dim[i]
        elseif i == N 
            sum += site_dim[i]*link_dim[i - 1]
        else 
            sum += site_dim[i]*link_dim[i - 1]*link_dim[i]
        end
    end
    return sum 
end

function total_element_evolution(site_inds, bd_list)
    link, step = size(bd_list)
    evo = zeros(step)
    for i in 1:step 
        evo[i] = get_total_elements(site_inds, bd_list[:,i])
    end
    return evo 
end

function init_separable(sites, q_state)
    N = length(sites)
    M = MPS(N)
    link_size = Int64.(ones(N - 1))
    link_ind = create_linkinds(N, link_size)
    for i in 1:N
        if i == 1
            core = zeros(2, 1)
            core[q_state[i] + 1,1] = 1.0
            core_ten = ITensor(core, sites[i], link_ind[i])
        elseif i == N 
            core = zeros(2, 1)
            core[q_state[i] + 1,1] = 1.0
            core_ten = ITensor(core, sites[i], link_ind[i - 1])
        else 
            core = zeros(1, 2, 1)
            core[1,q_state[i] + 1,1] = 1.0
            core_ten = ITensor(core, sites[i], link_ind[i - 1], link_ind[i])
        end

        M[i] = core_ten 
    end
    return M 
end

function vec_separable(q_state)
    N = length(q_state)
    init = zeros(2)
    init[q_state[1] + 1] = 1.0
    for i in 2:N
        state_vec = zeros(2)
        state_vec[q_state[i] + 1] = 1.0
        init = kron(init, state_vec)
    end
    return init 
end

q_state = [0,0,0,0,1,1,1,1]
N = 8
sites = siteinds("Qubit", N)
magnet_site = 1
J = 1.0
U = 1.0 
hj_list = rand(N)
hp_list = rand(N)

H = graziani_H(N, J, U, hj_list, hp_list)

H_MPO = graziani_H_MPO(N, sites, J, U, hj_list, hp_list)

H_MPO_mat = matrix_form(H_MPO, sites)

# init = all_up_spin(N, sites)
# init = domain_state(N, sites)
init = init_separable(sites, q_state)
init_vec = ComplexF64.(vectorize_mps(init; order = "reverse"))
t0 = 0 
T = 10.0
steps = 200
svd_cutoff = 1E-2
# M_evolve, storage_arr, _, bd = tdvp2_constant(H_MPO, init, t0, T, steps, 0.0)
M_evolve, magnet_arr, _, bd = tdvp2_constant_magnet(H_MPO, init, t0, T, steps, svd_cutoff,magnet_site)
# M_evolve, storage_arr, _, bd = tdvp2_constant(H_MPO, init, t0, T, steps, 0.0)
# M_evolve, storage_arr = tdvp_constant(H_MPO, init, t0, T, steps, 2)

step_sol = zeros(ComplexF64, (2^N, steps + 1))
step_sol[:,1] = init_vec
h = (T - t0)/steps 
magnet_op = s_op(sz, Int64(N - magnet_site + 1), N)
magnet_sol = zeros(steps + 1)
magnet_sol[1] = init_vec'*magnet_op*init_vec

init_vec_copy = copy(init_vec)
for i in 1:steps 
    evolve_vec = exp(-im*H*h)*init_vec_copy
    magnet_sol[i + 1] = evolve_vec'*magnet_op*evolve_vec
    step_sol[:,i + 1] = evolve_vec 
    init_vec_copy .= evolve_vec 
end
# println("Error by final timestep: ", norm(abs2.(init_vec_copy) - abs2.(storage_arr[:,end]))^2)



p1 = plot(LinRange(t0,T, steps + 1), magnet_arr, label = "TDVP2", linewidth = 2, alpha = 0.5, title = "SVD Cutoff: $(svd_cutoff)",xlabel = "time", ylabel = "Magnetization Site $(magnet_site)", titlefontsize = 5, legend=:bottomleft, legend_background_color=RGBA(1,1,1,0.5), xguidefontsize = 5, yguidefontsize = 5, legendfontsize = 5)
plot!(LinRange(t0, T, steps + 1), magnet_sol, label = "Mat Exp", linestyle=:dot)

# p1 = plot(LinRange(t0,T, steps + 1), magnet_arr, label = "TDVP", linewidth = 2, alpha = 0.5, title = "SVD Cutoff: $(svd_cutoff)", ylabel = "Magnet Site $(Int64(N - magnet_site + 1))", titlefontsize = 5, yguidefontsize = 5)
# plot!(LinRange(t0, T, steps + 1), magnet_sol, label = "Mat Exp", linestyle=:dot)

bd_plot = plot(LinRange(t0, T, steps + 1), bd', label = ["1-2" "2-3" "3-4" "4-5" "5-6" "6-7" "7-8"], legendfontsize = 5, legend_background_color=RGBA(1,1,1,0.5), title = "Bond Dimension", titlefontsize = 4, legend =:topleft)
display(bd_plot)


total_entries_plot = plot(LinRange(t0, T, steps + 1), total_element_evolution(sites, bd), label = "# of entries", title = "Total Entries", titlefontsize = 5, legend =:bottomright, legend_background_color=RGBA(1,1,1,0.5), legendfontsize = 5)
plot!([0.0, 10.0], [2^N, 2^N], label = "Vector Storage")

total_plot = plot(p1, bd_plot, total_entries_plot, layout = (3, 1), dpi = 150)
display(total_plot)
# savefig("total_plot3.png")
exact_sol = exp(-im*H*(T - t0))*init_vec
last_magnet = real(exact_sol'*magnet_op*exact_sol)

# println("Last magnetization: $(last_magnet - magnet_arr[end])")
# savefig(total_plot, "total_plot2.png")
# display(total_entries_plot)
# steps_number = 14
# magnet_list = zeros(steps_number)
# n_list = [10*(2^i) for i in 1:steps_number]
# # _,_,test_mag,_ = tdvp2_constant_magnet(H_MPO, init, t0, T, )
# for i in 1:length(n_list) 
#     M_evolve, magnet_arr, _, bd = tdvp2_constant_magnet(H_MPO,init,t0, T, n_list[i], svd_cutoff, magnet_site)
#     magnet_list[i] = magnet_arr[end]
# end

# magnet_err = abs.(magnet_list .- last_magnet)
# h_list = (T - t0) ./ n_list
# magnet_plot = plot(h_list, magnet_err, xscale=:log10, yscale =:log10,xlabel = "h", ylabel = "magnetization error", dpi = 250, label = "MPS", legend=:topleft)
# plot!(h_list, h_list.^3, label = L"O(h^3)")
# display(magnet_plot)

#now run convergence study, compare the state vector to matrix exponentiation, pick some coarse grid and evolve with both matrix exponentiation and MPS methods. Get norm at each time step on this coarse grid to see what happens throughout evolution 
refinements = 3
coarse_grid_steps = 20
new_grids = [coarse_grid_steps*(2^i) for i in 0:refinements]
# new_grids = [200]

function error_analysis(coarse_steps, refined_points, svd_cutoff)
    step_sol = zeros(ComplexF64, (2^N, coarse_steps + 1))
    init_copy = copy(init_vec)
    # println("init_copy")
    # display(init_copy)
    step_sol[:,1] .= init_copy
    h_coarse = (T - t0)/coarse_steps
    for i in 1:coarse_steps 
        vec_evolve = exp(-im*H*h_coarse)*init_copy 
        step_sol[:,i + 1] .= vec_evolve 
        init_copy .= vec_evolve 
    end
    err_history = zeros(length(refined_points))
    for j in 1:length(refined_points)
        sum = 0.0
        _,tdvp_hist, _,_ = tdvp2_constant(H_MPO, init, t0,T,refined_points[j], svd_cutoff)
        println("tdvp_history: ")
        display(tdvp_hist)
        # sum += norm(step_sol[:,1] - tdvp_hist[:,1])
        factor = Int64(refined_points[j]/coarse_steps) 
        for k in 1:coarse_steps
            # println("Step $k")
            # display(step_sol[:,k])
            # display(tdvp_hist[:,factor*k]) 
            # println("TDVP Step: ", factor*k + 1)
            println("this is it: ", 1 + factor*k)
            sum += norm(step_sol[:,k] - tdvp_hist[:,1 + factor*(k-1)])
        end
        # println("tdvp_history")
        # display(tdvp_hist)
        sum = sum/(coarse_steps +1)
        err_history[j] = sum 
        display(tdvp_hist[:,17])
    end
    println("mat_exp history")
    display(step_sol[:,5])
    return err_history
end

function error_analysis2(coarse_steps, refined_points, svd_cutoff)

    # println("init_copy")
    # display(init_copy)
    err_history = zeros(length(refined_points))
    err_history_norm = zeros(length(refined_points))
    for j in 1:length(refined_points)
        init_copy = copy(init_vec)
        h = (T - t0)/refined_points[j]
        step_sol = zeros(ComplexF64, (2^N, refined_points[j] + 1))
        sum = 0.0
        sum_normal = 0.0
        step_sol[:,1] = init_copy
        for i in 1:refined_points[j]
            vec_evolve = exp(-im*H*h)*init_copy 
            step_sol[:,i + 1] .= vec_evolve 
            init_copy .= vec_evolve
        end 
        _,tdvp_hist, _,_ = tdvp2_constant(H_MPO, init, t0,T,refined_points[j], svd_cutoff, false)
        _,tdvp_hist_norm,_,_ = tdvp2_constant(H_MPO, init, t0,T,refined_points[j], svd_cutoff, true)
        println("tdvp_history: ")
        display(tdvp_hist)
        # sum += norm(step_sol[:,1] - tdvp_hist[:,1])
        factor = Int64(refined_points[j]/coarse_steps) 
        for k in 1:refined_points[j] + 1
            # println("Step $k")
            # display(step_sol[:,k])
            # display(tdvp_hist[:,factor*k]) 
            # println("TDVP Step: ", factor*k + 1)
            sum += norm(step_sol[:,k] - tdvp_hist[:,k])^2
            sum_normal += norm(step_sol[:,k] - tdvp_hist_norm[:,k])^2
        end
        # println("tdvp_history")
        # display(tdvp_hist)
        sum = sqrt((sum)/(refined_points[j] + 1))
        sum_normal = sqrt((sum_normal)/(refined_points[j] + 1))
        err_history[j] = sum
        err_history_norm[j] = sum_normal
        display(tdvp_hist)
        display(step_sol)
    end
    return err_history, err_history_norm
end

function magnet_heatmap(t0,T, steps, cutoff)
    q_state = [0,0,0,0,1,1,1,1]
    N = length(q_state)
    init = init_separable(sites, q_state)
    init_vec = ComplexF64.(vec_mps(init, order = "reverse"))
    magnet_storage_exp = zeros(N, steps + 1)
    magnet_storage_mps = zeros(N, steps + 1) 
    h = (T - t0)/steps
    for i in 1:N
        init_vec_copy = copy(init_vec)
        m_mat = s_op(sz, N -i + 1, N)
        magnet_storage_exp[i,1] = real(init_vec'*m_mat*init_vec)
        for j in 1:steps 
            evolve_vec = exp(-im*H*h)*init_vec_copy 
            magnet_storage_exp[i,j + 1] = real(evolve_vec'*m_mat*evolve_vec)
            init_vec_copy .= evolve_vec 
        end
        _,magnet_history,_,_ = tdvp2_constant_magnet(H_MPO, init, t0,T,steps, cutoff, i)
        magnet_storage_mps[i,:] = magnet_history
    end
    x = LinRange(t0,T,steps + 1)
    y = collect(1:N)
    h_plot_exp = heatmap(x, y, magnet_storage_exp, c=:bluesreds, ylabel = "Site Index", xlabel = "time", yflip = true; colorrange = (-1,1))
    h_plot_mps = heatmap(x, y, magnet_storage_mps, c=:bluesreds, yflip = true; colorrange = (-1,1))
    h_diff = heatmap(abs.(magnet_storage_exp - magnet_storage_mps), c=:bluesreds, yflip = true)
    # h_diff = heatmap(abs.(magnet_storage_exp - magnet_storage_mps), yflip = true, c=:bluesreds,clims = (1E-8, 1E-4), colorbar_ticks = log10.([1E-9, 1E-8, 1E-7, 1E-6, 1E-5, 1E-4]))
    # display(magnet_storage_exp - magnet_storage_mps)
    # difference = abs.(x, y, magnet_storage_exp - magnet_storage_mps)
    return h_plot_exp, h_plot_mps, h_diff
end


# h_grids = (T - t0)./new_grids
# err1, err1norm = error_analysis2(coarse_grid_steps, new_grids, 0.0)
# err2, err2norm = error_analysis2(coarse_grid_steps, new_grids, 1E-10)
# err3, err3norm = error_analysis2(coarse_grid_steps, new_grids, 1E-9)
# err4, err4norm = error_analysis2(coarse_grid_steps, new_grids, 1E-4)
# err_plot1 = plot(h_grids, [err1 err1norm], xlabel = "h", label = ["Error Non-normalize" "Error Normalize"], xscale =:log10, yscale =:log10, legend=:topleft, legend_background_color=RGBA(1,1,1,0.5), title = "SVD Cutoff: 0.0", titlefontsize = 5)
# plot!(h_grids, h_grids.^3, label = L"O(h^3)", legendfontsize = 5)
# err_plot2 = plot(h_grids, [err2 err2norm], label = ["Error Non-normalize" "Error Normalize"], xscale =:log10, yscale =:log10, legend=:topleft, legend_background_color=RGBA(1,1,1,0.5), title = "SVD Cutoff: 1E-10", titlefontsize = 5)
# plot!(h_grids, h_grids.^3, label = L"O(h^3)", legendfontsize = 5)
# err_plot3 = plot(h_grids, [err3 err3norm], label = ["Error Non-normalize" "Error Normalize"], xscale =:log10, yscale =:log10, legend=:topleft, legend_background_color=RGBA(1,1,1,0.5), title = "SVD Cutoff: 1E-9", titlefontsize = 5)
# plot!(h_grids, h_grids.^3, label = L"O(h^3)", legendfontsize = 5)
# err_plot4 = plot(h_grids, [err4 err4norm], label = "Evolution Error", xscale =:log10, yscale =:log10, legend=:topleft, legend_background_color=RGBA(1,1,1,0.5), title = "SVD Cutoff: 1E-8", titlefontsize = 5)
# plot!(h_grids, h_grids.^3, label = L"O(h^3)", legendfontsize = 5)

# total_err = plot(err_plot1, err_plot2, err_plot3, err_plot4, layout = (2,2), dpi = 250)
# display(total_err)
# display(err_plot1)
# savefig("total_err_2_normal.png")
# display(err1)
N = 8
U = 0.0
J = 1.0
sites = siteinds("Qubit", N)
hp_list = 0*ones(N)
hj_list = 0*rand(N)
H = graziani_H(N, J, U, hj_list, hp_list)

H_MPO = graziani_H_MPO(N, sites, J, U, hj_list, hp_list)


h1, h2, h3 = magnet_heatmap(0.0,10.0,200,1E-2)
total_h = plot(h1, h2,h3, layout = (3,1))
