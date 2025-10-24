using LinearAlgebra, DelimitedFiles, ProgressMeter, Plots, LaTeXStrings


include("../Tucker_Matrices.jl")
include("../hamiltonian(8-4).jl")

function tuple_index_natural(alpha, index_size_list)
    s_list = [1;cumprod(index_size_list[1:end - 1])]
    n_indices = length(index_size_list)
    index_list = zeros(n_indices)
    for i in 1:length(index_list)
        index_list[i] = 1 + floor(((alpha - 1)%(index_size_list[i]*s_list[i]))/s_list[i])
    end
    return Int64.(index_list) 
end

function tuple_index_reverse(alpha, index_size_list)
    s_list = [1;cumprod(index_size_list[1:end - 1])]
    n_indices = length(index_size_list)
    index_list = zeros(n_indices)
    # println("S_list:" ,s_list)
    for i in 1:length(index_list)
        # println(s_list[length(index_list) - i + 1])
        index_list[i] = 1 + floor(((alpha - 1)%(index_size_list[i]*s_list[length(index_list) - i + 1]))/s_list[length(index_list) - i + 1])
    end
    return Int64.(index_list) 
end

function linear_index_natural(index_list, index_size_list)
    n = length(index_list)
    alpha = 1
    s = 1
    for i in 1:length(index_list)
        if i != 1
            s*= index_size_list[i - 1]
        end
        alpha += s*(index_list[i] - 1)
    end
    return alpha 
end

function linear_index_reverse(index_list, index_size_list)
    n = length(index_list)
    alpha = 1
    s = 1
    for i in length(index_list):-1:1
        if i != n 
            s *= index_size_list[i + 1]
        end
        alpha += s*(index_list[i]-1)
    end
    return alpha 
end

# for i in 0:7
#     println("Init $i")
#     rho = readdlm("rho_Re.iinit00$(lpad(i, 2, '0')).dat")
#     println("Size of vector: ", size(rho[1,2:end]))
#     init_vec = rho[1,2:end]
#     init_loc = findall(x -> x != 0, init_vec)
#     println(init_loc)
#     println(rho[1,2:end])
#     if length(init_loc) == 1
#         println("Natural Index loc: ", tuple_index_natural(init_loc[1], [4, 4, 4]))
#         println("Reverse Index loc: ", tuple_index_reverse(init_loc[1], [4, 4, 4]))
#     end
# end

init_cond = 0
init_cond = lpad(init_cond, 2, '0')

steps, N = size(readdlm("./rho_case2/rho_Re.iinit0000.dat"))
state_history = zeros(ComplexF64, steps, N - 1)
rho_re = readdlm("./rho_case2/rho_Re.iinit00$(init_cond).dat")
rho_im = readdlm("./rho_case2/rho_Im.iinit00$(init_cond).dat")
time_range = rho_re[:,1]
steps = 100
bd_history = zeros(steps, 3)
cut = 1E-15
println("Cutoff: $cut")
@showprogress 1 "Adding vector" for i in 1:steps
    println("Step $i")
    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    state_history[i,:] = rho_re[i,2:end] + im*rho_im[i,2:end]
    #Get bond dimension of state
    state_reshape = reshape(state_history[i,:], 4, 4, 4)
    core, factors = tucker(state_reshape; cutoff = cut, verbose = true)
    # println([size(factors[j], 2) for j in 1:length(factors)])
    bd_history[i,:] = collect(size(core))
    # println("Norm of state: ", norm(state_history[i,2:end]))
    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
end

function get_bd(init_cond, cut)
    steps, N = size(readdlm("./rho_case2/rho_Re.iinit0000.dat"))
    bd_history = zeros(steps, 3)
    state_history = zeros(ComplexF64, steps, N - 1)
    rho_re = readdlm("./rho_case2/rho_Re.iinit00$(init_cond).dat")
    rho_im = readdlm("./rho_case2/rho_Im.iinit00$(init_cond).dat")
    @showprogress 1 "Adding vector" for i in 1:steps
        state_history[i,:] = rho_re[i,2:end] + im*rho_im[i,2:end]
        #Get bond dimension of state
        state_reshape = reshape(state_history[i,:], 4, 4, 4)
        core, factors = tucker(state_reshape; cutoff = cut, verbose = false)
        # println([size(factors[j], 2) for j in 1:length(factors)])
        bd_history[i,:] = collect(size(core))
        # println("Norm of state: ", norm(state_history[i,2:end]))
    end
    return bd_history 
end


function get_state_history(init_conds, init_cond)
    steps, N = size(readdlm("./rho_case2/rho_Re.iinit0000.dat"))
    state_history = zeros(ComplexF64, steps, N - 1)
    state_essential = zeros(ComplexF64, steps, length(init_conds))
    rho_re = readdlm("./rho_case2/rho_Re.iinit00$(init_cond).dat")
    rho_im = readdlm("./rho_case2/rho_Im.iinit00$(init_cond).dat")
    @showprogress "Adding Vector" for i = 1:steps 
        state_history[i,:] = rho_re[i,2:end] + im*rho_im[i,2:end]
    end
    for i = 1:length(init_conds)
        loc = linear_index_reverse(init_conditions[i], [4, 4, 4])
        # println("Loc: $loc")
        state_essential[:,i] = state_history[:,loc]
    end
    return state_essential
end

function plot_essential(init_conds)
    p = plot(layout = (4,2), dpi = 200)
    for i = 0:7 
        init_cond = lpad(i, 2, '0')
        loc_index = tuple_index_reverse(i + 1, [2,2,2])
        loc_index_adjust = loc_index .- 1
        state_essential = get_state_history(init_conds, init_cond)
        if i == 0
            plot!(p[i + 1], time_range, abs2.(state_essential), xlabel = "t", ylabel = "Population", labels = [L"|000\rangle" L"|001\rangle" L"|010\rangle" L"|011\rangle" L"|100\rangle" L"|101\rangle" L"|110\rangle" L"|111\rangle"], legend_background_color = RGBA(1,1,1,0.25), legendfontsize = 4, title = "|$(tuple(loc_index_adjust)...)>", titlefontsize= 4, legend =:top, legend_columns = length(init_conds))
        else
            plot!(p[i + 1], time_range, abs2.(state_essential), legend = false, title = "|$(loc_index_adjust)>", titlefontsize = 4)
        end
    end
    return p 
end


init_conditions = [[1,1,1], [1,1,2], [1,2,1], [1,2,2], [2,1,1], [2,1,2], [2,2,1], [2,2,2]]
function essential_plot(state_vector, init_conditions)
    steps, N = size(state_vector)
    state_vector_essential = zeros(ComplexF64, steps, length(init_conditions))
    p = plot(layout = (4, 2), dpi = 200)
    for j = 0:7
        
        for i in 1:length(init_conditions)
            loc = linear_index_reverse(init_conditions[i], [4, 4, 4])
            println("Loc: $loc")
            state_vector_essential[:,i] = state_vector[:,loc]
        end
        loc_index = tuple_index_reverse(j + 1, [2,2,2])
        loc_index_adjust = loc_index .- 1
        if j == 0
            plot!(p[j + 1], time_range, abs2.(state_vector_essential), xlabel = "t", ylabel = "Population", labels = [L"|000\rangle" L"|001\rangle" L"|010\rangle" L"|011\rangle" L"|100\rangle" L"|101\rangle" L"|110\rangle" L"|111\rangle"], legend_background_color = RGBA(1,1,1,0.25), legendfontsize = 4, title = "|$(tuple(loc_index_adjust)...)>", titlefontsize= 4)
        else
            plot!(p[j + 1], time_range, abs2.(state_vector_essential), legend = false, title = "|$(loc_index_adjust)>", titlefontsize = 4)
        end
    end
    return p
end

function rank_plot(init_conditions, cutoff)
    # bd_history = zeros(steps, length(init_conditions))
    p = plot(layout = (4, 2), dpi = 200)
    for i in 0:7 
        init_cond = lpad(i, 2, '0')
        loc_index = tuple_index_reverse(i + 1, [2,2,2])
        loc_index_adjust = loc_index .- 1
        bd = get_bd(init_cond, cutoff)
        if i == 0
            plot!(p[i + 1], time_range, bd, xlabel = "t", ylabel = "Bond Dimension", xguidefontsize = 4, yguidefontsize = 4, labels = ["Qubit 1" "Qubit 2" "Qubit 3"], title = "|$(tuple(loc_index_adjust)...)>", titlefontsize = 4,
            legend_background_color = RGBA(1,1,1,0.25), legendfontsize = 4)
        else
            plot!(p[i + 1], time_range, bd, legend = false, title = "|$(loc_index_adjust)>", titlefontsize = 4)
        end
    end
    return p
end



# bd_plot = plot(time_range[1:steps], bd_history)
# p1 = plot_essential(init_conditions)
# p2 = rank_plot(1,0.0)