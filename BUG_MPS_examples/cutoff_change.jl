using ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles, Plots, CPUTime
using TensorMethods

#See how changing the cutoff value changes the error with a fixed time-step 
plot_tdvp = true 
plot_bug_tucker = true 
plot_bug_mps = true


N = 10
sites = siteinds("Qubit", N)
N_levels = fill(2, N)
q_state = Int64.(fill(1, N))
q_state[1] = 0
# q_state[N] = 0
init_MPS = init_separable(sites, q_state)
# init_MPS = random_mps(sites, linkdims = 16)
q_state_arr = q_state .+ 1
init_ten[q_state_arr...] = 1.0 + 0.0*im
init_core, init_factors = tucker(init_ten; cutoff = 0.0)
init_vec = vectorize_mps(init_MPS; order = "reverse")


t0 = 0.0
T = 10.0

J = 1.0
g = 0.5
H_mpo = xxx_mpo_scaled(N, sites, J, g)
H_mat = xxx_scaled(N, J, g)
H_ops_xxx = ops_xxx_scaled(N, J, g)


pts = 2
pts_range = LinRange(-30, -6, pts)
diff = pts_range[1] - pts_range[2]
cutoff_list = 2.0 .^pts_range

steps = 2^10
steps_list = collect(9:12)
cutoff_err_tdvp_list = []
cutoff_err_bug_tucker_list = []
cutoff_err_bug_mps_list = []

bd_list_tdvp2 = []
bd_list_bug_tucker = []
bd_list_bug_mps = []
true_sol = exp(-im*(T - t0)*H_mat)*init_vec

for i in steps_list
    init_MPS_copy = deepcopy(init_MPS)
    steps = 2^i
    
    cutoff_err_tdvp = zeros(pts)
    cutoff_err_bug_tucker = zeros(pts)
    cutoff_err_bug_mps = zeros(pts)
    bd_tdvp = []
    bd_bug_tucker = []
    bd_bug_mps = []
    
    for i in 1:pts
        init_MPS_copy = deepcopy(init_MPS)
        println("Cutoff: $(cutoff_list[i]^2)")

        #Three different simulation methods: TDVP2, BUG for Tucker-tensors, BUG for MPS

        if plot_tdvp == true 
            ans_mps, bd_history_tdvp, magnet_history, energy_history, trunc_err = tdvp2_constant(H_mpo, init_MPS_copy, t0, T, Int64(steps/2); cutoff = cutoff_list[i]^2)
            tdvp_vec = vectorize_mps(ans_mps; order = "reverse")
            cutoff_err_tdvp[i] = norm(tdvp_vec - true_sol)
            push!(bd_tdvp, bd_history_tdvp)
        end 

        if plot_bug_tucker == true 
            ans_core, ans_factors, state, nrg, bd_history_bug_tucker = bug_integrator_mat_ra(H_ops_xxx, init_core, init_factors, t0, T, steps; cutoff = cutoff_list[i]^2)
            bug_array = Multi_TTM_recursive(ans_core, ans_factors)
            bug_ans = vec(permutedims(bug_array, reverse(1:ndims(bug_array))))
            cutoff_err_bug_tucker[i] = norm(bug_ans - true_sol)
            push!(bd_bug_tucker, bd_history_bug_tucker)
        end

        if plot_bug_mps == true 
            ans_bug_mps, bd_history_bug_mps, _,_ = mps_bug_constant(H_mpo, init_MPS_copy, t0, T, steps; cutoff = cutoff_list[i]^2)'
            bug_mps_vec = vectorize_mps(ans_MPS; order = "reverse")
            cutoff_err_bug_mps[i] = norm(bug_mps_vec - true_sol)
            push!(bd_bug_mps, bd_history_bug_mps)
        end

    end

    if plot_tdvp == true 
        push!(cutoff_err_tdvp_list, cutoff_err_tdvp)
        push!(bd_list_tdvp2, bd_tdvp)
    end

    if plot_bug_tucker == true 
        push!(cutoff_err_bug_tucker_list, cutoff_err_bug_tucker)
        push!(bd_list_bug_tucker, bd_bug_tucker)
    end

    if plot_bug_mps == true 
        push!(cutoff_err_bug_mps_list, cutoff_err_bug_mps)
        push!(bd_list_bug_mps, bd_bug_mps)
    end

end

if plot_tdvp == true 
    entries_list_tdvp = [[count_MPS(bd_list_tdvp2[i][j][end,:], N_levels) for j in 1:pts] for i in 1:length(steps_list)]
    entries_plot_tdvp = plot(cutoff_list, entries_list_tdvp, labels = [L"2^8 steps" L"2^9 steps" L"2^{10} steps" L"2^{11} steps" L"2^{12} steps"], xlabel = "SVD Truncation Error (ε)", xscale =:log10, dpi = 250)
    cutoff_err_plot_tdvp = plot(cutoff_list, cutoff_err_tdvp_list, labels = [L"2^8 steps" L"2^9 steps" L"2^{10} steps" L"2^{11} steps" L"2^{12} steps"], xlabel = "SVD Truncation Error (ε)")

end

if plot_bug_tucker == true
    entries_list_bug_tucker = [[count_tucker(bd_list_bug_tucker[i][j][end,:], N_levels) for j in 1:pts] for i in 1:length(steps_list)]
    entries_plot_bug_tucker = plot(cutoff_list, entries_list_bug_tucker, labels = [L"2^8 steps" L"2^9 steps" L"2^{10} steps" L"2^{11} steps" L"2^{12} steps"], xlabel = "SVD Truncation Error (ε)", xscale =:log10, dpi = 250)
    cutoff_err_plot_bug_tucker = plot(cutoff_list, cutoff_err_bug_tucker_list, [L"2^8 steps" L"2^9 steps" L"2^{10} steps" L"2^{11} steps" L"2^{12} steps"], xlabel = "SVD Truncation Error (ε)")
end

if plot_bug_mps == true 
    entries_list_bug_mps = [[count_MPS(bd_list_bug_mps[i][j][end,:], N_levels) for j in 1:pts] for i in 1:length(steps_list)]
    entries_plot_bug_mps = plot(cutoff_list, entries_list_bug_mps, labels = [L"2^8 steps" L"2^9 steps" L"2^{10} steps" L"2^{11} steps" L"2^{12} steps"], xlabel = "SVD Truncation Error (ε)", xscale =:log10, dpi = 250)
    cutoff_err_plot_bug_mps = plot(cutoff_list, cutoff_err_bug_mps_list, [L"2^8 steps" L"2^9 steps" L"2^{10} steps" L"2^{11} steps" L"2^{12} steps"], xlabel = "SVD Truncation Error (ε)")
end



# cutoff_err_plot = plot(cutoff_list, cutoff_err_list, labels = "Truncation", xlabel = "SVD Truncation Error (ε)", xscale=:log10, yscale=:log10, legend=:topleft, legendfontsize = 7, dpi = 250)
# plot!(cutoff_list, 10^4*cutoff_list, label = "O(ε)", linestyle =:dash, xscale=:log10, yscale=:log10, legend=:topleft, legendfontsize = 7, dpi = 250)