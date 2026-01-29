using ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles, Plots, BenchmarkTools
using TensorMethods


plot_pulse = false
plot_tdvp = true 
plot_bug = true

N = 10
nlevels = fill(2, N)
# sites = siteinds("Qubit", N)
sites = qudit_siteinds(N, nlevels)
freq01_all = [5.18, 5.12, 5.06, 5.0, 4.94, 4.88, 4.82, 4.76, 4.7, 4.74].*2pi # [GHz] 0-1 frequency
self_kerr = zeros(N)
zz = zeros(N, N)
J = 5E-3*2pi
Jkl = zeros(N,N)
for i = 2:N 
    Jkl[i - 1, i] = J 
end
t0 = 0.0
T = 40.0 				# [ns]  Pulse duration
splines = 6
steps = 2620*4

freq01 = reverse(freq01_all)[1:N]
favg = sum(freq01)/N 
rot_freq = ones(N).*favg 
pcof = vec(readdlm("../spline_parms/params_10_coupled.dat"))

carrier_frequency_list = Vector{Vector{Float64}}(undef, N)
carrier_frequency_list[1] = [-0.17999999999999972, -0.21999999999999975].*2pi
carrier_frequency_list[2] = [-0.17999999999999972, -0.21999999999999975, -0.16000000000000014].*2pi
carrier_frequency_list[3] = [-0.21999999999999975, -0.16000000000000014, -0.09999999999999964].*2pi
carrier_frequency_list[4] = [-0.16000000000000014, -0.09999999999999964, -0.040000000000000036].*2pi
carrier_frequency_list[5] = [-0.09999999999999964, -0.040000000000000036, 0.020000000000000462].*2pi
carrier_frequency_list[6] = [-0.040000000000000036, 0.020000000000000462, 0.08000000000000007].*2pi
carrier_frequency_list[7] = [0.020000000000000462, 0.08000000000000007, 0.13999999999999968].*2pi
carrier_frequency_list[8] = [0.08000000000000007, 0.13999999999999968, 0.20000000000000018].*2pi 
carrier_frequency_list[9] = [0.13999999999999968, 0.20000000000000018, 0.2599999999999998].*2pi 
carrier_frequency_list[10] = [0.20000000000000018, 0.2599999999999998].*2pi

bc_params = bcparams(T, splines, carrier_frequency_list, pcof)


q_state = fill(0, N)
init_MPS = init_separable(sites, q_state)
init_vec = vectorize_mps(init_MPS)
init_ten = reshape(init_vec, fill(2, N)...)
init_core, init_factors = tucker(init_ten; cutoff = 0.0)
H_s = drift_MPO(N, sites, freq01, rot_freq, self_kerr, zz, Jkl)
H_s_ops = H_sys_rot(N, nlevels, freq01, rot_freq, self_kerr, Jkl, zz)

ans_mps, link_history, _, _, _ = tdvp2(H_s, init_MPS, t0, T, Int64(steps/2), bc_params; cutoff = 1E-5, strang = true)
tdvp_ans = vectorize_mps(ans_mps; order = "reverse")


#Now compare to data from quandary 
rho_re = readdlm("../Rho_data/rho_Re.iinit0000_10_coupled.dat")
rho_im = readdlm("../Rho_data/rho_Im.iinit0000_10_coupled.dat")
quandary_ans = rho_re[end,2:end] + im*rho_im[end,2:end]
println("Error between quandary and tdvp: ", norm(tdvp_ans - quandary_ans))
println("bond dim at end: ", linkdims(ans_mps))

pts = 10
pts_range = LinRange(-35, -3, pts)
cutoff_list = 2.0 .^pts_range
err_list_tdvp = zeros(pts)
err_list_bug = zeros(pts)
bd_list_tdvp = []
bd_list_bug = []
ans_tdvp = []
ans_bug = []
time_tdvp_cpu = zeros(pts)
time_bug_cpu = zeros(pts)
q_state = fill(0, N)
init_MPS = init_separable(sites, q_state)
H_s = drift_MPO(N, sites, freq01, rot_freq, self_kerr, zz, Jkl)

for i in 1:pts 
    
    println("Cutoff: ", cutoff_list[i]^2)
    if plot_tdvp == true 
        time_tdvp_cpu[i] = @belapsed begin 
        ans_mps, link_history, _, _, _ = tdvp2(H_s, init_MPS, t0, T, steps, bc_params; cutoff = cutoff_list[i]^2, strang = false)
        end 
        push!(ans_tdvp, ans_mps)
        tdvp_sol = vectorize_mps(ans_mps; order = "reverse")
        err_list_tdvp[i] = norm(tdvp_sol - quandary_ans)
        push!(bd_list_tdvp, link_history)
    end

    if plot_bug == true 
        time_bug_cpu[i] = @belapsed begin 
        ans_core, ans_factors, _,_, bd = bug_integrator_mat_ra(H_s_ops, bc_params, init_core, init_factors, t0, T, steps; cutoff = cutoff_list[i]^2)
        end
        push!(ans_bug, [ans_core, ans_factors])
        bug_array = Multi_TTM_recursive(ans_core, ans_factors)
        bug_ans = vec(permutedims(bug_array, reverse(1:ndims(bug_array))))
        err_list_bug[i] = norm(bug_ans - quandary_ans)
        push!(bd_list_bug, bd)
    end

    
end

entries_list_tdvp = [count_MPS(bd_list_tdvp[i][end,:], nlevels) for i in 1:pts]
entries_list_bug = [count_tucker(bd_list_bug[i][end,:], nlevels) for i in 1:pts]

if plot_tdvp == true & plot_bug == true 
    cutoff_plot = plot(cutoff_list, [err_list_tdvp, err_list_bug], label = ["TDVP2 Err" "BUG Err"], xscale =:log10, yscale=:log10, legend =:topleft, dpi = 250, yticks = [10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1], xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1], xlabel = "SVD Truncation Parameter(ε)")

    entries_plot = plot(cutoff_list, [entries_list_tdvp, entries_list_bug], label = ["MPS Storage" "Tucker Tensor Storage"], dpi = 250, xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1], xlabel = "SVD Truncation Parameter(ε)", xscale =:log10)

    plot!(cutoff_list, fill(2^N, pts), label = "Vector storage", linestyle =:dash)
    
elseif plot_tdvp == false 
    cutoff_plot_bug = plot(cutoff_list, err_list_bug, label = "BUG Err", xscale =:log10, yscale=:log10, legend =:topleft, dpi = 250, yticks = [10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1], xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1], xlabel = "SVD Truncation Parameter(ε)")

    entries_plot = plot(cutoff_list, entries_list_tdvp, label = "Tensor-train storage", dpi = 250, xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1], xlabel = "SVD Truncation Parameter(ε)", xscale =:log10)

elseif plot_bug == false 
    cutoff_plot_tdvp = plot(cutoff_list, err_list_tdvp, label = "TDVP2 Err", xscale =:log10, yscale=:log10, legend =:topleft, dpi = 250, yticks = [10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1], xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1], xlabel = "SVD Truncation Parameter(ε)")

    # plot!(cutoff_list, 10^3*cutoff_list.^1.06, label = L"O(\epsilon^{1.06})")

    entries_plot = plot(cutoff_list, entries_list_bug, label = "Tucker-tensor storage", dpi = 250, xticks = [10^-16, 10^-15, 10^-14, 10^-13, 10^-12, 10^-11, 10^-10, 10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1], xlabel = "SVD Truncation Parameter(ε)", xscale =:log10)

    plot!(cutoff_list, fill(2^N, pts), label = "Vector storage", linestyle =:dash)

end

if plot_pulse == true
    pulse = 1
    time_range = LinRange(0, T, steps)
    p_eval = zeros(length(time_range))
    q_eval = zeros(length(time_range))
    for j = 1:steps 
        p_eval[j] = bcarrier2(time_range[j], bc_params, 2*(pulse - 1))*(500/pi)
        q_eval[j] = bcarrier2(time_range[j], bc_params, 2*(pulse - 1) + 1)*(500/pi)
    end
    pulse_plot = plot(time_range, [p_eval, q_eval], labels = ["p(t)" "q(t)"], xlabel = "time(ns)", ylabel = "MHz")
end