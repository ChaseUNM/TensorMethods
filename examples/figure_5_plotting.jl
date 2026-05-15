# using necessary plotting and data I/O packages
using Plots 
using JLD2
using LaTeXStrings

# Parameters for the runs and plotting
p = 5 
steps = 500
t0 = 0.0
T = 5.0
# create time grid used in the simulations
time_range = LinRange(t0, T, steps + 1)
# range of system sizes (number of qubits)
N_min = 3
N_max = 5
N_list = collect(N_min:N_max)

# load precomputed runtimes and bond-dimension data for g = 0.5
g = 0.5
time_tdvp_g_half = load_object("time_tdvp_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max).jld2")
time_bug_g_half = load_object("time_bug_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max)_single_site_truncation_mo.jld2")
bd_tdvp_g_half = load_object("bd_tdvp_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max).jld2")
bd_bug_g_half = load_object("bd_bug_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max)_single_site_truncation_mo.jld2")

# load precomputed runtimes and bond-dimension data for g = 0.0
g = 0.0
time_tdvp_g_0 = load_object("time_tdvp_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max).jld2")
time_bug_g_0 = load_object("time_bug_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max)_single_site_truncation_mo.jld2")
bd_tdvp_g_0 = load_object("bd_tdvp_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max).jld2")
bd_bug_g_0 = load_object("bd_bug_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max)_single_site_truncation_mo.jld2")

# Fit power-law scaling of runtime ~ C * N^p by linearizing with logs:
# log(time) ≈ beta[1]*log(N) + beta[2]
X = [log.(N_list) ones(length(N_list))]
# for the "bug" data length may differ; construct design matrix accordingly
X_bug = [log.(N_list) ones(length(time_bug_g_half))]
beta_tdvp_g_half = X \ log.(time_tdvp_g_half)
beta_bug_g_half = X_bug \ log.(time_bug_g_half)
beta_tdvp_g_0 = X \ log.(time_tdvp_g_0)
beta_bug_g_0 = X \ log.(time_bug_g_0)

# Extract exponents and prefactors from fit results
p_tdvp_g_half = beta_tdvp_g_half[1]
C_tdvp_g_half = exp(beta_tdvp_g_half[2])
p_bug_g_half = beta_bug_g_half[1]
C_bug_g_half = exp(beta_bug_g_half[2])

p_tdvp_g_0 = beta_tdvp_g_0[1]
C_tdvp_g_0 = exp(beta_tdvp_g_0[2])
p_bug_g_0 = beta_bug_g_0[1]
C_bug_g_0 = exp(beta_bug_g_0[2])

# pick colors from the default palette for consistent plotting
cols = palette(:default)

# colors for g = 0.0
c_tdvp_g0 = cols[1]
c_bug_g0  = cols[2]

# colors for g = 0.5
c_tdvp_g05 = cols[3]
c_bug_g05  = cols[4]

# Main runtime plot: TDVP2 and MPS-BUG for g = 0.0 (plotted together)
t_plot = plot(
    N_list,
    [time_tdvp_g_0 time_bug_g_0],
    xlabel = "# of qubits (N)",
    labels = ["TDVP2 | g = 0.0" "MPS-BUG | g = 0.0"],
    title = "Runtime [s] of TDVP2 and MPS-BUG",
    dpi = 250,
    legend = :topleft,
    color = [c_tdvp_g0 c_bug_g0], ylims = (0,70),
    legendfontsize = 10, 
    xticks = [10,20,30,40,50,60,70,80,90,100]
)

# add fitted scaling lines for g = 0.0 (matching colors)
plot!(N_list, C_tdvp_g_0*(N_list).^p_tdvp_g_0,
    label = latexstring("O\\left(N^{", round(p_tdvp_g_0, digits=2), "}\\right)"),
    linestyle = :dash,
    alpha = 0.5,
    color = c_tdvp_g0
)

plot!(N_list, C_bug_g_0*(N_list).^p_bug_g_0,
    label = latexstring("O\\left(N^{", round(p_bug_g_0, digits=2), "}\\right)"),
    linestyle = :dash,
    alpha = 0.5,
    color = c_bug_g0
)

# add g = 0.5 data curves to the same plot
plot!(N_list,
    time_tdvp_g_half,
    label = "TDVP2 | g = 0.5",
    color = [c_tdvp_g05 c_bug_g05]
)
plot!(N_list, 
    time_bug_g_half, label = "MPS-BUG | g = 0.5", 
    color = [c_bug_g05]
)
# add fitted scaling lines for g = 0.5 (matching colors)
plot!(N_list, C_tdvp_g_half*(N_list).^p_tdvp_g_half,
    label = latexstring("O\\left(N^{", round(p_tdvp_g_half, digits=2), "}\\right)"),
    linestyle = :dash,
    alpha = 0.5,
    color = c_tdvp_g05
)

plot!(N_list, C_bug_g_half*(N_list).^p_bug_g_half,
    label = latexstring("O\\left(N^{", round(p_bug_g_half, digits=2), "}\\right)"),
    linestyle = :dash,
    alpha = 0.5,
    color = c_bug_g05
)

# Compute maximum bond dimension over the time evolution for each N
max_bond_tdvp_g_half = [maximum(bd_tdvp_g_half[i]) for i in 1:length(bd_tdvp_g_half)]
max_bond_tdvp_g_0 = [maximum(bd_tdvp_g_0[i]) for i in 1:length(bd_tdvp_g_0)]
max_bond_bug_g_half = [maximum(bd_bug_g_half[i]) for i in 1:length(bd_bug_g_half)]
max_bond_bug_g_0 = [maximum(bd_bug_g_0[i]) for i in 1:length(bd_bug_g_0)]

# Plot maximum bond dimension for g = 0.5
bd_plot = plot(N_list, [max_bond_tdvp_g_half, max_bond_bug_g_half], 
    labels = ["TDVP2 | g = 0.5" "MPS-BUG | g = 0.5"], 
    xlabel = "# of qubits",
    ylabel = "Max bond dimension",
    xticks = [10,20,30,40,50,60,70,80,90,100],
    colors = [cols[1] cols[2]],
    dpi = 250,
    legend=:topleft,
    legendfontsize = 10, 
    yticks = [1, 3, 5, 7, 9, 11])
# add g = 0.0 bond-dimension curves with different linestyles
plot!(N_list, [max_bond_tdvp_g_0, max_bond_bug_g_0], 
    labels = ["TDVP2 | g = 0.0" "MPS-BUG | g = 0.0"], 
    colors = [cols[1] cols[2]], linestyle = [:dash :dot])