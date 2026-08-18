using LinearAlgebra, ITensors, ITensorMPS, Plots, BenchmarkTools, Printf, JLD2
using TensorMethods

###################################################################################################
# Generate data for Figure 5 (arxiv 2603.13990) which can then be passed into a plotting script (figure_5_plotting.jl) to create the final figure. This script focuses on collecting bond dimension histories and runtimes for TDVP and BUG methods across a range of system sizes (3 to 100 qubits) for a specific Hamiltonian evolution.
###################################################################################################

# Get timings for n = 3:100 qubits

# Set up initial time, final time, and number of time steps
t0 = 0.0
T = 5.0
steps = 500

# Set the smallest and largest number of subsystems (qubits)
N_min = 3
N_max = 100

# Create a list of system sizes to loop over
N_list = collect(N_min:N_max)

# Store the total number of tensor entries used during evolution
total_entries_list = []

# Store truncation errors if desired (currently not being pushed to)
trunc_err_list = []

# Preallocate arrays to store runtime for TDVP and BUG methods
t_list_tdvp = zeros(length(N_list))
t_list_bug = zeros(length(N_list))

# Store bond dimension histories for each system size
bd_tdvp = []
bd_bug = []

# Set desired tolerance exponent so eps = 10^{-p}
p = 5
eps = 10.0 ^ -p 

# center = 1
g = 0.0
# For each qubit count n, evolve the system with tdvp2 and BUG
# and measure the execution time
for n in N_list 
    println("$n qubits")

    # Each qubit has 2 levels, so define local dimensions [2,2,...,2]
    N_levels = fill(2, n)

    # Create the ITensor site indices for n qubits
    sites = siteinds("Qubit", n)

    # Construct the Hamiltonian MPO
    # Here this appears to be a scaled XXX Hamiltonian with parameters J=1.0, g=0.0
    
    H = xxx_mpo_scaled(n, sites, 1.0, g)

    # Define the initial separable product state |0,0,...,0>
    q_state = Int64.(fill(0, n))
    init_MPS = init_separable(sites, q_state)

    # Make a copy so the original initial state is preserved
    init_MPS_copy = copy(init_MPS)

    # Time the TDVP2 evolution
    # Use a sufficiently small cutoff to avoid significant truncation
    # strang = true (indicates Strang splitting)
    t_list_tdvp[n - N_min + 1] = @elapsed begin
        _,_,_,_,_= tdvp2_constant(H, init_MPS_copy, t0, T, Int64(steps/2);cutoff = eps^2, verbose = false, strang = true)
    end

    # Run TDVP2 again to collect the bond dimension history
    _,bd_history_tdvp,_,_,_= tdvp2_constant(H, init_MPS_copy, t0, T, Int64(steps/2);cutoff = eps^2, verbose = false, strang = true)
    
    # Store the TDVP bond dimension history for this system size
    push!(bd_tdvp, bd_history_tdvp)

    # Time the BUG evolution
    t_list_bug[n - N_min + 1] = @elapsed begin
        _,_,_,_ = mps_bug_constant(H, init_MPS_copy, t0, T, steps; cutoff = eps^2, verbose = false)
    end

    # Run BUG again to collect the bond dimension history
    _,bd_history_bug,_,_ = mps_bug_constant(H, init_MPS_copy, t0, T, steps; cutoff = eps^2, verbose = false)

    # Store the BUG bond dimension history for this system size
    push!(bd_bug, bd_history_bug)

    # Run TDVP2 once more with a very small cutoff to estimate
    # bond dimensions / truncation error more accurately
    # _,bd_history,_,_,trunc_err= tdvp2_constant(H, init_MPS_copy, t0, T, Int64(steps/2);cutoff = 1E-15^2)

    # Count the total number of tensor entries used over the evolution history
    # entries_history = count_MPS_history(bd_history, N_levels)
    # push!(total_entries_list, entries_history)

    # Optionally store truncation error history
    # push!(trunc_err_list, trunc_err)
    
end

save_data = true

if save_data == true
    # Save BUG bond dimension histories to disk
    save_object("bd_bug_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max)_single_site_truncation_mo.jld2", bd_bug)

    # Save TDVP bond dimension histories to disk
    save_object("bd_tdvp_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max).jld2", bd_tdvp)

    # Save BUG runtimes to disk
    save_object("time_bug_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max)_single_site_truncation_mo.jld2", t_list_bug)

    # Save TDVP runtimes to disk
    save_object("time_tdvp_eps_minus_$(p)_$(steps)steps_g_$(g)_Nmin_$(N_min)_Nmax_$(N_max).jld2", t_list_tdvp)
end

