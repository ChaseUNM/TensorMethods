# Increase ITensor warning threshold for tensor order
using ITensors, ITensorMPS 
using Plots 
using DelimitedFiles, BenchmarkTools
using TensorMethods


ITensors.set_warn_order!(100)

################################################################################
# Generate data and plots for blue lines in Figure 10 (arxiv 2603.13990)
################################################################################


function run_pulses(nqubits_min::Int, nqubits_max::Int; run_TDVP::Bool = true, run_TDVP2::Bool = true, run_BUG_MPS::Bool = false, run_BUG_Tucker::Bool = false, strang_splitting::Bool = true, plot_spline::Bool = false)

    # create NamedTuple to store results 
    results = (;)

    # -------------------------------------------------------------------
    # Loop over each number of qubits 
    # -------------------------------------------------------------------
    N = length(nqubits_min:nqubits_max)
    time_tdvp = zeros(N)
    time_tdvp2 = zeros(N)
    time_bug_mps = zeros(N)
    time_bug_tucker = zeros(N)
    infidelity_tdvp = zeros(N)
    infidelity_tdvp2 = zeros(N)
    infidelity_bug_mps = zeros(N)
    infidelity_bug_tucker = zeros(N)
    linkdims_tdvp2 = Vector{Vector{<:Real}}(undef, N)
    linkdims_bug_mps = Vector{Vector{<:Real}}(undef, N)
    linkdims_bug_tucker = Vector{Vector{<:Real}}(undef, N)

    MPS_final_tdvp = Vector{MPS}(undef, N)
    MPS_final_tdvp2 = Vector{MPS}(undef, N)
    MPS_final_bug = Vector{MPS}(undef, N)
    Tucker_final_bug = Vector{Any}(undef, N)
    subsystem_fidelity_tdvp = Vector{Vector{Float64}}(undef, N)
    subsystem_fidelity_tdvp2 = Vector{Vector{Float64}}(undef, N)
    subsystem_fidelity_bug = Vector{Vector{Float64}}(undef, N)

    for nqubits in nqubits_min:nqubits_max
        println("# of qubits: $nqubits")

        # Local Hilbert space dimension at each site
        nlevels = fill(2, nqubits)

        # Create qudit site indices
        sites = qudit_siteinds(nqubits, nlevels)

        # -------------------------------------------------------------------
        # Set up qubit transition frequencies
        # -------------------------------------------------------------------

        # Bare 0-1 transition frequencies [GHz]
        freq01_all = zeros(nqubits)

        for i in 0:nqubits - 1
            # Determine which block of 4 qubits this site belongs to
            block = fld(i, 4)

            # Position within the current block
            k = i % 4

            if block % 2 == 0
                # "Up" block: descending frequency ladder
                f = 5.18 - 0.06 * k
            else
                # "Down" block: shifted to reduce frequency collisions
                f = 5.18 - 0.06 * (3 - k) + 0.03
            end 

            freq01_all[i + 1] = f
        end

        # Convert frequencies to angular frequency units
        freq01_all = freq01_all * 2pi

        # -------------------------------------------------------------------
        # Time interval and rotating frame setup
        # -------------------------------------------------------------------

        # Rotating-frame frequency: average over all qubit frequencies
        favg = sum(freq01_all) / length(freq01_all)

        # Initial and final time [ns]
        t0 = 0.0
        T = 40.0

        # -------------------------------------------------------------------
        # Hamiltonian coupling parameters
        # -------------------------------------------------------------------

        # Nearest-neighbor coupling strength [GHz] converted to angular frequency
        Jkl_coupling_strength = 0.0 * 2pi

        # Coupling matrix for chain topology
        Jkl = zeros(nqubits, nqubits)

        for i in 1:nqubits
            for j in i+1:nqubits
                if j == i + 1
                    Jkl[i, j] = Jkl_coupling_strength
                end
            end
        end 

        # Self-Kerr nonlinearities (set to zero here)
        self_kerr = zeros(nqubits) * 2pi 

        # ZZ interaction matrix (set to zero here)
        zz = zeros(nqubits, nqubits) * 2pi

        # -------------------------------------------------------------------
        # Carrier frequencies and rotating frame frequencies
        # -------------------------------------------------------------------

        # Each qubit gets a single carrier frequency relative to rotating frame
        carrier_freq = [[freq01_all[iq] - favg] for iq in 1:nqubits] 

        # Rotating-frame frequencies for all qubits
        rotfreq = favg * ones(nqubits)

        # -------------------------------------------------------------------
        # Time discretization and pulse parameterization
        # -------------------------------------------------------------------

        # Time step size [ns]
        dT = 0.01

        # Number of time steps
        steps = Int64((T - t0) / dT)

        # Number of spline segments for pulse parameterization
        splines = 6

        # -------------------------------------------------------------------
        # Import optimized spline pulse parameters
        # -------------------------------------------------------------------

        # File containing spline coefficients for this nqubits configuration
        datafile = joinpath(@__DIR__, "spline_params", "params_$(nqubits).dat")

        if !isfile(datafile)
            throw("No control parameters for $nqubits qubits.")
        end
        # Load spline coefficients
        pcof = vec(readdlm(datafile))

        # Build pulse parameter object
        bc_params = bcparams(T, splines, carrier_freq, pcof)

        # -------------------------------------------------------------------
        # Initial state and target state
        # -------------------------------------------------------------------

        # SVD truncation cutoff
        cutoff_val = 1E-5

        # Initial computational basis state |0,0,...,0⟩
        q_state = fill(0, nqubits)

        # Target state: equal superposition state
        if nqubits < 5
            target_vec = fill(1 / sqrt(2)^nqubits, 2^nqubits)
        end
        
        # Same target state represented as an MPS
        target_mps = equal_separable(sites)

        # -------------------------------------------------------------------
        # Build initial condition and Hamiltonian for MPS-based methods
        # -------------------------------------------------------------------

        if run_TDVP == true || run_BUG_MPS == true
            # Initial separable MPS state
            init_mps = init_separable(sites, q_state)

            # Convert initial MPS to full vector form
            if nqubits < 5
                init_vec = vectorize_mps(init_mps)
            end

            # Construct drift Hamiltonian MPO
            H_d = drift_MPO(nqubits, sites, freq01_all, rotfreq, self_kerr = nothing, zz = nothing, dipole = nothing)
        end 

        # -------------------------------------------------------------------
        # Run BUG-Tucker method (if enabled)
        # -------------------------------------------------------------------

        if run_BUG_Tucker == true 
            # Construct separable Tucker tensor initial condition
            init_core, init_factors = tucker_separable(q_state)

            # Reconstruct dense tensor from Tucker form
            init_array = Multi_TTM_recursive(init_core, init_factors)

            # Convert dense tensor to vector form
            init_vec = vec(permutedims(init_array, reverse(1:ndims(init_array))))

            # Construct Hamiltonian in operator/tensor form
            H_s_ops = H_sys_rot(nqubits, nlevels, freq01_all, rotfreq, self_kerr, Jkl, zz)

            t = @elapsed begin 
                # Run BUG-Tucker time evolution
                ans_core, ans_factors, state_bug_tucker, energy_bug_tucker, link_bug_tucker =
                    bug_integrator_mat_ra(
                        H_s_ops,
                        bc_params,
                        init_core,
                        init_factors,
                        t0,
                        T,
                        steps;
                        cutoff = cutoff_val^2,
                        state = false,
                        energy = false
                    )
            end

            # Reconstruct final dense tensor
            bug_array = Multi_TTM_recursive(ans_core, ans_factors)

            # Convert final tensor to vector form
            bug_tucker_vec = vec(permutedims(bug_array, reverse(1:ndims(bug_array))))

            # Compute fidelity with target state
            bug_tucker_fidelity = abs2(bug_tucker_vec' * target_vec)
            println("BUG Tucker time: $t seconds")
            println("BUG Tucker infidelity: $(1 - bug_tucker_fidelity)")
            time_bug_tucker[nqubits - nqubits_min + 1] = t 
            infidelity_bug_tucker[nqubits - nqubits_min + 1] = 1 - bug_tucker_fidelity
            ans_bug_tucker = (core = ans_core, 
                                factors = ans_factors)
            Tucker_final_bug[nqubits - nqubits_min + 1] = ans_bug_tucker
        end

        # ----------------------------------------------------------------------
        # Run TDVP method (if enabled)
        # ----------------------------------------------------------------------
        if run_TDVP == true 
            t = @elapsed begin 
                # Run TDVP2 time evolution
                ans_tdvp, magnet_tdvp, energy_tdvp =
                    TensorMethods.tdvp(
                        H_d,
                        init_mps,
                        t0,
                        T,
                        strang_splitting ? Int(steps/2) : steps,
                        bc_params;
                        strang = strang_splitting,
                        magnet = false,
                        energy = false,
                        verbose = false
                    )
            end
            # calculate infidelity of each qubit 
            N_qubits = length(ans_tdvp)
            fidelity_subsystems = zeros(N_qubits)
            for qubit in 1:N_qubits
                site = siteinds_tensor(ans_tdvp[qubit])
                link = linkinds_tensor(ans_tdvp[qubit])
                array_vals = [1/sqrt(2), 1/sqrt(2)]
                T = ITensor(array_vals, site, link)
                fidelity_subsystems[qubit] = abs2.(inner(conj(ans_tdvp[qubit]), T))
            end
            # Compute fidelity with target state
            tdvp_fidelity = abs2(inner(conj(ans_tdvp), target_mps))

            println("TDVP time: $t seconds")
            println("TDVP infidelity: $(1 - tdvp_fidelity)")
            time_tdvp[nqubits - nqubits_min + 1] = t 
            infidelity_tdvp[nqubits - nqubits_min + 1] = 1 - tdvp_fidelity
            MPS_final_tdvp[nqubits - nqubits_min + 1] = ans_tdvp
            subsystem_fidelity_tdvp[nqubits - nqubits_min + 1] = fidelity_subsystems
        end

        # -------------------------------------------------------------------
        # Run TDVP2 method (if enabled)
        # -------------------------------------------------------------------
        if run_TDVP2 == true 
            t = @elapsed begin 
                # Run TDVP2 time evolution
                ans_tdvp, link_tdvp, magnet_tdvp, energy_tdvp, cutoff_error =
                    tdvp2(
                        H_d,
                        init_mps,
                        t0,
                        T,
                        strang_splitting ? Int(steps/2) : steps,
                        bc_params;
                        cutoff = 1E-30,
                        maxdim = nothing,
                        strang = strang_splitting,
                        magnet = false,
                        energy = false,
                        verbose = true
                    )
            end
            N_qubits = length(ans_tdvp)
            fidelity_subsystems = zeros(N_qubits)
            for qubit in 1:N_qubits
                site = siteinds_tensor(ans_tdvp[qubit])
                link = linkinds_tensor(ans_tdvp[qubit])
                array_vals = [1/sqrt(2), 1/sqrt(2)]
                T = ITensor(array_vals, site, link)
                fidelity_subsystems[qubit] = abs2.(inner(conj(ans_tdvp[qubit]), T))
            end
            # Compute fidelity with target state
            tdvp_fidelity = abs2(inner(conj(ans_tdvp), target_mps))

            println("TDVP time: $t seconds")
            println("TDVP infidelity: $(1 - tdvp_fidelity)")
            time_tdvp[nqubits - nqubits_min + 1] = t 
            infidelity_tdvp[nqubits - nqubits_min + 1] = 1 - tdvp_fidelity
            linkdims_tdvp[nqubits - nqubits_min + 1] = linkdims(ans_tdvp)
            
            MPS_final_tdvp2[nqubits - nqubits_min + 1] = ans_tdvp
            subsystem_fidelity_tdvp2[nqubits - nqubits_min + 1] = fidelity_subsystems

        end

        # -------------------------------------------------------------------
        # Run BUG-MPS method (if enabled)
        # -------------------------------------------------------------------

        if run_BUG_MPS == true 
            t = @elapsed begin 
                # Run BUG-MPS time evolution
                ans_bug_mps, link_bug_mps, magnet_bug_mps, energy_bug_mps, trunc_err =
                    mps_bug(
                        H_d,
                        bc_params,
                        init_mps,
                        t0,
                        T,
                        steps;
                        cutoff = cutoff_val^2,
                        magnet = false,
                        energy = false
                    )
            end
            N_qubits = length(ans_bug_mps)
            fidelity_subsystems = zeros(N_qubits)
            for qubit in 1:N_qubits
                site = siteinds_tensor(ans_bug_mps[qubit])
                link = linkinds_tensor(ans_bug_mps[qubit])
                array_vals = [1/sqrt(2), 1/sqrt(2)]
                T = ITensor(array_vals, site, link)
                fidelity_subsystems[qubit] = abs2.(inner(conj(ans_bug_mps[qubit]), T))
            end
            # Compute fidelity with target state
            bug_mps_fidelity = abs2(inner(conj(ans_bug_mps), target_mps))
            println("BUG MPS time: $t seconds")
            println("BUG MPS infidelity: $(1 - bug_mps_fidelity)")
            time_bug_mps[nqubits - nqubits_min + 1] = t 
            infidelity_bug_mps[nqubits - nqubits_min + 1] = 1 - bug_mps_fidelity
            linkdims_bug_mps[nqubits - nqubits_min + 1] = linkdims(ans_bug_mps)
            MPS_final_bug[nqubits - nqubits_min + 1] = ans_bug_mps
            
            subsystem_fidelity_bug[nqubits - nqubits_min + 1] = fidelity_subsystems


        end
    end
    # -------------------------------------------------------------------
    # Optional: plot control pulse for a chosen qubit
    # -------------------------------------------------------------------

    if plot_spline == true 
        # Choose qubit whose control pulse is plotted
        qubit = 1

        # Time grid for evaluating pulse
        time_range = LinRange(t0, T, steps)

        # Arrays for in-phase and quadrature pulse components
        p_eval = zeros(length(time_range))
        q_eval = zeros(length(time_range))

        # Evaluate pulse over time
        for j = 1:steps 
            p_eval[j] = bcarrier2(time_range[j], bc_params, 2 * (qubit - 1)) * (500 / pi)
            q_eval[j] = bcarrier2(time_range[j], bc_params, 2 * (qubit - 1) + 1) * (500 / pi)
        end

        # Plot pulse components
        pulse_plot = plot(
            time_range,
            [p_eval, q_eval],
            labels = ["p(t)" "q(t)"],
            xlabel = "time(ns)",
            ylabel = "MHz"
        )
    end

    if run_TDVP 
        results = merge(results, (
        tdvp = (
            ans = MPS_final_tdvp,
            time = time_tdvp,
            infidelity = infidelity_tdvp,
            subsystem_fidelity = subsystem_fidelity_tdvp
        ),
    ))
    end

    if run_TDVP2
        results = merge(results, (
        tdvp2 = (
            ans = MPS_final_tdvp2, 
            time = time_tdvp2,
            infidelity = infidelity_tdvp2,
            linkdims = linkdims_tdvp2,
            subsystem_fidelity = subsystem_fidelity_tdvp2
        ),
    ))
    end 

    if run_BUG_MPS
        results = merge(results, (
        bug_mps = (
            ans = MPS_final_bug, 
            time = time_bug_mps,
            infidelity = infidelity_bug_mps,
            linkdims = linkdims_bug_mps,
            subsystem_fidelity = subsystem_fidelity_bug
        ),
    ))
    end 

    if run_BUG_Tucker
        results = merge(results, (
        bug_tucker = (
            ans = Tucker_final_bug, 
            time = time_bug_tucker,
            infidelity = infidelity_bug_tucker,
            linkdims = linkdims_bug_tucker,
        ),
    ))
    end

    return results
end

# -------------------------------------------------------------------
# Flags to determine which algorithm(s) are run for this test problem
# -------------------------------------------------------------------
TDVP = true
TDVP2 = false
BUG_MPS = false
BUG_Tucker = false
plot_spline = false
strang_splitting = true

nqubits_min = 28
nqubits_max = 28 

results = run_pulses(nqubits_min, nqubits_max, run_TDVP = TDVP, run_TDVP2 = TDVP2, run_BUG_MPS = BUG_MPS, run_BUG_Tucker = BUG_Tucker, strang_splitting = strang_splitting)

load_data = false
if load_data 
    if isfile("examples/Timings.dat")
        data = readdlm("examples/Timings.dat")
    end


    time_tdvp = results.tdvp2.time 
    infidelity_tdvp = results.tdvp2.infidelity 

    quandary_times = data[2:end,1]
    quandary_infidelity = data[2:end,2]
    nqubits_quandary = length(quandary_infidelity)
    nqubits_range = collect(nqubits_min: nqubits_max)
    nqubits_quandary_range = collect(nqubits_min: nqubits_min + nqubits_quandary-1)
    timing_plot = plot(nqubits_range, 
        time_tdvp, 
        label = "TDVP2", 
        xlabel = "# of qubits (N)",
        linestyle = :solid,
        ylabel = "Wall-clock time (s)",
        title = "Runtime of Quandary and TDVP2 (seconds)",
        titlefontsize = 15,
        xguidefontsize = 14, 
        yguidefontsize = 14, 
        xtickfontsize = 12, 
        ytickfontsize = 12,
        # xticks = [6,9,12,15,18,21],
        yticks = [2^-1, 2^1, 2^3, 2^5,2^7, 2^9],
        yscale=:log2, 
        legend=:topleft, 
        legendfontsize = 14, 
        linewidth = 3, 
        dpi = 250)
    plot!(nqubits_quandary_range, 
        quandary_times, 
        label = "Quandary", 
        linewidth = 3,
        linestyle=:dash)
    infidelity_plot = plot(nqubits_range, 
        infidelity_tdvp, 
        label = "TDVP2", 
        xlabel = "# of qubits (N)",
        linestyle =:solid,
        ylabel = "Infidelity",
        title = "State-to-State Infidelity of Quandary and TDVP2",
        titlefontsize = 15,
        xguidefontsize = 14, 
        yguidefontsize = 14, 
        xtickfontsize = 12, 
        ytickfontsize = 12,
        # xticks = [6,9,12,15,18,21], 
        legend=:topleft, 
        legendfontsize = 14,
        linewidth = 3, 
        yscale=:log10, 
        alpha = 0.8, dpi = 250)
    plot!(nqubits_quandary_range,
        quandary_infidelity, 
        label = "Quandary", 
        linewidth = 2, 
        alpha = 0.6, 
        linestyle =:dash)
    savefig(timing_plot, "timing_plot_time_dependent.png")
    savefig(infidelity_plot, "infidelity_plot_time_dependent.png")
end


# M_final = results.tdvp.ans[1]