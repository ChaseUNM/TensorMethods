using ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles, Plots, BenchmarkTools, NPZ, ProgressMeter, LaTeXStrings
using JLD2
using TensorMethods

N_groups_min = 2
N_groups_max = 2


function make_couplings(n_groups; group_size=5)
    base = [(1,4), (2,4), (2,5), (3,5)]
    reduce(vcat, (map(t -> t .+ g*group_size, base) for g in 0:n_groups-1))
end

function group_couplings(N_groups_min::Int,N_groups_max::Int)
    
    N_groups_total = N_groups_max - N_groups_min + 1
    time_storage = zeros(N_groups_total)
    fidelity_storage = zeros(N_groups_total)
    fidelity_storage_root = zeros(N_groups_total)
    subsystem_fidelities = Vector{Vector{Float64}}(undef, N_groups_total)

    for N_groups in N_groups_min:N_groups_max
        println("# of groups: ", N_groups)
        N = 5
        # Local Hilbert space dimension for each subsystem
        N_total = N*N_groups
        nlevels = fill(2, N)
        nlevels_total = fill(2, N_total)
        sites = qudit_siteinds(N, nlevels)
        sites_total = qudit_siteinds(N_total, nlevels_total)


        # -------------------------
        # Time evolution parameters
        # -------------------------

        t0 = 0.0
        T = 800.0 			# [ns] Pulse duration

        # Bare 0-1 transition frequencies for each qubit [GHz], converted to angular frequency
        freq01 = ([5.18, 5.12, 5.06, 4.94, 5.02]) .* 2pi

        freq01_total = repeat(freq01, N_groups)

        # Self-Kerr nonlinearities (set to zero here)
        self_kerr = ones(N) .* 0.211 .* 2pi
        self_kerr_total = repeat(self_kerr, N_groups)

        # ZZ coupling matrix (set to zero here)
        zz = zeros(N, N)
        zz_total = cat(fill(zz, N_groups)...; dims = (1,2))
        # Nearest-neighbor coupling strength
        J = 5E-3 * 2pi

        # Coupling matrix Jkl, not nearest neighbor coupling
        Jkl = zeros(N_total, N_total)
        coupled_indices = make_couplings(N_groups)
        t_vals_base = [(0.0,200.0), (200.0,400.0), (400.0,600.0), (600.0,800.0)]
        t_vals = repeat(t_vals_base, N_groups)
        if N_groups > 1
            for i in 1:N_groups-1 
                push!(coupled_indices, (5 + (i - 1)*N, 6 + (i - 1)*N))
                push!(t_vals, (t0, T))
            end
        end
        for i in eachindex(coupled_indices)
            Jkl[coupled_indices[i]...] = J
        end
        strength = 0.001
        dipole_off_strength = fill(strength, length(coupled_indices))
        Jkl_total = cat(fill(Jkl, N_groups)...; dims = (1,2))
        bond_dict = Dict{Any, Any}()
        for pair_idx in eachindex(coupled_indices)
            bond_key = (off_strength = dipole_off_strength[pair_idx], t_range = t_vals[pair_idx])
            bond_dict[coupled_indices[pair_idx]] = bond_key
        end

        # t_starts = [0,200,400,600]
        # t_ends = [200,400,600,800]
        # coupling_speed = 5
        # coupling_fraction = 0.01
        # max_coupling = J
        # for i in eachindex(coupled_indices)
        #     J[i,j] = t -> coupling_time_dependence(
        #         max_coupling,
        #         coupling_fraction,
        #         t_start[i],
        #         t_end[i],
        #         coupling_speed,
        #         t,
        #         t_ends[end]
        #     )
        # end

        # J_t = [J[i,j](t) for i in 1:N, j in 1:N]



        # Number of time steps used in evolution


        # Reverse frequency ordering and truncate to N qubits
        # freq01 = reverse(freq01_all)[1:N]
        # Average frequency used for rotating frame
        favg = sum(freq01) / N 

        # Rotating frame frequencies (all set equal to the average)
        rot_freq = ones(N) .* favg 
        rot_freq_total = repeat(rot_freq, N_groups)
        # -------------------------
        # Load pulse parameters
        # -------------------------

        pulse_data = npzread("examples/spline_params/concatenated_pulse_data.npz")
        pulse_real = pulse_data["p_concat"] .* pi/500
        pulse_imag = pulse_data["q_concat"] .* pi/500


        # make pulse coarser to make simulation time shorter
        pulse_resolution = 1
        pulse_real_downsample = pulse_real[:,1:pulse_resolution:end]
        pulse_imag_downsample = pulse_imag[:,1:pulse_resolution:end]

        pulse_real_total = repeat(pulse_real_downsample, N_groups, 1)
        pulse_imag_total = repeat(pulse_imag_downsample, N_groups, 1)

        steps = size(pulse_real_total)[2] - 1
        initial_groups, qec_circuits, total_qec_circuit, total_initial_state, initial_groups_MPS, initial_state_MPS, QEC_groups_MPS, total_QEC_MPS = build_qec_groups(N, N_groups, starting_index = 8)
        display(initial_groups)
        display(qec_circuits)

        maxdim_vec = [2,4,4,2,4]
        maxdim_total = repeat(maxdim_vec, N_groups-1)
        maxdim_total = vcat(maxdim_total, [2,4,4,2])

        # sites_total = siteinds(init_MPS)
        # construct drift Hamiltonian
        sites_total = siteinds(initial_state_MPS)

        
        # coupling_inds = [(1,4), (2, 4), (2,5), (3,5)]
        c = 0.0
        t = 200

        # J_new = create_dipole_matrix(Jkl_total, t_vals, coupling_inds, c, t)

        # N_gates = length(t_vals)
        # H_list = Vector{MPO}(undef, N_gates)
        # for i in eachindex(H_list)
        #     t = 100*(2*i-1)
        #     Jkl_new = create_dipole_matrix(Jkl_total, t_vals, coupling_inds, c, t)
        #     H_list[i] = drift_MPO(N_total, sites_total, freq01_total, rot_freq_total, dipole = Jkl_new)
        # end

        h_params = Drift_Hamiltonian(N_total, sites_total, freq01_total, rot_freq_total, dipole = Jkl_total, bond_dict = bond_dict)
        display(bond_dict)
        run_tdvp = true

        if run_tdvp == true

            t = @elapsed begin ans_mps, link_history, mps_history, _, _ , trunc_history = tdvp2_changing_dipole(
                h_params,
                initial_state_MPS,
                t0,
                T,
                steps,
                pulse_real_total, 
                pulse_imag_total;
                cutoff = 0.0,
                maxdim = maxdim_total,
                strang = true, 
                save_history = true, 
                normalize = false,
                verbose = false
            )
            end
            

            println("Calculating State-to-State Fidelity")
            fidelity = abs2.(inner(ans_mps, total_QEC_MPS))
            fidelity_storage[N_groups - N_groups_min + 1] = fidelity
            fidelity_storage_root[N_groups - N_groups_min + 1] = fidelity^(1/N_groups)
            println(fidelity)
            println("######################################################")
            println("######################################################")
            group_fidelities = zeros(N_groups)
            time_storage[N_groups - N_groups_min + 1] = t
            linkdims_mps = linkdims(ans_mps)
            if all(linkdims_mps[N:N:end] .== 1)
                for group in 1:N_groups
                    group_MPS = MPS_subset(ans_mps, (group - 1)*N + 1, group*N) 
                    group_MPS = remove_dim1_links(group_MPS)
                    fidelity = abs2.(inner(conj(group_MPS), QEC_groups_MPS[group]))
                    group_fidelities[group] = fidelity
                end
                subsystem_fidelities[N_groups - N_groups_min + 1] = group_fidelities
            end
        end
    end
    return time_storage, fidelity_storage, fidelity_storage_root, subsystem_fidelities
end
N_groups_min = 2
N_groups_max = 2

t, f, f_root, fs = group_couplings(N_groups_min, N_groups_max)
