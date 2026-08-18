using ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles, Plots, BenchmarkTools, NPZ, ProgressMeter, LaTeXStrings, Statistics
using JLD2
using TensorMethods

vcatnothing(xs...) = vcat(filter(!isnothing, xs)...)
Statistics.mean(::Nothing) = nothing

function QEC_pulse(freq_01_group_1, freq_01_group_2; favg::Union{Float64, Nothing} = nothing, dipole_off_strength::Real = 0.001, cutoff::Real = 0.0)

    ###########################################################################
    # Problem setup
    ###########################################################################

    N_groups = count(!isnothing, (freq_01_group_1, freq_01_group_2))
    
    N = 5
    N_total = N * N_groups

    data_qubit_idx = [1, 3, 5]
    ancilla_qubit_idx = [2, 4]

    ###########################################################################
    # Build initial states and QEC circuits
    ###########################################################################

    initial_groups,
    qec_circuits,
    total_QEC_circuit,
    total_initial_state,
    initial_groups_MPS,
    initial_state_MPS,
    QEC_groups_MPS,
    total_QEC_MPS = build_qec_groups(
        N,
        N_groups;
        data_qubit_idx = data_qubit_idx,
        ancilla_qubit_idx = ancilla_qubit_idx
    )

    ###########################################################################
    # Qubit frequencies
    ###########################################################################

    # freq_01_group_1 = [5.18, 5.12, 5.06, 4.94, 5.02] .* 2π
    # freq_01_group_2 = [5.38, 5.32, 5.26, 5.14, 5.22] .* 2π
    # freq_01_group_2 = [5.37, 5.30, 5.26, 5.12, 5.19] .* 2π

    
    freq_01_total = vcatnothing(freq_01_group_1, freq_01_group_2)

    # Each QEC group is simulated in its own rotating frame.
    # Could alternatively average only over qubits participating in each gate.
    # rot_freq_group_1 = isnothing(freq_01_group_1) ? nothing : fill(mean(freq_01_group_1), N)
    # rot_freq_group_2 = isnothing(freq_01_group_2) ? nothing : fill(mean(freq_01_group_2), N)
    # rot_freq_total = vcatnothing(rot_freq_group_1, rot_freq_group_2)
    if isnothing(favg)
        rot_freq_total = fill(mean(freq_01_total), N_total)
    else 
        rot_freq_total = fill(favg, N_total)
    end
    println(rot_freq_total)
    # uncomment to average over entire system
    # println("rot freq total 1")
    # display(rot_freq_total)
    # println("diff 1")
    # display(freq_01_total .- rot_freq_total)
    # rot_freq_total = fill(mean(vcat(freq_01_group_1, freq_01_group_2)), N_total)
    # println("rot freq total 2")
    # display(rot_freq_total)
    # println("diff 2")
    # display(freq_01_total .- rot_freq_total)
    ###########################################
    # Coupling graph
    ###########################################################################

    Jkl_coupling_strength = 5e-3 * 2π
    Jkl = zeros(N_total, N_total)

    coupled_indices = make_couplings_QEC(
        N_groups;
        data_qubit_idx = data_qubit_idx,
        ancilla_qubit_idx = ancilla_qubit_idx
    )

    t_vals_base = [(0.0, 800.0), (0.0, 800.0), (0.0, 800.0), (0.0, 800.0)]
    t_vals = repeat(t_vals_base, N_groups)

    # Optional inter-group couplings.
    if N_groups > 1
        for i in 1:N
            push!(coupled_indices, (i, i + N))
            push!(t_vals, (NaN, NaN))
        end
    end

    for pair in coupled_indices
        Jkl[pair...] = Jkl_coupling_strength
    end

    ###########################################################################
    # Bond dictionary (time-dependent dipole couplings)
    ###########################################################################

    
    dipole_off_strength = fill(dipole_off_strength, length(coupled_indices))

    bond_dict = Dict{Any, Any}()

    for pair_idx in eachindex(coupled_indices)
        bond_key = (
            off_strength = dipole_off_strength[pair_idx],
            t_range = t_vals[pair_idx]
        )
        bond_dict[coupled_indices[pair_idx]] = bond_key
    end

    ###########################################################################
    # Drift Hamiltonian
    ###########################################################################

    sites_total = siteinds(initial_state_MPS)

    h_params = Drift_Hamiltonian(
        N_total,
        sites_total,
        freq_01_total,
        rot_freq_total;
        dipole = Jkl,
        bond_dict = bond_dict
    )

    ###########################################################################
    # Import optimized control pulses
    ###########################################################################
    if !isnothing(freq_01_group_1)
        pulse_data_group_1 = load_object(
            "examples/spline_params/QEC_group_1_controls.jld2"
        )

        pulse_real_group_1 = pulse_data_group_1["p_optim"] .* 2π
        pulse_imag_group_1 = pulse_data_group_1["q_optim"] .* 2π
    else 
        pulse_data_group_1 = nothing 
    end

    if !isnothing(freq_01_group_2)
        pulse_data_group_2 = load_object(
            "examples/spline_params/QEC_group_2_controls.jld2"
        )

        pulse_real_group_2 = pulse_data_group_2["p_optim"] .* 2π
        pulse_imag_group_2 = pulse_data_group_2["q_optim"] .* 2π
    else 
        pulse_data_group_2 = nothing 
    end

    ###########################################################################
    # Downsample pulses if desired
    ###########################################################################

    pulse_resolution = 1

    if !isnothing(pulse_data_group_1)
        pulse_real_downsample_group_1 =
            pulse_real_group_1[:, 1:pulse_resolution:end]

        pulse_imag_downsample_group_1 =
            pulse_imag_group_1[:, 1:pulse_resolution:end]
    else
        pulse_real_downsample_group_1 = nothing 
        pulse_imag_downsample_group_2 = nothing
    end

    if !isnothing(pulse_data_group_2)
        pulse_real_downsample_group_2 =
            pulse_real_group_2[:, 1:pulse_resolution:end]

        pulse_imag_downsample_group_2 =
            pulse_imag_group_2[:, 1:pulse_resolution:end]
    else
        pulse_real_downsample_group_2 = nothing 
        pulse_imag_downsample_group_2 = nothing
    end


    ###########################################################################
    # Combine controls for both groups
    ###########################################################################
    pulse_real_total = vcatnothing(
        pulse_real_downsample_group_1,
        pulse_real_downsample_group_2
    )
    
    pulse_imag_total = vcatnothing(
        pulse_imag_downsample_group_1,
        pulse_imag_downsample_group_2
    )

    ###########################################################################
    # TDVP parameters
    ###########################################################################

    maxdim_vec = [2, 4, 4, 2, 1]

    maxdim_total = repeat(maxdim_vec, N_groups - 1)
    maxdim_total = vcat(maxdim_total, [2, 4, 4, 2])

    run_tdvp = true

    t0 = 0.0
    T = 800.0

    if !isnothing(pulse_data_group_1)
        steps = size(pulse_real_downsample_group_1, 2) - 1
    elseif !isnothing(pulse_data_group_2)
        steps = size(pulse_real_downsample_group_2, 2) - 1
    else
        throw("no pulse data")
    end
    ###########################################################################
    # TDVP evolution
    ###########################################################################
    
    if run_tdvp

        t = @elapsed begin

            ans_mps,
            link_history,
            mps_history,
            _,
            _,
            trunc_history = tdvp2_changing_dipole(
                h_params,
                initial_state_MPS,
                t0,
                T,
                steps,
                pulse_real_total,
                pulse_imag_total;
                cutoff = cutoff,
                maxdim = nothing,
                strang = true,
                save_history = true,
                normalize = false,
                verbose = false
            )

        end

    end

    ###########################################################################
    # Fidelity of the complete state
    ###########################################################################

    fidelity = abs2(inner(ans_mps, total_QEC_MPS))

    ###########################################################################
    # Fidelity of each QEC block
    ###########################################################################

    group_fidelities = zeros(N_groups)

    linkdims_mps = linkdims(ans_mps)

    # Only compute subsystem fidelities if the QEC groups remain disentangled.
    if all(linkdims_mps[N:N:end] .== 1)

        for group in 1:N_groups

            group_MPS = MPS_subset(
                ans_mps,
                (group - 1) * N + 1,
                group * N
            )

            group_MPS = remove_dim1_links(group_MPS)

            group_fidelities[group] =
                abs2(inner(conj(group_MPS), QEC_groups_MPS[group]))

        end

    end

    ###########################################################################
    # Return results
    ###########################################################################

    return (
        ans = ans_mps,
        fidelity = fidelity,
        group_fidelities = group_fidelities,
        tdvp_time = t,
    )

end


freq_01_group_1 = [5.18, 5.12, 5.06, 4.94, 5.02] .* 2π
freq_01_group_2 = [5.38, 5.32, 5.26, 5.14, 5.22] .* 2π
favg = mean(vcat(freq_01_group_1, freq_01_group_2))
output = QEC_pulse(freq_01_group_1, nothing, favg = favg, dipole_off_strength = 0.0, cutoff = 0.0)
# output_1 = QEC_pulse(freq_01_group_1, freq_01_group_2, dipole_off_strength = 0.001, cutoff = 1E-10)
# output_2 = QEC_pulse(freq_01_group_1, freq_01_group_2, dipole_off_strength = 0.001, cutoff = 1E-5)

output_list = []
cutoff_list = 10 .^ LinRange(-15, -1, 15)
dipole_list = [0.0001, 0.001, 0.01, 0.1, 1.0]

# for i in eachindex(cutoff_list)
#     for j in eachindex(dipole_list)
#         output = QEC_pulse(freq_01_group_1, freq_01_group_2, dipole_off_strength = dipole_list[j], cutoff = cutoff_list[i])
#         push!(output_list, output)
#     end
# end

# save_object("examples/outout_list_3.jld2", output_list)