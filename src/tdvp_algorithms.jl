using ITensors
using LinearAlgebra, ProgressMeter


#######################################################################################################################################
##############         HELPER FUNCTIONS   for 3 qubit QEC    ##############
#######################################################################################################################################

function QEC_operator() 
    Ident_ops = [Matrix{Float64}(I, 2, 2) for i in 1:5]
    I_mat = reduce(kron, Ident_ops)
    ops_14 = [Matrix{Float64}(I, 2, 2) for i in 1:5]
    ops_24 = [Matrix{Float64}(I, 2, 2) for i in 1:5]
    ops_25 = [Matrix{Float64}(I, 2, 2) for i in 1:5]
    ops_35 = [Matrix{Float64}(I, 2, 2) for i in 1:5]
    ops_14[1] = [0 0; 0 1]
    ops_14[4] = [-1 1; 1 -1]
    ops_24[2] = [0 0; 0 1]
    ops_24[4] = [-1 1; 1 -1]
    ops_25[2] = [0 0; 0 1]
    ops_25[5] = [-1 1; 1 -1]
    ops_35[3] = [0 0; 0 1]
    ops_35[5] = [-1 1; 1 -1]

    CNOT_14 = I_mat + reduce(kron, ops_14)
    CNOT_24 = I_mat + reduce(kron, ops_24)
    CNOT_25 = I_mat + reduce(kron, ops_25)
    CNOT_35 = I_mat + reduce(kron, ops_35)

    return CNOT_35*CNOT_25*CNOT_24*CNOT_14
end
    
function QEC_initial_states()
    initial_states_mat = zeros(32, 8)
    vec_basis = [[1,0],[0,1]]
    for i in 1:8
        q_state = bitstring(i - 1)[end - 2:end]
        q_state = parse.(Int, collect(q_state))
        ops = [vec_basis[q_state[1] + 1], vec_basis[q_state[2] + 1], vec_basis[q_state[3] + 1], vec_basis[1], vec_basis[1]]
        initial_states_mat[:,i] = reduce(kron, ops)
    end
    return initial_states_mat
end

function remove_dim1_links(psi::MPS)
    psi_copy = deepcopy(psi)
    for i in 1:length(psi)
        T = psi[i]

        for ind in inds(T)
            if dim(ind) == 1 && hastags(ind, "Link")
                # Create a tensor that selects the only value of ind
                e = onehot(ind => 1)

                # Contract away the dimension-1 index
                T = T * e
            end
        end

        psi_copy[i] = T
    end
    return psi_copy
end

function mps_element(psi::MPS, state::AbstractVector{<:Integer})
    length(state) == length(psi) ||
        throw(ArgumentError("State vector has wrong length"))

    T = psi[1]

    for n in 1:length(psi)
        s = siteind(psi, n)

        # Basis vector |state[n]⟩
        e = ITensor(s)
        e[s => state[n]] = 1.0

        T *= e

        if n < length(psi)
            T *= psi[n+1]
        end
    end

    return scalar(T)
end


##################################################################################################
##################################################################################################
##################################################################################################

function create_initial(q_array::Vector{Int64})
    n = length(q_array)
    vec_basis = [[1,0],[0,1]]
    ops = [Vector{Float64}([1.0,0.0]) for i in 1:n]
    for i in 1:n 
        ops[i] = vec_basis[q_array[i] + 1]
    end
    return reduce(kron, ops)
end

function coupling_time_dependence(max_coupling::Float64, coupling_fraction::Float64, t_start::Float64, t_end::Float64, t_min::Float64, t_max::Float64, coupling_speed::Float64, t::Float64)
    # account for minimum and maximum allowed times
    
    if t_start == t_min
        
    # maximum coupling during gate
    elseif t_start + c <= t <= t_end - coupling_speed
        return max_coupling 
    # during ramp up to max coupling 
    elseif t_start - c <= t < t_start + coupling_speed
        f_val = (1/(2*coupling_speed))*(max_coupling - coupling_fraction*max_coupling)*(t - t_start + coupling_speed) + max_coupling*coupling_fraction
        return f_val 
    # during ramp down to minimum coupling
    elseif t_end - c < t <= t_end + coupling_speed
        f_val = (1/(2*coupling_speed))*(coupling_fraction*max_coupling - max_coupling)*(t - t_end + coupling_speed) + max_coupling
        return f_val
    # minimum coupling everywhere else
    elseif t_end == t_max
        return max_coupling
    else
        return max_coupling * coupling_fraction
    end
end

# function QEC_circuit(q_state::Vector{Int64})
#     q_state = BitVector(q_state)
#     ancilla_state = BitVector([0,0])
#     if q_state[1]
#         ancilla_state[1] = !ancilla_state[1]
#     end
#     if q_state[2]
#         ancilla_state[1] = !ancilla_state[1]
#     end
#     if q_state[2]
#         ancilla_state[2] = !ancilla_state[2]
#     end
#     if q_state[3]
#         ancilla_state[2] = !ancilla_state[2]
#     end
#     return Int.(vcat(q_state, ancilla_state)), Int.(vcat(q_state, [0,0]))
# end

# function build_qec_groups(N, N_groups; starting_index::Union{Int, Nothing}=nothing)
#     n_data = 3

#     initial_groups = Vector{Vector{Int}}(undef, N_groups)
#     qec_circuits = Vector{Any}(undef, N_groups)

#     initial_groups_MPS = Vector{MPS}(undef, N_groups)
#     QEC_groups_MPS = Vector{MPS}(undef, N_groups)
#     initial_state_MPS = MPS(N_groups * N)
#     QEC_MPS = MPS(N_groups*N)
#     first_group = isnothing(starting_index) ? 1 : starting_index

#     for local_g in 1:N_groups
#         global_g = first_group + local_g - 1

#         sites = qudit_siteinds(
#             N,
#             fill(2, N),
#             tag_set = collect(N * (local_g - 1) + 1: N * local_g)
#         )

#         initial_state = fill(0, N - 2)

#         # Cycle through all 2^n_data basis states
#         state_num = (global_g - 1) % (2^n_data)
#         bits = digits(state_num, base=2, pad=n_data)
#         initial_state[1:n_data] .= reverse(bits)

#         qec_circuit, initial_state = QEC_circuit(initial_state)

#         initial_groups[local_g] = initial_state
#         initial_groups_MPS[local_g] = init_separable(sites, initial_state)
#         QEC_groups_MPS[local_g] = init_separable(sites, qec_circuit)

#         for n in 1:N
#             initial_state_MPS[(local_g - 1) * N + n] = initial_groups_MPS[local_g][n]
#             QEC_MPS[(local_g - 1)*N + n] = QEC_groups_MPS[local_g][n]
#         end

#         qec_circuits[local_g] = qec_circuit
#     end

#     total_QEC_circuit = vcat(qec_circuits...)
#     total_initial_state = vcat(initial_groups...)

#     return initial_groups,
#            qec_circuits,
#            total_QEC_circuit,
#            total_initial_state,
#            initial_groups_MPS,
#            initial_state_MPS,
#            QEC_groups_MPS,
#            QEC_MPS
# end

function QEC_circuit(q_state::Vector{Int64}; data_qubit_idx::Union{Vector{Int}, Nothing} = nothing, ancilla_qubit_idx::Union{Vector{Int}, Nothing} = nothing)
    if isnothing(data_qubit_idx)
        data_qubit_idx = [1,2,3]
    end
    if isnothing(ancilla_qubit_idx)
        ancilla_qubit_idx = [4,5]
    end
    q_state = BitVector(q_state)
    ancilla_state = BitVector([0,0])
    N = length(q_state) + length(ancilla_state)
    if q_state[1]
        ancilla_state[1] = !ancilla_state[1]
    end
    if q_state[2]
        ancilla_state[1] = !ancilla_state[1]
    end
    if q_state[2]
        ancilla_state[2] = !ancilla_state[2]
    end
    if q_state[3]
        ancilla_state[2] = !ancilla_state[2]
    end
    QEC_state = zeros(Int, N)
    initial_state = zeros(Int, N)
    for i in eachindex(q_state) 
        QEC_state[data_qubit_idx[i]] = q_state[i]
        initial_state[data_qubit_idx[i]] = q_state[i]
    end
    QEC_state[ancilla_qubit_idx[1]] = ancilla_state[1]
    QEC_state[ancilla_qubit_idx[2]] = ancilla_state[2] 
    initial_state[ancilla_qubit_idx[1]] = 0 
    initial_state[ancilla_qubit_idx[2]] = 0
    return QEC_state, initial_state
end

function build_qec_groups(N::Int, N_groups::Int; data_qubit_idx::Union{Vector{Int}, Nothing} = nothing, ancilla_qubit_idx::Union{Vector{Int}, Nothing} = nothing, starting_index::Union{Int, Nothing}=nothing)
    n_data = 3

    initial_groups = Vector{Vector{Int}}(undef, N_groups)
    qec_circuits = Vector{Any}(undef, N_groups)

    initial_groups_MPS = Vector{MPS}(undef, N_groups)
    QEC_groups_MPS = Vector{MPS}(undef, N_groups)
    initial_state_MPS = MPS(N_groups * N)
    QEC_MPS = MPS(N_groups*N)
    first_group = isnothing(starting_index) ? 1 : starting_index

    for local_g in 1:N_groups
        global_g = first_group + local_g - 1

        sites = qudit_siteinds(
            N,
            fill(2, N),
            tag_set = collect(N * (local_g - 1) + 1: N * local_g)
        )

        initial_state = fill(0, N - 2)
        # Cycle through all 2^n_data basis states
        state_num = (global_g - 1) % (2^n_data)
        bits = digits(state_num, base=2, pad=n_data)
        initial_state[1:n_data] .= reverse(bits)

        qec_circuit, initial_state = QEC_circuit(initial_state, data_qubit_idx = data_qubit_idx, ancilla_qubit_idx = ancilla_qubit_idx)

        initial_groups[local_g] = initial_state
        initial_groups_MPS[local_g] = init_separable(sites, initial_state)
        QEC_groups_MPS[local_g] = init_separable(sites, qec_circuit)

        for n in 1:N
            initial_state_MPS[(local_g - 1) * N + n] = initial_groups_MPS[local_g][n]
            QEC_MPS[(local_g - 1)*N + n] = QEC_groups_MPS[local_g][n]
        end

        qec_circuits[local_g] = qec_circuit
    end

    total_QEC_circuit = vcat(qec_circuits...)
    total_initial_state = vcat(initial_groups...)

    return initial_groups,
           qec_circuits,
           total_QEC_circuit,
           total_initial_state,
           initial_groups_MPS,
           initial_state_MPS,
           QEC_groups_MPS,
           QEC_MPS
end



contains(x, interval::Tuple{<:Real,<:Real}) =
    interval[1] <= x <= interval[2]

# Exponential solver using full matrix exponential for verification.
# H: Hamiltonian matrix, init_vec: initial state vector, N: number of sites,
# t0, T: start/end times, steps: number of time steps.
function exp_solver(H, init_vec, N, t0, T, steps)
    h = (T - t0)/steps 
    sol_op = exp(-im*H*h)                      # time evolution operator for a single step
    magnet_history = zeros(steps + 1, N)       # store local magnetizations
    energy_history = zeros(steps + 1)          # store energies
    for j = 1:N
        m_mat = s_op([1 0; 0 -1], j, N)        # local sigma_z operator on site j
        magnet_history[1,j] = real(init_vec'*m_mat*init_vec)
    end
    energy_history[1] = real(init_vec'*H*init_vec)
    @showprogress 1 "Exponential solver" for i = 1:steps 
        init_vec = sol_op*init_vec              # apply exponential propagator
        for j = 1:N 
            m_mat = s_op([1 0; 0 -1], j, N)
            magnet_history[i + 1,j] = real(init_vec'*m_mat*init_vec)
        end
        energy_history[i + 1] = real(init_vec'*H*init_vec)
    end
    return init_vec, magnet_history, energy_history 
end

# Count total number of tensor entries in an MPS by summing sizes of cores.
function count_MPS(M::MPS)
    return sum(prod.(dims.(M)))
end

# Count total number of entries in an MPS given bond dimensions and local levels.
# bd: bond-dimension vector, N_levels: physical dimensions at each site.
function count_MPS(bd::Vector, N_levels::Vector{Int64})
    entries = 0
    N = length(N_levels)
    for i in 1:N 
        if i == 1
            entries += N_levels[i]*bd[i]        # left boundary core: (phys x right)
        elseif i == N 
            entries += N_levels[i]*bd[i - 1]   # right boundary core: (phys x left)
        else
            entries += N_levels[i]*bd[i]*bd[i - 1]  # bulk core: phys x left x right
        end
    end
    return entries 
end

# Count total number of entries for a history of bond-dimension arrays.
# bd: matrix of bond dims per step, N_levels: physical dims.
function count_MPS_history(bd::Array, N_levels::Vector{Int64})
    steps = size(bd, 1)
    entries_list = zeros(steps)
    for i = 1:steps
        entries_list[i] = count_MPS(bd[i,:], N_levels)
    end
    return entries_list 
end

# Create link Index objects for an MPS given a vector of link sizes.
function create_linkinds(N::Int64, link_size::Vector{Int64})
    ind_vec = Index{Int64}[]
    for i in 1:N-1
        ind = Index(link_size[i];tags="Link, l = $i")  # tag links for later selection
        push!(ind_vec, ind)
    end
    return ind_vec
end

# Check if an ITensor core is left-orthogonal to specified tolerance.
function is_left_orthogonal(A::ITensor; tol=1e-12)
    site, _, r = get_site_and_links(A)    # get site and right link index
    r === nothing && return true   # left boundary: trivially left-orthogonal

    Ac = dag(A)
    prime!(Ac, r)                       # prime the right link for contraction

    T = A * Ac                          # should be identity on left indices
    T_arr = Array(T, inds(T))
    row, col = size(T_arr)
    err = norm(T_arr - Matrix(1.0*I, row, col))
    println("err: ", err)
    if err < tol
        println("left orthogonal: true")
    else
        println("left orthogonal: false")
    end
    # return is_identity(T; tol=tol)
end

# Check if an ITensor core is right-orthogonal to specified tolerance.
function is_right_orthogonal(A::ITensor; tol=1e-12)
    site, l, _ = get_site_and_links(A)    # get site and left link index
    l === nothing && return true   # right boundary: trivially right-orthogonal

    Ac = dag(A)
    prime!(Ac, l)                       # prime the left link for contraction

    T = A * Ac                          # should be identity on right indices
    T_arr = Array(T, inds(T))
    row, col = size(T_arr)
    err = norm(T_arr - Matrix(1.0*I, row, col))
    println("err: ", err)
    if err < tol
        println("right orthogonal: true") 
    else
        println("right orthogonal: false")
    end
    # return is_identity(T; tol=tol)
end

# Print orthogonality properties (left/right) for every site in an MPS.
function ortho_properties(M::MPS; tol = 1e-12)
    N = length(M)
    for i in 1:N 
        println("Site $i")
        println("-----------------------------------------------")
        println("left-orthogonal: ")
        is_left_orthogonal(M[i])
        println("right-orthogonal: ")
        is_right_orthogonal(M[i])
        println("-----------------------------------------------")
    end
end

# Create an initial separable MPS corresponding to given qubit states.
# sites: site indices, q_state: vector of 0/1 values for each site.
function init_separable(sites, q_state)
    N = length(sites)
    M = MPS(N)
    link_size = Int64.(ones(N - 1))        # trivial bond dims = 1
    link_ind = create_linkinds(N, link_size)
    for i in 1:N
        if i == 1
            core = zeros(2, 1)
            core[q_state[i] + 1,1] = 1.0
            core_ten = ITensor(core, sites[i], link_ind[i])   # leftmost tensor
        elseif i == N 
            core = zeros(2, 1)
            core[q_state[i] + 1,1] = 1.0
            core_ten = ITensor(core, sites[i], link_ind[i - 1])  # rightmost tensor
        else 
            core = zeros(1, 2, 1)
            core[1,q_state[i] + 1,1] = 1.0
            core_ten = ITensor(core, sites[i], link_ind[i - 1], link_ind[i])  # bulk
        end

        M[i] = core_ten 
    end
    return M 
end

# Create an MPS where every site is the equal superposition (|0> + |1>)/sqrt(2).
function equal_separable(sites)
    N = length(sites)
    M = MPS(N)
    equal_arr = 1/sqrt(2)*ones(2)
    link_size = Int64.(ones(N-1)) 
    link_ind = create_linkinds(N, link_size)
    for i in 1:N 
        if i == 1
            core = zeros(2, 1)
            core[:,1] = equal_arr 
            core_ten = ITensor(core, sites[i], link_ind[i])
        elseif i == N 
            core = zeros(2, 1)
            core[:,1] = equal_arr 
            core_ten = ITensor(core, sites[i], link_ind[i - 1])
        else 
            core = zeros(1, 2, 1)
            core[1,:,1] = equal_arr 
            core_ten = ITensor(core, sites[i], link_ind[i - 1], link_ind[i])
        end
        M[i] = core_ten 
    end
    return M 
end

# get vector of link indices for a single Tensor 
function linkinds_tensor(M::ITensor)
    link_vec = Vector{Index}()
    N_inds = length(inds(M))
    for i in 1:N_inds 
        if hastags(inds(M)[i], "Link")
            push!(link_vec, inds(M)[i])
        end
    end
    return link_vec
end

# get vector of site indices for a single Tensor 
function siteinds_tensor(M::ITensor)
    site_vec = Vector{Index}()
    N_inds = length(inds(M))
    for i in 1:N_inds 
        if hastags(inds(M)[i], "Site")
            push!(site_vec, inds(M)[i])
        end
    end
    return site_vec
end

# create a new MPS as a subset of the old MPS
function MPS_subset(M::MPS, start_idx::Int, end_idx::Int)
    N = end_idx - start_idx + 1
    M_n = MPS(N)
    site = 1
    for i in start_idx:end_idx
        M_n[site] = M[i]
        site += 1
    end
    return M_n
end 

# Apply effective single-site Hamiltonian contribution: L * M * H_site * R (no primes).
function applyH_eff(H, M, L, R, site)
    return noprime(L*M*H[site]*R)
end

# Apply effective two-site Hamiltonian contribution: L * H_site * M * H_site+1 * R.
function applyH2_eff(H, M, L, R, site)
    return noprime(L*H[site]*M*H[site + 1]*R)
end

# Apply K effective (used for bond or zero-site blocks): L * C * R
function applyK_eff(H, C, L, R, site)
    return noprime(L*C*R)
end

# Contract left environment up to termination_site and return list of left blocks.
# H: MPO, M: MPS, termination_site: last site included (inclusive).
function contract_left(H::MPO, M::MPS, termination_site::Int64)
    L_list = []
    L = 1
    push!(L_list, L)
    for i = 1:termination_site
        L = L*M[i]*H[i]*conj(M[i]')
        push!(L_list, L)
    end
    return L_list 
end

# Contract right environment starting from termination_site to the end, return reversed list.
function contract_right(H::MPO, M::MPS, termination_site::Int64)
    N = length(M)
    R_list = []
    R = 1
    push!(R_list, R)
    for i = reverse(termination_site:N)
        R = R*M[i]*H[i]*conj(M[i]')
        push!(R_list, R)
    end
    return reverse(R_list) 
end

# Fixed-point iteration for two-site implicit midpoint (TT) update.
# H: MPO, init: two-site tensor, L/R: environments, h: step, site: left site index
function TT_fp_2site_new(H, init, L, R, h, site, maxiter, tol, verbose)
    k_init = ITensor(inds(init))                 # start with zero-like tensor of same shape
    # count = 0
    for i = 1:maxiter 
        k = -im*applyH2_eff(H, init + 0.5*h*k_init, L, R, site)
        err = norm(k - k_init)
        # count += 1
        if verbose == true 
            println("Iteration $i")
            println("Error: ", err)
        end
        if err < tol 
            break 
        end
        k_init = copy(k)
        
    end
    # println("Converged in $count iterations")
    return k_init 
end

# Fixed-point iteration for single-site implicit midpoint update (forward direction).
function TT_fp_1site_new(H, init, L, R, h, site, maxiter, tol, verbose)
    k_init = ITensor(inds(init))
    # println("inds k_init: ", inds(k_init))
    for i = 1:maxiter
        k = -im*applyH_eff(H, init + 0.5*h*k_init, L, R, site)
        # println("inds k", inds(k))
        err = norm(k - k_init)
        if verbose == true 
            println("Iteration $i")
            println("Error: ", err)
        end
        if err < tol 
            break 
        end
        k_init = copy(k)
    end
    return k_init 
end

# Fixed-point iteration for single-site implicit midpoint update used when evolving backwards.
function TT_fp_1site_new_backwards(H, init, L, R, h, site, maxiter, tol, verbose)
    k_init = ITensor(inds(init))
    # count = 0
    for i = 1:maxiter
        k = -im*applyH_eff(H, init - 0.5*h*k_init, L, R, site + 1)
        err = norm(k - k_init)
        # count += 1
        if verbose == true 
            println("Iteration $i")
            println("Error: ", err)
        end
        if err < tol 
            break 
        end
        k_init = copy(k)
    end
    # println("Converged in $count iterations")
    return k_init 
end

# Fixed-point iteration for zero-site (bond) update (implicit).
function TT_fp_0site_new(H, init, L, R, h, site, maxiter, tol, verbose)
    k_init = ITensor(inds(init))
    for i = 1:maxiter 
        k = im*applyK_eff(H, init - 0.5*h*k_init, L, R, site)
        err = norm(k - k_init)
        if verbose == true 
            println("Iteration $i")
            println("Error: ", err)
        end
        if err < tol 
            break 
        end
        k_init = copy(k)
    end
    return k_init 
end

# Implicit midpoint (IMR) two-site update helper that runs the fixed point and returns updated two-site tensor.
function TT_IMR_2site_new(H, init, L, R, h, site)
    k = TT_fp_2site_new(H, init, L, R, h, site, 100, 1E-15, false)
    update = init + h*k 
    return update 
end

# IMR single-site forward update helper.
function TT_IMR_1site_new(H, init, L, R, h, site)
    k = TT_fp_1site_new(H, init, L, R, h, site, 100, 1E-15, false)
    update = init + h*k 
    return update 
end

# IMR single-site backwards update helper.
function TT_IMR_1site_new_backwards(H, init, L, R, h, site)
    k = TT_fp_1site_new_backwards(H, init, L, R, h, site, 100, 1E-15, false)
    update = init - h*k 
    return update 
end

# IMR zero-site (bond) update helper.
function TT_IMR_0site_new(H, init, L, R, h, site)
    k = TT_fp_0site_new(H, init, L, R, h, site, 100, 1E-15, false)
    update = init - h*k 
    return update 
end

# Left-to-right single-site TDVP sweep using IMR updates.
# H: MPO, M: MPS (modified in-place and returned), R_list: precomputed right environments,
# t: current time (unused in body here), h: timestep.
function lr_sweep_new_new(H::MPO, M::MPS, R_list::Vector{Any}, t::Float64, h::Float64)
    N = length(M)
    L_list = []
    L = 1
    push!(L_list, L)
    for i = 1:N - 1
        # perform IMR one-site update for site i using left env L and right env R_list[i]
        M_evolve = TT_IMR_1site_new(H, M[i], L, R_list[i], h, i)
        if i==1
            # perform QR such that site index remains physical "n = 1" and new link tagged
            Q, R = qr(M_evolve, inds(M[i]; tags = "n = 1"); tags = "Link, l = 1")
        else
            # generalized QR: split by physical and left-link indices and tag new link
            Q, R = qr(M_evolve, inds(M[i]; tags = "n = $i")[1], inds(M[i]; tags = "l = $(i-1)")[1], ; tags = "Link, l = $i")
        end
        # update left environment with transformed site Q
        L = L*Q*H[i]*conj(Q)'
        push!(L_list, L)
        M[i] = Q 
        # Evolve the R tensor (right block) with zero-site IMR and absorb into next site
        R_evolve = TT_IMR_0site_new(H, R, L, R_list[i], h, i)
        M[i + 1] = R_evolve*M[i + 1]
    end
    # final site update (no QR needed)
    M_N_evolve = TT_IMR_1site_new(H, M[N], L, R_list[N], h, N)
    M[N] = M_N_evolve 
    return M, L_list
end

# Right-to-left single-site TDVP sweep using IMR updates.
function rl_sweep_new_new(H::MPO, M::MPS, L_list::Vector{Any}, t::Float64, h::Float64)
    N = length(M)
    R_list = []
    R_block = 1
    push!(R_list, R_block)
    for i = N:-1:2 
        # IMR update for site i using left env L_list[i] and right env R_block
        M_evolve = TT_IMR_1site_new(H, M[i], L_list[i], R_block, h, i)
        # factorize returns R (right-triangular) and Q; choose right-orthogonal factorization
        R, Q = factorize(M_evolve, inds(M[i]; tags = "l = $(i - 1)"); ortho = "right", tags = "Link, l = $(i-1)")
        R_block = R_block*Q*H[i]*conj(Q)'   # update right environment
        push!(R_list, R_block)
        M[i] = Q 
        # evolve the left piece and absorb into previous site
        R_evolve = TT_IMR_0site_new(H, R, L_list[i], R_block, h, i)
        M[i - 1] = R_evolve*M[i-1]
    end
    M_1_evolve = TT_IMR_1site_new(H, M[1], L_list[1], R_block, h, 1)
    M[1] = M_1_evolve 
    return M, reverse(R_list)
end

# Left-to-right two-site TDVP sweep performing two-site IMR updates then SVD truncation.
# cutoff, maxdim: SVD truncation controls. normalize: renormalize singular values if true.
function lr_sweep_2site_new(H::MPO, M::MPS, R_list::Vector{Any}, h::Float64, cutoff_vec::AbstractVector, maxdim::Union{Vector{Int64}, Nothing}=nothing; normalize::Bool = false)
    N = length(M)
    L_list = []    
    L = 1
    push!(L_list, L)
    # store per-bond truncation errors
    trunc_err = zeros(N-1)
    spectrum_list = []
    for i = 1:N-1
        two_site = M[i]*M[i + 1]                          # combine two neighboring cores
        two_site_evolve = TT_IMR_2site_new(H, two_site, L, R_list[i + 1], h, i)

        M_inds = inds(two_site_evolve)
        
        if isnothing(maxdim)
            maxdim = fill(nothing, N - 1)
        end

        if i == 1
            if N > 2
                bd = min(dim(M_inds[1]), dim(M_inds[2])*dim(M_inds[3]))
                # SVD with lefttags to define new link tag and optional cutoff/maxdim
                
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = 1")[1], cutoff = cutoff_vec[i]; lefttags = "Link, l = 1", maxdim = maxdim[i])
                if normalize == true 
                    S_trunc = S_trunc/norm(S_trunc)
                end
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = 1")[1], cutoff = cutoff_vec[i]; lefttags = "Link, l = 1", maxdim = maxdim[i])
                if normalize == true 
                    S_trunc = S_trunc/norm(S_trunc)
                end
            end
        else
            if i != N - 1
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3])*dim(M_inds[4]))
            else 
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3]))
            end
            # SVD splitting two-site block into left and right parts, tagging new link
            U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $i"), inds(two_site_evolve; tags = "l = $(i - 1)"), cutoff = cutoff_vec[i]; lefttags = "Link, l = $(i)", maxdim = maxdim[i])
            if normalize == true 
                S_trunc = S_trunc/norm(S_trunc)
            end
        end
        # store truncation error (sqrt of spectrum.truncerr)
        trunc_err[i] = sqrt(spectrum.truncerr)
        # also store spectrum
        push!(spectrum_list, spectrum)
        # update left environment using truncated U
        L = L*U_trunc*H[i]*conj(U_trunc)'
        push!(L_list, L)
        M[i] = U_trunc
        M_n = S_trunc*V_trunc
        if i != N - 1
            # evolve the right-hand piece backwards and absorb into next site
            M_evolve = TT_IMR_1site_new_backwards(H, M_n, L, R_list[i + 1], h, i)
            M[i + 1] = M_evolve 
        elseif i == N - 1
            M[i + 1] = S_trunc*V_trunc
        end 
    end
    return M, L_list, trunc_err, spectrum_list
end

# Right-to-left two-site TDVP sweep with SVD truncation and environments.
function rl_sweep_2site_new(H::MPO, M::MPS, L_list::Vector{Any}, h::Float64, cutoff_vec::AbstractVector, maxdim::Union{Int64, Nothing}=nothing; normalize::Bool = false)
    N = length(M)
    R_list = []
    R_block = 1
    push!(R_list, R_block)
    trunc_err = zeros(N - 1)
    for i = N:-1:2
        two_site = M[i]*M[i-1]
        two_site_evolve = TT_IMR_2site_new(H, two_site, L_list[i - 1], R_block, h, i - 1)
        M_inds = inds(two_site_evolve)
        if i == N
            if N > 2
                bd = min(dim(M_inds[1]), dim(M_inds[2])*dim(M_inds[3]))
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $(N-1)")[1], inds(two_site_evolve; tags = "l = $(N - 2)")[1], cutoff = cutoff_vec[i]; righttags = "l = $(N - 1)", maxdim = maxdim)
                if normalize == true 
                    S_trunc = S_trunc/norm(S_trunc)
                end
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $N")[1], inds(two_site_evolve; tags = "l = $(N - 1)"), cutoff = cutoff_vec[i]; righttags = "l = $(N - 1)", maxdim = maxdim)
                if normalize == true 
                    S_trunc = S_trunc/norm(S_trunc)
                end
            end
        else
            if i != 2
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3])*dim(M_inds[4]))
            else 
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3]))
            end
            U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $(i - 1)")[1], inds(two_site_evolve; tags = "l = $(i - 2)"), cutoff = cutoff_vec[i]; righttags = "l = $(i - 1)", maxdim = maxdim)

            if normalize == true 
                S_trunc = S_trunc/norm(S_trunc)
            end
        end
        trunc_err[i - 1] = sqrt(spectrum.truncerr)
        R_block = R_block*(V_trunc*H[i]*conj(V_trunc)')
        push!(R_list, R_block)
        M[i] = V_trunc 
        M_n = U_trunc*S_trunc
        if i != 2
            # evolve left-hand piece backwards and absorb into previous site
            M_evolve = TT_IMR_1site_new_backwards(H, M_n, L_list[i - 1], R_block, h, i - 2)
            M[i - 1] = M_evolve 
        elseif i == 2
            M[i - 1] = U_trunc*S_trunc 
        end
    end 
    return M, reverse(R_list), trunc_err
end

# TDVP single-site-like sweep variant using IMR updates with constant MPO (no time-dependence).
# Returns evolved MPS.
function tdvp_constant_adjoint(H, init, t0, T, steps, verbose = false)
    N = length(init)
    orthogonalize!(init, 1)                     # ensure canonical form starting at site 1
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    # Get step size
    h = (T - t0)/steps
    L_list_init = Vector{Any}(undef, N)
    R_list_init = contract_right(H, init, 2)
    @showprogress 1 "TDVP" for i = 1:steps
        if verbose == true
            println("Step: ", i)
        end
        # left-to-right half sweep (IMR single-site)
        init_copy, L_list_init = lr_sweep_new_new(H, init_copy, R_list_init, t0, h/2)
        t0 += h/2
        # right-to-left half sweep
        init_copy, R_list_init = rl_sweep_new_new(H, init_copy, L_list_init, t0, h/2)
        t0 += h/2
    end
    return init_copy
end

function tdvp(H::MPO, init::MPS, t0::Real, T::Real, steps::Int64, bc_params::bcparams;strang::Bool = false, magnet::Bool = false, energy::Bool = false, verbose::Bool = false, save_history::Bool = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = deepcopy(init)
    d = prod(dim(sites))
    # Get step size
    h = (T - t0)/steps
    magnet_history = zeros(steps + 1, N)
    if magnet == true 
        magnet_history[1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
    end
    energy_history = zeros(steps + 1)   
    if energy == true 
        energy_history[1] = real(inner(init_copy', H, init_copy))
    end
    state_history = Vector{MPS}(undef, steps)
    if strang == true
        t0 = t0
        # Run Strang-split time stepping with time-dependent MPO updates via bc_params
        L_list = Vector{Any}(undef, N)
        @showprogress 1 "TDVP2 Strang Splitting" for i = 1:steps
            if verbose == true
                println("Step: ", i)
            end
            # update MPO for first half-step and perform left-to-right half-step
            update_MPO!(H, bc_params, t0 + h/4)
            R_list = contract_right(H, init_copy, 2)
            init_copy, _ = lr_sweep_new_new(H, init_copy, R_list, t0, h/2)
            # update for second half-step and perform right-to-left half-step
            t0 += h/2
            update_MPO!(H, bc_params, t0 + h/4)
            L_list = contract_left(H, init_copy, N - 1)
            init_copy, _ = rl_sweep_new_new(H, init_copy, L_list, t0, h/2)
            if save_history 
                state_history[i] = init_copy 
            end

            if magnet == true 
                magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
            end
            if energy == true 
                energy_history[i + 1] = real(inner(init_copy', H, init_copy))
            end
            t0 += h/2
        end
    elseif strang == false 
        t0 = t0
        trunc_err = zeros(steps, N - 1)
        @showprogress 1 "TDVP2 Lie-Trotter Splitting" for i = 1:steps
            update_MPO!(H, bc_params, t0 + h/2)
            R_list = contract_right(H, init_copy, 2)
            if verbose == true 
                println("Step: ", i)
            end
            init_copy, _ = lr_sweep_new_new(H, init_copy, R_list, t0, h)
            if save_history 
                state_history[i] = init_copy 
            end

            if magnet == true 
                magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
            end
            if energy == true 
                energy_history[i + 1] = real(inner(init_copy', H, init_copy))
            end
            t0 += h

            orthogonalize!(init_copy, 1)
        end
    end

    return init_copy, state_history, magnet_history, energy_history
end

"""
tdvp2_constant(H::MPO, init::MPS, t0::Real, T::Real, steps::Int64;
               cutoff::Union{Float64,Nothing}=nothing,
               maxdim::Union{Int64,Nothing}=nothing,
               magnet::Bool=false,
               energy::Bool=false,
               verbose::Bool=false,
               normalize::Bool=false,
               strang::Bool=true)

Perform two-site TDVP time evolution with a constant time-step.

Arguments
- H::MPO
  Hamiltonian (MPO) driving the evolution. Must be compatible with `init` (same sites/ordering).

- init::MPS
  Initial matrix product state to be evolved.

- t0::Real
  Initial time.

- T::Real
  Final time.

- steps::Int64
  Number of equal time steps between `t0` and `T` (dt = (T - t0) / steps).

Keyword arguments
- cutoff::Union{Float64,Nothing} = nothing
  Truncation tolerance for SVD. If `nothing`, no explicit cutoff is applied (implementation-dependent default).
  Typical call sites may pass a squared tolerance; follow the convention used by the implementation.

- maxdim::Union{Int64,Nothing} = nothing
  Maximum allowed bond dimension during truncation. If `nothing`, no explicit cap beyond algorithmic limits.

- magnet::Bool = false
  If true, compute and return local magnetization history during the evolution.

- energy::Bool = false
  If true, compute and return energy expectation values during the evolution.

- verbose::Bool = false
  If true, emit progress and diagnostic information.

- normalize::Bool = false
  If true, re-normalize the MPS at appropriate steps to control norm drift.

- strang::Bool = true
  If true, use Strang (symmetric) splitting ordering for two-site updates; otherwise use a non-symmetric (Lie-Trotter) ordering.

Returns
A tuple typically containing:
- evolved MPS at final time,
- bond-dimension history,
- magnetization history (if requested),
- energy history (if requested),
- truncation error(s) or other diagnostics.
"""
function tdvp2_constant(H::MPO, init::MPS, t0::Real, T::Real, steps::Int64; cutoff::Union{Float64, Nothing}=nothing, maxdim::Union{Int64, Nothing} = nothing, magnet::Bool = false, energy::Bool = false, verbose::Bool = false, normalize::Bool = false, strang::Bool = true)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = deepcopy(init)
    d = prod(dim(sites))
    # Get step size
    h = (T - t0)/steps
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(init_copy)
    if strang == true 
        magnet_history = zeros(2*steps + 1, N)   # store magnetization at half-steps too
    else
        magnet_history = zeros(steps + 1, N)
    end
    if magnet == true 
        magnet_history[1,:] = expect(init_copy, [1 0; 0 -1])
    end
    energy_history = zeros(steps + 1)   
    if energy == true 
        energy_history[1] = real(inner(init_copy', H, init_copy))
    end
    L_list_list = []
    R_list_list = []
    if strang == true
        trunc_err = zeros(2*steps, N - 1)
        # Run time stepper with Strang splitting (left half-step, right half-step)
        L_list = Vector{Any}(undef, N)
        R_list = contract_right(H, init, 2)
        @showprogress 1 "TDVP2 Strang Splitting" for i = 1:steps
            if verbose == true
                println("Step: ", i)
                println("Bond dimensions before evolution: ", linkdims(init_copy))
            end
            # left-to-right half-step using two-site sweeps
            init_copy, L_list, trunc1 = lr_sweep_2site_new(H, init_copy, R_list, h/2, cutoff, maxdim; normalize = normalize)
            if magnet == true 
                magnet_history[2*i,:] = expect(init_copy, [1 0; 0 -1])
            end
            # right-to-left half-step
            init_copy, R_list, trunc2 = rl_sweep_2site_new(H, init_copy, L_list, h/2, cutoff, maxdim; normalize = normalize)
            trunc_err[2*i - 1, :] = trunc1 
            trunc_err[2*i, :] = trunc2
            link_dim[i + 1,:] = linkdims(init_copy)
            if magnet == true 
                magnet_history[2*i + 1,:] = expect(init_copy, [1 0; 0 -1])
            end
            if energy == true 
                energy_history[i + 1] = real(inner(init_copy', H, init_copy))
            end
        end
    elseif strang == false 
        trunc_err = zeros(steps, N - 1)
        @showprogress 1 "TDVP2 Lie-Trotter Splitting" for i = 1:steps
            R_list = contract_right(H, init_copy, 2)
            push!(R_list_list, R_list)
            if verbose == true 
                println("Step: ", i)
                println("Bond dimensions before evolution: ", linkdims(init_copy))
            end
            init_copy, L_list, trunc = lr_sweep_2site_new(H, init_copy, R_list, h, cutoff, maxdim; normalize = normalize)
            link_dim[i + 1,:] = linkdims(init_copy)
            if magnet == true 
                magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
            end
            if energy == true 
                energy_history[i + 1] = real(inner(init_copy', H, init_copy))
            end
            trunc_err[i,:] = trunc
            orthogonalize!(init_copy, 1)
        end
    end

    return init_copy, link_dim, magnet_history, energy_history, trunc_err
end


"""
tdvp2(H::MPO, init::MPS, t0::Real, T::Real, steps::Int64, bc_params::bcparams;
               cutoff::Union{Float64,Nothing}=nothing,
               maxdim::Union{Int64,Nothing}=nothing,
               magnet::Bool=false,
               energy::Bool=false,
               verbose::Bool=false,
               normalize::Bool=false,
               strang::Bool=true)

Evolve an MPS under a Hamiltonian MPO using a two-site TDVP integrator.

Performs time evolution of the matrix product state `init` under the Hamiltonian `H`
from time `t0` to `T` using `steps` discrete time steps. The integrator works on
two-site updates and supports optional SVD truncation and bond-dimension control.
A symmetric second-order (Strang) splitting is used by default.

Arguments
- H::MPO
    The Hamiltonian as an MPO that generates the time evolution (may be time-independent).
- init::MPS
    The initial MPS to be evolved. This state is modified or copied depending on implementation.
- t0::Real
    Initial time of the evolution.
- T::Real
    Final time of the evolution.
- steps::Int64
    Number of time steps. The step size used is dt = (T - t0) / steps.
- bc_params::bcparams
    Boundary-condition parameters (type depends on implementation) controlling edge terms/closures.

Keyword arguments
- cutoff::Union{Float64, Nothing}=nothing
    SVD truncation tolerance: singular values smaller than `cutoff` are discarded.
    If `nothing`, no truncation by tolerance is performed.
- maxdim::Union{Int64, Nothing}=nothing
    Maximum allowed bond dimension during truncation. If `nothing`, bond dimensions are
    not explicitly limited (only controlled by `cutoff`).
- magnet::Bool=false
    If true, compute and record site magnetizations (or a user-defined local observable)
    at each saved time point.
- energy::Bool=false
    If true, compute and record the energy ⟨H⟩ at each saved time point.
- verbose::Bool=false
    Print progress and diagnostic information during the evolution.
- normalize::Bool=false
    If true, renormalize the MPS (to unit norm) after each time step/update.
- strang::Bool=true
    Use Strang (second-order symmetric) splitting for the integrator when true.
    If false, a first-order integrator is used.

Returns
A tuple typically containing:
- evolved MPS at final time,
- bond-dimension history,
- magnetization history (if requested),
- energy history (if requested),
- truncation error(s) or other diagnostics.

- verbose::Bool = false
  If true, print progress information and diagnostics during the time evolution to assist with monitoring and debugging.

"""
function tdvp2(H::MPO, init::MPS, t0::Real, T::Real, steps::Int64, bc_params::bcparams; cutoff::Union{Vector{Float64}, Float64, Nothing}=nothing, maxdim::Union{Vector{Int64}, Nothing} = nothing, magnet::Bool = false, energy::Bool = false, verbose::Bool = false, normalize::Bool = false, strang::Bool = true, save_history::Bool = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = deepcopy(init)
    d = prod(dim(sites))
    # Get step size
    h = (T - t0)/steps
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(init_copy)
    magnet_history = zeros(steps + 1, N)

    # create cutoff vector 
    if cutoff isa Number 
        cutoff_vec = fill(cutoff, N - 1)
    elseif cutoff isa Vector 
        cutoff_vec = cutoff
    else
        cutoff_vec = fill(nothing, N - 1)
    end

    if magnet == true 
        magnet_history[1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
    end
    energy_history = zeros(steps + 1)   
    if energy == true 
        energy_history[1] = real(inner(init_copy', H, init_copy))
    end
    state_history = Vector{MPS}(undef, steps)
    if strang == true
        t0 = t0
        trunc_err = zeros(2*steps, N - 1)
        # Run Strang-split time stepping with time-dependent MPO updates via bc_params
        L_list = Vector{Any}(undef, N)
        @showprogress 1 "TDVP2 Strang Splitting" for i = 1:steps
            if verbose == true
                println("Step: ", i)
            end
            # update MPO for first half-step and perform left-to-right half-step
            update_MPO!(H, bc_params, t0 + h/4)
            R_list = contract_right(H, init_copy, 2)
            init_copy, _, trunc1, _ = lr_sweep_2site_new(H, init_copy, R_list, h/2, cutoff_vec, maxdim; normalize = normalize)
            # update for second half-step and perform right-to-left half-step
            t0 += h/2
            update_MPO!(H, bc_params, t0 + h/4)
            L_list = contract_left(H, init_copy, N - 1)
            init_copy, _, trunc2 = rl_sweep_2site_new(H, init_copy, L_list, h/2, cutoff_vec, maxdim; normalize = normalize)
            if save_history 
                state_history[i] = init_copy 
            end
            trunc_err[2*i - 1, :] = trunc1 
            trunc_err[2*i, :] = trunc2
            link_dim[i + 1,:] = linkdims(init_copy)
            if magnet == true 
                magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
            end
            if energy == true 
                energy_history[i + 1] = real(inner(init_copy', H, init_copy))
            end
            t0 += h/2
        end
    elseif strang == false 
        t0 = t0
        trunc_err = zeros(steps, N - 1)
        @showprogress 1 "TDVP2 Lie-Trotter Splitting" for i = 1:steps
            update_MPO!(H, bc_params, t0 + h/2)
            R_list = contract_right(H, init_copy, 2)
            if verbose == true 
                println("Step: ", i)
                println("Bond dimensions before evolution: ", linkdims(init_copy))
            end
            init_copy, _, trunc, _ = lr_sweep_2site_new(H, init_copy, R_list, h, cutoff_vec, maxdim; normalize = normalize)
            if save_history 
                state_history[i] = init_copy 
            end
            link_dim[i + 1,:] = linkdims(init_copy)
            if magnet == true 
                magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
            end
            if energy == true 
                energy_history[i + 1] = real(inner(init_copy', H, init_copy))
            end
            t0 += h
            trunc_err[i,:] = trunc
            orthogonalize!(init_copy, 1)
        end
    end

    return init_copy, link_dim, state_history, magnet_history, energy_history, trunc_err
end



function tdvp2(H::MPO, init::MPS, t0::Real, T::Real, steps::Int64, pulse_real::AbstractMatrix, pulse_imag::AbstractMatrix; cutoff::Union{Vector{<:Real}, Real, Nothing}=nothing, maxdim::Union{Vector{Int64}, Nothing} = nothing, magnet::Bool = false, energy::Bool = false, verbose::Bool = false, normalize::Bool = false, strang::Bool = true, save_history::Bool = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = deepcopy(init)
    d = prod(dim(sites))
    # Get step size
    h = (T - t0)/steps
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(init_copy)
    magnet_history = zeros(steps + 1, N)
    if magnet == true 
        magnet_history[1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
    end
    energy_history = zeros(steps + 1)   
    if energy == true 
        energy_history[1] = real(inner(init_copy', H, init_copy))
    end
    state_history = Vector{MPS}(undef, steps + 1)
    if save_history
        state_history[1] = init
    end
    t0 = t0
    trunc_err = zeros(steps, N - 1)
    spectrum_history = []
    if cutoff isa Number 
        cutoff_vec = fill(cutoff, N - 1)
    elseif cutoff isa Vector 
        cutoff_vec = cutoff
    else
        cutoff_vec = fill(nothing, N - 1)
    end
    @showprogress 1 "TDVP2 Lie-Trotter Splitting" for i = 2:steps
        update_MPO!(H, pulse_real, pulse_imag, i)
        R_list = contract_right(H, init_copy, 2)
        if verbose == true 
            println("Step: ", i)
            println("Bond dimensions before evolution: ", linkdims(init_copy))
        end
        init_copy, _, trunc, spectrum_list = lr_sweep_2site_new(H, init_copy, R_list, h, cutoff_vec, maxdim; normalize = normalize)
        if save_history 
            state_history[i] = init_copy 
        end
        link_dim[i,:] = linkdims(init_copy)
        if magnet == true 
            magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
        end
        if energy == true 
            energy_history[i + 1] = real(inner(init_copy', H, init_copy))
        end
        t0 += h
        trunc_err[i-1,:] = trunc
        push!(spectrum_history, spectrum_list)
        orthogonalize!(init_copy, 1)
    end


    return init_copy, link_dim, state_history, magnet_history, energy_history, trunc_err, spectrum_history
end


# implement TDVP2 with a changing dipole-dipole coupling value
function tdvp2_changing_dipole(H_list::Vector{MPO}, init::MPS, t0::Real, T::Real, steps::Int64, pulse_real::AbstractMatrix, pulse_imag::AbstractMatrix, t_vals::Vector{<:Tuple{<:Real, <:Real}}; cutoff::Union{Vector{<:Real}, Real, Nothing}=nothing, maxdim::Union{Vector{Int64}, Nothing} = nothing, magnet::Bool = false, energy::Bool = false, verbose::Bool = false, normalize::Bool = false, strang::Bool = true, save_history::Bool = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = deepcopy(init)
    d = prod(dim(sites))
    # Get step size
    h = (T - t0)/steps
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(init_copy)
    magnet_history = zeros(steps + 1, N)
    if magnet == true 
        magnet_history[1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
    end
    energy_history = zeros(steps + 1)   
    if energy == true 
        energy_history[1] = real(inner(init_copy', H, init_copy))
    end
    state_history = Vector{MPS}(undef, steps + 1)
    if save_history
        state_history[1] = init
    end
    t0 = t0
    trunc_err = zeros(steps, N - 1)

    # create cutoff vector 
    if cutoff isa Number 
        cutoff_vec = fill(cutoff, N - 1)
    elseif cutoff isa Vector 
        cutoff_vec = cutoff
    else
        cutoff_vec = fill(nothing, N - 1)
    end

    @showprogress 1 "TDVP2 Lie-Trotter Splitting" for i = 2:steps
        t_ind = findfirst(contains.(t0, t_vals))
        H = H_list[t_ind]
        update_MPO!(H, pulse_real, pulse_imag, i)
        R_list = contract_right(H, init_copy, 2)
        if verbose == true 
            println("Step: ", i)
            println("Bond dimensions before evolution: ", linkdims(init_copy))
        end
        init_copy, _, trunc, spectrum_list = lr_sweep_2site_new(H, init_copy, R_list, h, cutoff_vec, maxdim; normalize = normalize)
        if save_history 
            state_history[i] = init_copy 
        end
        link_dim[i + 1,:] = linkdims(init_copy)
        if magnet == true 
            magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
        end
        if energy == true 
            energy_history[i + 1] = real(inner(init_copy', H, init_copy))
        end
        t0 += h
        trunc_err[i,:] = trunc
        orthogonalize!(init_copy, 1)
    end


    return init_copy, link_dim, state_history, magnet_history, energy_history, trunc_err
end

# implement TDVP2 with a changing dipole-dipole coupling value
# pass in object of values, construct new hamiltonian each time something changes. 
function tdvp2_changing_dipole(h_params::Drift_Hamiltonian, init::MPS, t0::Real, T::Real, steps::Int64, pulse_real::AbstractMatrix, pulse_imag::AbstractMatrix; cutoff::Union{Vector{<:Real}, Real, Nothing}=nothing, maxdim::Union{Vector{Int64}, Nothing} = nothing, magnet::Bool = false, energy::Bool = false, verbose::Bool = false, normalize::Bool = false, strang::Bool = true, save_history::Bool = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = deepcopy(init)
    d = prod(dim(sites))
    # Get step size
    h = (T - t0)/steps
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(init_copy)
    magnet_history = zeros(steps + 1, N)
    if magnet == true 
        magnet_history[1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
    end
    energy_history = zeros(steps + 1)   
    if energy == true 
        energy_history[1] = real(inner(init_copy', H, init_copy))
    end
    state_history = Vector{MPS}(undef, steps + 1)
    if save_history
        state_history[1] = init
    end
    t0 = t0
    trunc_err = zeros(steps, N - 1)

    H_sites = h_params.sites
    freq01 = h_params.transition_freq
    rot_freq = h_params.rot_freq
    zz = h_params.zz
    self_kerr = h_params.self_kerr
    Jkl_total = h_params.dipole
    bond_dict = h_params.bond_dict
    all_times = Vector{Any}()
    for pair in keys(bond_dict)
        push!(all_times, bond_dict[pair].t_range)
    end
    # create sorted list 
    t_list_sorted = sort(unique(collect(Iterators.flatten(all_times))))
    t_list = filter(!isnan, t_list_sorted)
        # create cutoff vector 
    if cutoff isa Number 
        cutoff_vec = fill(cutoff, N - 1)
    elseif cutoff isa Vector 
        cutoff_vec = cutoff
    else
        cutoff_vec = fill(nothing, N - 1)
    end
    H = nothing
    @showprogress 1 "TDVP2 Lie-Trotter Splitting" for i = 2:steps
        if t0 >= t_list[1]
            
            Jkl_updated = create_dipole_matrix(Jkl_total, bond_dict, t0)
            # if verbose
            #     println("Time: $t0, Jkl")
            #     display(Jkl_updated)
            # end
            H = drift_MPO(N, H_sites, freq01, rot_freq, self_kerr = self_kerr, zz = zz, dipole = Jkl_updated)
            popfirst!(t_list)
        end

        update_MPO!(H, pulse_real, pulse_imag, i)
        R_list = contract_right(H, init_copy, 2)
        if verbose == true 
            println("Step: ", i)
            println("Bond dimensions before evolution: ", linkdims(init_copy))
        end
        init_copy, _, trunc, spectrum_list = lr_sweep_2site_new(H, init_copy, R_list, h, cutoff_vec, maxdim; normalize = normalize)
        if save_history 
            state_history[i] = init_copy 
        end
        link_dim[i + 1,:] = linkdims(init_copy)
        if magnet == true 
            magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
        end
        if energy == true 
            energy_history[i + 1] = real(inner(init_copy', H, init_copy))
        end
        t0 += h
        trunc_err[i,:] = trunc
        orthogonalize!(init_copy, 1)
    end


    return init_copy, link_dim, state_history, magnet_history, energy_history, trunc_err
end