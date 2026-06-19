using ITensors
using LinearAlgebra, ProgressMeter

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
    k = TT_fp_2site_new(H, init, L, R, h, site, 100, 1E-12, false)
    update = init + h*k 
    return update 
end

# IMR single-site forward update helper.
function TT_IMR_1site_new(H, init, L, R, h, site)
    k = TT_fp_1site_new(H, init, L, R, h, site, 100, 1E-12, false)
    update = init + h*k 
    return update 
end

# IMR single-site backwards update helper.
function TT_IMR_1site_new_backwards(H, init, L, R, h, site)
    k = TT_fp_1site_new_backwards(H, init, L, R, h, site, 100, 1E-12, false)
    update = init - h*k 
    return update 
end

# IMR zero-site (bond) update helper.
function TT_IMR_0site_new(H, init, L, R, h, site)
    k = TT_fp_0site_new(H, init, L, R, h, site, 100, 1E-12, false)
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
function lr_sweep_2site_new(H::MPO, M::MPS, R_list::Vector{Any}, h::Float64, cutoff::Union{Float64, Nothing}, maxdim::Union{Int64, Nothing}=nothing; normalize::Bool = false)
    N = length(M)
    L_list = []    
    L = 1
    push!(L_list, L)
    # store per-bond truncation errors
    trunc_err = zeros(N-1)
    for i = 1:N-1
        two_site = M[i]*M[i + 1]                          # combine two neighboring cores
        two_site_evolve = TT_IMR_2site_new(H, two_site, L, R_list[i + 1], h, i)
        M_inds = inds(two_site_evolve)
        
        if i == 1
            if N > 2
                bd = min(dim(M_inds[1]), dim(M_inds[2])*dim(M_inds[3]))
                # SVD with lefttags to define new link tag and optional cutoff/maxdim
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = 1")[1], cutoff = cutoff; lefttags = "Link, l = 1", maxdim = maxdim)
                if normalize == true 
                    S_trunc = S_trunc/norm(S_trunc)
                end
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = 1")[1], cutoff = cutoff; lefttags = "Link, l = 1", maxdim = maxdim)
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
            U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $i"), inds(two_site_evolve; tags = "l = $(i - 1)"), cutoff = cutoff; lefttags = "Link, l = $(i)", maxdim = maxdim)
            if normalize == true 
                S_trunc = S_trunc/norm(S_trunc)
            end
        end
        # store truncation error (sqrt of spectrum.truncerr)
        trunc_err[i] = sqrt(spectrum.truncerr)
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
    return M, L_list, trunc_err
end

# Right-to-left two-site TDVP sweep with SVD truncation and environments.
function rl_sweep_2site_new(H::MPO, M::MPS, L_list::Vector{Any}, h::Float64, cutoff::Union{Float64, Nothing}, maxdim::Union{Int64, Nothing}=nothing; normalize::Bool = false)
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
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $(N-1)")[1], inds(two_site_evolve; tags = "l = $(N - 2)")[1], cutoff = cutoff; righttags = "l = $(N - 1)", maxdim = maxdim)
                if normalize == true 
                    S_trunc = S_trunc/norm(S_trunc)
                end
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $N")[1], inds(two_site_evolve; tags = "l = $(N - 1)"), cutoff = cutoff; righttags = "l = $(N - 1)", maxdim = maxdim)
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
            U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $(i - 1)")[1], inds(two_site_evolve; tags = "l = $(i - 2)"), cutoff = cutoff; righttags = "l = $(i - 1)", maxdim = maxdim)

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
function tdvp2(H::MPO, init::MPS, t0::Real, T::Real, steps::Int64, bc_params::bcparams; cutoff::Union{Float64, Nothing}=nothing, maxdim::Union{Int64, Nothing} = nothing, magnet::Bool = false, energy::Bool = false, verbose::Bool = false, normalize::Bool = false, strang::Bool = true, save_history::Bool = false)
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
            init_copy, _, trunc1 = lr_sweep_2site_new(H, init_copy, R_list, h/2, cutoff, maxdim; normalize = normalize)
            # update for second half-step and perform right-to-left half-step
            t0 += h/2
            update_MPO!(H, bc_params, t0 + h/4)
            L_list = contract_left(H, init_copy, N - 1)
            init_copy, _, trunc2 = rl_sweep_2site_new(H, init_copy, L_list, h/2, cutoff, maxdim; normalize = normalize)
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
            init_copy, _, trunc = lr_sweep_2site_new(H, init_copy, R_list, h, cutoff, maxdim; normalize = normalize)
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
