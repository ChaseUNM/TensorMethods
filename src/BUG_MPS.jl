using ITensors, ITensorMPS, LinearAlgebra 

# Returns maximum dimension of the left and right site indices for a quantum system
# where each subsystem has 2 energy levels. Useful to reason about maximum possible
# bond dimensions when building/testing MPS structures.
function max_bond_dimension(i, N)
    middle_site = (N + 1)/2
    if i == 1
        max_left = nothing 
        max_right = 2
    elseif i == N 
        max_left = 2
        max_right = nothing 
    else
        if i < middle_site 
            max_left = 2^(i - 1)
            max_right = 2^i 
        elseif i > middle_site 
            max_left = 2^(N - i + 1)
            max_right = 2^(N - i)
        elseif i == middle_site 
            max_left = 2^(i - 1)
            max_right = 2^(i - 1)
        end
    end
    return max_left, max_right 
end

# Helper: given an ITensor representing a single MPS site, find the site Index
# and the Link Indices to the left and right. Also extract the site number n from tags.
function get_site_and_links(T::ITensor)
    site_idx = nothing
    left_link = nothing
    right_link = nothing
    n = nothing

    # 1. Find site index and extract site number n
    for idx in inds(T)
        if hastags(idx, "Site")
            site_idx = idx
            for tag in tags(idx)
                t = String(tag)
                if occursin("n=", t)
                    n = parse(Int, split(t, "n=")[2])
                end
            end
            break
        end
    end

    n === nothing && error("Site index does not contain tag n=<int>")

    # 2. Classify link indices by checking tags "Link" and "l = <int>".
    for idx in inds(T)
        hastags(idx, "Link") || continue

        for tag in tags(idx)
            t = String(tag)

            # look for "l = <int>"
            occursin("l", t) || continue
            occursin("=", t) || continue

            parts = split(t, "=")
            length(parts) == 2 || continue

            lval = tryparse(Int, strip(parts[2]))
            lval === nothing && continue

            if lval == n - 1
                left_link = idx
            elseif lval == n
                right_link = idx
            end
        end
    end

    return site_idx, left_link, right_link
end

# Perform a series of 1-site SVD moves from left to center and from right to center
# to truncate bond dimensions in a TDVP-like style. This returns a new MPS.
function TDVP1_style_truncation(M::MPS, center::Int64; cutoff::Float64 = 1E-10)
    M_trunc = deepcopy(M)
    N = length(M)

    # Left-to-center orthogonalization and truncation
    for i in 1:center - 1
        site_idx, left_idx, right_idx = get_site_and_links(M_trunc[i])
        if i == 1 
            row_inds = Vector{Index}(undef, 1)
            row_inds[1] = site_idx
        else
            row_inds = Vector{Index}(undef, 2)
            row_inds[1] = left_idx 
            row_inds[2] = site_idx
        end
        U, S, V, spectrum = svd(M_trunc[i], row_inds; cutoff = cutoff, lefttags = "Link, l = $(i)")
        M_trunc[i] = U 
        M_trunc[i + 1] = M_trunc[i + 1]*S*V
    end

    # Right-to-center orthogonalization and truncation
    for i in N:-1:center + 1 
        site_idx, left_idx, right_idx = get_site_and_links(M_trunc[i])
        if i == N 
            row_inds = Vector{Index}(undef, 1)
            row_inds = left_idx 
        else
            row_inds = Vector{Index}(undef, 1)
            row_inds[1] = left_idx
        end
        U, S, V, spectrum = svd(M_trunc[i], row_inds; cutoff = cutoff, righttags = "Link, l = $(i-1)")
        M_trunc[i] = V 
        M_trunc[i-1] = M_trunc[i - 1]*U*S 
    end

    return M_trunc 
end

# Orthogonalize an MPS from right to left between start_site down to end_site+1.
# This returns a new MPS (non-mutating).
function orthogonalize_right_to_left(M::MPS, start_site::Int64, end_site::Int64)
    M_ortho = deepcopy(M)
    for i in start_site:-1:end_site + 1
        M_site = M_ortho[i]
        site_idx, left_idx, right_idx = get_site_and_links(M_site)
        R, Q = factorize(M_site, left_idx; ortho="right", tags = "Link, l = $(i-1)")
        M_ortho[i] = Q 
        M_ortho[i - 1] = M_ortho[i - 1]*R
    end
    return M_ortho 
end

# In-place version of orthogonalize_right_to_left: modifies M directly.
function orthogonalize_right_to_left!(M::MPS, start_site::Int64, end_site::Int64)

    for i in start_site:-1:end_site + 1
        M_site = M[i]
        site_idx, left_idx, right_idx = get_site_and_links(M_site)
        R, Q = factorize(M_site, left_idx; ortho="right", tags = "Link, l = $(i-1)")
        M[i] = Q 
        M[i - 1] = M[i - 1]*R
    end
end

# Orthogonalize an MPS from left to right between start_site to end_site-1.
# Returns a new MPS with left-to-right orthogonality enforced.
function orthogonalize_left_to_right(M::MPS, start_site::Int64, end_site::Int64)
    M_ortho = deepcopy(M)
    for i in start_site:end_site - 1
        M_site = M_ortho[i]
        site_idx, left_idx, right_idx = get_site_and_links(M_site)
        if i == 1
            row_inds = site_idx 
        else
            row_inds = Vector{Index}(undef, 2)
            row_inds[1] = left_idx 
            row_inds[2] = site_idx 
        end
        Q, R = qr(M_site, row_inds; tags = "Link, l = $i")
        M_ortho[i] = Q 
        M_ortho[i + 1] = R*M_ortho[i + 1]
    end
    return M_ortho 
end

# In-place left-to-right orthogonalization.
function orthogonalize_left_to_right!(M::MPS, start_site::Int64, end_site::Int64)

    for i in start_site:end_site - 1
        M_site = M[i]
        site_idx, left_idx, right_idx = get_site_and_links(M_site)
        if i == 1
            row_inds = site_idx 
        else
            row_inds = Vector{Index}(undef, 2)
            row_inds[1] = left_idx 
            row_inds[2] = site_idx 
        end
        Q, R = qr(M_site, row_inds; tags = "Link, l = $i")
        M[i] = Q 
        M[i + 1] = R*M[i + 1]
    end

end

# TDVP-like truncation that first moves orthogonality to the given center (in-place),
# then performs left-to-center SVDs and right-to-center SVDs. Optionally enforces maxdim.
function TDVP1_style_truncation_in_move_orthogonal(M::MPS, center::Int64; cutoff::Float64 = 1E-10, maxdim::Union{Nothing, Int64} = nothing)
    orthogonalize_right_to_left!(M, center, 1)
    M_trunc = deepcopy(M)
    N = length(M)
    
    for i in 1:center - 1
        site_idx, left_idx, right_idx = get_site_and_links(M_trunc[i])
        if i == 1 
            row_inds = Vector{Index}(undef, 1)
            row_inds[1] = site_idx
        else
            row_inds = Vector{Index}(undef, 2)
            row_inds[1] = left_idx 
            row_inds[2] = site_idx
        end
        U, S, V, spectrum = svd(M_trunc[i], row_inds; cutoff = cutoff, lefttags = "Link, l = $(i)", maxdim = maxdim)

        M_trunc[i] = U 
        M_trunc[i + 1] = M_trunc[i + 1]*S*V
    end
    orthogonalize_left_to_right!(M_trunc, center, N)
    for i in N:-1:center + 1 
        site_idx, left_idx, right_idx = get_site_and_links(M_trunc[i])
        if i == N 
            row_inds = Vector{Index}(undef, 1)
            row_inds = left_idx 
        else
            row_inds = Vector{Index}(undef, 1)
            row_inds[1] = left_idx
        end
        U, S, V, spectrum = svd(M_trunc[i], row_inds; cutoff = cutoff, righttags = "Link, l = $(i-1)", maxdim = maxdim)

        M_trunc[i] = V 
        M_trunc[i-1] = M_trunc[i - 1]*U*S 
    
    end

    return M_trunc 
end

### Make new truncation code, going from the orthogonality center to the ends of the MPS. 
function TDVP1_style_truncation_out_move_orthogonal(M::MPS, center::Int64; cutoff::Float64 = 1E-10, maxdim::Union{Nothing, Int64} = nothing)
    # orthogonalize_right_to_left!(M, center, 1)
    M_trunc = deepcopy(M)
    N = length(M)
    for i in center:N-1
        site_idx, left_idx, right_idx = get_site_and_links(M_trunc[i])
        row_inds = Vector{Index}(undef, 2)
        row_inds[1] = left_idx 
        row_inds[2] = site_idx
        U, S, V, spectrum = svd(M_trunc[i], row_inds; cutoff = cutoff, lefttags = "Link, l = $(i)", maxdim = maxdim)

        M_trunc[i] = U 
        M_trunc[i + 1] = S*V*M_trunc[i + 1]
    end
    orthogonalize_right_to_left!(M_trunc, N, center)
    for i in center:-1:2
        site_idx, left_idx, right_idx = get_site_and_links(M_trunc[i])
        row_inds = Vector{Index}(undef, 1)
        row_inds[1] = left_idx 
        U, S, V, spectrum = svd(M_trunc[i], row_inds; cutoff = cutoff, righttags = "Link, l = $(i - 1)", maxdim = maxdim)

        M_trunc[i] = V 
        M_trunc[i - 1] = M_trunc[i - 1]*U*S 
    end
    orthogonalize_left_to_right!(M_trunc, 1, center)

    return M_trunc 
end

# TDVP2-style truncation: combines adjacent sites via SVD and truncates moving from
# both ends towards the center, finally truncating bonds around the orthogonality center.
function TDVP2_style_truncation(M::MPS, center::Int64; cutoff::Float64 = 1E-10)
    M_trunc = deepcopy(M)
    N = length(M)
    # Left-to-center combine+SVD steps
    for i in 1:center - 2
        M_combine = M_trunc[i]*M_trunc[i + 1]
        M_inds = inds(M_combine)
        if i == 1 
            row_inds = M_inds[1]
        else
            row_inds = M_inds[1:2]
        end
        U, S, V, spectrum = svd(M_combine, row_inds; cutoff = cutoff, lefttags = "Link, l = $(i)")
        M_trunc[i] = U 
        M_trunc[i + 1] = S*V
    end
    # Right-to-center combine+SVD steps
    for i in N:-1:center + 2 
        M_combine = M_trunc[i - 1]*M_trunc[i]
        M_inds = inds(M_combine)
        if i == N 
            row_inds = M_inds[1:2]
        else
            row_inds = M_inds[1:2]
        end
        U, S, V, spectrum = svd(M_combine, row_inds; cutoff = cutoff, righttags = "Link, l = $(i-1)")
        M_trunc[i] = V 
        M_trunc[i-1] = U*S 
    end

    # Truncate left bond next to center
    M_combine_left = M_trunc[center - 1]*M_trunc[center]
    M_inds = inds(M_combine_left)
    if length(M_inds) == 3 
        row_inds = M_inds[1]
    else
        row_inds = M_inds[1:2]
    end
    U, S, V, spectrum = svd(M_combine_left, row_inds; cutoff = cutoff, lefttags = "Link, l = $(center - 1)")
    M_trunc[center - 1] = U 
    M_trunc[center] = S*V 

    # Truncate right bond next to center
    M_combine_right = M_trunc[center]*M_trunc[center + 1]
    M_inds = inds(M_combine_right)

    if length(M_inds) == 3
        row_inds = M_inds[1:2]
    else
        row_inds = M_inds[1:2]
    end
    U, S, V, spectrum = svd(M_combine_right, row_inds; cutoff = cutoff, righttags = "Link, l = $(center)")
    M_trunc[center + 1] = V 
    M_trunc[center] = U*S 

    return M_trunc
end

# Like TDVP2 above but moves orthogonality (via orthogonalize!) before truncation.
function TDVP2_style_truncation_move_orthogonal(M::MPS, center::Int64; cutoff::Float64 = 1E-10)
    M_trunc = deepcopy(M)
    N = length(M)
    orthogonalize!(M_trunc, 1)
    for i in 1:center - 2
        M_combine = M_trunc[i]*M_trunc[i + 1]
        M_inds = inds(M_combine)
        if i == 1 
            row_inds = M_inds[1]
        else
            row_inds = M_inds[1:2]
        end
        U, S, V, spectrum = svd(M_combine, row_inds; cutoff = cutoff, lefttags = "Link, l = $(i)")
        M_trunc[i] = U 
        M_trunc[i + 1] = S*V
    end
    orthogonalize!(M_trunc, N)
    for i in N:-1:center + 2 
        M_combine = M_trunc[i - 1]*M_trunc[i]
        M_inds = inds(M_combine)
        if i == N 
            row_inds = M_inds[1:2]
        else
            row_inds = M_inds[1:2]
        end
        U, S, V, spectrum = svd(M_combine, row_inds; cutoff = cutoff, righttags = "Link, l = $(i-1)")
        M_trunc[i] = V 
        M_trunc[i-1] = U*S 
    end

    M_combine_left = M_trunc[center - 1]*M_trunc[center]
    M_inds = inds(M_combine_left)
    if length(M_inds) == 3 
        row_inds = M_inds[1]
    else
        row_inds = M_inds[1:2]
    end
    U, S, V, spectrum = svd(M_combine_left, row_inds; cutoff = cutoff, lefttags = "Link, l = $(center - 1)")
    M_trunc[center - 1] = U 
    M_trunc[center] = S*V 

    M_combine_right = M_trunc[center]*M_trunc[center + 1]
    M_inds = inds(M_combine_right)

    if length(M_inds) == 3
        row_inds = M_inds[1:2]
    else
        row_inds = M_inds[1:2]
    end
    U, S, V, spectrum = svd(M_combine_right, row_inds; cutoff = cutoff, righttags = "Link, l = $(center)")
    M_trunc[center + 1] = V 
    M_trunc[center] = U*S 

    return M_trunc
end

# Sweep routine: update sites left-to-right until the given center.
# Returns the updated left-block MPS, a projected tensor for the center, and left env L.
function sweep_right(H_mpo, M, h, center)
    N = length(M)
    # new_MPS will store orthonormal Q tensors for the left sweep
    new_MPS = MPS(N)
    # Build right environments up to site 2 (used later in local effective Hamiltonians)
    R_list = contract_right(H_mpo, M, 2)
    L = 1
    M_proj = M[1]
    for i in 1:center - 1 
        # Get site, left, and right indices so matricization is consistent
        site_idx, left_idx, right_idx = get_site_and_links(M_proj)
        # Update site by evolving 1-site with effective Hamiltonian
        M_update = TT_IMR_1site_new(H_mpo, M_proj, L, R_list[i], h, i)
        if i == 1
            # For first site: work with 2-index tensor [site,right] form.
            M_update_arr = Array(M_update, right_idx, site_idx)
            M_old_arr = Array(M_proj, right_idx, site_idx)
            # Concatenate updated and old (transposed to align columns), then orthonormalize
            M_combine = hcat(transpose(M_update_arr), transpose(M_old_arr))
            Q, _ = qr(M_combine)
            row, col = size(M_combine)
            # Keep only first min(row,col) columns
            Q = Q[:,1:min(row, col)]
            new_right_index = Index(min(row, col); tags = "Link, l = 1")
            Q_ten = ITensor(Q, new_right_index, siteinds(M)[1])

        elseif i != 1
            # For interior sites: work with 3-index [left,site,right] tensors.
            M_update_arr = Array(M_update, left_idx, site_idx, right_idx)
            M_proj_arr = Array(M_proj, left_idx, site_idx, right_idx)
            M_proj_mat = matricization(M_proj_arr, 3)
            M_update_mat = matricization(M_update_arr, 3)
            M_combine = hcat(transpose(M_update_mat), transpose(M_proj_mat))

            # Orthonormalize combined matrix via QR
            Q, _ = qr(M_combine)
            row, col = size(M_combine)
            Q = Q[:, 1:min(row, col)]

            # Reshape Q back into tensor with new right link
            new_right_index = Index(min(row, col); tags = "Link, l = $i")
            Q_reshape = reshape(Q, dim(left_idx), dim(site_idx), dim(new_right_index))
            Q_ten = ITensor(Q_reshape, left_idx, site_idx, new_right_index)
        end 

        # Update left environment by contracting the MPO with the new site
        L *= H_mpo[i]*Q_ten*conj(Q_ten)'
        # Store orthonormal Q in new_MPS
        new_MPS[i] = Q_ten
        
        # Update M_proj (the tensor to be fed into next step)
        if i < center - 1
            M_proj *= conj(new_MPS[i])*M[i + 1]
        elseif i == center - 1
            M_proj *= conj(new_MPS[i])
        end
          
    end
    if center == 1
        return new_MPS, 1, L 
    else
        return new_MPS, M_proj, L 
    end
end

# Sweep routine: update sites right-to-left until the given center.
# Returns the updated right-block MPS, a projected tensor for the center, and right env R.
function sweep_left(H::MPO, M::MPS, h::Float64, center::Int64)
    N = length(M)
    L_list = contract_left(H, M, N - 1)
    R_block = 1
    new_MPS = MPS(N)
    M_proj = M[N]
    for i in N:-1:center + 1
        # Get indices for current projection tensor
        site_idx, left_idx, right_idx = get_site_and_links(M_proj)
        # Compute evolved tensor for site i using effective Hamiltonian
        M_evolve = TT_IMR_1site_new(H, M_proj, L_list[i], R_block, h, i)
        
        # Matricize and orthonormalize depending on number of indices
        if length(inds(M_proj)) == 2
            M_mat = Array(M_proj, left_idx, site_idx)
            M_evolve_mat = Array(M_evolve, left_idx, site_idx)
            M_combine = hcat(transpose(M_evolve_mat), transpose(M_mat))
            Q, R = qr(M_combine)
            
            row, col = size(M_combine)
            Q = Q[1:row, 1:min(row, col)]
            new_link = Index(min(row, col); tags="Link, l = $(i - 1)")
            Q_ten = ITensor(Q, siteinds(M)[i], new_link)
        end

        if length(inds(M_proj)) == 3
            M_arr = Array(M_proj, left_idx, site_idx, right_idx)
            M_evolve_arr = Array(M_evolve, left_idx, site_idx, right_idx)
            M_mat = matricization(M_arr, 1)
            M_evolve_mat = matricization(M_evolve_arr, 1)
            M_combine = vcat(M_evolve_mat, M_mat)

            # QR on transpose then appropriately conjugate to preserve right-orthogonality
            Q, R = qr(M_combine')
            row, col = size(M_combine')
            Q = Q[:, 1:min(row, col)]
            Q = transpose(conj(Q))

            row, col = size(Q)
            new_left_idx = Index(min(row, col); tags = "Link, l = $(i - 1)")
            Q = Array(reshape(Q, dim(new_left_idx), dim(site_idx), dim(right_idx)))

            Q_ten = ITensor(Q, new_left_idx, site_idx, right_idx)
        
        end
        # Update right environment and store orthonormal Q in new_MPS
        R_block *= H[i]*Q_ten*conj(Q_ten)'
        new_MPS[i] = Q_ten
        # Update M_proj for next iteration
        if i > center + 1
            M_proj *= conj(new_MPS[i])*M[i - 1]
        elseif i == center + 1
            M_proj *= conj(new_MPS[i])
        end

    end
    if center == N 
        return new_MPS, 1, R_block 
    else
        return new_MPS, M_proj, R_block 
    end
end


# Perform one full BUG-MPS integrator step:
# - sweep right updating left block
# - sweep left updating right block
# - update center with combined projected environment
# Return the updated MPS.
function mps_bug_step(H_mpo, M, h, center)
    N = length(M)
    # sweep-right: get updated left sites, left projection, left environment
    M_l, M_l_proj, L_block = sweep_right(H_mpo, M, h, center)

    # sweep-left: get updated right sites, right projection, right environment
    M_r, M_r_proj, R_block = sweep_left(H_mpo, M, h, center)
    # Build the projected center tensor from left and right projected pieces
    center_proj = M_l_proj*M[center]*M_r_proj
    # Update center site using 1-site integrator with the left/right environments
    center_update = TT_IMR_1site_new(H_mpo, center_proj, L_block, R_block, h, center)
    updated_MPS = MPS(N)
    # Assemble updated MPS from left updates, right updates, and updated center
    for i in 1:center - 1
        updated_MPS[i] = M_l[i]
    end

    for i in N:-1:center + 1
        updated_MPS[i] = M_r[i]
    end

    updated_MPS[center] = center_update 

    return updated_MPS 
end


"""
mps_bug_constant(H::MPO, M::MPS, t0::Real, T::Real, steps::Int64;
                 center::Union{Nothing,Int64}=nothing,
                 cutoff::Union{Nothing,Float64}=nothing,
                 maxdim::Union{Nothing,Int64}=nothing,
                 magnet::Bool=false,
                 energy::Bool=false,
                 verbose::Bool=false)

Evolve an MPS in time under an MPO using the mps_bug_constant routine.

Arguments
- H::MPO
  The Hamiltonian (or more generally the time-evolution generator) represented as a Matrix Product Operator.
  Must be compatible with the lattice and local physical dimensions of M.

- M::MPS
  The initial Matrix Product State to be evolved. Should match H in number of sites and physical dimensions.

- t0::Real
  The initial time of the evolution.

- T::Real
  The final time to which the state should be evolved.

- steps::Int64
  The number of equal time steps between t0 and T. The time step used is (T - t0) / steps.

Keyword arguments
- center::Union{Nothing,Int64} = nothing
  Optional orthogonality-center site index for the MPS. If given, the algorithm will treat this site as the canonical center.
  If nothing, the routine will set the midpoint as the center by default.

- cutoff::Union{Nothing,Float64} = nothing
  Singular-value cutoff used when truncating bond dimensions during evolution.
  Singular values below this threshold are discarded. If nothing, truncation is disabled or a library default is applied.
  Typical values are very small (e.g. 1e-15) when high accuracy is required.

- maxdim::Union{Nothing,Int64} = nothing
  Maximum allowed bond dimension during truncation. If nothing, no explicit cap is imposed beyond algorithmic or memory limits.
  Use this to limit memory/compute when truncation alone is insufficient.

- magnet::Bool = false
  If true, compute and return magnetization (local spin/particle expectation values) at each time step as part of diagnostics.

- energy::Bool = false
  If true, compute and return the energy expectation value at each time step.

- verbose::Bool = false
  If true, print progress information and diagnostics during the time evolution to assist with monitoring and debugging.

"""
function mps_bug_constant(H::MPO, M::MPS, t0::Real, T::Real, steps::Int64; center::Union{Nothing,Int64} = nothing, cutoff::Union{Nothing,Float64} = nothing, maxdim::Union{Nothing,Int64} = nothing, magnet::Bool = false, energy::Bool = false, verbose::Bool = false)
    h = (T - t0)/steps 
    M_copy = deepcopy(M)
    N = length(M)
    if center == nothing 
        center = Int64(ceil(N/2))

    end

    # Preallocate histories for diagnostics
    magnet_history = zeros(steps + 1, N)
    energy_history = zeros(steps + 1)
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(M_copy)
    if magnet == true 
        magnet_history[1,:] = (expect(M_copy, [1 0; 0 -1]))
    end
    if energy == true 
        energy_history[1] = real(inner(M_copy', H, M_copy))
    end 

    @showprogress 1 "BUG for Tensor-trains" for i in 1:steps 
        M_copy = mps_bug_step(H, M_copy, h, center)
        # Apply truncation policy if requested
        if cutoff != nothing
            M_copy = TDVP1_style_truncation_in_move_orthogonal(M_copy, center; cutoff = cutoff, maxdim = maxdim)
        else
            M_copy = TDVP1_style_truncation_in_move_orthogonal(M_copy, center; cutoff = 1E-15, maxdim = maxdim)
        end
        link_dim[i + 1, :] = linkdims(M_copy)
        
        if magnet == true 
            magnet_history[i + 1,:] = (expect(M_copy, [1 0; 0 -1]))
        end
        if energy == true 
            energy_history[i + 1] = real(inner(M_copy', H, M_copy))
        end
        link_dim[i + 1,:] = linkdims(M_copy)
        if verbose == true 
            println("Step $i")
            println("Bond Dimensions: ", linkdims(M_copy))
        end
    end
    return M_copy, link_dim, magnet_history, energy_history
end


"""
mps_bug(H::MPO, bc_params::bcparams, M::MPS, t0::Real, T::Real, steps::Int64;
                 center::Union{Nothing,Int64}=nothing,
                 cutoff::Union{Nothing,Float64}=nothing,
                 maxdim::Union{Nothing,Int64}=nothing,
                 magnet::Bool=false,
                 energy::Bool=false,
                 verbose::Bool=false)

Evolve an MPS under a Hamiltonian MPO using the BUG-MPS integrator.

Evolve a matrix product state (MPS) in time under a matrix product operator (MPO) Hamiltonian,
sampling diagnostics along the way.

Arguments
- H::MPO
    The Hamiltonian expressed as an MPO that generates the time evolution.
- bc_params::bcparams
    Boundary-condition parameters or other auxiliary data required by the evolution routine.
- M::MPS
    The initial state provided as an MPS. This state will be evolved from time `t0` to `T`.
- t0::Real
    Initial time of the evolution.
- T::Real
    Final time of the evolution.
- steps::Int64
    Number of time steps to take between `t0` and `T`. The times at which diagnostics are
    sampled are determined by this parameter.

Keyword arguments
- center::Union{Nothing, Int64} = nothing
    Optional site index to enforce or use as the orthogonality center of the MPS. When `nothing`,
    the routine may choose or preserve the current center.
- cutoff::Union{Nothing, Float64} = nothing
    Singular value truncation threshold. If provided, singular values below `cutoff` are discarded
    when truncating bond dimensions during the update steps. A `nothing` value disables truncation
    by threshold.
- maxdim::Union{Nothing, Float64} = nothing
    Maximum allowed bond dimension. If provided, bond dimensions are capped at `maxdim` during
    truncation. A `nothing` value disables a strict cap.
- magnet::Bool = false
    If true, compute and record magnetization (or other local observable specified by the
    implementation) at each sampled time.
- energy::Bool = false
    If true, compute and record the expectation value of the Hamiltonian (energy) at each sampled time.
- verbose::Bool = false
    Toggle verbose logging or progress output to aid debugging or monitor the evolution.

Returns
A tuple typically containing:
- evolved MPS at final time,
- bond-dimension history,
- magnetization history (if requested),
- energy history (if requested),lementation-specific scalars

"""
function mps_bug(H::MPO, bc_params::bcparams, M::MPS, t0::Real, T::Real, steps::Int64; center::Union{Nothing, Int64} = nothing, cutoff::Union{Nothing, Float64}=nothing, maxdim::Union{Nothing, Float64}=nothing, magnet::Bool=false, energy::Bool=false, verbose::Bool=false)
    h = (T - t0)/steps 
    M_copy = deepcopy(M)
    N = length(M)
    if center == nothing 
        center = Int64(ceil(N/2))

    end
    magnet_history = zeros(steps + 1, N)
    energy_history = zeros(steps + 1)
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(M_copy)
    if magnet == true 
        magnet_history[1,:] = (expect(M_copy, [1 0; 0 -1]))
    end
    if energy == true
        energy_history[1] = real(inner(M_copy', H, M_copy))
    end
    @showprogress 1 "BUG for Tensor-trains" for i in 1:steps 
        # Update time-dependent MPO (boundary conditions / time dependency)
        update_MPO!(H, bc_params, t0 + h/2)
        M_copy = mps_bug_step(H, M_copy, h, center)
        t0 += h
        # Optionally apply truncation policies
        if cutoff != nothing
            M_copy = TDVP1_style_truncation_out_move_orthogonal(M_copy, center; cutoff = cutoff, maxdim = maxdim)
        end
        if maxdim != nothing 
            M_copy = TDVP1_style_truncation_out_move_orthogonal(M_copy, center; cutoff = 1E-15, maxdim = maxdim)
        end
        link_dim[i + 1, :] = linkdims(M_copy)
        if magnet == true 
            magnet_history[i + 1,:] = (expect(M_copy, [1 0; 0 -1]))
        end
        if energy == true 
            energy_history[i + 1] = real(inner(M_copy', H, M_copy))
        end
        link_dim[i + 1,:] = linkdims(M_copy)
        if verbose == true 
            println("Step $i")
            println("Bond Dimensions: ", linkdims(M_copy))
        end
    end
    return M_copy, link_dim, magnet_history, energy_history
end
