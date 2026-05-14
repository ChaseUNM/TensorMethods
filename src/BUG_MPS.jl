using ITensors, ITensorMPS, LinearAlgebra 

#returns maximum dimension of the left and right site indices for a quantum system where each subsystem has 2 energy levels
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

#Helper functions to get site index, left link index, and right link index. 
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

    # 2. Classify link indices
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

function TDVP1_style_truncation(M::MPS, center::Int64; cutoff::Float64 = 1E-10)
    M_trunc = deepcopy(M)
    N = length(M)

    for i in 1:center - 1
        site_idx, left_idx, right_idx = get_site_and_links(M_trunc[i])
        # println("M_trunc[$(i)] inds: ", inds(M_trunc[i]))
        # println("M_trunc[$(i + 1)] inds: ", inds(M_trunc[i + 1]))
        if i == 1 
            row_inds = Vector{Index}(undef, 1)
            row_inds[1] = site_idx
        else
            row_inds = Vector{Index}(undef, 2)
            row_inds[1] = left_idx 
            row_inds[2] = site_idx
        end
        U, S, V, spectrum = svd(M_trunc[i], row_inds; cutoff = cutoff, lefttags = "Link, l = $(i)")
        # println("U inds", inds(U))
        # println("S inds: ", inds(S))
        # println("V inds: ", inds(V))
        
        M_trunc[i] = U 
        M_trunc[i + 1] = M_trunc[i + 1]*S*V
    end

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

function orthogonalize_right_to_left!(M::MPS, start_site::Int64, end_site::Int64)

    for i in start_site:-1:end_site + 1
        M_site = M[i]
        site_idx, left_idx, right_idx = get_site_and_links(M_site)
        R, Q = factorize(M_site, left_idx; ortho="right", tags = "Link, l = $(i-1)")
        M[i] = Q 
        M[i - 1] = M[i - 1]*R
    end
end

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

function TDVP1_style_truncation_move_orthogonal(M::MPS, center::Int64; cutoff::Float64 = 1E-10)
    orthogonalize_right_to_left!(M, center, 1)
    M_trunc = deepcopy(M)
    N = length(M)
    
    for i in 1:center - 1
        site_idx, left_idx, right_idx = get_site_and_links(M_trunc[i])
        # println("M_trunc[$(i)] inds: ", inds(M_trunc[i]))
        # println("M_trunc[$(i + 1)] inds: ", inds(M_trunc[i + 1]))
        if i == 1 
            row_inds = Vector{Index}(undef, 1)
            row_inds[1] = site_idx
        else
            row_inds = Vector{Index}(undef, 2)
            row_inds[1] = left_idx 
            row_inds[2] = site_idx
        end
        U, S, V, spectrum = svd(M_trunc[i], row_inds; cutoff = cutoff, lefttags = "Link, l = $(i)")
        # println("U inds", inds(U))
        # println("S inds: ", inds(S))
        # println("V inds: ", inds(V))
        
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
        U, S, V, spectrum = svd(M_trunc[i], row_inds; cutoff = cutoff, righttags = "Link, l = $(i-1)")
        M_trunc[i] = V 
        M_trunc[i-1] = M_trunc[i - 1]*U*S 
    
    end

    return M_trunc 
end

function TDVP2_style_truncation(M::MPS, center::Int64; cutoff::Float64 = 1E-10)
    M_trunc = deepcopy(M)
    N = length(M)
    # M_copy = deepcopy(M)
    # spectrum_list = []
    for i in 1:center - 2
        M_combine = M_trunc[i]*M_trunc[i + 1]
        M_inds = inds(M_combine)
        if i == 1 
            row_inds = M_inds[1]
        else
            row_inds = M_inds[1:2]
        end
        U, S, V, spectrum = svd(M_combine, row_inds; cutoff = cutoff, lefttags = "Link, l = $(i)")
        # println("Bond $i-$(i + 1) Spectrum: ")
        # display(spectrum.eigs)
        # push!(spectrum_list, spectrum)
        M_trunc[i] = U 
        M_trunc[i + 1] = S*V
    end
    for i in N:-1:center + 2 
        M_combine = M_trunc[i - 1]*M_trunc[i]
        M_inds = inds(M_combine)
        # println("right inds for site $i: ", M_inds)
        if i == N 
            row_inds = M_inds[1:2]
        else
            row_inds = M_inds[1:2]
        end
        U, S, V, spectrum = svd(M_combine, row_inds; cutoff = cutoff, righttags = "Link, l = $(i-1)")
        # println("Bond $i-$(i + 1) Spectrum: ")
        # display(spectrum.eigs)
        # push!(spectrum_list, spectrum)
        M_trunc[i] = V 
        M_trunc[i-1] = U*S 
    end

    #Now truncate the bond dimensions to the left and right of the orthogonality center starting with left bond dimension
    M_combine_left = M_trunc[center - 1]*M_trunc[center]
    M_inds = inds(M_combine_left)
    if length(M_inds) == 3 
        row_inds = M_inds[1]
    else
        row_inds = M_inds[1:2]
    end
    U, S, V, spectrum = svd(M_combine_left, row_inds; cutoff = cutoff, lefttags = "Link, l = $(center - 1)")
    # println("Site $(center-1)-$center Spectrum: ")
    # display(spectrum.eigs)
    # push!(spectrum_list, spectrum)
    M_trunc[center - 1] = U 
    M_trunc[center] = S*V 

    #Now right bond dimension 
    M_combine_right = M_trunc[center]*M_trunc[center + 1]
    M_inds = inds(M_combine_right)

    if length(M_inds) == 3
        row_inds = M_inds[1:2]
    else
        row_inds = M_inds[1:2]
    end
    U, S, V, spectrum = svd(M_combine_right, row_inds; cutoff = cutoff, righttags = "Link, l = $(center)")
    # println("Site $center-$(center + 1) Spectrum: ")
    # display(spectrum.eigs)
    # push!(spectrum_list, spectrum)
    M_trunc[center + 1] = V 
    M_trunc[center] = U*S 

    return M_trunc
end

function TDVP2_style_truncation_move_orthogonal(M::MPS, center::Int64; cutoff::Float64 = 1E-10)
    M_trunc = deepcopy(M)
    N = length(M)
    # M_copy = deepcopy(M)
    # spectrum_list = []
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
        # push!(spectrum_list, spectrum)
        M_trunc[i] = U 
        M_trunc[i + 1] = S*V
    end
    orthogonalize!(M_trunc, N)
    for i in N:-1:center + 2 
        M_combine = M_trunc[i - 1]*M_trunc[i]
        M_inds = inds(M_combine)
        # println("right inds for site $i: ", M_inds)
        if i == N 
            row_inds = M_inds[1:2]
        else
            row_inds = M_inds[1:2]
        end
        U, S, V, spectrum = svd(M_combine, row_inds; cutoff = cutoff, righttags = "Link, l = $(i-1)")
        # push!(spectrum_list, spectrum)
        M_trunc[i] = V 
        M_trunc[i-1] = U*S 
    end

    #Now truncate the bond dimensions to the left and right of the orthogonality center starting with left bond dimension
    M_combine_left = M_trunc[center - 1]*M_trunc[center]
    M_inds = inds(M_combine_left)
    if length(M_inds) == 3 
        row_inds = M_inds[1]
    else
        row_inds = M_inds[1:2]
    end
    U, S, V, spectrum = svd(M_combine_left, row_inds; cutoff = cutoff, lefttags = "Link, l = $(center - 1)")
    # push!(spectrum_list, spectrum)
    M_trunc[center - 1] = U 
    M_trunc[center] = S*V 

    #Now right bond dimension 
    M_combine_right = M_trunc[center]*M_trunc[center + 1]
    M_inds = inds(M_combine_right)

    if length(M_inds) == 3
        row_inds = M_inds[1:2]
    else
        row_inds = M_inds[1:2]
    end
    U, S, V, spectrum = svd(M_combine_right, row_inds; cutoff = cutoff, righttags = "Link, l = $(center)")
    # push!(spectrum_list, spectrum)
    M_trunc[center + 1] = V 
    M_trunc[center] = U*S 

    return M_trunc
end

function sweep_right(H_mpo, M, h, center)
    #Create new MPS to store updated sites
    N = length(M)
    new_MPS = MPS(N)
    #Create left and right environments for effective Hamiltonian
    R_list = contract_right(H_mpo, M, 2)
    L = 1
    M_proj = M[1]
    for i in 1:center - 1 
        #Get site, left, and right indices so things will get matricized correctly
        site_idx, left_idx, right_idx = get_site_and_links(M_proj)
        #Update site
        M_update = TT_IMR_1site_new(H_mpo, M_proj, L, R_list[i], h, i)
        if i == 1
            #Matricize updated and old tensors
            M_update_arr = Array(M_update, right_idx, site_idx)
            M_old_arr = Array(M_proj, right_idx, site_idx)
            #Concatenate the matrices and then perform an orthogonalization factorization (either QR or SVD)
            M_combine = hcat(transpose(M_update_arr), transpose(M_old_arr))
            Q, _ = qr(M_combine)
            # Q, _ = LLSV(M_combine)
            row, col = size(M_combine)
            #Convert Q back into an ITensor
            Q = Q[:,1:min(row, col)]
            new_right_index = Index(min(row, col); tags = "Link, l = 1")
            Q_ten = ITensor(Q, new_right_index, siteinds(M)[1])

        elseif i != 1
            #Matricize the updated and old tensors, and then concatenate
            M_update_arr = Array(M_update, left_idx, site_idx, right_idx)
            M_proj_arr = Array(M_proj, left_idx, site_idx, right_idx)
            M_proj_mat = matricization(M_proj_arr, 3)
            M_update_mat = matricization(M_update_arr, 3)
            M_combine = hcat(transpose(M_update_mat), transpose(M_proj_mat))

            #Orthogonalize M_combine, either using QR or SVD

            Q, _ = qr(M_combine)
            row, col = size(M_combine)
            Q = Q[:, 1:min(row, col)]
            # Q, _ = LLSV(M_combine)
            # row, col = size(Q)

            #Reshape Q into a tensor
            new_right_index = Index(min(row, col); tags = "Link, l = $i")
            Q_reshape = reshape(Q, dim(left_idx), dim(site_idx), dim(new_right_index))
            Q_ten = ITensor(Q_reshape, left_idx, site_idx, new_right_index)
        end 

        #Update left environment for effective Hamiltonian 
        L *= H_mpo[i]*Q_ten*conj(Q_ten)'
        #Set i-th site in new MPS to be Q
        new_MPS[i] = Q_ten
        
        #Update initial conditions
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

function sweep_left(H::MPO, M::MPS, h::Float64, center::Int64)
    #Start with updating right-to-left until we get to the orthogonality center
    #Create new MPS to store updated sites, and create left and right environments
    #for effective Hamiltonain
    N = length(M)
    L_list = contract_left(H, M, N - 1)
    R_block = 1
    new_MPS = MPS(N)
    M_proj = M[N]
    for i in N:-1:center + 1
        #Get site, left, and right indices

        site_idx, left_idx, right_idx = get_site_and_links(M_proj)
        #Update the i-th core
        M_evolve = TT_IMR_1site_new(H, M_proj, L_list[i], R_block, h, i)
        
        #Now need to matricize M_evolve and M_proj
        if length(inds(M_proj)) == 2
            #Matricize old and updated sites
            M_mat = Array(M_proj, left_idx, site_idx)
            M_evolve_mat = Array(M_evolve, left_idx, site_idx)
            #Concatenate old and updated matrices and perform an orthogonalization (QR or SVD)
            M_combine = hcat(transpose(M_evolve_mat), transpose(M_mat))
            Q, R = qr(M_combine)
            
            #Convert Q back into a ITensor
            row, col = size(M_combine)
            Q = Q[1:row, 1:min(row, col)]
            new_link = Index(min(row, col); tags="Link, l = $(i - 1)")
            Q_ten = ITensor(Q, siteinds(M)[i], new_link)
        end

        if length(inds(M_proj)) == 3
            #Matricize old and updated sites
            M_arr = Array(M_proj, left_idx, site_idx, right_idx)
            M_evolve_arr = Array(M_evolve, left_idx, site_idx, right_idx)
            M_mat = matricization(M_arr, 1)
            M_evolve_mat = matricization(M_evolve_arr, 1)
            #Concate old and updated matricized sites
            M_combine = vcat(M_evolve_mat, M_mat)

            #Perform an orthogonalization of M_combine (either QR or SVD)
            #Note: if using QR decomposition in order to maintain right-orthogonality a QR decomposition should be done
            # on the conjugate transpose M_combine and then Q should be conjugate transposed back
            Q, R = qr(M_combine')
            # _, _, Q = svd(M_combine)
            row, col = size(M_combine')
            Q = Q[:, 1:min(row, col)]
            Q = transpose(conj(Q))

            # Q, _ = RLSV(M_combine)

            #Convert back into an ITensor object.
            row, col = size(Q)
            new_left_idx = Index(min(row, col); tags = "Link, l = $(i - 1)")
            Q = Array(reshape(Q, dim(new_left_idx), dim(site_idx), dim(right_idx)))

            Q_ten = ITensor(Q, new_left_idx, site_idx, right_idx)
        
        end
        #Update right environment
        R_block *= H[i]*Q_ten*conj(Q_ten)'
        #Set new site to be Q
        new_MPS[i] = Q_ten
        #Update initial condition for next site update
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


function mps_bug_step(H_mpo, M, h, center)
    N = length(M)
    #sweep-right to return updated left sites
    M_l, M_l_proj, L_block = sweep_right(H_mpo, M, h, center)

    M_r, M_r_proj, R_block = sweep_left(H_mpo, M, h, center)
    #Get initial conditions for center update
    center_proj = M_l_proj*M[center]*M_r_proj
    #Update center site
    center_update = TT_IMR_1site_new(H_mpo, center_proj, L_block, R_block, h, center)
    updated_MPS = MPS(N)
    #In new MPS set sites using M_l, M_r, and center_update
    for i in 1:center - 1
        updated_MPS[i] = M_l[i]
    end

    for i in N:-1:center + 1
        updated_MPS[i] = M_r[i]
    end

    updated_MPS[center] = center_update 

    return updated_MPS 
end

function mps_bug_constant(H, M, t0, T, steps ; center::Union{Nothing,Int64} = nothing, cutoff::Union{Nothing,Float64} = nothing, maxdim::Union{Nothing,Int64} = nothing, magnet::Bool = false, energy::Bool = false, verbose::Bool = false)
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
        # println("Step $i")
        M_copy = mps_bug_step(H, M_copy, h, center)
        # display(M_copy)
        if cutoff != nothing
            # M_copy = TDVP2_style_truncation(M_copy, center; cutoff = cutoff)
            M_copy = TDVP1_style_truncation_move_orthogonal(M_copy, center; cutoff = cutoff)
            # truncate!(M_copy; cutoff = cutoff)
        else
            # M_copy = TDVP2_style_truncation(M_copy, center; cutoff = 1E-15)
            M_copy = TDVP1_style_truncation_move_orthogonal(M_copy, center; cutoff = 1E-15)
            # truncate!(M_copy; cutoff = 1E-15)
        end
        # display(M_copy)
        # if maxdim == nothing 
        #     M_copy = TDVP2_style_truncation(M_copy, center; cutoff = 1E-15)
        # end
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

function mps_bug(H::MPO, bc_params::bcparams, M::MPS, t0::Float64, T::Float64, steps::Int64; center::Union{Nothing, Int64} = nothing, cutoff::Union{Nothing, Float64}=nothing, maxdim::Union{Nothing, Float64}=nothing, magnet::Bool=false, energy::Bool=false)
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
        update_MPO!(H, bc_params, t0 + h/2)
        M_copy = mps_bug_step(H, M_copy, h, center)
        t0 += h
        if cutoff != nothing
            # truncate!(M_copy; cutoff = cutoff)
            M_copy = TDVP1_style_truncation_move_orthogonal(M_copy, center; cutoff = cutoff)
        end
        if maxdim != nothing 
            # truncate!(M_copy; maxdim = maxdim)
            M_copy = TDVP1_style_truncation_move_orthogonal(M_copy, center; cutoff = 1E-15)
        end
        link_dim[i + 1, :] = linkdims(M_copy)
        if magnet == true 
            magnet_history[i + 1,:] = (expect(M_copy, [1 0; 0 -1]))
        end
        if energy == true 
            energy_history[i + 1] = real(inner(M_copy', H, M_copy))
        end
        link_dim[i + 1,:] = linkdims(M_copy)
    end
    return M_copy, link_dim, magnet_history, energy_history
end