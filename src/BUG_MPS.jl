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

function sweep_right(H_mpo, M, h, center)
    N = length(M)
    new_MPS = MPS(N)
    R_list = contract_right(H_mpo, M, 2)
    L = 1
    M_proj = M[1]
    for i in 1:center - 1 
        site_idx, left_idx, right_idx = get_site_and_links(M_proj)
        # println(init_MPS[i])
        # println(L)
        # println(R_list[i])
        M_update = TT_IMR_1site_new(H_mpo, M_proj, L, R_list[i], h, i)
        # println("Updated Site $i")
        # max_left, max_right = max_bond_dimension(i, N)
        if i == 1
            M_update_arr = Array(M_update, right_idx, site_idx)
            M_old_arr = Array(M_proj, right_idx, site_idx)
            #Concatenate the matrices and then perform a QR factorization 
            M_combine = hcat(transpose(M_update_arr), transpose(M_old_arr))
            # println("M_combine: ")
            # display(M_combine)
            # Q, _ = qr(M_combine)
            Q, _ = LLSV(M_combine)
            row, col = size(M_combine)
            Q = Q[:,1:min(row, col)]
            # println("Q")
            # display(Q)
            new_right_index = Index(min(row, col); tags = "Link, l = 1")
            Q_ten = ITensor(Q, new_right_index, siteinds(M)[1])
            # println(Q_ten)
        elseif i != 1
            M_update_arr = Array(M_update, left_idx, site_idx, right_idx)
            M_proj_arr = Array(M_proj, left_idx, site_idx, right_idx)
            M_proj_mat = matricization(M_proj_arr, 3)
            M_update_mat = matricization(M_update_arr, 3)
            # M_combine = hcat(transpose(M_update_mat), transpose(M_proj_mat))
            # M_proj_mat = reshape(M_proj_arr, dim(right_idx), dim(site_idx)*dim(left_idx))
            # M_update_mat = reshape(M_update_arr, dim(right_idx), dim(site_idx)*dim(left_idx))
            M_combine = hcat(transpose(M_update_mat), transpose(M_proj_mat))

            # Q, _ = qr(M_combine)
            # println("max_right: ", max_right)
            # println("Size M_combine: ")
            # println(size(M_combine))
            Q, _ = LLSV(M_combine)
            # println("Size Q: ")
            # println(size(Q))
            row, col = size(Q)
            # row, col = size(M_combine)
            # Q = Q[:, 1:min(row, col)]
            new_right_index = Index(min(row, col); tags = "Link, l = $i")
            Q_reshape = reshape(Q, dim(left_idx), dim(site_idx), dim(new_right_index))
            Q_ten = ITensor(Q_reshape, left_idx, site_idx, new_right_index)
        end 
        L *= H_mpo[i]*Q_ten*conj(Q_ten)'
        new_MPS[i] = Q_ten
        if i < center - 1
            M_proj *= conj(new_MPS[i])*M[i + 1]
        elseif i == center - 1
            M_proj *= conj(new_MPS[i])
        end
          
        # println(M_proj)
    end
    if center == 1
        return new_MPS, 1, L 
    else
        return new_MPS, M_proj, L 
    end
end

function sweep_left(H::MPO, M::MPS, h::Float64, center::Int64)
    #Start with updating right-to-left until we get to the orthogonality center
    N = length(M)
    L_list = contract_left(H, M, N - 1)
    R_block = 1
    new_MPS = MPS(N)
    M_proj = M[N]
    for i in N:-1:center + 1
        #Update the i-th core
        site_idx, left_idx, right_idx = get_site_and_links(M_proj)
        # max_left, max_right = max_bond_dimension(i, N)
        # println("M_proj")
        # println(M_proj)
        # println("L_list[$i]")
        # println(L_list[i])
        # println("R_block")
        # println(R_block)
        M_evolve = TT_IMR_1site_new(H, M_proj, L_list[i], R_block, h, i)
        # println("Site $i updated from right to left")
        # println(M_copy[i])
        # println(M_evolve)
        #Now need to matricize M_evolve and M[i]
        if length(inds(M_proj)) == 2
            M_mat = Array(M_proj, left_idx, site_idx)
            M_evolve_mat = Array(M_evolve, left_idx, site_idx)
            
            M_combine = hcat(transpose(M_evolve_mat), transpose(M_mat))
            # M_combine = hcat(M_evolve_mat)
            Q, R = qr(M_combine)
            
            # println("M combine")
            # display(M_combine)

            row, col = size(M_combine)

            Q = Q[1:row, 1:min(row, col)]
            # println("Q")
            # println(Q)
            # Q = Q[1:row, 1:min(row, col)]
            new_link = Index(min(row, col); tags="Link, l = $(i - 1)")
            # println("Got here")
            # println(siteinds(M)[i])
            # println(new_link)
            # display(Q)
            Q_ten = ITensor(Q, siteinds(M)[i], new_link)
        end

        if length(inds(M_proj)) == 3
            # site_idx, left_idx, right_idx = get_site_and_links(M_proj)
            M_arr = Array(M_proj, left_idx, site_idx, right_idx)
            M_evolve_arr = Array(M_evolve, left_idx, site_idx, right_idx)
            # M_arr2 = Array(M_copy[i], site_idx, right_idx, left_idx)
            # M_evolve_arr2 = Array(M_evolve, site_idx, right_idx, left_idx)
            # M_mat = reshape(M_arr, dim(left_idx), dim(site_idx)*dim(right_idx))
            # M_evolve_mat = reshape(M_evolve_arr, dim(left_idx), dim(site_idx)*dim(right_idx))
            M_mat = matricization(M_arr, 1)
            M_evolve_mat = matricization(M_evolve_arr, 1)
            # M_mat2 = reshape(M_arr2, dim(right_idx)*dim(site_idx), dim(left_idx))
            # M_evolve_mat2 = reshape(M_evolve_arr2, dim(right_idx)*dim(site_idx), dim(left_idx))
            # println("Size M_Mat2: ", size(M_mat2))
            # M_combine = hcat(transpose(M_evolve_mat), transpose(M_mat))
            # if size(M_mat, 1) >= 2^(i - 1)
            #     M_combine = M_mat 
            # else
            #     M_combine = vcat(M_evolve_mat, M_mat)
            # end
            # if dim(left_idx) < max_left
            #     M_combine = vcat(M_evolve_mat, M_mat)
            # else 
            #     M_combine = vcat(M_evolve_mat)
            # end
            M_combine = vcat(M_evolve_mat, M_mat)
            # M_combine = hcat(M_evolve_mat2, M_mat2)

            # Q, R = qr(M_combine')
            # _, _, Q = svd(M_combine)
            Q, _ = RLSV(M_combine)
            # display(Q1)
            # display(Q2)
            # println("Size of M_combine:", size(M_combine))
            
            # println(new_left_idx)
            # row, col = size(M_combine')
            # Q = Q[:, 1:min(row, col)]

            # Q = transpose(conj(Q))
            row, col = size(Q)
            new_left_idx = Index(min(row, col); tags = "Link, l = $(i - 1)")
            # println("Size of Q: ", size(Q))
            # println("size new_left_index: ", dim(new_left_idx))
            # println("size right_index: ", dim(right_idx))
            # println("size site_index: ", dim(site_idx))
            Q = Array(reshape(Q, dim(new_left_idx), dim(site_idx), dim(right_idx)))

            Q_ten = ITensor(Q, new_left_idx, site_idx, right_idx)
            
            # println("Got here")
        end
        R_block *= H[i]*Q_ten*conj(Q_ten)'
        new_MPS[i] = Q_ten
        if i > center + 1
            M_proj *= conj(new_MPS[i])*M[i - 1]
        elseif i == center + 1
            M_proj *= conj(new_MPS[i])
        end
        # M_copy[i-1] = M_copy[i-1]*M_copy[i]*conj(Q_ten)
        # M_copy[i] = Q_ten
    end
    if center == N 
        return new_MPS, 1, R_block 
    else
        return new_MPS, M_proj, R_block
    end
end


function mps_bug_step(H_mpo, M, h, center)
    N = length(M)
    M_l, M_l_proj, L_block = sweep_right(H_mpo, M, h, center)
    M_r, M_r_proj, R_block = sweep_left(H_mpo, M, h, center)
    center_proj = M_l_proj*M[center]*M_r_proj
    center_update = TT_IMR_1site_new(H_mpo, center_proj, L_block, R_block, h, center)

    updated_MPS = MPS(N)
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
        # center = 1
    end
    println("Center : $center")
    magnet_history = zeros(steps + 1, N)
    energy_history = zeros(steps + 1)
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(M_copy)
    if magnet == true 
        magnet_history[1,:] = reverse(expect(M_copy, [1 0; 0 -1]))
    end
    if energy == true 
        energy_history[1] = real(inner(M_copy', H, M_copy))
    end 

    @showprogress 1 "BUG for Tensor-trains" for i in 1:steps 
        # println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # println("Step $i")
        # println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        M_copy = mps_bug_step(H, M_copy, h, center)
        # println("M[center]")
        # println(M[center])
        if cutoff != nothing
            truncate!(M_copy; cutoff = cutoff)
        end
        if maxdim == nothing 
            truncate!(M_copy; maxdim = 2^Int64(floor(N/2)))
        end
        link_dim[i + 1, :] = linkdims(M_copy)
        if magnet == true 
            magnet_history[i + 1,:] = reverse(expect(M_copy, [1 0; 0 -1]))
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
        # center = Int64(ceil(N/2))
        center = 1
    end
    magnet_history = zeros(steps + 1, N)
    energy_history = zeros(steps + 1)
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(M_copy)
    if magnet == true 
        magnet_history[1,:] = reverse(expect(M_copy, [1 0; 0 -1]))
    end
    if energy == true 
        energy_history[1] = real(inner(M_copy', H, M_copy))
    end
    @showprogress 1 "BUG for Tensor-trains" for i in 1:steps 
        update_MPO!(H, bc_params, t0 + h/2)
        M_copy = mps_bug_step(H, M_copy, h, center)
        t0 += h
        if cutoff != nothing
            truncate!(M_copy; cutoff = cutoff)
        end
        if maxdim != nothing 
            truncate!(M_copy; maxdim = maxdim)
        end
        link_dim[i + 1, :] = linkdims(M_copy)
        if magnet == true 
            magnet_history[i + 1,:] = reverse(expect(M_copy, [1 0; 0 -1]))
        end
        if energy == true 
            energy_history[i + 1] = real(inner(M_copy', H, M_copy))
        end
        link_dim[i + 1,:] = linkdims(M_copy)
    end
    return M_copy, link_dim, magnet_history, energy_history
end