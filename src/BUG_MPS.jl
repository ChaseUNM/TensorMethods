using ITensors, ITensorMPS, LinearAlgebra 


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


function project_ortho_center(M::MPS, center::Int64, left_updated::MPS, right_updated::MPS)
    N = length(M)
    L_block = 1
    R_block = 1
    for i = 1:center - 1
        L_block *= M[i]*conj(left_updated[i])
    end
    for i = N:-1:center + 1
        # println(i)
        R_block *= M[i]*conj(right_updated[i])
    end
    updated_center = L_block*M[center]*R_block 
    return updated_center 
end

#Choose an orthogonal center, this will remain the orthogonal center 
function sweep_left_bug(H::MPO, M::MPS, h::Float64, center::Int64)
    #Start with updating right-to-left until we get to the orthogonality center
    N = length(M)
    L_list = contract_left(H, M, N - 1)
    R_block = 1
    M_copy = deepcopy(M)
    for i in N:-1:center + 1
        
        #Update the i-th core
        M_evolve = TT_IMR_1site_new(H, M_copy[i], L_list[i], R_block, h, i)
        # println("Site $i updated")
        # println(M_copy[i])
        # println(M_evolve)
        #Now need to matricize M_evolve and M[i]
        if length(inds(M_copy[i])) == 2
            M_mat = Array(M_copy[i], inds(M_copy[i]))
            M_evolve_mat = Array(M_evolve, inds(M_evolve))
            # M_combined = hcat(M_evolve_mat, M_mat)
            M_combine = hcat(M_evolve_mat)
            Q, R = qr(M_combine)



            row, col = size(M_combine)

            # Q = Q[1:row, 1:min(row, col, 2^i)]
            Q = Q[1:row, 1:min(row, col)]
            new_link = Index(min(row, col); tags="Link, l = $(i - 1)")
            # println("Got here")
            # println(siteinds(M)[i])
            # println(new_link)
            # display(Q)
            Q_ten = ITensor(Q, siteinds(M_copy)[i], new_link)
        end

        if length(inds(M_copy[i])) == 3
            site_idx, left_idx, right_idx = get_site_and_links(M_copy[i])
            M_arr = Array(M_copy[i], left_idx, site_idx, right_idx)
            M_evolve_arr = Array(M_evolve, left_idx, site_idx, right_idx)
            # M_arr2 = Array(M_copy[i], site_idx, right_idx, left_idx)
            # M_evolve_arr2 = Array(M_evolve, site_idx, right_idx, left_idx)
            M_mat = reshape(M_arr, dim(left_idx), dim(site_idx)*dim(right_idx))
            M_evolve_mat = reshape(M_evolve_arr, dim(left_idx), dim(site_idx)*dim(right_idx))
            # M_mat2 = reshape(M_arr2, dim(right_idx)*dim(site_idx), dim(left_idx))
            # M_evolve_mat2 = reshape(M_evolve_arr2, dim(right_idx)*dim(site_idx), dim(left_idx))
            # println("Size M_Mat2: ", size(M_mat2))
            # M_combine = hcat(transpose(M_evolve_mat), transpose(M_mat))
            
            # M_combine = vcat(M_evolve_mat, M_mat)
            M_combine = hcat(M_evolve_mat)

            # M_combine = hcat(M_evolve_mat2, M_mat2)

            Q, R = qr(M_combine')
            # println("Size of M_combine:", size(M_combine))
            
            # println(new_left_idx)
            row, col = size(M_combine')
            # Q = Q[1:row, 1:min(row, col, 2^i)]
            Q = Q[1:row, 1:min(row, col)]

            Q = transpose(conj(Q))
            new_left_idx = Index(min(row, col); tags = "Link, l = $(i - 1)")
            # println("Size of Q: ", size(Q))
            # println("size new_left_index: ", dim(new_left_idx))
            # println("size right_index: ", dim(right_idx))
            # println("size site_index: ", dim(site_idx))
            Q = Array(reshape(Q, dim(new_left_idx), dim(site_idx), dim(right_idx)))

            Q_ten = ITensor(Q, new_left_idx, site_idx, right_idx)
            
            # println("Got here")
        end
        R_block = R_block*H[i]*Q_ten*conj(Q_ten)'
        M_copy[i-1] = M_copy[i-1]*M_copy[i]*conj(Q_ten)
        M_copy[i] = Q_ten
    end
    return M_copy, R_block 
end

function sweep_right_bug(H::MPO, M::MPS, h::Float64, center::Int64)
    N = length(M)
    R_list = contract_right(H, M, 2)
    L_block = 1
    M_copy = deepcopy(M)

    for i in 1:center - 1

        M_evolve = TT_IMR_1site_new(H, M_copy[i], L_block, R_list[i], h, i)

        # println("Site $i updated")
        if length(inds(M_copy[i])) == 2
            M_mat = Array(M_copy[i], inds(M_copy[i]))
            M_evolve_mat = Array(M_evolve, inds(M_evolve))
            # M_combine = hcat(M_evolve_mat, M_mat)
            M_combine = hcat(M_evolve_mat)
            Q, R = qr(M_combine)
            row, col = size(M_combine)
            # Q = Q[1:row, 1:min(row, col, 2^i)]
            Q = Q[1:row, 1:min(row, col)]
            new_link = Index(min(row, col); tags = "Link, l = $i")
            # println("Got here")
            Q_ten = ITensor(Q, siteinds(M_copy)[i], new_link)
        end

        if length(inds(M[i])) == 3
            site_idx, left_idx, right_idx = get_site_and_links(M_copy[i])
            M_arr = Array(M_copy[i], left_idx, site_idx, right_idx)
            M_evolve_arr = Array(M_evolve, left_idx, site_idx, right_idx)
            M_mat = reshape(M_arr, dim(right_idx), dim(left_idx)*dim(site_idx))
            M_evolve_mat = reshape(M_evolve_arr, dim(right_idx), dim(left_idx)*dim(site_idx))
            # M_mat2 = reshape(M_arr, dim(left_idx)*dim(site_idx), dim(right_idx))
            # M_evolve_mat2 = reshape(M_evolve_arr, dim(left_idx)*dim(site_idx), dim(right_idx))
            # println("diff matrices: ", norm(transpose(M_mat) - M_mat2))
            # M_combine = hcat(transpose(M_evolve_mat), transpose(M_mat))
            M_combine = hcat(M_evolve_mat)

            # M_combine = hcat(M_evolve_mat2, M_mat2)
            Q, R = qr(M_combine)
            println("size M_combine: ", size(M_combine))
            # println("Size of M combine: ", size(M_combine))
            row, col = size(M_combine)
            
            # Q = Q[1:row, 1:min(row, col, 2^i)]
            Q = Q[1:row, 1:min(row, col)]
            println("size Q: ", size(Q))
            new_right_idx = Index(min(row, col), tags = "Link, l = $i")
            Q = reshape(Q, dim(left_idx), dim(site_idx), dim(new_right_idx))
            Q_ten = ITensor(Q, left_idx, site_idx, new_right_idx)
            # println("Got here")
        end
        L_block = L_block*H[i]*Q_ten*conj(Q_ten)'
        M_copy[i + 1] = M_copy[i + 1]*M_copy[i]*conj(Q_ten)
        M_copy[i] = Q_ten 
    end
    return M_copy, L_block
end

function mps_bug_step(H::MPO, M::MPS, h::Float64, center::Int64)
    N = length(M)
    
    M_r, R_block = sweep_left_bug(H, M, h, center)
    M_l, L_block = sweep_right_bug(H, M, h, center)
    M_center = project_ortho_center(M, center, M_l, M_r)
    # println("Norm M_center: ", norm(M_center))
    # println("inds M_center: ", inds(M_center))
    # println("inds L_block: ", inds(L_block))
    # println("inds R_block: ", inds(R_block))
    # println("inds H[center]: ", inds(H[center]))
    M_center_evolve = TT_IMR_1site_new(H, M_center, L_block, R_block, h, center)
    # println("Site $center updated")
    updated_MPS = MPS(N)
    for i in 1:center - 1
        updated_MPS[i] = M_l[i]
    end
    for i in N:-1:center + 1
        updated_MPS[i] = M_r[i]
    end
    updated_MPS[center] = M_center_evolve 
    return updated_MPS 
end

function mps_bug_constant(H::MPO, M::MPS, t0::Float64, T::Float64, steps::Int64, center::Union{Nothing, Int64} = nothing; cutoff::Union{Nothing, Float64}=nothing, maxdim::Union{Nothing, Float64}=nothing, magnet::Bool=false, energy::Bool=false)
    # if orthoCenter(M) != center 
    #     orthogonalize!(M, center)
    # end
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
        magnet_history[1,:] = reverse(expect(M_copy, [1 0; 0 -1]))
    end
    if energy == true 
        energy_history[1] = real(inner(M_copy', H, M_copy))
    end 

    @showprogress 1 "BUG for Tensor-trains" for i in 1:steps 
        # println("Step $i")
        M_copy = mps_bug_step(H, M_copy, h, center)
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