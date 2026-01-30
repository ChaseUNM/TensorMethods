using LinearAlgebra, ITensors, ITensorMPS, Plots, ProgressMeter


function create_links_tucker(N::Int64, link_size::Vector{Int64})
    ind_vec = Index{Int64}[]
    for i in 1:N 
        ind = Index(link_size[i]; tags="Link, l = $i")
        push!(ind_vec, ind)
    end
    return ind_vec 
end

#Count total number of entries in a tucker tensor given the bond dimension and total size of the site indices
function count_tucker(bd::Vector, N_levels::Vector{Int64})
    entries = 0 
    N = length(N_levels)
    entries += prod(bd)
    for i in 1:N 
        entries += bd[i]*N_levels[i]
    end
    return entries 
end

function matricize_factors(factors)
    N = length(factors)
    factor_matrices = []
    for i in 1:N 
        u_mat = Array(factors[i], inds(factors[i]))
        push!(factor_matrices, u_mat)
    end
    return factor_matrices 
end



function Multi_TTM_recursive(tensor::Array, matrices::Vector{<:AbstractMatrix}, mode::Int=1)
    if mode > length(matrices)
        return tensor
    else
        return Multi_TTM_recursive(TTM(tensor, matrices[mode], mode), matrices, mode + 1)
    end
end


function trim_by_tolerance(v::Vector{T}, tol::Real) where T<:Real
    idx = length(v)
    s = zero(T)
    v = v/norm(v)
    count = 0 
    while idx >= 1 
        
        s += v[idx]^2
        
        if s > tol 
            break
        end
        idx -= 1
        count += 1
    end
    return idx
end



function matricization(tensor::Array, mode::Int64)
    if mode == 1 
        return reshape(tensor, size(tensor)[1], Int64(prod(size(tensor))/size(tensor)[1]))
    else
        # A = permutedims(tensor, (mode, reverse(setdiff(1:ndims(tensor), mode))...))
        A = permutedims(tensor, (mode, setdiff(1:ndims(tensor), mode)...))
        # A = permutedims(tensor, (setdiff(1:ndims(tensor), mode)..., mode))
        # println("A size: ", size(A))
        # return reshape(A, size(tensor, mode), Int64(prod(size(tensor))/size(tensor, mode)))
        return reshape(A, size(tensor, mode), Int64(prod(size(tensor))/size(tensor, mode)))
    end
end

function refold_mat(mat::Array, original_dim::Tuple{Vararg{Int64}}, mode::Int64)
    d = length(original_dim)
    if mode == 1
        return reshape(mat, original_dim)
    else
        perm = (mode, setdiff(1:d, mode)...)
        tensor_perm = reshape(mat, (original_dim[mode], original_dim[setdiff(1:d, mode)]...))
        # return tensor_perm 
        return permutedims(tensor_perm, invperm(perm))
    end
end


#Calculate left-leading singular vectors
function LLSV(Y::Array; cutoff::Union{Nothing,Float64}=nothing, target_rank::Union{Nothing,Int64}=nothing, verbose::Bool=false, mode::Int64 = nothing)
    U, S, Vt = svd(Y)
    
    if (cutoff === nothing) == (target_rank === nothing)
        error("Specify either cutoff or target_rank, but not both.")
    end
    if cutoff !== nothing
        rank = trim_by_tolerance(S, cutoff)
        # println("Truncated rank by cutoff: ", rank)
    else
        rank = target_rank
    end
    W = U[:,1:rank]
    if verbose == true
        println("Singular Values for mode $mode: ", S)
        println("Removed Singular Values for mode $mode: ", S[rank + 1:end])
        println("Rank of factor $mode: $rank")
        println("---------------------------------------------------------")
    end
    err = sqrt(sum(S[rank+1:end].^2))
    return W, err
end


#Calculate right-leading singular vectors
function RLSV(Y::Array; cutoff::Union{Nothing,Float64}=nothing, target_rank::Union{Nothing,Int64}=nothing, verbose::Bool=false, mode::Int64 = nothing)
    U, S, Vt = svd(Y)
    
    if (cutoff === nothing) == (target_rank === nothing)
        error("Specify either cutoff or target_rank, but not both.")
    end
    if cutoff !== nothing
        rank = trim_by_tolerance(S, cutoff)
        # println("Truncated rank by cutoff: ", rank)
    else
        rank = target_rank
    end
    W = U[1:rank,:]
    if verbose == true
        println("Singular Values for mode $mode: ", S)
        println("Removed Singular Values for mode $mode: ", S[rank + 1:end])
        println("Rank of factor $mode: $rank")
        println("---------------------------------------------------------")
    end
    err = sqrt(sum(S[rank+1:end].^2))
    return W, err
end

function applyHV(op, factors, site)
    N = length(factors)
    new_factors = [ComplexF64.(Matrix(1.0*I, size(factors[i], 2), size(factors[i], 2))) for i in 1:N]
    # new_factors[site] = zeros(ComplexF64, size(factors[site], 1), size(factors[site], 2))
    if length(op) == 2
        op_site = op[2]
        if op_site == site 
            new_factors[op[2]] = op[1]*factors[op[2]]
        elseif op_site != site
            new_factors[op[2]] = (op[1]*factors[op[2]])'*factors[op[2]]
            new_factors[site] = factors[site]
        end
        # new_factors[op[4]] = op[3]*new_factors[op[4]]
    elseif length(op) == 4
        op_site1 = op[2]
        op_site2 = op[4]
        if op_site1 == site
            new_factors[op[2]] = op[1]*factors[op[2]]
            new_factors[op[4]] = (op[3]*factors[op[4]])'*factors[op[4]]
        elseif op_site2 == site 
            new_factors[op[2]] = (op[1]*factors[op[2]])'*factors[op[2]]
            new_factors[op[4]] = op[3]*factors[op[4]]
        else 
            new_factors[op[2]] = (op[1]*factors[op[2]])'*factors[op[2]]
            new_factors[op[4]] = (op[3]*factors[op[4]])'*factors[op[4]]
            new_factors[site] = factors[site]
        end
    end
    return new_factors 
end

function tucker(tensor::Array; cutoff::Union{Nothing,Float64}=nothing, target_rank::Union{Nothing,Vector{Int64}}=nothing, verbose::Bool=false)
    d = length(size(tensor))
    if target_rank === nothing && cutoff == 0.0
        target_rank_vec = fill(nothing, d)
    else
        target_rank_vec = fill(nothing, d)
    end
    U_list = Matrix{eltype(tensor)}[]
    core = copy(tensor)
    core_copy = copy(tensor)
    err_list = zeros(d)
    cutoff_bar = cutoff !== nothing ? cutoff*norm(tensor)/sqrt(d) : nothing
    for i in 1:d
        U, err = LLSV(matricization(core_copy, i); cutoff = cutoff, target_rank = target_rank_vec[i], verbose = verbose, mode = i)
        push!(U_list, U)
        err_list[i] = err
        core = TTM(core, Array(U'), i)
    end
    total_err = sqrt(sum(err_list.^2))
    return core, U_list, total_err
end

function truncate_tucker(tensor::AbstractArray, factors::Vector{<:AbstractMatrix}; cutoff::Union{Nothing,Float64}=nothing, target_rank::Union{Nothing,Vector{Int64}}=nothing)
    d = length(size(tensor))
    if target_rank === nothing
        target_rank_vec = fill(nothing, d)
    else
        target_rank_vec = target_rank
    end
    U_list = Matrix{eltype(tensor)}[]
    core = copy(tensor)
    core_copy = copy(tensor)
    err_list = zeros(d)
    cutoff_bar = cutoff !== nothing ? cutoff*norm(tensor)/sqrt(d) : nothing
    for i in 1:d 
        U, err = LLSV(matricization(core_copy, i); cutoff = cutoff, target_rank = target_rank_vec[i])
        push!(U_list, factors[i]*U)
        err_list[i] = err 
        core = TTM(core, Array(U'), i)
    end
    total_err = sqrt(sum(err_list.^2))
    return core, U_list, total_err 
end

function tucker_sequential(tensor::Array; cutoff::Union{Nothing,Float64}=nothing, target_rank::Union{Nothing,Vector{Int64}}=nothing)
    d = length(size(tensor))
    if target_rank === nothing
        target_rank_vec = fill(nothing, d)
    else
        target_rank_vec = target_rank
    end
    U_list = Matrix{eltype(tensor)}[]
    core = copy(tensor)
    err_list = zeros(d)
    cutoff_bar = cutoff !== nothing ? cutoff*norm(tensor)/sqrt(d) : nothing
    for i in 1:d
        U, err = LLSV(matricization(core, i); cutoff = cutoff, target_rank = target_rank_vec[i])
        push!(U_list, U)
        err_list[i] = err
        core = TTM(core, Array(U'), i)
    end
    total_err = sqrt(sum(err_list.^2))
    return core, U_list, total_err
end

function tucker_separable(q_state)
    N = length(q_state)
    link_size = Int64.(ones(N))
    links_tuple = (link_size...,)
    factors = Vector{Matrix{Float64}}(undef, N)
    core = Array{Float64}(undef, links_tuple)
    core[1] = 1.0
    for i in 1:N
        factor = zeros(2, 1)
        factor[q_state[i] + 1, 1] = 1.0
        factors[i] = factor
    end
    return core, factors
end

function fixed_point_iter_C_mat(H_ops, core, h, factors, M_list, P_list, Y_list, maxiter, tol, verbose)
    K_init = zeros(eltype(core), size(core))
    for i in 1:maxiter 
        K = C_dot_im_mat(core + 0.5*h*K_init, factors, H_ops, M_list, P_list, Y_list)
        error = norm(K_init - K)
        if verbose == true 
            println("Iteration $i")
            println("Error: ", error)
        end
        K_init .= K
        if error < tol 
            break 
        end
    end
    return K_init 
end

#This is the one
function IMR_core_mat(H_ops, core, h, factors_matrices, M_list, P_list, Y_list, maxiter, tol, verbose)
    K = fixed_point_iter_C_mat(H_ops, core, h, factors_matrices, M_list, P_list, Y_list, maxiter, tol, verbose)
    core .+= h*K
    return core 
end

function bug_step_itensor(H_ten, core, factors, h, sites)
    #Need to update basis matrices from core 
    N = length(factors)
    core_inds = collect(inds(core))
    factors_matrix = matricize_factors(factors)
    factors_matrix_T = transpose_U(factors_matrix)
    factors_update = []
    M_list = []
    for i in 1:N 

        sites_copy = copy(sites)
        deleteat!(sites_copy, i)
        core_inds_copy = copy(core_inds)
        deleteat!(core_inds_copy, i)
        core_mat = matricization(core, i)
        Q, St = qr(core_mat')
        row_S, col_S = size(St)
        Q = Q*I 
        Q = Q[:,1:row_S]
        V_T = Q'*factor_kron(factors_matrix_T, i)
        K0 = factors_matrix[i]*St' 
        # println("K0: ")
        # display(K0)
        Y0 = K0*V_T
        orig_order = collect(1:N)
        permutation = vcat([i], setdiff(orig_order, i))
        sites_copy2 = copy(sites)
        permute!(sites_copy2, permutation)
        Y0_ten = ITensor(Y0, sites_copy2)
        # println("Inefficient Factor $i: ")
        # @time begin 
        Y0_dot = H_ten*Y0_ten
        # println("Y0_dot")
        # println(Y0_dot)
        Y0_dot_mat = matricization(Y0_dot, i)*V_T'
        # end
        # println("K[$i] derivative")
        # display(Y0_dot_mat)
        K1 = K0 - h*im*Y0_dot_mat
        row_K, col_K = size(K1) 
        U, R = qr(K1)
        U = U*I 
        U = U[:,1:col_K]
        row_U, col_U = size(U)
        col_ind = Index(col_U;tags="link, $i")
        U_ten = ITensor(U, sites[i], col_ind)
        
        push!(factors_update, U_ten)
        M = conj(U_ten)*factors[i]
        push!(M_list, M)
        # println(U_ten)
        # println(factors[i])
    end

    init_C = reconstruct(core, M_list)
    # println("C_INIT 1")
    # println(init_C)
    # C_update = init_C
    C_update = IMR_core_itensor(H_ten, init_C, h, factors_update, 100, 1E-14, false)

    return C_update, factors_update
end

function bug_step_eff(H_ops, core, factors, h, sites)
    N = length(factors)
    factors_update = Vector{ITensor}(undef, N)
    M_list = Vector{ITensor}(undef, N)
    for i in 1:N 
        # K_dot, K0 = K_evolution_itensor(core, factors, i, H_ops, sites)
        K_dot, K0 = K_evolution_itensor2(core, factors, i, H_ops, sites)
        # println("K[$i] derivative")
        # println(K_dot)
        # println(K0)
        K1 = K0 - h*im*K_dot
        # println(K1)
        site_ind = inds(K1, "Site")
        U, R = qr(K1, site_ind)
        # U_link = inds(U,"Link")
        # U = U*delta(U_link,inds(core)[i])
        # push!(factors_update, U)
        factors_update[i] = U
        M = conj(U)*factors[i]
        # println(U)
        # println(factors[i])
        # push!(M_list, M)
        M_list[i] = M
    end

    init_C = reconstruct(core, M_list)
    # println("M_list:")
    # println(M_list)
    # println("init_C: ")
    # display(Array(init_C, inds(init_C)))
    C_update = IMR_core_ten(H_ops, init_C, h, factors_update, 100, 1E-14, false)

    return C_update, factors_update
end


    


function TTM(tensor::AbstractArray, matrix::AbstractMatrix, mode::Int)
    tensor_dim = collect(size(tensor))
    # println("Tensor dims: ", tensor_dim)
    d = length(tensor_dim)
    mat_row, mat_col = size(matrix)
    # println("Mode $mode")
    if mode == 1
        M = tensor_dim[1]
        P = copy(tensor_dim)
        deleteat!(P, 1)
        col = Int(prod(size(tensor)) / size(tensor, 1))
        row = size(tensor, 1)
        # println("M: ", row)
        # println("P: ", col)
        # @time begin 
        Y = matrix * reshape(tensor, row, col)
        # end
        # println("Size Y: ", size(Y))
        return reshape(Y, mat_row, P...)
    else
        M = prod(tensor_dim[1:mode - 1])
        P = prod(tensor_dim[mode + 1:d])
        # println("M: $M")
        # println("P: $P")
        # println("M: $M, size(matrix, 1): $(size(matrix, 1)), P: $P")
        X_bar = reshape(tensor, M, tensor_dim[mode], P)
        #most of the memory is here
        # println("Mode: $mode")
        # @time begin 
        Y = zeros(eltype(X_bar), M, size(matrix, 1), P)
        # end
        matT = transpose(matrix)
        @views for l = 1:P
            mul!(Y[:, :, l], X_bar[:, :, l], matT)
            # Y[:,:,l] .= X_bar[:,:,l]*matT
        end
        return reshape(Y, tensor_dim[1:mode - 1]..., mat_row, tensor_dim[mode + 1:d]...)
    end
end



function Multi_TTM(tensor::AbstractArray, matrices::Vector{<:AbstractMatrix})
    Y = copy(tensor)  
    for i in 1:length(matrices)
        Y = TTM(Y, matrices[i], i)
    end
    return Y 
end



#This is the one
function K_evolution_mat2(core,factors,site, total_H)
    Q, S = qr(transpose(matricization(core, site)))
    K = ComplexF64.(factors[site]*transpose(S))
    factors_copy = copy(factors)
    factors_intermediate = factors_copy 
    factors_intermediate[site] = K
    # println("Site $site")
    # println("Size core: ", size(core))
    # println("size transpose: ", size(transpose(matricization(core, site))))
    # println("Size Q: ", size(Q))
    # println(size(core, site))
    Q = Array(Q)[:,1:min(size(core, site), size(Q, 2))]
    size_Q = zeros(Int64, length(factors_intermediate))
    for i in 1:length(factors_intermediate)
        size_Q[i] = size(factors_intermediate[i], 2)
    end
    size_Q = tuple(size_Q...)
    # println(size_Q)
    Q_ten = refold_mat(Array(transpose(Q)), size_Q, site)

    
    
    # K_inds = inds(factors[site]*S)
    # K_ten = ITensor(K, K_inds)
    
    N_ops = length(total_H)
    # @time begin 
    new_factors = applyHV(total_H[1], factors_intermediate, site)
    # new_factors = applyH_mat(total_H[1], factors_intermediate, site)
    init = Multi_TTM_recursive(Q_ten, new_factors)
    # init = Multi_TTM(Q_ten, new_factors)
    
    for i in 2:N_ops
        new_factors = applyHV(total_H[i], factors_intermediate, site)
        # new_factors = applyH_mat(total_H[i], factors_intermediate, site)
        init .+= Multi_TTM_recursive(Q_ten, new_factors)
        # init .+= Multi_TTM(Q_ten, new_factors)
    end
    K_dot = matricization(init, site)*conj(Q)
    return K_dot, K
end

#This is the one
#Non-allocated memory version
function C_dot_test2(core, factors, total_H)
    # println("C_dot_test2")
    N_ops = length(total_H)
    N_factors = length(factors)
    init = zeros(eltype(core), size(core)...)
    

    for i in 1:N_ops 
        # println("Op $i")
        new_factors = [ComplexF64.(Matrix(1.0*I, size(factors[i], 2), size(factors[i], 2))) for i in 1:N_factors]
        if length(total_H[i]) == 2 
            new_factors[total_H[i][2]] = factors[total_H[i][2]]'*total_H[i][1]*factors[total_H[i][2]]
        elseif length(total_H[i]) == 4
            new_factors[total_H[i][2]] = factors[total_H[i][2]]'*total_H[i][1]*factors[total_H[i][2]]
            new_factors[total_H[i][4]] = factors[total_H[i][4]]'*total_H[i][3]*factors[total_H[i][4]]
        end 
        init .+= -im*Multi_TTM_recursive(core, new_factors)
        # init .+= -im*Multi_TTM(core, new_factors)
        # println("new factors: ")
        # display(new_factors)
    end
    return init 
end



#This is the one
#Non-allocated memory version
function fixed_point_iter_C_mat(H_ops, core, h, factors, maxiter, tol, verbose)
    # println("Called the right one!")
    K_init = zeros(eltype(core), size(core))
    for i in 1:maxiter 
        K = C_dot_test2(core + 0.5*h*K_init, factors, H_ops)
        # display(K)
        error = norm(K_init - K)
        if verbose == true 
            println("Iteration $i")
            println("Error: ", error)
            # display(K)
        end
        K_init .= K
        if error < tol 
            break 
        end
    end
    return K_init 
end

#This is for switching to a fixed-point method for evolving the factors
function fixed_point_factor(H_ops, core, h, factors, site, maxiter, tol, verbose)
    K_init = zeros(eltype(factors[site]), size(factors[site]))
    for i in 1:maxiter 
        K,_ = K_evolution_mat2(core, factors, site, H_ops)
        K .*= -im
        error = norm(K_init - K)
        if verbose == true 
            println("Iteration $i")
            println("Error: ", error)
        end
        K_init .= K 
        if error < tol 
            break 
        end
    end
    return K_init 
end



#Non-allocated memory version
function IMR_core_mat(H_ops, core, h, factors_matrices, maxiter, tol, verbose)
    K = fixed_point_iter_C_mat(H_ops, core, h, factors_matrices, maxiter, tol, verbose)
    core .+= h*K
    return core 
end

function energy_tucker(total_H, core::AbstractArray, factors)
    energy = 0.0
    vec_core = vec(core)
    N_ops = length(total_H)
    for i = 1:N_ops 
        new_factors = [ComplexF64.(Matrix(1.0*I, size(factors[i], 2), size(factors[i], 2))) for i in 1:length(factors)]
        if length(total_H[i]) == 2 
            new_factors[total_H[i][2]] = factors[total_H[i][2]]'*total_H[i][1]*factors[total_H[i][2]]
        elseif length(total_H[i]) == 4
            new_factors[total_H[i][2]] = factors[total_H[i][2]]'*total_H[i][1]*factors[total_H[i][2]]
            new_factors[total_H[i][4]] = factors[total_H[i][4]]'*total_H[i][3]*factors[total_H[i][4]]
        end 
        energy += vec_core'*vec(Multi_TTM(core, new_factors))
    end
    return real(energy)
end

function bug_step_mat_ra(H_ops, core, factors, h, cutoff)
    N = length(factors)
    # println(N)
    factors_update = Vector{AbstractMatrix}(undef, N)
    core_size = size(core)
    # M_storage = [zeros(ComplexF64, min(size(factors[i],1), 2*size(factors[i], 2)), size(factors[i],2)) for i in 1:N]
    M_storage = [zeros(ComplexF64, min(size(factors[i],1), size(factors[i], 2) + min(size(factors[i], 2), prod(deleteat!(copy(collect(core_size)), i)))), size(factors[i],2)) for i in 1:N]
    # M_storage = Vector{AbstractMatrix}(undef, N)
    for i in 1:N
        # println("Factor $i")
        # @time begin
        K_dot, K0 = K_evolution_mat2(core, factors, i, H_ops)
        # end
        # K1 = K0 - h*im*K_dot
        K0 .-= h*im*K_dot 
        combined_K = hcat(K0, factors[i])
        U, R = qr(combined_K)
        # println("Size combined_K: ", size(combined_K))
        # println("Size U: ", size(U))
        U = U[1:size(combined_K, 1),1:min(size(combined_K, 1), size(combined_K, 2))]
        factors_update[i] = U 
        # M_storage[i] = U'*factors[i]
        # println("Size M_Storage $i: ", size(M_storage[i]))
        # println("Size U: ", size(U))
        # println("Size factors $i: ", size(factors[i]))
        # M_storage[i] = U'*factors[i]
        mul!(M_storage[i], U', factors[i])
        # M = U'*factors[i]
        # push!(M_storage, M)
    end 
    # println(M_list)
    # println(P_list)
    # println(Y_list)
    # println(M_storage[4])
    # println("init_C")
    # @time begin 
    init_C = Multi_TTM_recursive(core, M_storage)
    # end
    # init_C = Multi_TTM(core, M_storage)
    # init_C = copy(Multi_TTM_allocate_recursive(core, M_storage, M_list, P_list, Y_list))
    
    # println("init_C_2")
    # display(init_C_2)
    # println("init_C")
    # display(init_C)
    # println(norm(init_C_2 - init_C))
    # println("init_C: ")
    # display(init_C)
    # println("Core update: ")
    # @time begin 
    C_update = IMR_core_mat(H_ops, init_C, h, factors_update, 100, 1E-13, false)
    # end
    # C_update = heun_core(H_ops, init_C, h, factors_update)
    C_trunc, factors_trunc = truncate_tucker_arr(C_update, factors_update; cutoff = cutoff)
    return C_trunc, factors_trunc
end

function truncate_tucker_arr(tensor::AbstractArray, factors::Vector{<:AbstractMatrix}; cutoff::Union{Nothing,Float64}=nothing, target_rank::Union{Nothing,Vector{Int64}}=nothing)
    d = length(size(tensor))
    if target_rank === nothing
        target_rank_vec = fill(nothing, d)
    else
        target_rank_vec = target_rank
    end
    U_list = Matrix{eltype(tensor)}[]
    core = copy(tensor)
    core_copy = copy(tensor)
    err_list = zeros(d)
    cutoff_bar = cutoff !== nothing ? cutoff*norm(tensor)/sqrt(d) : nothing
    for i in 1:d 
        U, err = LLSV(matricization(core_copy, i); cutoff = cutoff, target_rank = target_rank_vec[i], verbose = false, mode = i)
        push!(U_list, factors[i]*U)
        err_list[i] = err 
        core = TTM(core, Array(U'), i)
    end
    total_err = sqrt(sum(err_list.^2))
    return core, U_list, total_err 
end


#This is for a constant hamiltonian
function bug_integrator_mat(H_ten, init_core, init_factors, t0, T, steps)
    h = (T - t0)/steps 
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    N = length(init_factors)
    energy_history = zeros(steps + 1)
    # state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    @showprogress 1 "BUG" for i in 1:steps + 1
        energy_history[i] = energy_tucker(H_ten, init_core_copy, init_factors_copy)
        # state_history[i,:] = Multi_TTM_recursive(init_core_copy, init_factors_copy)
        if i == steps + 1
            break 
        end
        C_update, factors_update = bug_step_mat(H_ten, init_core_copy, init_factors_copy, h)
        init_core_copy .= C_update 
        init_factors_copy = copy(factors_update)
    end 
    # return init_core_copy, init_factors_copy, state_history, energy_history 
    return init_core_copy, init_factors_copy, energy_history 
end

#This is for a time-varying control hamiltonian
function bug_integrator_mat(H_s, bcparams, init_core, init_factors, t0, T, steps)
    h = (T - t0)/steps 
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    N = length(init_factors)
    energy_history = zeros(steps + 1)
    # state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    H_s_copy = deepcopy(H_s)
    @showprogress 1 "BUG" for i in 1:steps + 1
        # println("H_s_copy: ")
        # display(H_s_copy)
        energy_history[i] = energy_tucker(H_s_copy, init_core_copy, init_factors_copy)
        # state_history[i,:] = Multi_TTM_recursive(init_core_copy, init_factors_copy)
        if i == steps + 1
            break 
        end
        H_s_copy = updateH(H_s, bcparams, t0 + 0.5*h)
        init_core_copy, init_factors_copy = bug_step_mat(H_s_copy, init_core_copy, init_factors_copy, h)
        t0 += h
        # H_s_copy = control_h_rot(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz, bcparams, t0)
    end 
    # return init_core_copy, init_factors_copy, state_history, energy_history 
    return init_core_copy, init_factors_copy, energy_history 
end


#This is the one 
#This is for a constant Hamiltonian
function bug_integrator_mat_ra(H_ten, init_core, init_factors, t0, T, steps; cutoff::Float64 = 0.0, energy::Bool = false, state::Bool = false)
    h = (T - t0)/steps
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    N = length(init_factors)
    energy_history = nothing 
    if energy == true
        energy_history = zeros(steps + 1)
    end
    state_history = nothing 
    if state == true 
        state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    end
    bd = zeros(steps + 1, N)
    @showprogress 1 "Rank-adaptive Tucker" for i in 1:steps + 1
        if energy == true 
            energy_history[i] = energy_tucker(H_ten, init_core_copy, init_factors_copy)
        end
        if state == true 
            state_history[i,:] = Multi_TTM_recursive(init_core_copy, init_factors_copy)
        end
        bd[i,:] = [size(init_factors_copy[j], 2) for j in 1:length(init_factors_copy)]
        if i == steps + 1
            break 
        end
        C_update, factors_update = bug_step_mat_ra(H_ten, init_core_copy, init_factors_copy, h, cutoff)
        init_core_copy = copy(C_update) 
        init_factors_copy = copy(factors_update)
    end 
    return init_core_copy, init_factors_copy, state_history, energy_history, bd
    # return init_core_copy, init_factors_copy, energy_history, bd
end

#This is the one
#This is for a time-varying control Hamiltonian
function bug_integrator_mat_ra(H_s, bcparams::bcparams, init_core::AbstractArray, init_factors, t0::Float64, T::Float64, steps::Int64; cutoff::Float64 = 0.0, state::Bool = false, energy::Bool = false, normalize::Bool = false)
    h = (T - t0)/steps 
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    N = length(init_factors)
    state_history = nothing 
    if state == true 
        state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    end
    energy_history = nothing
    if energy == true
        energy_history = zeros(steps + 1)
    end
    bd = zeros(steps + 1, N)
    H_s_copy = deepcopy(H_s)
    @showprogress 1 "BUG for Tucker-tensors" for i in 1:steps + 1
        if energy == true 
            energy_history[i] = energy_tucker(H_s_copy, init_core_copy, init_factors_copy)
        end
        if state == true 
            state_history[i,:] = Multi_TTM_recursive(init_core_copy, init_factors_copy)
        end
        # state_history[i,:] = Multi_TTM(init_core_copy, init_factors_copy)
        bd[i,:] = [size(init_factors_copy[j], 2) for j in 1:length(init_factors_copy)]
        if i == steps + 1
            break 
        end
        H_s_copy = updateH(H_s, bcparams, t0 + 0.5*h)
        # println("Total")
        # @time begin 
        init_core_copy, init_factors_copy = bug_step_mat_ra(H_s_copy, init_core_copy, init_factors_copy, h, cutoff)
        # end
        if normalize == true 
            init_core_copy = init_core_copy/norm(init_core_copy)
        end
        t0 += h
        
    end 
    return init_core_copy, init_factors_copy, state_history, energy_history, bd
    # return init_core_copy, init_factors_copy, energy_history, bd
end



function vec_separable(q_state)
    N = length(q_state)
    init = zeros(2)
    init[q_state[1] + 1] = 1.0
    for i in 2:N
        state_vec = zeros(2)
        state_vec[q_state[i] + 1] = 1.0
        init = kron(init, state_vec)
    end
    return init 
end

function expect_tucker(op_core, op_factors, state_core, state_factors)
    contract_factors = []
    N = length(state_factors)
    for i in 1:N 
        push!(contract_factors, op_factors[i]*state_factors[i]*conj(state_factors[i]'))
    end
    tens = op_core
    for i in 1:N 
        tens = tens*contract_factors[i]
    end
    tens = tens*state_core
    tens = tens*conj(state_core') 
    return scalar(tens) 
end

function entries_tucker(core, factors)
    entries = length(Array(core, inds(core)))
    for i in 1:length(factors)
        entries += length(Array(factors[i], inds(factors[i])))
    end
    return entries 
end

