using LinearAlgebra, BenchmarkTools


include("BUG_tucker(8-27).jl")
#todo: be able to do this without ITensor.jl 


#Need to able to calculate TTM, with that comes proper unfolding and SVD 

"""
    TTM(tensor::Array, matrix::Array, mode::Int64)

Performs the Tensor Times Matrix (TTM) operation along the specified mode.

# Arguments
- `tensor::Array`: The input tensor to be multiplied.
- `matrix::Array`: The matrix to multiply with the tensor along the given mode.
- `mode::Int64`: The mode (dimension) along which to perform the multiplication.

# Returns
- `Y::Array`: The resulting tensor after the TTM operation.
"""

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
        # println("Size matrix: ", size(matrix))
        # println("Size reshape: ", size(reshape(tensor, row, col)))
        Y = matrix * reshape(tensor, row, col)
        # end
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

function TTM_allocate(tensor::AbstractArray, matrix::AbstractMatrix, M::Int, P::Int, Y_alloc::AbstractArray, mode::Int)
    # println("Mode $mode")
    tensor_dim = collect(size(tensor))
    d = length(tensor_dim)
    mat_row, mat_col = size(matrix)
    # println("Mode: $mode")
    # println("M: $M")
    # println("P: $P")
    if mode == 1
        # M = tensor_dim[1]
        P_inds = copy(tensor_dim)
        deleteat!(P_inds, 1)
        # col = Int(prod(size(tensor)) / size(tensor, 1))
        # row = size(tensor, 1)
        # M = Int(prod(size(tensor)) / size(tensor, 1))
        # P = size(tensor, 1)
        # @time begin 
        # Y = matrix * reshape(tensor, row, col)
        M = size(tensor, 1)
        P = Int(prod(size(tensor)) / size(tensor, 1))
        # println("Size: ", size(matrix))
        # println("Size tensor: ", size(tensor))
        # println("M: $M")
        # println("P: $P")
        # println("Size 2: ", size(reshape(tensor, M, P)))
        # println("Sizes 3: ", size(Y_alloc))
        # println("Size matrix: ", size(matrix))
        subY = view(Y_alloc, 1:size(matrix, 1), 1:P)
        mul!(subY, matrix, reshape(tensor, M, P))
        # end
        return reshape(subY, mat_row, P_inds...)
    else
        # println(size(Y_bar))
        M = Int(prod(size(tensor)[1:mode - 1]))
        P = Int(prod(size(tensor)[mode + 1:d]))
        # println("M: $M")
        # println("P: $P")
        # println("tensor size: ", size(tensor))
        X_bar = reshape(tensor, M, tensor_dim[mode], P)
        #most of the memory is here
        
        # matT = transpose(matrix)
        # time_total = 0.0
        # mul!(Y_bar, matrix, matricization(tensor, mode))
        # Y_bar = matrix*matricization(tensor, mode)
        @views for l = 1:P
            # subY = view(Y_alloc[1:M,1:mat_row,l])
            # println("Slice size: ", size(Y_alloc[:,:,l]))
            # println("Product Size: ", size(X_bar[:,:,l]*transpose(matrix)))
            # println("Size matrix: ", size(matrix))
            # t_start = time_ns()
            mul!(Y_alloc[:, :, l], X_bar[:, :, l], transpose(matrix))
            # mul!(view(Y_alloc, 1:M, 1:mat_row, l), X_bar[:, :, l], transpose(matrix))
            # Y_alloc[:,:,l] = X_bar[:,:,l]*transpose(matrix)
            # t_end = time_ns()
            # time_total += (t_end - t_start)/1E9
        end
        # println("Size Y_alloc: ", size(Y_alloc))
        # subY = view(Y_alloc, 1:M, 1:mat_row, 1:P)
        # println("Size Y_alloc view: ", size(view(Y_alloc, 1:M, 1:mat_row, 1:P)))
        # @views for l = 1:M 
        #     mul!(Y_bar[l,:,:], matrix, X_bar[l,:,:])
        # end
        # println("Time for Multiplcation: $time_total")
        return reshape(view(Y_alloc, 1:M, 1:mat_row, 1:P), tensor_dim[1:mode - 1]..., mat_row, tensor_dim[mode + 1:d]...)
    end
end

function TTM_allocate2(tensor::AbstractArray, matrix::AbstractMatrix, M::Int, P::Int, Y_bar::AbstractArray, mode::Int)
    tensor_dim = collect(size(tensor))
    d = length(tensor_dim)
    mat_row, mat_col = size(matrix)
    if mode == 1
        # M = tensor_dim[1]
        P_inds = copy(tensor_dim)
        deleteat!(P_inds, 1)
        # col = Int(prod(size(tensor)) / size(tensor, 1))
        # row = size(tensor, 1)
        # @time begin 
        # Y = matrix * reshape(tensor, row, col)

        mul!(Y_bar, matrix, reshape(tensor, M, P))
        # end
        return reshape(Y_bar, mat_row, P_inds...)
    else
        X_bar = reshape(tensor, M, tensor_dim[mode], P)
        #most of the memory is here
        
        # matT = transpose(matrix)
        # time_total = 0.0
        # mul!(Y_bar, matrix, matricization(tensor, mode))
        # Y_bar = matrix*matricization(tensor, mode)
        # @views for l = 1:P
        #     # t_start = time_ns()
        #     mul!(Y_bar[:, :, l], X_bar[:, :, l], matT)
        #     # t_end = time_ns()
        #     # time_total += (t_end - t_start)/1E9
        # end
        @views for l = 1:M 
            mul!(Y_bar[l,:,:], matrix, X_bar[l,:,:])
        end
        # println("Time for Multiplcation: $time_total")
        return reshape(Y_bar, tensor_dim[1:mode - 1]..., mat_row, tensor_dim[mode + 1:d]...)
    end
end

function TTM_allocate3(tensor::AbstractArray, matrix::AbstractMatrix, Y_bar::AbstractArray, mode::Int, tensor_perm_arr::AbstractArray, Y_tensor_arr::AbstractArray)
    tensor_dim = collect(size(tensor))
    d = length(tensor_dim)
    mat_row, mat_col = size(matrix)
    if mode == 1
        # M = tensor_dim[1]
        P_inds = copy(tensor_dim)
        deleteat!(P_inds, 1)
        # col = Int(prod(size(tensor)) / size(tensor, 1))
        # row = size(tensor, 1)
        # @time begin 
        # Y = matrix * reshape(tensor, row, col)
        row = size(tensor, 1)
        col = Int(prod(size(tensor)) / size(tensor, 1))
        # println(size(matrix))
        # println(size(reshape(tensor, row, col)))
        # println(size(Y_bar))
        mul!(Y_bar, matrix, reshape(tensor, row, col))
        # end
        return reshape(Y_bar, mat_row, P_inds...)
    else
        # X_bar = reshape(tensor, M, tensor_dim[mode], P)
        #most of the memory is here
        # println(size(matrix))
        # println(size(matricization(tensor, mode)))
        # println(size(Y_bar))
        perm = (mode, setdiff(1:d, mode)...)
        # tensor_perm = permutedims(tensor, perm)
        # println("tensor: ", size(tensor))
        permutedims!(tensor_perm_arr, tensor, perm)
        X = reshape(tensor_perm_arr, tensor_dim[mode], :)
        # println(size(matrix))
        # println(size(X))
        # println(size(Y_bar))
        mul!(Y_bar, matrix, X)
        new_dims = (mat_row, tensor_dim[setdiff(1:d, mode)]...)
        Y_tensor = reshape(Y_bar, new_dims)
        # invperm = invpermute(perm)
        # matT = transpose(matrix)
        # time_total = 0.0
        # mul!(Y_bar, matrix, matricization(tensor, mode))
        # Y_bar = matrix*matricization(tensor, mode)
        # @views for l = 1:P
        #     # t_start = time_ns()
        #     mul!(Y_bar[:, :, l], X_bar[:, :, l], matT)
        #     # t_end = time_ns()
        #     # time_total += (t_end - t_start)/1E9
        # end
        # @views for l = 1:M 
        #     mul!(Y_bar[l,:,:], matrix, X_bar[l,:,:])
        # end
        # println("Time for Multiplcation: $time_total")
        # return refold_mat(Y_bar, (tensor_dim[1:mode - 1]..., mat_row, tensor_dim[mode + 1:d]...), mode)
        # return reshape(Y_bar, tensor_dim[1:mode - 1]..., mat_row, tensor_dim[mode + 1:d]...)
        return permutedims!(Y_tensor_arr, Y_tensor, invperm(perm))
    end
end

function pre_allocate_tensor(tensor::Array, matrices::Vector{<:AbstractMatrix})
    Y_arr = Vector{Array}(undef, length(matrices))
    tensor_permute = Vector{Array}(undef, length(matrices))
    tensor_dim = collect(size(tensor)) 
    d = length(tensor_dim) 
    # mat_dims = collect(size.(matrices)) 
    Y_list = Vector{Array}(undef, d)
    for i in 1:d 
        row = size(matrices[i], 1)
        col = Int(prod(tensor_dim)/tensor_dim[i])
        Y_arr[i] = zeros(ComplexF64, tensor_dim[1:i - 1]..., row, tensor_dim[i + 1:end]...)
        tensor_permute[i] = zeros(ComplexF64, tensor_dim[i], tensor_dim[setdiff(1:d, i)]...)
        # tensor_dim[i] = size(matrices[i], 1)
    end
    return Y_arr, tensor_permute
end

function pre_allocate(tensor::Array, matrices::Vector{<:AbstractMatrix})
    tensor_dim = collect(size(tensor)) 
    d = length(tensor_dim) 
    # mat_dims = collect(size.(matrices)) 
    M_list, P_list = zeros(Int, d), zeros(Int, d)
    Y_list = Vector{Array}(undef, d)
    for i in 1:d
        if i == 1
            M_list[i] = tensor_dim[1]
            P_list[i] = prod(tensor_dim[2:end])
            Y_list[i] = rand(ComplexF64, size(matrices[1], 1), P_list[i])
        else
            #The P's are correct here, but the M's change as we go through our Multi-TTM
            M_list[i] = prod(tensor_dim[1:i-1])
            P_list[i] = prod(tensor_dim[i+1:end])
            Y_list[i] = rand(ComplexF64, M_list[i], size(matrices[i], 1), P_list[i])
        end
        tensor_dim[i] = size(matrices[i], 1)
    end
    return M_list, P_list, Y_list 
end

function pre_allocate2(tensor::Array, matrices::Vector{<:AbstractMatrix})
    tensor_dim = collect(size(tensor)) 
    d = length(tensor_dim) 
    # mat_dims = collect(size.(matrices)) 
    M_list, P_list = zeros(Int, d), zeros(Int, d)
    Y_list = Vector{Array}(undef, d)
    for i in 1:d
        if i == 1
            M_list[i] = tensor_dim[1]
            P_list[i] = prod(tensor_dim[2:end])
            Y_list[i] = zeros(ComplexF64, size(matrices[1], 1), P_list[i])
        else
            #The P's are correct here, but the M's change as we go through our Multi-TTM
            M_list[i] = prod(tensor_dim[1:i-1])
            P_list[i] = prod(tensor_dim[i+1:end])
            Y_list[i] = zeros(ComplexF64, M_list[i], size(matrices[i], 1), P_list[i])
        end
        # tensor_dim[i] = size(matrices[i], 1)
    end
    return M_list, P_list, Y_list 
end

function pre_allocate_matrices(tensor::Array, matrices::Vector{<:AbstractMatrix})
    tensor_dim = collect(size(tensor)) 
    d = length(tensor_dim) 
    # mat_dims = collect(size.(matrices)) 
    # M_list, P_list = zeros(Int, d), zeros(Int, d)
    Y_list = Vector{Array}(undef, d)
    # for i in 1:d
    #     if i == 1
    #         M_list[i] = tensor_dim[1]
    #         P_list[i] = prod(tensor_dim[2:end])
    #         Y_list[i] = zeros(ComplexF64, size(matrices[1], 1), P_list[i])
    #     else
    #         #The P's are correct here, but the M's change as we go through our Multi-TTM
    #         M_list[i] = prod(tensor_dim[1:i-1])
    #         P_list[i] = prod(tensor_dim[i+1:end])
    #         Y_list[i] = zeros(ComplexF64, M_list[i], size(matrices[i], 1), P_list[i])
    #     end
    #     tensor_dim[i] = size(matrices[i], 1)
    # end
    for i in 1:d 
        row = size(matrices[i], 1)
        col = Int(prod(tensor_dim)/tensor_dim[i])
        Y_list[i] = zeros(ComplexF64, row, col)
        tensor_dim[i] = size(matrices[i], 1)
    end
    return Y_list 
end

function pre_allocate_matrices2(tensor::Array, matrices::Vector{<:AbstractMatrix})
    tensor_dim = collect(size(tensor)) 
    d = length(tensor_dim) 
    # mat_dims = collect(size.(matrices)) 
    # M_list, P_list = zeros(Int, d), zeros(Int, d)
    Y_list = Vector{Array}(undef, d)
    # for i in 1:d
    #     if i == 1
    #         M_list[i] = tensor_dim[1]
    #         P_list[i] = prod(tensor_dim[2:end])
    #         Y_list[i] = zeros(ComplexF64, size(matrices[1], 1), P_list[i])
    #     else
    #         #The P's are correct here, but the M's change as we go through our Multi-TTM
    #         M_list[i] = prod(tensor_dim[1:i-1])
    #         P_list[i] = prod(tensor_dim[i+1:end])
    #         Y_list[i] = zeros(ComplexF64, M_list[i], size(matrices[i], 1), P_list[i])
    #     end
    #     tensor_dim[i] = size(matrices[i], 1)
    # end
    for i in 1:d 
        row = size(matrices[i], 1)
        col = Int(prod(tensor_dim)/tensor_dim[i])
        Y_list[i] = zeros(ComplexF64, row, col)
        # tensor_dim[i] = size(matrices[i], 1)
    end
    return Y_list 
end

function Multi_TTM(tensor::Array, matrices::Vector{<:AbstractMatrix})
    Y = copy(tensor)  
    for i in 1:length(matrices)
        Y = TTM(Y, matrices[i], i)
    end
    return Y 
end

function Multi_TTM_recursive(tensor::Array, matrices::Vector{<:AbstractMatrix}, mode::Int=1)
    if mode > length(matrices)
        return tensor
    else
        return Multi_TTM_recursive(TTM(tensor, matrices[mode], mode), matrices, mode + 1)
    end
end

function Multi_TTM_allocate(tensor::Array, matrices::Vector{<:AbstractMatrix}, M_list::Union{Nothing, Vector{Int}}=nothing, P_list::Union{Nothing, Vector{Int}}=nothing, Y_list::Vector{Array}=nothing)
    Y = copy(tensor)
    for i in 1:length(matrices)
        Y = TTM_allocate(Y, matrices[i], M_list[i], P_list[i], Y_list[i], i)
    end
    return Y
end

function Multi_TTM_allocate_recursive(tensor::Array, matrices::Vector{<:AbstractMatrix}, M_list::Union{Nothing, Vector{Int}}=nothing, P_list::Union{Nothing, Vector{Int}}=nothing, Y_list::Vector{Array}=nothing, mode::Int=1)
    if mode > length(matrices)
        return tensor
    else
        # Y = TTM_allocate(tensor, matrices[mode], M_list[mode], P_list[mode], Y_list[mode], mode)
        return Multi_TTM_allocate_recursive(TTM_allocate(tensor, matrices[mode], M_list[mode], P_list[mode], Y_list[mode], mode), matrices, M_list, P_list, Y_list, mode + 1)
    end
end


function trim_by_tolerance(v::Vector{T}, tol::Real) where T<:Real
    idx = length(v)
    s = zero(T)
    println("Vector: ", v)
    while idx >= 1
        s += v[idx]^2
        if s >= tol^2
            break
        end
        idx -= 1
    end
    println("Trimmed to rank: ", idx)
    return idx
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

# function LLSV(Y::Array; cutoff::Union{Nothing,Float64}=nothing, target_rank::Union{Nothing,Int64}=nothing)
#     U, S, Vt = svd(Y)
#     # println("Singular Values: ", S)
#     if (cutoff === nothing) == (target_rank === nothing)
#         error("Specify either cutoff or target_rank, but not both.")
#     end
#     if cutoff !== nothing
#         rank = trim_by_tolerance(S, cutoff)
#         # println("Truncated rank by cutoff: ", rank)
#     else
#         rank = target_rank
#     end
#     # println("Chosen rank: ", rank)
#     W = U[:, 1:rank]
#     err = sqrt(sum(S[rank+1:end].^2))
#     return W, err
# end

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
    W = U[:, 1:rank]
    if verbose == true
        println("Singular Values for mode $mode: ", S)
        println("Removed Singular Values for mode $mode: ", S[rank + 1:end])
        println("Rank of factor $mode: $rank")
        println("---------------------------------------------------------")
    end
    err = sqrt(sum(S[rank+1:end].^2))
    return W, err
end

# function tucker(tensor::Array; cutoff::Union{Nothing,Float64}=nothing, target_rank::Union{Nothing,Vector{Int64}}=nothing)
#     d = length(size(tensor))
#     if target_rank === nothing && cutoff == 0
#         target_rank_vec = fill(nothing, d)
#     else
#         target_rank_vec = target_rank
#     end
#     U_list = Matrix{eltype(tensor)}[]
#     core = copy(tensor)
#     core_copy = copy(tensor)
#     err_list = zeros(d)
#     cutoff_bar = cutoff !== nothing ? cutoff*norm(tensor)/sqrt(d) : nothing
#     for i in 1:d
#         U, err = LLSV(matricization(core_copy, i); cutoff = cutoff, target_rank = target_rank_vec[i])
#         push!(U_list, U)
#         err_list[i] = err
#         core = TTM(core, Array(U'), i)
#     end
#     total_err = sqrt(sum(err_list.^2))
#     return core, U_list, total_err
# end

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

#Test that refold and matricization are inverses
# A = rand(ComplexF64, 4,5,6,4)
# for mode in 1:4
#     A_mat = matricization(A, mode)
#     A_refold = refold(A_mat, size(A), mode)
#     println("Norm difference matricization and refold mode $mode: ", norm(A - A_refold))
# end

#Test Tucker decomposition
# truncation_cutoff = 1E-1
# A = rand(ComplexF64, 4,5,6)
# A = zeros(2, 3, 3)
# A[:,:,1] = [-1 -2 -3; -4 -5 -6]
# A[:,:,2] = [1 2 3; 4 5 6]
# A[:,:,3] = [7 8 9; 10 11 12]

# core, U_list, err = tucker(A; cutoff = truncation_cutoff)

# #Compare with ITensor.jl
# using ITensors
# i = Index(4; tags="i")
# j = Index(5; tags="j")
# k = Index(6; tags="k")
# sites = [i,j,k]
# A_itensor = ITensor(A, i,j,k)

# core_iten, factors_iten = tucker_itensor(A_itensor;cutoff = truncation_cutoff)

# #Compare results
# A_reconstructed = Multi_TTM(core, U_list)
# println("Norm difference Tucker: ", norm(A - A_reconstructed))

# #Compare now with itensor 
# A_iten_reconstructed = core_iten*factors_iten[1]*factors_iten[2]*factors_iten[3]
# A_iten_reconstructed = permute(A_iten_reconstructed, (i,j,k))
# println("Norm difference Tucker with ITensor.jl: ", norm(A - Array(A_iten_reconstructed, inds(A_iten_reconstructed))))

# #Now compare cores and factors between the two methods
# println("Norm difference cores: ", norm(core - Array(core_iten, inds(core_iten))))
# for n in 1:3
#     println("Norm difference factor $n: ", norm(U_list[n] - Array(factors_iten[n], inds(factors_iten[n]))))
# end

#Test Multi_TTM
# A = rand(ComplexF64, 2,3,4)
# B = rand(ComplexF64, 5,2)
# C = rand(ComplexF64, 6,3)
# D = rand(ComplexF64, 7,4)
# @btime begin 
# A_new = Multi_TTM(A, [B,C,D])
# end
# #Test with sequential TTM
# A_new2 = TTM(TTM(TTM(A, B, 1), C, 2), D, 3)
# println("Norm difference Multi_TTM: ", norm(A_new - A_new2))
# #Now test with ITensor.jl
# using ITensors
# i = Index(2; tags="i")
# j = Index(3; tags="j")
# k = Index(4; tags="k")
# l = Index(5; tags="l")
# m = Index(6; tags="m")
# n = Index(7; tags="n")
# A_itensor = ITensor(A, i,j,k)
# B_itensor = ITensor(B, l,i)
# C_itensor = ITensor(C, m,j)
# D_itensor = ITensor(D, n,k)
# @btime begin 
# A_itensor_new = A_itensor*B_itensor*C_itensor*D_itensor
# end
# A_itensor_new = permute(A_itensor_new, (l,m,n))
# println("Norm difference Multi_TTM with ITensor.jl: ", norm(A_new - Array(A_itensor_new, inds(A_itensor_new))))


#Test TTM compared to ITensor.jl TTM 
# using ITensors
# A = rand(ComplexF64, 2,3,4)
# B = rand(ComplexF64, 5,3)
# C = rand(ComplexF64, 6,2)
# i = Index(2; tags="i")
# j = Index(3; tags="j")
# k = Index(4; tags="k")
# l = Index(5; tags="l")
# m = Index(6; tags="m")
# A_itensor = ITensor(A, i,j,k)
# B_itensor = ITensor(B, l,j)
# C_itensor = ITensor(C, m,i)
# #Test TTM on mode 1
# A_new = TTM(A, C, 1)
# A_itensor_new = A_itensor*C_itensor
# A_itensor_new = permute(A_itensor_new, (m,j,k))
# println("Norm difference TTM mode 1: ", norm(A_new - Array(A_itensor_new, inds(A_itensor_new))))
# #Test TTM on mode 2
# A_new2 = TTM(A, B, 2)
# A_itensor_new2 = A_itensor*B_itensor
# A_itensor_new2 = permute(A_itensor_new2, (i,l,k))
# println("Norm difference TTM mode 2: ", norm(A_new2 - Array(A_itensor_new2, inds(A_itensor_new2))))
# #Test TTM on mode 3
# D = rand(ComplexF64, 7,4)
# n = Index(7; tags="n")
# A_new3 = TTM(A, D, 3)
# D_itensor = ITensor(D, n,k)
# A_itensor_new3 = A_itensor*D_itensor
# A_itensor_new3 = permute(A_itensor_new3, (i,j,n))
# println("Norm difference TTM mode 3: ", norm(A_new3 - Array(A_itensor_new3, inds(A_itensor_new3))))

# #Now try with order 4 tensor 
# A4 = rand(ComplexF64, 2,3,4,5)
# B4 = rand(ComplexF64, 6,2)
# C4 = rand(ComplexF64, 7,3)
# D4 = rand(ComplexF64, 8,4)
# E4 = rand(ComplexF64, 9,5)
# i4 = Index(2; tags="i4")
# j4 = Index(3; tags="j4")
# k4 = Index(4; tags="k4")
# l4 = Index(5; tags="l4")
# m4 = Index(6; tags="m4")
# n4 = Index(7; tags="n4")
# o4 = Index(8; tags="o4")
# p4 = Index(9; tags="p4")
# A4_itensor = ITensor(A4, i4,j4,k4,l4)
# B4_itensor = ITensor(B4, m4,i4)
# C4_itensor = ITensor(C4, n4,j4)
# D4_itensor = ITensor(D4, o4,k4)
# E4_itensor = ITensor(E4, p4,l4)
# #Test TTM on mode 1
# @btime begin 
# A4_new = TTM(A4, B4, 1)
# end 
# @btime begin 
# A4_itensor_new = A4_itensor*B4_itensor
# end
# # A4_itensor_new = permute(A4_itensor_new, (m4,j4,k4,l4))
# # println("Norm difference TTM mode 1 (order 4): ", norm(A4_new - Array(A4_itensor_new, inds(A4_itensor_new))))
# #Test TTM on mode 2
# @btime begin
# A4_new2 = TTM(A4, C4, 2)
# end
# println("End")
# @btime begin
# A4_itensor_new2 = A4_itensor*C4_itensor
# end
# # A4_itensor_new2 = permute(A4_itensor_new2, (i4,n4,k4,l4))
# # println("Norm difference TTM mode 2 (order 4): ", norm(A4_new2 - Array(A4_itensor_new2, inds(A4_itensor_new2))))
# #Test TTM on mode 3
# @btime begin
# A4_new3 = TTM(A4,D4, 3)
# end
# @btime begin
# A4_itensor_new3 = A4_itensor*D4_itensor
# end
# # A4_itensor_new3 = permute(A4_itensor_new3, (i4,j4,o4,l4))
# # println("Norm difference TTM mode 3 (order 4): ", norm(A4_new3 - Array(A4_itensor_new3, inds(A4_itensor_new3))))
# #Test TTM on mode 4
# @btime begin
# A4_new4 = TTM(A4, E4, 4)
# end
# @btime begin
# A4_itensor_new4 = A4_itensor*E4_itensor
# end
# A4_itensor_new4 = permute(A4_itensor_new4, (i4,j4,k4,p4))
# println("Norm difference TTM mode 4 (order 4): ", norm(A4_new4 - Array(A4_itensor_new4, inds(A4_itensor_new4))))    


#Test Multi_TTM with Multi_TTM versus ITensor method
# A4 = rand(ComplexF64, 4,4,4,4)
# B4 = rand(ComplexF64, 4,4)
# C4 = rand(ComplexF64, 4,4)
# D4 = rand(ComplexF64, 4,4)
# E4 = rand(ComplexF64, 4,4)
# factors = [B4, C4, D4, E4]
# M_wa, P_wa, Y_wa = pre_allocate(A4, factors)
#This is fine, the issue begins when I have factors of smaller size
# @btime begin 
# ans1 = Multi_TTM_recursive(A4, factors)
# end 
# @btime begin 
# ans2 = Multi_TTM_allocate_recursive(A4, factors, M_wa, P_wa, Y_wa)
# end
# println("Error: ", norm(ans1 - ans2))

# A4_2 = rand(ComplexF64, 3, 3, 3, 3)
# B4_2 = rand(ComplexF64, 4,3)
# C4_2 = rand(ComplexF64, 4,3)
# D4_2 = rand(ComplexF64, 4,3)
# E4_2 = rand(ComplexF64, 4,3)
# factors_2 = [B4_2, C4_2, D4_2, E4_2]
# @btime begin 
# ans1 = Multi_TTM_recursive(A4_2, factors_2)
# end 
# @btime begin 
# ans2 = Multi_TTM_allocate_recursive(A4_2, factors_2, M_wa, P_wa, Y_wa)
# end
# println("Error 2: ", norm(ans1 - ans2))

# i4 = Index(20; tags="i4")
# j4 = Index(21; tags="j4")
# k4 = Index(22; tags="k4")
# l4 = Index(23; tags="l4")
# m4 = Index(6; tags="m4")
# n4 = Index(7; tags="n4")
# o4 = Index(8; tags="o4")
# p4 = Index(9; tags="p4")
# A4_itensor = ITensor(A4, i4,j4,k4,l4)
# B4_itensor = ITensor(B4, m4,i4)
# C4_itensor = ITensor(C4, n4,j4)
# D4_itensor = ITensor(D4, o4,k4)
# E4_itensor = ITensor(E4, p4,l4)

# A = rand(ComplexF64, 2, 2, 2)
# B = rand(ComplexF64, 2, 2)
# C = rand(ComplexF64, 2, 2)
# D = rand(ComplexF64, 2, 2)

# core = A
# factors = [B, C, D]
# M_list, P_list, Y_list = pre_allocate(core, factors)
# M_list_l, P_list_l, Y_list_l = pre_allocate2(core, factors)
# Y_list2 = pre_allocate_matrices(core, factors)
# Y_list3 = pre_allocate_matrices2(core, factors)
# Y_arr, Y_arr_permute = pre_allocate_tensor(core, factors)
# A4 = TTM_allocate3(core, C4, Y_list3[2], 2, Y_arr_permute[2], Y_arr[2])
# @btime begin 
# A4_slow = Multi_TTM(core, factors)
# end
# @btime begin
# A_recursive = Multi_TTM_recursive(core, factors)
# A_recursive_allocate = Multi_TTM_allocate_recursive(core, factors, M_list, P_list, Y_list)
# A_recursive2 = Multi_TTM_recursive(core, factors)
# println("Diff: ", norm(A_recursive - A_recursive2))
# A_recursive_allocate2 = Multi_TTM_allocate_recursive(core, factors, M_list, P_list, Y_list)
# println("Diff: ", norm(A_recursive_allocate - A_recursive_allocate2))
# end
# println("Multi-TTM ITensor: ")
# @btime begin 
# A4 = A4_itensor*B4_itensor*C4_itensor*D4_itensor*E4_itensor
# end


# println("Multi-TTM Batched Matrix-Matrix: ")
# @btime begin
# A4_recursive_allocate = Multi_TTM_allocate_recursive(core, factors, M_list, P_list, Y_list)
# end
# println("Norm difference Multi_TTM and recursive: ", norm(A4_slow - A4_recursive_allocate))
# @btime begin 
# A4_fast = Multi_TTM_allocate(core, factors, M_list, P_list, Y_list)
# end
# #compare speed results 

#Test single TTM without ITensor vs with ITensor
# @time begin 
#     A4_new = TTM(TTM(TTM(TTM(A4, B4, 1), C4, 2), D4, 3), E4, 4)
# end
# @benchmark TTM_allocate(A4, B4, M_list[1], P_list[1], Y_list[1], 1)
# A4_n = TTM_allocate(A4, B4, M_list[1], P_list[1], Y_list[1], 1)
# A4_n2 = TTM_allocate2(A4, B4, M_list[1], P_list[1], Y_list[1], 1)
# A4_n3 = TTM_allocate3(A4, B4, M_list[1], P_list[1], Y_list2[1], 1)
# println("Norm difference TTM_allocate and TTM_allocate2: ", norm(A4_n - A4_n2))
# println("Norm difference TTM_allocate and TTM_allocate3: ", norm(A4_n - A4_n3))
# println("Norm difference TTM_allocate and TTM_allocate2: ", norm(A4_new - A4_new2))
# A4_n = TTM_allocate(A4, B4, M_list[1], P_list[1], Y_list[1], 1)
# println("Batched Matrix-Matrix TTM Mode 2: ")
# @btime begin
#     TTM_allocate(A4_n, C4, M_list[2], P_list[2], Y_list[2], 2)
# end
# A4_nn2 = TTM_allocate2(A4_n2, C4, M_list[2], P_list[2], Y_list[2], 2)
# A4_nn3 = TTM_allocate3(A4_n3, C4, M_list[2], P_list[2], Y_list2[2], 2)
# println("Norm difference TTM_allocate and TTM_allocate2: ", norm(A4_nn - A4_nn2))
# println("Norm difference TTM_allocate and TTM_allocate3: ", norm(A4_nn - A4_nn3))

# A4_new = TTM_allocate(A4_new, C4, M_list[2], P_list[2], Y_list[2], 2)
# @btime begin TTM_allocate(A4_new, D4, M_list[3], P_list[3], Y_list[3], 3) end
# A4_new = TTM_allocate(A4_new, D4, M_list[3], P_list[3], Y_list[3], 3)
# @btime begin TTM_allocate(A4_new, E4, M_list[4], P_list[4], Y_list[4], 4) end
# A4_new = TTM_allocate(A4_new, E4, M_list[4], P_list[4], Y_list[4], 4)


# @btime begin A4_itensor*B4_itensor end
# A4_itensor_new = A4_itensor*B4_itensor
# println("ITensor TTM Mode 2: ")
# @btime begin A4_itensor_new*C4_itensor end
# A4_itensor_new = A4_itensor_new*C4_itensor
# @btime begin A4_itensor_new*D4_itensor end
# A4_itensor_new = A4_itensor_new*D4_itensor
# @btime begin A4_itensor_new*E4_itensor end
# A4_itensor_new = A4_itensor_new*E4_itensor
# println("Norm Difference: ", norm(A4_slow - A4_new))
# println("Norm Difference ITensor: ", norm(A4_slow - Array(A4_itensor_new, inds(A4_itensor_new))))
# end
# println("End Multi_TTM")
# A4_itensor_new = A4_itensor*B4_itensor*C4_itensor*D4_itensor*E4_itensor

# A4_itensor_new = permute(A4_itensor_new, (m4,n4,o4,p4))
# println("Norm difference Multi_TTM (order 4): ", norm(A4_new - Array(A4_itensor_new, inds(A4_itensor_new))))
# println("ITensor TTM Mode 1: ")
# @btime begin
#     A4_itensor*B4_itensor
# end
# println("ITensor TTM Mode 2: ")
# @btime begin
#     A4_itensor*C4_itensor
# end
# println("ITensor TTM Mode 3: ")
# @btime begin
#     A4_itensor*D4_itensor
# end
# println("ITensor TTM Mode 4: ")
# @btime begin
#     A4_itensor*E4_itensor
# end

# println("Batched Matrix-Matrix TTM Mode 1: ")
# @btime begin
#     TTM_allocate(core, B4, M_list_l[1], P_list_l[1], Y_list_l[1], 1)
# end
# println("Batched Matrix-Matrix TTM Mode 2: ")
# @btime begin
#     TTM_allocate(core, C4, M_list_l[2], P_list_l[2], Y_list_l[2], 2)
# end
# println("Batched Matrix-Matrix TTM Mode 3: ")
# @btime begin
#     TTM_allocate(core, D4, M_list_l[3], P_list_l[3], Y_list_l[3], 3)
# end
# println("Batched Matrix-Matrix TTM Mode 4: ")
# @btime begin
#     TTM_allocate(core, E4, M_list_l[4], P_list_l[4], Y_list_l[4], 4)
# end

# # TTM_allocate(core, C4, M_list_l[2], P_list_l[2], Y_list_l[2], 2)
# println("Unfolding TTM Mode 1: ")
# @btime begin
#     TTM_allocate3(core, B4, Y_list3[1], 1, Y_arr_permute[1], Y_arr[1])
# end
# println("Unfolding TTM Mode 2: ")
# @btime begin
#     TTM_allocate3(core, C4, Y_list3[2], 2, Y_arr_permute[2], Y_arr[2])
# end
# println("Unfolding TTM Mode 3: ")
# @btime begin
#     TTM_allocate3(core, D4, Y_list3[3], 3, Y_arr_permute[3], Y_arr[3])
# end
# println("Unfolding TTM Mode 4: ")
# @btime begin
#     TTM_allocate3(core, E4, Y_list3[4], 4, Y_arr_permute[4], Y_arr[4])
# end


# println("End")
