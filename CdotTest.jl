using ITensors, ITensorMPS, LinearAlgebra
include("Tucker_Matrices.jl")
include("BUG_small.jl")



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
        # @time begin 
        # Y = matrix * reshape(tensor, row, col)

        mul!(Y_alloc, matrix, reshape(tensor, M, P))
        # end
        return reshape(Y_alloc, mat_row, P_inds...)
    else
        # println(size(Y_bar))
        X_bar = reshape(tensor, M, tensor_dim[mode], P)
        #most of the memory is here
        
        # matT = transpose(matrix)
        # time_total = 0.0
        # mul!(Y_bar, matrix, matricization(tensor, mode))
        # Y_bar = matrix*matricization(tensor, mode)
        @views for l = 1:P
            # t_start = time_ns()
            mul!(Y_alloc[:, :, l], X_bar[:, :, l], transpose(matrix))
            # Y_alloc[:,:,l] = X_bar[:,:,l]*transpose(matrix)
            # t_end = time_ns()
            # time_total += (t_end - t_start)/1E9
        end
        # @views for l = 1:M 
        #     mul!(Y_bar[l,:,:], matrix, X_bar[l,:,:])
        # end
        # println("Time for Multiplcation: $time_total")
        return reshape(Y_alloc, tensor_dim[1:mode - 1]..., mat_row, tensor_dim[mode + 1:d]...)
    end
end

function Multi_TTM(tensor::Array, matrices::Vector{<:AbstractMatrix})
    Y = copy(tensor)  
    for i in 1:length(matrices)
        Y = TTM(Y, matrices[i], i)
    end
    return Y 
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

function C_dot_test(core, factors, total_H, Ms, Ps, Ys)
    N_ops = length(total_H)
    
    N_factors = length(factors)

    init = zeros(eltype(core), size(core)...)

    new_factors = [zeros(eltype(factors[i]), size(factors[i], 2), size(factors[i], 2)) for i in 1:length(factors)]
    temp_factors = [zeros(eltype(factors[i]), size(factors[i], 2), size(factors[i], 2)) for i in 1:length(factors)]

    for i in 1:N_ops
        # println("Operation $i: ") 
        for j in 1:N_factors
            mul!(temp_factors[j], total_H[i][j], factors[j])
            mul!(new_factors[j], temp_factors[j]', factors[j])

        end
        # init .+= -im*Multi_TTM_allocate_recursive(core, new_factors, Ms, Ps, Ys)
        init .+= -im*Multi_TTM_recursive(core, new_factors)
    end
    return init
end

function fixed_point_iter_C_mat(H_ops, core, h, factors, M_list, P_list, Y_list, maxiter, tol, verbose)
    K_init = zeros(eltype(core), size(core))
    for i in 1:maxiter 
        K = C_dot_test(core + 0.5*h*K_init, factors, H_ops, M_list, P_list, Y_list)
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

function IMR_core_mat(H_ops, core, h, factors_matrices, M_list, P_list, Y_list, maxiter, tol, verbose)
    K = fixed_point_iter_C_mat(H_ops, core, h, factors_matrices, M_list, P_list, Y_list, maxiter, tol, verbose)
    core .+= h*K
    return core 
end

N = 3
ops = [identity_ops(N)] 
A = rand(ComplexF64, collect(fill(2, N))...)

core_arr, factors_arr = tucker(A; cutoff = 0.0)

M_list, P_list, Y_list = pre_allocate(core, factors)

core_new = fixed_point_iter_C_mat(ops, core_arr, 0.1, factors_arr, M_list, P_list, Y_list, 10, 1E-14, true)
