using LinearAlgebra, ITensors, ITensorMPS, Random, Plots, ProgressMeter

# include("controlH.jl")
# pyplot()
ITensors.set_warn_order(18)
# include("Hamiltonian.jl")
#Create tucker-tensor format for ITensor 
# function tucker_tensor(A, rank)
#     factors = []
#     U = reduced_SVD(unfolding(A, 1), rank)
#     push!(factors, U)
#     V = reduced_SVD(unfolding(A, 2), rank)
#     push!(factors, V)
#     W = reduced_SVD(unfolding(A, 3), rank)
#     push!(factors, W)

#     # core = ttm_mode_k(A, U', 1)
#     # println("TTM Mode 1")
#     # display(core)
#     # core = ttm_mode_1_diff(A, U')
#     # core = ttm_mode_k(core, V', 2)
#     # println("TTM Mode 2")
#     # display(core)
#     core = ttm_mode_1_diff(A, U')
#     core = ttm_mode_2_diff(core, V')
#     # core = ttm_mode_k(core, W', 3)
#     core = ttm_mode_3_diff(core, W')
#     # println("Core")
#     return core, factors
# end

Random.seed!(42)

function create_linkinds(N::Int64, link_size::Vector{Int64})
    ind_vec = Index{Int64}[]
    for i in 1:N-1
        ind = Index(link_size[i];tags="Link, $i")
        push!(ind_vec, ind)
    end
    return ind_vec
end

function create_links_tucker(N::Int64, link_size::Vector{Int64})
    ind_vec = Index{Int64}[]
    for i in 1:N 
        ind = Index(link_size[i]; tags="Link, $i")
        push!(ind_vec, ind)
    end
    return ind_vec 
end

function magnetization_MPO(sites, magnetization_site)
    N = length(sites)
    H = MPO(N)
    links = Int64.(ones(N - 1))
    link_ind = create_linkinds(N, links)
    Ident = Matrix(1.0*I, 2, 2)
    sz = [1.0 0.0; 0.0 -1.0]
    for i in 1:N
        if i == 1 
            core = zeros(2, 2, links[i]) 
            if magnetization_site != i 
                core[:,:,1] = Ident
            else 
                core[:,:,1] = sz
            end
            core_ten = ITensor(core, sites[i], sites[i]', link_ind[i])
        elseif i == N
            core = zeros(links[N - 1], 2, 2)
            if magnetization_site != i 
                core[1,:,:] = Ident
            else 
                core[1,:,:] = sz
            end
            core_ten = ITensor(core, link_ind[i - 1], sites[i], sites[i]')
        else
            core = zeros(links[i - 1], 2, 2, links[i])
            if magnetization_site != i 
                core[1,:,:,1] = Ident
            else 
                core[1,:,:,1] = sz
            end
            core_ten = ITensor(core, link_ind[i - 1], sites[i], sites[i]', link_ind[i]) 
        end
        H[i] = core_ten
    end
    return H
end

function magnetization_tucker(sites, magnetization_site)
    N = length(sites)
    factors = []
    links = Int64.(ones(N))
    links_tuple = (links...,)
    link_ind = create_links_tucker(N, links)
    Ident = Matrix(1.0*I, 2, 2)
    sz = [1.0 0.0; 0.0 -1.0]
    core = Array{Float64}(undef, links_tuple)
    core[1] = 1.0
    for i in 1:N
        factor = zeros(2,2,1) 
        if i == magnetization_site 
            factor[:,:,1] = sz 
        else
            factor[:,:,1] = Ident 
        end
        factor_ten = ITensor(factor, sites[i], sites[i]', link_ind[i])
        push!(factors, factor_ten)
    end
    core_ten = ITensor(core, link_ind)

    return core_ten, factors
end

function tucker_itensor(A; cutoff::Float64 = 0.0)
    A_inds = inds(A)
    N = length(A_inds)
    factors = Vectors{ITensor}(undef, N)
    A_copy = copy(A)
    core = copy(A)
    for i in 1:N
        U, S, V = svd(A_copy, A_inds[i]; cutoff)
        # push!(factors, U)
        factors[i] = U
        core = core*conj(U)
    end
    return core, factors 
end

function tucker_itensor(A, sites; cutoff::Float64 = 0.0)
    A_ten = ITensor(A, sites)
    N = length(sites)
    factors = Vector{ITensor}(undef, N) 
    A_inds = inds(A_ten)
    A_copy = copy(A_ten)
    core = copy(A_ten)
    for i in 1:N
        # println("Index $i") 
        # println(A_inds[i])
        U, S, V = svd(A_copy, A_inds[i]; cutoff)
        # push!(factors, U)
        factors[i] = U
        core = core*conj(U)
    end
    return core, factors 
end


function reconstruct(core, factors)
    N = length(factors)
    for i in 1:N 
        core = core*factors[i]
    end 
    return core 
end

function TTM_alloc!(core::ITensor, factor::ITensor, alloc_iten::ITensor)
    ITensors.contract!(alloc_iten, core, factor)
    # return alloc_arr 
end

function Multi_TTM!(core::ITensor, factors::Vector{ITensor}, alloc_tensors::Vector{ITensor})
    N = length(factors)
    for i in 1:N
        if i == 1
            TTM_alloc!(core, factors[i], alloc_tensors[i])
        else
            TTM_alloc!(alloc_tensors[i - 1], factors[i], alloc_tensors[i])
        end
    end
end

function preallocate_itensor(core::ITensor, factors::Vector{ITensor})
    core_inds = collect(inds(core))
    N = length(factors)
    alloc_list = Vector{ITensor}(undef, N)
    for i in 1:N 
        core_inds[i] = uniqueinds(factors[i], core)[1]
        # println(core_inds)
        alloc_list[i] = ITensor(core_inds...)
    end
    return alloc_list 
end

function reconstruct_mat(core, factors)
    N = length(factors)
    core_ind = inds(core)
    # println("INDSSS:", inds(core))
    for i in 1:N 
        # println("FACTOR $i: ", factors[i])
        # println("FACTOR $i tensor: ", ITensor(factors[i], core_ind[i], prime(core_ind[i])))
        core = core*ITensor(factors[i], prime(core_ind[i]), core_ind[i])
    end 
    return core 
end


function factor_kron(factor_matrices, i)
    N = length(factor_matrices)
    which_factor = setdiff(1:N, i)
    init = factor_matrices[maximum(which_factor)]
    which_factor = setdiff(which_factor, maximum(which_factor))
    for i in reverse(which_factor)
        init = kron(init, factor_matrices[i])
    end
    return init 
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

function matricization(tensor, mode)
    tensor_inds = collect(inds(tensor))
    inds_copy = copy(tensor_inds)
    deleteat!(inds_copy, mode)
    C = combiner(inds_copy;tags="c")
    TC = tensor*C 
    # println("Inds C:", inds(C))
    # println("Inds TC: ", inds(TC))
    TC_arr = Array(TC, inds(TC))
    return TC_arr 
end

function conj_factors(factors)
    N = length(factors)
    factor_conj = []
    for i in 1:N 
        push!(factor_conj, prime(conj(factors[i]);tags="Site"))
    end
    return factor_conj 
end

function conj_factors2(factors)
    N = length(factors)
    factor_conj = []
    for i in 1:N 
        push!(factor_conj,prime(conj(factors[i])))
    end
    return factor_conj 
end

function fixed_point_iter_C(H_ten, core, h, factor_matrices, maxiter, tol, verbose)
    core_inds = collect(inds(core))
    K_init = ITensor(ComplexF64, core_inds)
    for i in 1:maxiter 
        # println("Look here core!")
        # println(core)
        intermediate = -im*H_ten*reconstruct(core + 0.5*h*K_init, factor_matrices)
        K = reconstruct(intermediate, conj_factors(factor_matrices))
        # println("LOook here")
        # println(K)
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

function fixed_point_iter_C_ten(H_ops, core, h, factors, maxiter, tol, verbose)
    core_inds = collect(inds(core))
    K_init = ITensor(ComplexF64, core_inds)
    for i in 1:maxiter 
        K = C_dot_itensor_im(core + 0.5*h*K_init, factors, H_ops, sites)
        error = norm(K_init - K)
        if verbose == true
            println("Iteration $i") 
            println("Error: ", error)
        end
        K_init = K 
        if error < tol 
            break 
        end
    end
    return K_init 
end

function fixed_point_iter_C_ten(H_ops, core, h, factors, alloc_arr, maxiter, tol, verbose)
    core_inds = collect(inds(core))
    K_init = ITensor(ComplexF64, core_inds)
    for i in 1:maxiter 
        K = C_dot_itensor_im(core + 0.5*h*K_init, factors, alloc_arr, H_ops, sites)
        error = norm(K_init - K)
        if verbose == true
            println("Iteration $i") 
            println("Error: ", error)
        end
        K_init = K 
        if error < tol 
            break 
        end
    end
    return K_init 
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

function IMR_core_itensor(H_ten, core, h, factor_matrices, maxiter, tol, verbose)
    K = fixed_point_iter_C(H_ten, core, h, factor_matrices, maxiter, tol, verbose)
    core = core + h*K
    return core 
end

function IMR_core_itensor_ten(H_ops, core, h, factor_matrices, maxiter, tol, verbose)
    K = fixed_point_iter_C_ten(H_ops, core, h, factor_matrices, maxiter, tol, verbose)
    core = core + h*K
    return core 
end

function IMR_core_itensor_ten(H_ops, core, h, factor_matrices, alloc_arr, maxiter, tol, verbose)
    K = fixed_point_iter_C_ten(H_ops, core, h, factor_matrices, alloc_arr, maxiter, tol, verbose)
    core = core + h*K
    return core 
end

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

#This allocates the maximum size array, will probably need to change it in the future when this array becomes too large to store
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

#Need to pre-allocate all possible arrays, so doesn't matter the size of the matrices we'll just the size of the tensor as 
#the maximum sizes, technically incorrect but in our case it's fine because this is for working with BUG
# function BUG_allocate(tensor::Array)
#     tensor_dim = collect(size(tensor))
    


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

function TTM_allocate(tensor::AbstractArray, matrix::AbstractMatrix, M::Int, P::Int, Y_alloc::AbstractArray, mode::Int)
    tensor_dim = collect(size(tensor))
    println(tensor_dim)
    d = length(tensor_dim)
    mat_row, mat_col = size(matrix)
    # println("Mode: $mode")
    # println("M: $M")
    # println("P: $P")
    println("Mode $mode")
    if mode == 1
        subY_alloc = @view Y_alloc[1:size(matrix, 1),1:P]
        # M = tensor_dim[1]
        P_inds = copy(tensor_dim)
        deleteat!(P_inds, 1)
        # col = Int(prod(size(tensor)) / size(tensor, 1))
        # row = size(tensor, 1)
        # @time begin 
        # Y = matrix * reshape(tensor, row, col)
        # println("Size Y_alloc: ", size(Y_alloc))
        # println("Size matrix: ", size(matrix))
        # println("Size reshape: ", size(reshape(tensor, M, P)))
        mul!(subY_alloc, matrix, reshape(tensor, M, P))
        # end
        return reshape(subY_alloc, mat_row, P_inds...)
    else
        # println(size(Y_bar))
        println("M: ", M)
        println("P: ", P)
        println("tensor_dim[mode]: ", tensor_dim[mode])
        X_bar = reshape(tensor, prod(tensor_dim[1:mode - 1]), tensor_dim[mode], prod(tensor_dim[1:mode + 1: d]))
        println("Size X_bar: ", size(X_bar))
        #most of the memory is here
        
        # matT = transpose(matrix)
        # time_total = 0.0
        # mul!(Y_bar, matrix, matricization(tensor, mode))
        # Y_bar = matrix*matricization(tensor, mode)
        @views for l = 1:tensor_dim[1:mode + 1: d]
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

function Multi_TTM(tensor::AbstractArray, matrices::Vector{<:AbstractMatrix})
    Y = copy(tensor)  
    for i in 1:length(matrices)
        Y = TTM(Y, matrices[i], i)
    end
    return Y 
end

function Multi_TTM_allocate(tensor::AbstractArray, matrices::Vector{<:AbstractMatrix}, M_list::Union{Nothing, Vector{Int}}=nothing, P_list::Union{Nothing, Vector{Int}}=nothing, Y_list::Vector{Array}=nothing)
    Y = copy(tensor)  
    for i in 1:length(matrices)
        Y = TTM_allocate(Y, matrices[i], M_list[i], P_list[i], Y_list[i], i)
    end
    return Y
end

function Multi_TTM_allocate_recursive(tensor::AbstractArray, matrices::Vector{<:AbstractMatrix}, M_list::Union{Nothing, Vector{Int}}=nothing, P_list::Union{Nothing, Vector{Int}}=nothing, Y_list::Vector{Array}=nothing, mode::Int=1)
    if mode > length(matrices)
        return tensor
    else
        # Y = TTM_allocate(tensor, matrices[mode], M_list[mode], P_list[mode], Y_list[mode], mode)
        return Multi_TTM_allocate_recursive(TTM_allocate(tensor, matrices[mode], M_list[mode], P_list[mode], Y_list[mode], mode), matrices, M_list, P_list, Y_list, mode + 1)
    end
end

#Allocated memory version
function C_dot_test(core, factors, total_H, Ms, Ps, Ys)
    println("C_dot_test")
    N_ops = length(total_H)
    N_factors = length(factors)

    init = zeros(eltype(core), size(core)...)
    # println("Size original factors: ")
    # println("Size: ", size(factors[1]))
    # println("Size: ", size(factors[2]))
    # println("Size: ", size(factors[3]))
    new_factors = [zeros(eltype(factors[i]), size(factors[i], 2), size(factors[i], 2)) for i in 1:length(factors)]
    temp_factors = [zeros(eltype(factors[i]), size(factors[i], 1), size(factors[i], 2)) for i in 1:length(factors)]

    for i in 1:N_ops
        # println("Operation $i: ") 
        for j in 1:N_factors
            mul!(temp_factors[j], total_H[i][j], factors[j])
            mul!(new_factors[j], temp_factors[j]', factors[j])
        end
        # println("Size new_factors: ")
        # println("Size ", size(new_factors[1]))
        # println("Size ", size(new_factors[2]))
        # println("Size ", size(new_factors[3]))
        # init .+= -im*Multi_TTM_allocate_recursive(core, new_factors, Ms, Ps, Ys)
        # println("new factors: ")
        # display(new_factors)
        init .+= -im*Multi_TTM_recursive(core, new_factors)
    end
    return init
end

#Non-allocated memory version
function C_dot_test(core, factors, total_H)
    # println("C_dot_test")
    N_ops = length(total_H)
    N_factors = length(factors)

    init = zeros(eltype(core), size(core)...)
    # println("Size original factors: ")
    # println("Size: ", size(factors[1]))
    # println("Size: ", size(factors[2]))
    # println("Size: ", size(factors[3]))
    new_factors = [zeros(eltype(factors[i]), size(factors[i], 2), size(factors[i], 2)) for i in 1:length(factors)]
    temp_factors = [zeros(eltype(factors[i]), size(factors[i], 1), size(factors[i], 2)) for i in 1:length(factors)]

    for i in 1:N_ops
        # println("Operation $i: ") 
        for j in 1:N_factors
            mul!(temp_factors[j], total_H[i][j], factors[j])
            mul!(new_factors[j], temp_factors[j]', factors[j])
        end
        # println("Size new_factors: ")
        # println("Size ", size(new_factors[1]))
        # println("Size ", size(new_factors[2]))
        # println("Size ", size(new_factors[3]))
        # init .+= -im*Multi_TTM_allocate_recursive(core, new_factors, Ms, Ps, Ys)
        # println("new factors: ")
        # display(new_factors)
        init .+= -im*Multi_TTM_recursive(core, new_factors)
    end
    return init
end

#Allocated memory version
function C_dot_test2(core, factors, total_H, Ms, Ps, Ys)
    # println("C_dot_test2")
    N_ops = length(total_H)
    N_factors = length(factors)
    init = zeros(eltype(core), size(core)...)
    

    for i in 1:N_ops 
        new_factors = [ComplexF64.(Matrix(1.0*I, size(factors[i], 2), size(factors[i], 2))) for i in 1:N_factors]
        if length(total_H[i]) == 2 
            new_factors[total_H[i][2]] = factors[total_H[i][2]]'*total_H[i][1]*factors[total_H[i][2]]
        elseif length(total_H[i]) == 4
            new_factors[total_H[i][2]] = factors[total_H[i][2]]'*total_H[i][1]*factors[total_H[i][2]]
            new_factors[total_H[i][4]] = factors[total_H[i][4]]'*total_H[i][3]*factors[total_H[i][4]]
        end 
        # init .+= -im*Multi_TTM_recursive(core, new_factors)
        init .+= -im*Multi_TTM_allocate_recursive(core, new_factors, Ms, Ps, Ys)
        # println("new factors: ")
        # display(new_factors)
    end
    return init 
end

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
        # println("new factors: ")
        # display(new_factors)
    end
    return init 
end

function C_dot_test2_tucker(core, factors, total_H)
    N_ops = length(total_H)
    N_factors = length(factors)

    init = ITensor(eltype(core), inds(core))
    for i in 1:N_ops 
        # println("Op $i")
        new_factors = [ITensor(ComplexF64.(Matrix(1.0*I, size(factors[i], 2), size(factors[i], 2))), inds(factors[i])[2], inds(factors[i])[2]') for i in 1:N_factors]
        if length(total_H[i]) == 2 
            new_factors[total_H[i][2]] = conj(factors[total_H[i][2]]')*total_H[i][1]*factors[total_H[i][2]]
        elseif length(total_H[i]) == 4
            new_factors[total_H[i][2]] = conj(factors[total_H[i][2]]')*total_H[i][1]*factors[total_H[i][2]]
            new_factors[total_H[i][4]] = conj(factors[total_H[i][4]]')*total_H[i][3]*factors[total_H[i][4]]
        end 
        init .+= -im*noprime(reconstruct(core, new_factors))
        # println("new factors: ")
        # display(new_factors)
    end
    return init
end

#Allocated memory version
function fixed_point_iter_C_mat(H_ops, core, h, factors, M_list, P_list, Y_list, maxiter, tol, verbose)
    # println("Called the right one!")
    K_init = zeros(eltype(core), size(core))
    for i in 1:maxiter 
        K = C_dot_test2(core + 0.5*h*K_init, factors, H_ops, M_list, P_list, Y_list)
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

function fixed_point_iter_C_ten(H_ops, core, h, factors, maxiter, tol, verbose)
    core_inds = collect(inds(core))
    K_init = ITensor(ComplexF64, core_inds)
    for i in 1:maxiter 
        K = C_dot_test2_tucker(core + 0.5*h*K_init, factors, H_ops)
        # println(K)
        error = norm(K_init - K)
        if verbose == true
            println("Iteration $i") 
            println("Error: ", error)
        end
        K_init = K 
        if error < tol 
            break 
        end
    end
    return K_init 
end

#Allocated memory version
function IMR_core_mat(H_ops, core, h, factors_matrices, M_list, P_list, Y_list, maxiter, tol, verbose)
    K = fixed_point_iter_C_mat(H_ops, core, h, factors_matrices, M_list, P_list, Y_list, maxiter, tol, verbose)
    core .+= h*K
    return core 
end

#time-independent Hamiltonian
function heun_core(H_ops, core, h, factor_matrices)
    k1 = C_dot_test2(core, factor_matrices, H_ops)
    k2 = C_dot_test2(core + (1/3)*h*k1, factor_matrices, H_ops)
    k3 = C_dot_test2(core + (2/3)*h*k2, factor_matrices, H_ops)

    C_update = core + 0.25*h*k1 + 0.75*h*k3 
    return C_update 
end

#time-dependent Hamiltonian
function heun_core(H_ops, core, t0, h, factor_matrices, bc_params)
    Hk2 = updateH(H_ops, bc_params, t0 + h/3)
    Hk3 = updateH(H_ops, bc_params, t0 + h*(2/3))
    k1 = C_dot_test2(core, factor_matrices, H_ops)
    k2 = C_dot_test2(core + (1/3)*h*k1, factor_matrices, Hk2)
    k3 = C_dot_test2(core + (2/3)*h*k2, factor_matrices, Hk3)

    C_update = core + 0.25*h*k1 + 0.75*h*k3 
    return C_update 
end

#Non-allocated memory version
function IMR_core_mat(H_ops, core, h, factors_matrices, maxiter, tol, verbose)
    K = fixed_point_iter_C_mat(H_ops, core, h, factors_matrices, maxiter, tol, verbose)
    core .+= h*K
    return core 
end

function IMR_core_ten(H_ops, core, h, factors_matrices, maxiter, tol, verbose)
    K = fixed_point_iter_C_ten(H_ops, core, h, factors_matrices, maxiter, tol, verbose)
    core = core + h*K
    return core 
end

function IMR_factor(H_ops, core, h, factor_matrices, site, maxiter, tol, verbose)
    K = fixed_point_factor(H_ops, core, h, factor_matrices, site, maxiter, tol, verbose)
    factor = factor_matrices[site] + h*K
    return factor
end

#Allocated memory version
function bug_step_mat(H_ops, core, factors, h, M_list, P_list, Y_list)
    N = length(factors)
    # println(N)
    factors_update = Vector{Matrix}(undef, N)
    M_storage = [zeros(ComplexF64, size(factors[i],2), size(factors[i],2)) for i in 1:N]
    for i in 1:N
        # println("Factor $i")
        # @time begin
        K_dot, K0 = K_evolution_mat2(core, factors, i, H_ops, M_list, P_list, Y_list)
        # end
        # K1 = K0 - h*im*K_dot
        K0 .-= h*im*K_dot 
        U, R = qr(K0)
        U = U[1:size(K0, 1),1:size(K0, 2)]
        factors_update[i] = U 
        # M_storage[i] = U'*factors[i]
        mul!(M_storage[i], U', factors[i])
        # M = U'*factors[i]
        # push!(M_storage, M)
    end 
    # println("M_list")
    # display(M_storage)
    # println(P_list)
    # println(Y_list)
    # println(M_storage[4])
    init_C = Multi_TTM_recursive(core, M_storage)
    # init_C = copy(Multi_TTM_allocate_recursive(core, M_storage, M_list, P_list, Y_list))
    
    # println("init_C_2")
    # display(init_C_2)
    # println("init_C")
    # display(init_C)
    # println(norm(init_C_2 - init_C))
    # println("init_C: ")
    # display(init_C)
    C_update = IMR_core_mat(H_ops, init_C, h, factors_update, M_list, P_list, Y_list, 100, 1E-14, false)
    return C_update, factors_update
end

#Non-allocated memory version
function bug_step_mat(H_ops, core, factors, h)
    N = length(factors)
    # println(N)
    factors_update = Vector{Matrix}(undef, N)
    M_storage = [zeros(ComplexF64, size(factors[i],2), size(factors[i],2)) for i in 1:N]
    for i in 1:N
        # println("Factor $i")
        # @time begin
        K_dot, K0 = K_evolution_mat2(core, factors, i, H_ops)
        # end
        # K1 = K0 - h*im*K_dot
        #Look at better way of evolving this, possibly using midpoint rule or exponential
        K0 .-= h*im*K_dot 
        # K0 = IMR_factor(H_ops, core, h, factors, i, 100, 1E-14, false)
        # K0 = exp(-im*K_dot*h)*K0
        U, R = qr(K0)
        U = U[1:size(K0, 1),1:size(K0, 2)]
        factors_update[i] = U 
        # M_storage[i] = U'*factors[i]
        mul!(M_storage[i], U', factors[i])
        # M = U'*factors[i]
        # push!(M_storage, M)
    end 
    # println("M_list")
    # display(M_storage)
    # println(P_list)
    # println(Y_list)
    # println(M_storage[4])
    init_C = Multi_TTM_recursive(core, M_storage)
    # init_C = copy(Multi_TTM_allocate_recursive(core, M_storage, M_list, P_list, Y_list))
    # println("init_C: ")
    # display(init_C)
    # println("init_C_2")
    # display(init_C_2)
    # println("init_C")
    # display(init_C)
    # println(norm(init_C_2 - init_C))
    # println("init_C: ")
    # display(init_C)
    C_update = IMR_core_mat(H_ops, init_C, h, factors_update, 100, 1E-15, false)
    # C_update = heun_core(H_ops, init_C, h, factors_update)
    return C_update, factors_update
end

function bug_step_midpoint(H_ops, core, factors, h, cutoff)
    #First take half step forward with bug
    N = length(factors)
    core_half, factors_half = bug_step_mat(H_ops, core, factors, h/2)
    factors_update = Vector{Matrix}(undef, N)
    M_storage = [zeros(ComplexF64, size(factors[i],2), size(factors[i],2)) for i in 1:N]
    for i = 1:N 
        K_dot, K0 = K_evolution_mat2(core_half, factors_half, i, H_ops)
        combined_K = hcat(factors_half[i], h*K_dot)
        # combined_K = hcat(factors[i],factors_half[i], h*K_dot)
        U_bar, _ = qr(combined_K)
        U_bar = U_bar[1:size(combined_K, 1), 1:min(size(combined_K, 1), size(combined_K, 2))]
        factors_update[i] = U_bar
        mul!(M_storage[i], U_bar', factors[i])
    end
    init_C = Multi_TTM_recursive(core, M_storage)
    C_update = IMR_core_mat(H_ops, init_C, h, factors_update, 100, 1E-14, false)
    if cutoff == 0.0
        return C_update, factors_update
    else
        C_trunc, factors_trunc = truncate_tucker_arr(C_update, factors_update; cutoff = cutoff)
        return C_trunc, factors_trunc 
    end
end
     

function bug_step_eff_ra(H_ops, core, factors, h, sites, cutoff)
    N = length(factors)
    factors_update = []
    M_list = []
    for i in 1:N 
        K_dot, K0 = K_evolution_itensor2(core, factors, i, H_ops, sites)
        # println(K_dot)
        # println(K0)
        K1 = K0 - h*im*K_dot
        # println(K1)
        site_ind = inds(K1, "Site")
        K1_mat = Array(K1, inds(K1))
        mat_factor = Array(factors[i], inds(factors[i]))
        combined_K = hcat(K1_mat, mat_factor)
        # display(combined_K)
        col_dim = dim(inds(K1,"Link")) + dim(inds(factors[i],"Link"))
        col_ind = Index(col_dim; tags = "Link, $i")
        combined_K_ten = ITensor(combined_K, site_ind, col_ind)
        # println(combined_K_ten)
        U, R = qr(combined_K_ten, site_ind)
        # U, R = qr(combined_K)
        # row_combined, col_combined = size(combined_K)
        # U = U*I 
        # U = U[:,1:row_combined]
        # row_U, col_U = size(U)
        # col_ind = Index(col_U;tags = "Link, $i")
        # U= ITensor(U, sites[i], col_ind)
        # U_link = inds(U,"Link")
        # U = U*delta(U_link,inds(core)[i])
        push!(factors_update, U)
        M = conj(U)*factors[i]
        # println(U)
        # println(factors[i])
        push!(M_list, M)
    end
    # display(core)
    # display(factors_update)
    # display(M_list)
    init_C = reconstruct(core, M_list)
    # display(init_C)
    # println("Factors Update: ")
    # display(factors_update)
    C_update = IMR_core_ten(H_ops, init_C, h, factors_update, 100, 1E-14, false)
    # C_update = init_C
    # println("C_INIT 1")
    # println(init_C)
    # C_update = init_C
    C_trunc, factors_trunc = truncate_tucker(C_update, factors_update, cutoff)
    return C_trunc, factors_trunc
end

#Allocated memory version
function bug_step_mat_ra(H_ops, core, factors, h, M_list, P_list, Y_list, cutoff)
    N = length(factors)
    # println(N)
    factors_update = Vector{Matrix}(undef, N)
    M_storage = [zeros(ComplexF64, size(factors[i],2), size(factors[i],2)) for i in 1:N]
    for i in 1:N
        # println("Factor $i")
        # @time begin
        K_dot, K0 = K_evolution_mat2(core, factors, i, H_ops, M_list, P_list, Y_list)
        # end
        # K1 = K0 - h*im*K_dot
        K0 .-= h*im*K_dot 
        combined_K = hcat(K0, factors[i])
        U, R = qr(combined_K)
        # println("Size combined_K: ", size(combined_K))
        # println("Size U: ", size(U))
        U = U[1:size(combined_K, 1),1:size(combined_K, 1)]
        factors_update[i] = U 
        # M_storage[i] = U'*factors[i]
        # println("Size U: ", size(U))
        # println("Size factors $i: ", size(factors[i]))
        mul!(M_storage[i], U', factors[i])
        # M = U'*factors[i]
        # push!(M_storage, M)
    end 
    # println(M_list)
    # println(P_list)
    # println(Y_list)
    # println(M_storage[4])
    # init_C = Multi_TTM_recursive(core, M_storage)
    init_C = copy(Multi_TTM_allocate_recursive(core, M_storage, M_list, P_list, Y_list))
    
    # println("init_C_2")
    # display(init_C_2)
    # println("init_C")
    # display(init_C)
    # println(norm(init_C_2 - init_C))
    # println("init_C: ")
    # display(init_C)
    C_update = IMR_core_mat(H_ops, init_C, h, factors_update, M_list, P_list, Y_list, 100, 1E-14, false)
    C_trunc, factors_trunc = truncate_tucker_arr(C_update, factors_update; cutoff = cutoff)
    return C_trunc, factors_trunc
end

#Non-allocated memory version
# function bug_step_mat_ra(H_ops, core, factors, h, cutoff)
#     N = length(factors)
#     # println(N)
#     factors_update = Vector{Matrix}(undef, N)
#     M_storage = [zeros(ComplexF64, size(factors[i],2), size(factors[i],2)) for i in 1:N]
#     for i in 1:N
#         println("Factor $i Update Runtime: ")
#         @time begin
#         K_dot, K0 = K_evolution_mat2(core, factors, i, H_ops, M_list, P_list, Y_list)
#         end
#         # K1 = K0 - h*im*K_dot
#         K0 .-= h*im*K_dot 
#         combined_K = hcat(K0, factors[i])
#         U, R = qr(combined_K)
#         # println("Size combined_K: ", size(combined_K))
#         # println("Size U: ", size(U))
#         U = U[1:size(combined_K, 1),1:size(combined_K, 1)]
#         factors_update[i] = U 
#         # M_storage[i] = U'*factors[i]
#         # println("Size U: ", size(U))
#         # println("Size factors $i: ", size(factors[i]))
#         mul!(M_storage[i], U', factors[i])
#         # M = U'*factors[i]
#         # push!(M_storage, M)
#     end 
#     # println(M_list)
#     # println(P_list)
#     # println(Y_list)
#     # println(M_storage[4])
#     init_C = Multi_TTM_recursive(core, M_storage)
#     # init_C = copy(Multi_TTM_allocate_recursive(core, M_storage, M_list, P_list, Y_list))
    
#     # println("init_C_2")
#     # display(init_C_2)
#     # println("init_C")
#     # display(init_C)
#     # println(norm(init_C_2 - init_C))
#     # println("init_C: ")
#     # display(init_C)
#     println("C_update runtime: ")
#     @time begin
#     C_update = IMR_core_mat(H_ops, init_C, h, factors_update, M_list, P_list, Y_list, 100, 1E-14, false)
#     end
#     C_trunc, factors_trunc = truncate_tucker_arr(C_update, factors_update; cutoff = cutoff)
#     return C_trunc, factors_trunc
# end

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
    init_C = Multi_TTM_recursive(core, M_storage)
    # init_C = copy(Multi_TTM_allocate_recursive(core, M_storage, M_list, P_list, Y_list))
    
    # println("init_C_2")
    # display(init_C_2)
    # println("init_C")
    # display(init_C)
    # println(norm(init_C_2 - init_C))
    # println("init_C: ")
    # display(init_C)
    # C_update = IMR_core_mat(H_ops, init_C, h, factors_update, 100, 1E-14, false)
    C_update = heun_core(H_ops, init_C, h, factors_update)
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

function truncate_tucker(core, factors, cutoff)
    N = length(factors)
    trunc_factors = []
    core_inds = inds(core)
    new_core = copy(core)
    for i in 1:N 
        P, S, Q = svd(core, core_inds[i]; cutoff = cutoff)
        push!(trunc_factors, factors[i]*P)
        new_core = new_core*prime(conj(P);tags="Site")
    end
    return new_core, trunc_factors 
end

function bug_step_itensor_ra(H_ten, core, factors, h, sites, cutoff)
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
        Y0 = K0*V_T
        orig_order = collect(1:N)
        permutation = vcat([i], setdiff(orig_order, i))
        sites_copy2 = copy(sites)
        permute!(sites_copy2, permutation)
        Y0_ten = ITensor(Y0, sites_copy2)
        Y0_dot = H_ten*Y0_ten
        Y0_dot_mat = matricization(Y0_dot, i)*V_T'
        K1 = K0 - h*im*Y0_dot_mat
        row_K, col_K = size(K1)
        
        combined = hcat(K1, factors_matrix[i])
        U, R = qr(combined)
        row_combined, col_combined = size(combined)
        U = U*I
        U = U[:,1:row_combined]
        row_U, col_U = size(U)
        col_ind = Index(col_U;tags="link, $i")
        U_ten = ITensor(U, sites[i], col_ind)
        push!(factors_update, U_ten)
        M = conj(U_ten)*factors[i]
        push!(M_list, M)
    end

    init_C = reconstruct(core, M_list)

    C_update = IMR_core_itensor(H_ten, init_C, h, factors_update, 100, 1E-14, false)
    C_trunc, factors_trunc = truncate_tucker(C_update, factors_update, cutoff)

    return C_trunc, factors_trunc
end

#Constant Hamiltonian
function bug_integrator_itensor(H_ten, init_core, init_factors, t0, T, steps, sites)
    h = (T - t0)/steps
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors) 
    N = length(init_factors)
    state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    energy_history = zeros(N + 1)
    # m_core, m_factors = magnetization_tucker(sites, magnet_site)
    # magnet_scalar = zeros(steps + 1)
    # magnet_scalar[1] = real(expect_tucker(m_core, m_factors, init_core, init_factors))

    @showprogress 1 "Evolving" for i in 1:steps + 1
        # C_1, update_U = bug_step_itensor(H_ten, init_core_copy, init_factors_copy, h, sites)
        # energy_history[i] = energy_tucker(H_ten, init_core_copy, init_factors_copy)
        # energy_history[i] = conj(init_core_copy)'*H_ten*init_core_copy
        state_history[i,:] = vec_itensor(reconstruct(init_core_copy, init_factors_copy))
        if i == steps + 1
            break 
        end
        C_1, update_U = bug_step_itensor(H_ten, init_core_copy, init_factors_copy, h, sites)
        # init_core_copy, init_factors_copy = bug_step_eff(H_ten, init_core_copy, init_factors_copy, h, sites)
        init_core_copy = copy(C_1) 
        init_factors_copy = copy(update_U)
        # state_history[:, i + 1] = vec_itensor(reconstruct(C_1, update_U))
    end
    return init_core_copy, init_factors_copy, state_history, energy_history
end

#Time-varying Hamiltonian 
function bug_integrator_itensor(H_ten, bcparams, init_core, init_factors, t0, T, steps, sites)
    h = (T - t0)/steps
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors) 
    N = length(init_factors)
    state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    energy_history = zeros(steps + 1)
    H_ten_copy = deepcopy(H_ten)
    # m_core, m_factors = magnetization_tucker(sites, magnet_site)
    # magnet_scalar = zeros(steps + 1)
    # magnet_scalar[1] = real(expect_tucker(m_core, m_factors, init_core, init_factors))

    @showprogress 1 "Evolving" for i in 1:steps + 1
        # C_1, update_U = bug_step_itensor(H_ten, init_core_copy, init_factors_copy, h, sites)
        energy_history[i] = energy_tucker(H_ten_copy, init_core_copy, init_factors_copy)
        state_history[i,:] = vec_itensor(reconstruct(init_core_copy, init_factors_copy))
        if i == steps + 1
            break 
        end
        C_1, update_U = bug_step_eff(H_ten_copy, init_core_copy, init_factors_copy, h, sites)
        init_core_copy = copy(C_1) 
        init_factors_copy = copy(update_U)
        # state_history[:, i + 1] = vec_itensor(reconstruct(C_1, update_U))
        t0 += h
        H_ten_copy = updateH_ten(H_ten, bcparams, t0)
    end
    return init_core_copy, init_factors_copy, state_history, energy_history
end

#Constant Hamiltonian 
function bug_integrator_itensor_ra(H_ten, init_core, init_factors, t0, T, steps, sites, cutoff)
    h = (T - t0)/steps
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors) 
    N = length(init_factors)
    state_history = zeros(ComplexF64, (2^N, steps + 1))
    # m_core, m_factors = magnetization_tucker(sites, magnet_site)
    bd = zeros(N, steps + 1)
    bd[:,1] = get_links_tucker(init_core_copy)
    state_history[:,1] = reverse_vec(reconstruct(init_core_copy, init_factors_copy))
    @showprogress 1 "Evolving Tucker" for i in 1:steps 
        # println("Iteration $i")
        # C_1, update_U = bug_step_itensor_ra(H_ten, init_core_copy, init_factors_copy, h, sites, cutoff)
        C_1, update_U = bug_step_eff_ra(H_ten, init_core_copy, init_factors_copy, h, sites, cutoff)
        bd[:,i + 1] = get_links_tucker(C_1)
        init_core_copy = copy(C_1)
        init_factors_copy = copy(update_U)
        state_history[:,i + 1] = reverse_vec(reconstruct(C_1, update_U))
    end
    return init_core_copy, init_factors_copy, state_history, bd
end

#Time varying Hamiltonian 
function bug_integrator_itensor_ra(H_ten, bcparams, init_core, init_factors, t0, T, steps, sites, cutoff)
    h = (T - t0)/steps
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors) 
    N = length(init_factors)
    state_history = zeros(ComplexF64, (2^N, steps + 1))
    # m_core, m_factors = magnetization_tucker(sites, magnet_site)
    bd = zeros(steps + 1, N)
    state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    energy_history = zeros(steps + 1)
    H_ten_copy = copy(H_ten)
    @showprogress 1 "Evolving Tucker" for i in 1:steps + 1
        state_history[i,:] = vec_itensor(reconstruct(init_core_copy, init_factors_copy))
        energy_history[i] = energy_tucker(H_ten_copy, init_core_copy, init_factors_copy)
        bd[i,:] = get_links_tucker(init_core_copy)
        if i == steps + 1
            break 
        end
        # println("Iteration $i")
        # C_1, update_U = bug_step_itensor_ra(H_ten, init_core_copy, init_factors_copy, h, sites, cutoff)
        # C_1, update_U = bug_step_eff_ra(H_ten, init_core_copy, init_factors_copy, h, sites, cutoff)
        init_core_copy, init_factors_copy = bug_step_eff_ra(H_ten_copy, init_core_copy, init_factors_copy, h, sites, cutoff)
        t0 += h
        H_ten_copy = updateH_ten(H, bcparams, t0)
        # init_core_copy = copy(C_1)
        # init_factors_copy = copy(update_U)
        # state_history[:,i + 1] = reverse_vec(reconstruct(C_1, update_U))
    end
    return init_core_copy, init_factors_copy, state_history, energy_history, bd
end

function bug_integrator_itensor_magnet(H_ten, init_core, init_factors, t0, T, steps, sites)
    h = (T - t0)/steps
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors) 
    N = length(init_factors)
    storage_arr = zeros(ComplexF64, (2^N, steps + 1))
    # m_core, m_factors = magnetization_tucker(sites, magnet_site)
    #Look specifically at magnetization observable
    magnet_history = zeros(N, steps + 1)
    # magnet_scalar[1] = real(expect_tucker(m_core, m_factors, init_core, init_factors))
    @showprogress 1 "Evolving" for i in 1:steps + 1
        for j in 1:N 
            m_core, m_factors = magnetization_tucker(sites, N - j + 1)
            magnet_history[j, i] = real(expect_tucker(m_core, m_factors, init_core_copy, init_factors_copy))
        end
        if i == steps + 1
            break 
        end
        C_1, update_U = bug_step_itensor(H_ten, init_core_copy, init_factors_copy, h, sites)
        # C_1, update_U = bug_step_eff(H_ten, init_core_copy, init_factors_copy, h, sites)
        init_core_copy = copy(C_1) 
        init_factors_copy = copy(update_U)
    end
    return init_core_copy, init_factors_copy, magnet_history 
end

function bug_integrator_itensor_ra_magnet(H_ten, init_core, init_factors, t0, T, steps, sites, cutoff)
    h = (T - t0)/steps
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors) 
    N = length(init_factors)
    storage_arr = zeros(ComplexF64, (2^N, steps + 1))
    #Look specifically at magnetization observable
    magnet_history = zeros(N, steps + 1)
    # magnet_scalar[1] = real(expect_tucker(m_core, m_factors, init_core, init_factors))
    bd = zeros(N, steps + 1)
    bd[:,1] = get_links_tucker(init_core_copy)
    @showprogress 1 "Evolving Tucker" for i in 1:steps + 1
        # println("Iteration $i")
        for j in 1:N 
            m_core, m_factors = magnetization_tucker(sites, N - j + 1)
            magnet_history[j, i] = real(expect_tucker(m_core, m_factors, init_core_copy, init_factors_copy))
        end
        if i == steps + 1
            break 
        end
        # C_1, update_U = bug_step_mat_ra(H_ten, init_core_copy, init_factors_copy, h, sites, cutoff)
        C_1, update_U = bug_step_itensor_ra(H_ten, init_core_copy, init_factors_copy, h, sites, cutoff)
        # C_1, update_U = bug_step_eff_ra(H_ten, init_core_copy, init_factors_copy, h, sites, cutoff)
        bd[:,i + 1] = get_links_tucker(C_1)
        init_core_copy = copy(C_1)
        init_factors_copy = copy(update_U)

    end
    return init_core_copy, init_factors_copy, magnet_history, bd
end

#This is for a constant hamiltonian
function bug_integrator_mat(H_ten, init_core, init_factors, t0, T, steps)
    h = (T - t0)/steps 
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    N = length(init_factors)
    energy_history = zeros(steps + 1)
    state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    @showprogress 1 "Evolving Tucker" for i in 1:steps + 1
        energy_history[i] = energy_tucker(H_ten, init_core_copy, init_factors_copy)
        state_history[i,:] = Multi_TTM_recursive(init_core_copy, init_factors_copy)
        if i == steps + 1
            break 
        end
        C_update, factors_update = bug_step_mat(H_ten, init_core_copy, init_factors_copy, h)
        init_core_copy .= C_update 
        init_factors_copy = copy(factors_update)
    end 
    return init_core_copy, init_factors_copy, state_history, energy_history 
end

#This is for a time-varying control hamiltonian
function bug_integrator_mat(H_s, bcparams, init_core, init_factors, t0, T, steps)
    h = (T - t0)/steps 
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    N = length(init_factors)
    energy_history = zeros(steps + 1)
    state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    H_s_copy = deepcopy(H_s)
    @showprogress 1 "Evolving Tucker" for i in 1:steps + 1
        # println("H_s_copy: ")
        # display(H_s_copy)
        energy_history[i] = energy_tucker(H_s_copy, init_core_copy, init_factors_copy)
        state_history[i,:] = Multi_TTM_recursive(init_core_copy, init_factors_copy)
        if i == steps + 1
            break 
        end
        t0 += h
        init_core_copy, init_factors_copy = bug_step_mat(H_s_copy, init_core_copy, init_factors_copy, h)
        H_s_copy = updateH(H_s, bcparams, t0)
        # H_s_copy = control_h_rot(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz, bcparams, t0)
    end 
    return init_core_copy, init_factors_copy, state_history, energy_history 
end

#For constant Hamiltonian
function bug_integrator_midpoint(H_s, init_core, init_factors, t0, T, steps, cutoff)
    h = (T - t0)/steps 
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    N = length(init_factors)
    energy_history = zeros(steps + 1)
    state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    bd = zeros(steps + 1, N)
    @showprogress 1 "Evolving Tucker" for i in 1:steps + 1
        # println("H_s_copy: ")
        # display(H_s_copy)
        energy_history[i] = energy_tucker(H_s, init_core_copy, init_factors_copy)
        state_history[i,:] = Multi_TTM_recursive(init_core_copy, init_factors_copy)
        bd[i,:] = [size(init_factors_copy[j], 2) for j in 1:length(init_factors_copy)]
        if i == steps + 1
            break 
        end
        init_core_copy, init_factors_copy = bug_step_midpoint(H_s, init_core_copy, init_factors_copy, h, cutoff)
        # H_s_copy = control_h_rot(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz, bcparams, t0)
    end 
    return init_core_copy, init_factors_copy, state_history, energy_history, bd
end
#This is for a constant Hamiltonian
function bug_integrator_mat_ra(H_ten, init_core, init_factors, t0, T, steps, cutoff)
    h = (T - t0)/steps 
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    N = length(init_factors)
    energy_history = zeros(steps + 1)
    state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    bd = zeros(steps + 1, N)

    @showprogress 1 "Evolving Tucker" for i in 1:steps + 1
        energy_history[i] = energy_tucker(H_ten, init_core_copy, init_factors_copy)
        state_history[i,:] = Multi_TTM_recursive(init_core_copy, init_factors_copy)
        bd[i,:] = [size(init_factors_copy[j], 2) for j in 1:length(init_factors_copy)]
        if i == steps + 1
            break 
        end
        C_update, factors_update = bug_step_mat_ra(H_ten, init_core_copy, init_factors_copy, h, cutoff)
        init_core_copy = copy(C_update) 
        init_factors_copy = copy(factors_update)
    end 
    return init_core_copy, init_factors_copy, state_history, energy_history, bd
end

#This is for a time-varying control Hamiltonian
function bug_integrator_mat_ra(H_s, bcparams, init_core, init_factors, t0, T, steps, cutoff)
    h = (T - t0)/steps 
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    N = length(init_factors)
    energy_history = zeros(steps + 1)
    state_history = zeros(ComplexF64, steps + 1, prod([size(init_factors[i], 1) for i in 1:N]))
    bd = zeros(steps + 1, N)
    H_s_copy = deepcopy(H_s)
    @showprogress 1 "Evolving Tucker" for i in 1:steps + 1
        energy_history[i] = energy_tucker(H_s_copy, init_core_copy, init_factors_copy)
        state_history[i,:] = Multi_TTM_recursive(init_core_copy, init_factors_copy)
        bd[i,:] = [size(init_factors_copy[j], 2) for j in 1:length(init_factors_copy)]
        if i == steps + 1
            break 
        end
        t0 += h
        init_core_copy, init_factors_copy = bug_step_mat_ra(H_s_copy, init_core_copy, init_factors_copy, h, cutoff)
        H_s_copy = updateH(H_s, bcparams, t0)
    end 
    return init_core_copy, init_factors_copy, state_history, energy_history, bd
end

function init_separable(sites, q_state)
    N = length(sites)
    M = MPS(N)
    link_size = Int64.(ones(N - 1))
    link_ind = create_linkinds(N, link_size)
    for i in 1:N
        if i == 1
            core = zeros(2, 1)
            core[q_state[i] + 1,1] = 1.0
            core_ten = ITensor(core, sites[i], link_ind[i])
        elseif i == N 
            core = zeros(2, 1)
            core[q_state[i] + 1,1] = 1.0
            core_ten = ITensor(core, sites[i], link_ind[i - 1])
        else 
            core = zeros(1, 2, 1)
            core[1,q_state[i] + 1,1] = 1.0
            core_ten = ITensor(core, sites[i], link_ind[i - 1], link_ind[i])
        end

        M[i] = core_ten 
    end
    return M 
end

function tucker_separable(sites, q_state)
    N = length(sites)
    link_size = Int64.(ones(N))
    link_ind = create_links_tucker(N, link_size)
    links_tuple = (link_size...,)
    factors = []
    core = Array{Float64}(undef, links_tuple)
    core[1] = 1.0
    for i in 1:N
        factor = zeros(2, 1)
        factor[q_state[i] + 1, 1] = 1.0
        factor_ten = ITensor(factor, sites[i], link_ind[i])
        push!(factors, factor_ten)
    end
    core_ten = ITensor(core, link_ind)

    return core_ten, factors
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

function get_links_tucker(core)
    dims = collect(dim.(inds(core)))
    return dims 
end

function entries_tucker2(linkdims)
    entries = prod(linkdims)
    for i in 1:length(linkdims)
        entries += 2*linkdims[i]
    end
    return entries 
end