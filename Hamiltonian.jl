using LinearAlgebra, Plots, LaTeXStrings, ProgressMeter

#Create quantum heisenberg model for 3 systems 

sz = [1 0; 0 -1]
sx = [0 1; 1 0]
sy = [0 -im; im 0]
hadamard = [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)]
Ident = Matrix(1.0*I, 2, 2)

function s_op(op, j, N)
    if j == 1 || j == N + 1
        Ident = Matrix(I, 2^(N - 1), 2^(N - 1))
        return kron(op, Ident)
    elseif j == N
        Ident = Matrix(I, 2^(j - 1), 2^(j - 1))
        return kron(Ident, op)
    else 
        I1 = Matrix(I, 2^(j - 1), 2^(j - 1))
        I2 = Matrix(I, 2^(N - j), 2^(N - j))
        return kron(I1, op, I2)
    end
end

function xxx(N, J, g)
    H = zeros(ComplexF64, (2^N, 2^N))
    for j in 1:N - 1
        H .+= -J*s_op(sz, j, N)*s_op(sz, j + 1, N)
    end
    for j in 1:N
        H .-= g*J*s_op(sx, j, N)
    end
    return H
end

function graziani_H(N, J, U, hj_list, hp_list)
    H = zeros(ComplexF64, 2^N, 2^N)
    # println("N: $N")
    for i =  1:N
        # println("i: $i")
        if i != N  
            H .+= -J*(s_op(sx, i, N)*s_op(sx, i + 1, N) + s_op(sy, i, N)*s_op(sy, i + 1, N))
            H .+= U*s_op(sz, i, N)*s_op(sz, i + 1, N)
            # println("H $i")
            # display(U*s_op(sz, i, N)*s_op(sz, i + 1, N))
        end
        H .+= hj_list[i]*s_op(sz, i, N)
        H .+= hp_list[i]*s_op(sx, i, N)
    end
    return H
end

function xxx_hamiltonian(J, g)
    H = zeros(2, 2, 2, 2, 2, 2)
    for i in 1:2 
        for j in 1:2 
            for k in 1:2 
                for iprime in 1:2 
                    for jprime in 1:2 
                        for kprime in 1:2
                            H[i,iprime,j,jprime,k,kprime] = -J*(sz[i,iprime]*sz[j,jprime]*Ident[k,kprime] + Ident[i,iprime]*sz[j,jprime]*sz[k,kprime]) - g*J*(sx[i,iprime]*Ident[j,jprime]*Ident[k,kprime] + Ident[i,iprime]*sx[j,jprime]*Ident[k,kprime]+Ident[i,iprime]*Ident[j,jprime]*sx[k,kprime])
                        end
                    end
                end
            end
        end
    end
    return H 
end 


# function reverse_vec(psi)
#     v = zeros(eltype(psi), 8)
#     for i in 1:2 
#         for j in 1:2 
#             for k in 1:2 
#                 index = 4*(i - 1) + 2*(j - 1) + k
#                 # index = i + 2*(j - 1) + 4*(k - 1) 
#                 v[index] = psi[i,j,k]
#             end
#         end
#     end
#     return v 
# end
# psi = rand(2,2,2) 
# psi_vec = vec(psi)
# psi_vec = reverse_vec(psi)

function mode1_unfold(ten)
    A = zeros(eltype(ten), 2, 4)
end

function mode2_unfold(ten) 
    A = zeros(eltype(ten), 2, 4)
end

function mode3_unfold(ten) 
    A = zeros(eltype(ten), 2, 4)
end
function linear_index_natural(index_list, index_size_list)
    n = length(index_list)
    alpha = 1
    s = 1
    for i in 1:length(index_list)
        if i != 1
            s*= index_size_list[i - 1]
        end
        alpha += s*(index_list[i] - 1)
    end
    return alpha 
end

function tuple_index_natural(alpha, index_size_list)
    s_list = [1;cumprod(index_size_list[1:end - 1])]
    n_indices = length(index_size_list)
    index_list = zeros(n_indices)
    for i in 1:length(index_list)
        index_list[i] = 1 + floor(((alpha - 1)%(index_size_list[i]*s_list[i]))/s_list[i])
    end
    return Int64.(index_list) 
end

function linear_index_reverse(index_list, index_size_list)
    n = length(index_list)
    alpha = 1
    s = 1
    for i in length(index_list):-1:1
        if i != n 
            s *= index_size_list[i + 1]
        end
        alpha += s*(index_list[i]-1)
    end
    return alpha 
end

function tuple_index_reverse(alpha, index_size_list)
    s_list = [1;cumprod(index_size_list[1:end - 1])]
    n_indices = length(index_size_list)
    index_list = zeros(n_indices)
    # println("S_list:" ,s_list)
    for i in 1:length(index_list)
        # println(s_list[length(index_list) - i + 1])
        index_list[i] = 1 + floor(((alpha - 1)%(index_size_list[i]*s_list[length(index_list) - i + 1]))/s_list[length(index_list) - i + 1])
    end
    return Int64.(index_list) 
end

function reverse_vec(tensor::Array)
    N = length(tensor)
    vec_tensor = zeros(eltype(tensor), N)
    size_tuple = size(tensor)
    size_vec = collect(size_tuple)
    for i = 1:N 
        index = tuple_index_reverse(i, size_vec)
        vec_tensor[i] = tensor[index...,]
    end
    return vec_tensor 
end

function reverse_vec(tensor::ITensor)
    tensor_array = Array(tensor, inds(tensor))
    N = length(tensor_array)
    vec_tensor = zeros(eltype(tensor_array), N)
    size_tuple = size(tensor_array)
    size_vec = collect(size_tuple)
    for i = 1:N 
        index = tuple_index_reverse(i, size_vec)
        vec_tensor[i] = tensor_array[index...,]
    end
    return vec_tensor 
end

function vec_itensor(tensor::ITensor; order = "natural")
    tensor_array = Array(tensor, inds(tensor))
    N = length(tensor_array)
    vec_tensor = zeros(eltype(tensor_array), N)
    size_tuple = size(tensor_array)
    size_vec = collect(size_tuple)
    for i = 1:N
        if order == "natural"
            index = tuple_index_natural(i, size_vec)
            vec_tensor[i] = tensor_array[index...,]
        elseif order == "reverse"
            index = tuple_index_natural(i, size_vec)
            vec_tensor[i] = tensor_array[index...,]
        end
    end
    return vec_tensor 
end

function unfolding(A, mode)
    index_size_list = collect(size(A))
    new_list = copy(index_size_list)
    deleteat!(new_list, mode)
    N = prod(new_list)
    mode_size = index_size_list[mode]
    unfold = zeros(eltype(A), mode_size, N)
    ranges = (1:s for s in new_list)
    for i = 1:mode_size
        for iter in Iterators.product(ranges...)
            index_list = collect(iter)
            # beta = linear_index_reverse(index_list, new_list)
            beta = linear_index_natural(index_list, new_list)
            #Insert index back into list 
            re_insert = copy(index_list)
            insert!(re_insert, mode, i)
            unfold[i,beta] = A[re_insert...]
            
        end
    end
    return unfold
end

function refold(A, new_size, mode)
    new_size_tuple = Tuple(new_size)
    A_ten = zeros(eltype(A), new_size_tuple)
    row, col = size(A)
    modified_new_size = copy(new_size)
    deleteat!(modified_new_size, mode)
    for i in 1:row 
        for j in 1:col
            # inds = tuple_index_reverse(j, modified_new_size)
            inds = tuple_index_natural(j, modified_new_size)
            if mode == 1
                total_inds = tuple(i, inds[1], inds[2])
            elseif mode == 2 
                total_inds = tuple(inds[1], i, inds[2])
            elseif mode == 3
                total_inds = tuple(inds[1], inds[2], i)
            end
            A_ten[total_inds...] = A[i,j]
        end
    end
    return A_ten 
end

# function mode_k_unfolding(ten, k)
#     indices = size(ten)
# end

function tensor_mult(H, psi)
    ans = zeros(ComplexF64, 2, 2, 2)
    for i in 1:2 
        for j in 1:2
            for k in 1:2 
                ans .+= H[i,:,j,:,k,:]*psi[i,j,k]
            end
        end
    end
    return ans 
end

function ttm_mode_1(A, U)
    A_1 = unfolding(A, 1)
    Y = U*A_1
    q, m = size(U)
    m,n,p = size(A) 
    Y_ten = refold(Y, [q, n, p], 1)
    return Y_ten 
end

# function ttm_mode_2(A, U)
#     A_2 = unfolding(A, 2)
#     for k = 1:3

#     Y = V*A_2
#     n, r = size(V)
#     m,n,p = size(A)
#     Y_ten = refold(Y, [m,r,p], 2)
#     return Y_ten 
# end

# function ttm_mode_2_real(A, U)



function ttm_mode_k(A, U, mode)
    A_k = unfolding(A, mode)
    # println("Mode $mode")
    # println("A_k")
    # display(A_k)
    # println("U")
    # display(U)
    if length(A_k) == 1
        A_k = A_k[1,1]
    end
    Y = U*A_k 
    row_u, col_u = size(U)
    i,j,k = size(A)
    # row_a, col_a = size(A_k)
    if mode == 1
        inds = (row_u, j, k)
    elseif mode == 2
        inds = (i,row_u,k)
    elseif mode == 3
        inds = (i,j,row_u)
    end
    Y_ten = refold(Y, collect(inds), mode)
    return Y_ten 
end


function ttm_mode_1_diff(A, U)
    a1, a2, a3 = size(A)
    # println("Size A: $(size(A))")
    row_u, col_u = size(U)
    # println("Size U: $(size(U))")
    Y = zeros(ComplexF64, row_u, a2, a3)
    for alpha in 1:row_u 
        for i in 1:a1 
            for j in 1:a2
                for k in 1:a3 
                    Y[alpha, j,k] += A[i,j,k]*U[alpha,i]
                end
            end
        end
    end
    return Y 
end

function ttm_mode_2_diff(A, U)
    a1, a2, a3 = size(A)
    row_u, col_u = size(U)
    Y = zeros(ComplexF64, a1, row_u, a3)
    for beta in 1:row_u 
        for i in 1:a1  
            for j in 1:a2 
                for k in 1:a3 
                    Y[i, beta, k] += A[i,j,k]*U[beta,j]
                end
            end
        end
    end
    return Y 
end

function ttm_mode_3_diff(A, U)
    a1, a2, a3 = size(A)
    row_u, col_u = size(U)
    Y = zeros(ComplexF64, a1, a2, row_u)
    for omega in 1:row_u
        for i in 1:a1 
            for j in 1:a2
                for k in 1:a3
                    Y[i, j, omega] += A[i,j,k]*U[omega,k]
                end
            end
        end
    end
    return Y 
end

# function ttm_mode_3(A, U)
#     A_3 = unfolding(A, 3)
#     Y = 


# H_ten = xxx_hamiltonian(1, 1)
# H_mat = xxx(3, 1, 1)
# ans_vec = H_mat*psi_vec 
# ans_ten = tensor_mult(H, psi)

# println("Vector formulation: ")
# display(ans_vec)
# println("Vectorized tensor: ")
# display(reverse_vec(ans_ten))

#Test TTM mode-k multiplication order-3 tensor

# A = zeros(ComplexF64, 2, 2, 2)
# A[:,:,1] = [1 0; 0 0]
# A[:,:,2] = [0 0;0 0]
# A[1,1,1] = 1.0 + 0.0*im
# U = [-2 -1; -4 -3]

# A_order3 = zeros(3, 3, 3)
# A_order3[:,:,1] = [1 1 0; 1 -1 -1; 0 2 1]
# A_order3[:,:,2] = [2 2 0; 1 -3 -2; 1 5 2]
# A_order3[:,:,3] = [1 1 0; 0 -2 -1; 1 3 1]

# Y_1 = ttm_mode_k(A, U, 1)
# Y_1_diff = ttm_mode_1_diff(A, U)
# Y_2 = ttm_mode_k(A, U, 2)
# Y_2_diff = ttm_mode_2_diff(A, U)
# Y_3 = ttm_mode_k(A, U, 3)
# Y_3_diff = ttm_mode_3_diff(A, U)

function trim_by_tolerance(v::Vector{T}, tol::Real) where T<:Real
    idx = length(v)
    s = zero(T)
    while idx >= 1
        s += v[idx]^2
        if s >= tol^2
            break
        end
        idx -= 1
    end
    return v[1:idx]
end

function reduced_SVD(X, r)
    U, S, V = svd(X)
    row, col = size(U)
    if r > col 
        error("Prescribed rank is higher than maximum")
    else
        U_reduced = U[:,1:r]
    end
    return U_reduced 
end

function reduced_SVD_tol(X, cutoff)
    U, S, V = svd(X)
    row, col = size(U)
    S_trunc = trim_by_tolerance(S, cutoff)
    r = length(S_trunc)
    U_reduced = U[:,1:r]
    return U_reduced 
end


function tucker_tensor(A, rank)
    factors = []
    U = reduced_SVD(unfolding(A, 1), rank)
    push!(factors, U)
    V = reduced_SVD(unfolding(A, 2), rank)
    push!(factors, V)
    W = reduced_SVD(unfolding(A, 3), rank)
    push!(factors, W)

    # core = ttm_mode_k(A, U', 1)
    # println("TTM Mode 1")
    # display(core)
    # core = ttm_mode_1_diff(A, U')
    # core = ttm_mode_k(core, V', 2)
    # println("TTM Mode 2")
    # display(core)
    core = ttm_mode_1_diff(A, U')
    core = ttm_mode_2_diff(core, V')
    # core = ttm_mode_k(core, W', 3)
    core = ttm_mode_3_diff(core, W')
    # println("Core")
    return core, factors
end

function tucker_tensor_cutoff(A, cutoff)
    factors = []
    U = reduced_SVD_tol(unfolding(A, 1), cutoff)
    push!(factors, U)
    V = reduced_SVD_tol(unfolding(A, 2), cutoff)
    push!(factors, V)
    W = reduced_SVD_tol(unfolding(A, 3), cutoff)
    push!(factors, W)
    core = ttm_mode_1_diff(A, U')
    core = ttm_mode_2_diff(core, V')
    core = ttm_mode_3_diff(core, W')

    return core, factors 
end

function reconstruct_tensor(core, U, V, W)
    reconstruct_core = ttm_mode_1_diff(core, U)
    reconstruct_core = ttm_mode_2_diff(reconstruct_core, V)
    reconstruct_core = ttm_mode_3_diff(reconstruct_core, W)

    return reconstruct_core 
end

function reconstruct_tensor2(core, factor_matrices)
    reconstruct_core = ttm_mode_1_diff(core, factor_matrices[1])
    reconstruct_core = ttm_mode_2_diff(reconstruct_core, factor_matrices[2])
    reconstruct_core = ttm_mode_3_diff(reconstruct_core, factor_matrices[3])

    return reconstruct_core 
end

# core, factors = tucker_tensor(A, 2)
# println("U'")
# display(U')
# println("TTM Mode 1")
# core1 = ttm_mode_1(A_order3, U')
# display(core1)
# println("TTM Mode 2")
# display(ttm_mode_2_diff(core1, V'))
# println("U: $(size(U))")
# println("V: $(size(V))") 
# println("W: $(size(W))")

# println("Core: ")
# display(core)
# reconstruct_core = reconstruct_tensor(core, U, V, W)
# display(reconstruct_core)
# println("Frob Norm Error: ", norm(A - reconstruct_core))



function kron_prod(factor_matrices, i)
    N = length(factor_matrices)
    which_factor = setdiff(1:N, i)
    init = factor_matrices[minimum(which_factor)]
    which_factor = setdiff(which_factor, minimum(which_factor))
    for i in which_factor 
        init = kron(init, factor_matrices[i])
    end
    return init 
end

function kron_prod2(factor_matrices, i)
    N = length(factor_matrices)
    which_factor = setdiff(1:N, i)
    init = factor_matrices[maximum(which_factor)]
    which_factor = setdiff(which_factor, maximum(which_factor))
    for i in reverse(which_factor) 
        init = kron(init, factor_matrices[i])
    end
    return init 
end

function kron_prod_test(factor_matrices, i)
    N = length(factor_matrices)
    which_factor = setdiff(1:N, i)
    # println("Which factor")
    # display(which_factor)
    init = factor_matrices[minimum(which_factor)]
    which_factor = setdiff(which_factor, minimum(which_factor))
    for i in which_factor 
        init = init*factor_matrices[i]
    end
    return init 
end

function transpose_U(factors)
    transpose_U = []
    for i in factors 
        push!(transpose_U, i')
    end
    return transpose_U 
end

function fixed_point_iter_C(H_ten, core, h, factor_matrices, factor_matrices_T, maxiter, tol, verbose)
    K_init = zeros(eltype(core), size(core))
    for i in 1:maxiter 
        intermediate = -im*tensor_mult(H_ten, reconstruct_tensor2(core + 0.5*h*K_init, factor_matrices))
        K = reconstruct_tensor2(intermediate, factor_matrices_T) 
        error = norm(K_init - K) 
        if verbose == true
            println("Error: ", error)
        end
        K_init .= K
        if error < tol 
            break 
        end
    end
    return K_init 
end

function IMR_core(H_ten, core, h, factor_matrices, factor_matrices_T, maxiter, tol, verbose)
    K = fixed_point_iter_C(H_ten, core, h, factor_matrices, factor_matrices_T, maxiter,tol, verbose)
    core = core + h*K 
    return core 
end

function new_trunc_factors(U, P)
    new_factors = []
    for i in 1:length(U)
        push!(new_factors, U[i]*P[i])
    end
    return new_factors
end


function tucker_bug_step(H_ten, core, factor_matrices, h)
    #Need initial size
    U = factor_matrices[1]
    V = factor_matrices[2]
    W = factor_matrices[3]
    Y0 = reconstruct_tensor(core, U, V, W)
    a1, a2, a3 = size(Y0)
    update_U = []
    M_list = []
    U_T = transpose_U(factor_matrices)
    for i in 1:3
        # println("Array Factors_matrix $i")
        # display(factor_matrices[i]) 
        # display(U_T[i])
        C_mat = unfolding(core, i)
        Q, S = qr(C_mat')
        row_S, col_S = size(S)
        Q = Q*I 
        Q = Q[:,1:row_S]
        # println("Array Q $i: ")
        # display(Q)
        if length(Q) == 1
            Q = Q[1,1]
        end
        V_T = Q'*kron_prod2(U_T, i)
        # println("factor_kron Array $i:")
        # display(kron_prod2(U_T, i))
        # println("V_T Array $i: ")
        # display(V_T)
        # println("Matricized Y0: ")
        # display(unfolding(Y0, i))   
        # println("U*S*V")
        # display(factor_matrices[i]*S'*V_T)

        #Now I need to evolve the factors forward
        K0 = factor_matrices[i]*S'
        #Now evolve forwards
        # println("Size Mat(C): $(size(C_mat'))")
        # println("Size factor matrix $i: ", size(factor_matrices[i]))
        # println("Array Size S: $(size(S))")
        # println("Array Size Q: $(size(Q))")
        # println("Array Size K0: $(size(K0))")
        # println("Array Size V_T: $(size(V_T))")
        # println("Array Y0 $i: ")
        # display(K0*V_T)
        Y0_ten = refold(K0*V_T, [a1, a2, a3], i)
        # println("Y0_ten")
        # display(Y0_ten)
        Y0_dot = unfolding(tensor_mult(H_ten, Y0_ten), i)*V_T' 
        # println("Array Y0_dot $i: ")
        # display(Y0_dot)
        # K_1 = exp(-im*Y0_dot*h)*K0
        K_1 = K0 - im*h*Y0_dot
        row_K, col_K = size(K_1)
        # K0_t = K0 - h*Y0_dot
        U, R = qr(K_1)
        # println("Size K_1: $(size(K_1))")
        # println("Size R: $(size(R))")
        U = U*I
        U = U[:,1:col_K]
        # println("Size U: $(size(U))")
        push!(update_U, U)
        M = U'*factor_matrices[i]
        # println("Size M: $(size(M))")
        push!(M_list, M)
        # println("M $i: ")
        # display(M)
    end
    update_U_T = transpose_U(update_U)

    #Now update core tensor, for now do explicit euler to see 1-step error
    # intermediate_Y = reconstruct_tensor2(core, factor_matrices)
    init_C = reconstruct_tensor2(core, M_list)
    # println("CORE SIZE: ")
    # println(size(init_C))
    # F_1 = tensor_mult(H_ten, reconstruct_tensor2(core, update_U))
    # C_dot = reconstruct_tensor2(F_1, update_U_T)
    # C_1 = init_C - im*h*C_dot
    C_1 = IMR_core(H_ten, init_C, h, update_U, update_U_T, 100,1E-15, false)
    # C_1 = reconstruct_tensor2(core, M_list) - h*im*tensor_mult(intermediate_Y, recupdate_U)
    return C_1, update_U 
end

function tucker_bug_step_RA(H_ten, core, factor_matrices, h, tol)
    #Need initial size
    U = factor_matrices[1]
    V = factor_matrices[2]
    W = factor_matrices[3]
    Y0 = reconstruct_tensor(core, U, V, W)
    a1, a2, a3 = size(Y0)
    update_U = []
    M_list = []
    U_T = transpose_U(factor_matrices)
    for i in 1:3 
        C_mat = unfolding(core, i)
        Q, S = qr(C_mat')
        row_S, col_S = size(S)
        Q = Q*I 
        Q = Q[:,1:row_S]
        
        if length(Q) == 1
            Q = Q[1,1]
        end
        V_T = Q'*kron_prod2(U_T, i)
        # println("Matricized Y0: ")
        # display(unfolding(Y0, i))   
        # println("U*S*V")
        # display(factor_matrices[i]*S'*V_T)

        #Now I need to evolve the factors forward
        K0 = factor_matrices[i]*S'

        #Now evolve forwards
        # println("Size Mat(C): $(size(C_mat'))")
        # println("Size factor matrix $i: ", size(factor_matrices[i]))
        # println("Size S: $(size(S))")
        # println("Size Q: $(size(Q))")
        # println("Size K0: $(size(K0))")
        # println("Size V_T: $(size(V_T))")
        Y0_ten = refold(K0*V_T, [a1, a2, a3], i)
        # println("Y0_ten")
        # display(Y0_ten)
        Y0_dot = unfolding(tensor_mult(H_ten, Y0_ten), i)*V_T' 
        K0_t = exp(-im*Y0_dot*h)*K0
        # K_1 = K0 - im*h*Y0_dot
        row_K, col_K = size(K0_t)
        # K0_t = K0 - h*Y0_dot
        combined = hcat(K0_t, factor_matrices[i])
        U, R = qr(combined)
        # println("Size K_1: $(size(K_1))")
        # println("Size R: $(size(R))")
        U = U*I
        U = U[:,1:col_K]
        # println("Size U: $(size(U))")
        push!(update_U, U)
        M = U'*factor_matrices[i]
        # println("Size M: $(size(M))")
        push!(M_list, M)
    end
    update_U_T = transpose_U(update_U)

    #Now update core tensor, for now do explicit euler to see 1-step error
    # intermediate_Y = reconstruct_tensor2(core, factor_matrices)
    init_C = reconstruct_tensor2(core, M_list)
    # println("CORE SIZE: ")
    # println(size(init_C))
    # F_1 = tensor_mult(H_ten, reconstruct_tensor2(core, update_U))
    # C_dot = reconstruct_tensor2(F_1, update_U_T)
    # C_1 = init_C - im*h*C_dot
    C_1 = IMR_core(H_ten, init_C, h, update_U, update_U_T, 100,1E-15, false)

    C_1_trunc, factors_trunc = tucker_tensor_cutoff(C_1, tol)
    new_factors = new_trunc_factors(update_U, factors_trunc)
    # C_1 = reconstruct_tensor2(core, M_list) - h*im*tensor_mult(intermediate_Y, recupdate_U)
    return C_1_trunc, new_factors
end


function tucker_integrator(H_ten, init_core, init_factors, t0, T, steps)
    h = (T - t0)/steps 
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    storage_arr = zeros(ComplexF64, (8, steps + 1))
    storage_arr[:,1] = reverse_vec(reconstruct_tensor2(init_core_copy, init_factors_copy))
    @showprogress 1 "Evolving" for i in 1:steps
        C_1, update_U = tucker_bug_step(H_ten, init_core_copy, init_factors_copy, h)
        init_core_copy .= C_1 
        init_factors_copy = update_U
        storage_arr[:,i + 1] = reverse_vec(reconstruct_tensor2(C_1, update_U)) 
    end
    return init_core_copy, init_factors_copy, storage_arr
end

function tucker_integrator_RA(H_ten, init_core, init_factors, t0, T, steps, tol)
    h = (T - t0)/steps 
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors)
    storage_arr = zeros(ComplexF64, (8, steps + 1))
    storage_arr[:,1] = reverse_vec(reconstruct_tensor2(init_core_copy, init_factors_copy))
    @showprogress 1 "Evolving" for i in 1:steps
        C_1, update_U = tucker_bug_step_RA(H_ten, init_core_copy, init_factors_copy, h, tol)
        init_core_copy .= C_1 
        init_factors_copy = update_U
        storage_arr[:,i + 1] = reverse_vec(reconstruct_tensor2(C_1, update_U)) 
    end
    return init_core_copy, init_factors_copy, storage_arr
end

# A = zeros(ComplexF64, 2, 2, 2)
# A[:,:,1] = [3 4; 0 0]
# A[:,:,2] = [0 0;2 1]
# A[1,1,1] = 1.0 + 0.0*im

# core, factors = tucker_tensor(A, 2)

# t0 = 0.0
# T = 10.0
# n = 100
# h = (T - t0)/n

# H_ten = xxx_hamiltonian(1, 1)
# H_mat = xxx(3, 1, 1)
# A_vec = reverse_vec(A)

# C_final, factors_final, storage_arr = tucker_integrator(H_ten, core, factors, t0, T, n)

# c1, u1 = tucker_bug_step(H_ten, core, factors, h)
# c2, u2 = tucker_bug_step(H_ten, c1, u1, h)

# exp_storage = zeros(ComplexF64, 8, n + 1)
# exp_storage[:,1] = A_vec
# for i in 1:n 
#     evolve = exp(-im*H_mat*h)*A_vec
#     exp_storage[:, i + 1] .= evolve 
#     A_vec .= evolve 
# end 

# p = plot(LinRange(t0,T,n + 1), [real(exp_storage[1,:]) imag(exp_storage[1,:])], label = ["Exponential Real" "Exponential Imaginary"], linestyle=:dash, alpha = 0.5)
# plot!(LinRange(t0, T, n + 1), [real(storage_arr[1,:]) imag(storage_arr[1,:])], label = ["BUG Real" "BUG Imaginary"], linewidth = 3, alpha = 0.8)

# display(p)

# true_answer = exp(-im*(T - t0)H_mat)*A_vec 

# tucker_answer = reverse_vec(reconstruct_tensor2(C_final, factors_final))

# println("Final Error: ", norm(true_answer - tucker_answer))



# N = 5
# n_list = 10 .^(collect(2:6))
# N = length(n_list)
# err_list = zeros(N)
# for i in 1:N
#     println("n_list[i] $(n_list[i])") 
#     C_final, factors_final = tucker_integrator(H_ten, core, factors, t0, T, n_list[i])
#     tucker_answer = reverse_vec(reconstruct_tensor2(C_final, factors_final))
#     err = norm(true_answer - tucker_answer)
#     err_list[i] = err 
# end

# p = plot(1 ./n_list, err_list, label = "BUG Error", xlabel = L"h", xscale =:log10, yscale=:log10, 
# legend =:topleft, dpi = 200)
# plot!(1 ./n_list, (1 ./n_list).^2, label = L"h^2")
# savefig(p, "QuadraticPlot2.png")
# set_x = [1,2,3]
# kron_prod_test(set_x, 3)

# factors = []
# U1 = [1 2; 3 4]
# push!(factors, U1)
# U2 = [-1 0; 1 4]
# push!(factors, U2)
# U3 = [0 3; 1 0]
# push!(factors, U3)

# println("kron(U1, U2)")
# display(kron(U1, U2))
# println("kron(U1, U3)")
# display(kron(U1, U3))
# println("kron(U2, U3)")
# display(kron(U2, U3))

# println("kron_prod1")
# display(kron_prod(factors, 3))
# println("kron_prod2")
# display(kron_prod(factors, 2))
# println("kron_prod3")
# display(kron_prod(factors, 1))

# A = A/norm(A)

# H_ten = xxx_hamiltonian(1, 1)
# core, factors = tucker_tensor(A, 2)

# h = 0.001

# C_1, update_U = tucker_bug_step(H_ten, core, factors, h)
# Y_1 = reconstruct_tensor2(C_1, update_U)
# Y_1_vec = reverse_vec(Y_1)


# A_1 = exp(h*H_mat)*A_vec
# display(A_1)
# display(Y_1_vec)
# println("Norm error: ", norm(A_1 - Y_1_vec))

# println("Checking tensor multiplication: ")
# display(reverse_vec(tensor_mult(H_ten, A)))
# println("Checking mat-vec multiplication")
# display(H_mat*A_vec)

# n = 100
# h_list = 10 .^ (-LinRange(1, 5, n))
# err_list = zeros(n)
# for i in 1:n 
#     C_1, update_U = tucker_bug_step(H_ten, core, factors, h_list[i])
#     Y_1 = reconstruct_tensor2(C_1, update_U)
#     Y_1_vec = reverse_vec(Y_1)
#     display(Y_1_vec)
#     H_mat = xxx(3, 1, 1)
#     A_vec = reverse_vec(A)
#     A_1 = exp(-im*h_list[i]*H_mat)*A_vec
#     display(A_1)
#     err = norm(A_1 - Y_1_vec)
#     # err = norm(A_1 - Y_1_vec)
#     err_list[i] = err 
# end

# plot(h_list, err_list, xscale =:log10, yscale=:log10, label = "Error", xlabel = "h", legend=:topleft)
# plot!(h_list, h_list.^3, label = L"h^3")
