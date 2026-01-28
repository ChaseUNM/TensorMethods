using LinearAlgebra, ITensors, ITensorMPS, Random, Plots

pyplot()
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
        ind = Index(link_size[i];tags="Link, $i \u2194 $(i + 1)")
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

function tucker_itensor(A, cutoff)
    factors = [] 
    A_inds = inds(A)
    N = length(A_inds)
    A_copy = copy(A)
    core = copy(A)
    for i in 1:N
        # println("Index $i") 
        # println(A_inds[i])
        U, S, V = svd(A_copy, A_inds[i]; cutoff)
        push!(factors, U)
        core = core*conj(U)
    end
    return core, factors 
end

function tucker_itensor(A, sites; cutoff::Float64 = 0.0)
    A_ten = ITensor(A, sites)
    N = length(sites)
    factors = [] 
    A_inds = inds(A_ten)
    N = length(A_inds)
    A_copy = copy(A_ten)
    core = copy(A_ten)
    for i in 1:N
        # println("Index $i") 
        # println(A_inds[i])
        U, S, V = svd(A_copy, A_inds[i]; cutoff)
        push!(factors, U)
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

function fixed_point_iter_C(H_ten, core, h, factor_matrices, maxiter, tol, verbose)
    core_inds = collect(inds(core))
    K_init = ITensor(ComplexF64, core_inds)
    for i in 1:maxiter 
        intermediate = -im*H_ten*reconstruct(core + 0.5*h*K_init, factor_matrices)
        K = reconstruct(intermediate, conj_factors(factor_matrices))
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
        U, R = qr(K1)
        U = U*I 
        U = U[:,1:col_K]
        row_U, col_U = size(U)
        col_ind = Index(col_U;tags="link, $i")
        U_ten = ITensor(U, sites[i], col_ind)
        push!(factors_update, U_ten)
        M = conj(U_ten)*factors[i]
        push!(M_list, M)
    end

    init_C = reconstruct(core, M_list)
    C_update = IMR_core_itensor(H_ten, init_C, h, factors_update, 100, 1E-14, false)

    return C_update, factors_update
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


function bug_integrator_itensor(H_ten, init_core, init_factors, t0, T, steps, sites)
    h = (T - t0)/steps
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors) 
    N = length(init_factors)
    state_history = zeros(ComplexF64, (2^N, steps + 1))
    m_core, m_factors = magnetization_tucker(sites, magnet_site)
    magnet_scalar = zeros(steps + 1)
    magnet_scalar[1] = real(expect_tucker(m_core, m_factors, init_core, init_factors))

    @showprogress 1 "Evolving" for i in 1:steps 
        C_1, update_U = bug_step_itensor(H_ten, init_core_copy, init_factors_copy, h, sites)
        init_core_copy = copy(C_1) 
        init_factors_copy = copy(update_U)
        state_history[:, i + 1] = reverse_vec(reconstruct(C_1, update_U))
    end
    return init_core_copy, init_factors_copy, state_history
end

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
        C_1, update_U = bug_step_itensor_ra(H_ten, init_core_copy, init_factors_copy, h, sites, cutoff)
        bd[:,i + 1] = get_links_tucker(C_1)
        init_core_copy = copy(C_1)
        init_factors_copy = copy(update_U)
        state_history[:,i + 1] = reverse_vec(reconstruct(C_1, update_U))
    end
    return init_core_copy, init_factors_copy, state_history, bd
end

function bug_integrator_itensor_magnet(H_ten, init_core, init_factors, t0, T, steps, sites, magnet_site)
    h = (T - t0)/steps
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors) 
    N = length(init_factors)
    storage_arr = zeros(ComplexF64, (2^N, steps + 1))
    m_core, m_factors = magnetization_tucker(sites, magnet_site)
    #Look specifically at magnetization observable
    magnet_scalar = zeros(steps + 1)
    magnet_scalar[1] = real(expect_tucker(m_core, m_factors, init_core, init_factors))
    @showprogress 1 "Evolving" for i in 1:steps 
        C_1, update_U = bug_step_itensor(H_ten, init_core_copy, init_factors_copy, h, sites)
        init_core_copy = copy(C_1) 
        init_factors_copy = copy(update_U)
        magnet_scalar[i + 1] = real(expect_tucker(m_core, m_factors, C_1, update_U))
    end
    return init_core_copy, init_factors_copy, magnet_scalar
end

function bug_integrator_itensor_ra_magnet(H_ten, init_core, init_factors, t0, T, steps, sites, magnet_site, cutoff)
    h = (T - t0)/steps
    init_core_copy = copy(init_core)
    init_factors_copy = copy(init_factors) 
    N = length(init_factors)
    storage_arr = zeros(ComplexF64, (2^N, steps + 1))
    m_core, m_factors = magnetization_tucker(sites, magnet_site)
    #Look specifically at magnetization observable
    magnet_scalar = zeros(steps + 1)
    magnet_scalar[1] = real(expect_tucker(m_core, m_factors, init_core, init_factors))
    bd = zeros(N, steps + 1)
    bd[:,1] = get_links_tucker(init_core_copy)
    @showprogress 1 "Evolving Tucker" for i in 1:steps 
        # println("Iteration $i")
        C_1, update_U = bug_step_itensor_ra(H_ten, init_core_copy, init_factors_copy, h, sites, cutoff)
        bd[:,i + 1] = get_links_tucker(C_1)
        init_core_copy = copy(C_1)
        init_factors_copy = copy(update_U)
        magnet_scalar[i + 1] = real(expect_tucker(m_core, m_factors, C_1, update_U))

    end
    return init_core_copy, init_factors_copy, magnet_scalar, bd
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