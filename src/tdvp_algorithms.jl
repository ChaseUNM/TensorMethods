using ITensors
using LinearAlgebra, ProgressMeter

#Exponential solver using matrix vector format, used in order for verification of true results. 
function exp_solver(H, init_vec, N, t0, T, steps)
    h = (T - t0)/steps 
    sol_op = exp(-im*H*h)
    magnet_history = zeros(steps + 1, N)
    energy_history = zeros(steps + 1)
    for j = 1:N
        m_mat = s_op_reverse([1 0; 0 -1], j, N)
        magnet_history[1,j] = real(init_vec'*m_mat*init_vec)
    end
    energy_history[1] = real(init_vec'*H*init_vec)
    @showprogress 1 "Exponential solver" for i = 1:steps 
        init_vec = sol_op*init_vec 
        for j = 1:N 
            m_mat = s_op_reverse([1 0; 0 -1], j, N)
            magnet_history[i + 1,j] = real(init_vec'*m_mat*init_vec)
        end
        energy_history[i + 1] = real(init_vec'*H*init_vec)
    end
    return init_vec, magnet_history, energy_history 
end

#Count total number of entries in an MPS given an MPS
function count_MPS(M::MPS)
    return sum(prod.(dims.(M)))
end

#Count total number of entries in an MPS given the bond dimensions
function count_MPS(bd::Vector, N_levels::Vector{Int64})
    entries = 0
    N = length(N_levels)
    for i in 1:N 
        if i == 1
            entries += N_levels[i]*bd[i]
        elseif i == N 
            entries += N_levels[i]*bd[i - 1]
        else
            entries += N_levels[i]*bd[i]*bd[i - 1]
        end
    end
    return entries 
end

#Count total number of entries given an array of bond dimensions
function count_MPS_history(bd::Array, N_levels::Vector{Int64})
    steps = size(bd, 1)
    entries_list = zeros(steps)
    for i = 1:steps
        entries_list[i] = count_MPS(bd[i,:], N_levels)
    end
    return entries_list 
end


function create_linkinds(N::Int64, link_size::Vector{Int64})
    ind_vec = Index{Int64}[]
    for i in 1:N-1
        ind = Index(link_size[i];tags="Link, l = $i")
        push!(ind_vec, ind)
    end
    return ind_vec
end

function is_left_orthogonal(A::ITensor; tol=1e-12)
    site, _, r = get_site_and_links(A)
    r === nothing && return true   # left boundary

    Ac = dag(A)
    prime!(Ac, r)

    T = A * Ac
    T_arr = Array(T, inds(T))
    row, col = size(T_arr)
    err = norm(T_arr - Matrix(1.0*I, row, col))
    println("err: ", err)
    if err < tol
        println("true")
    else
        println("false")
    end
    # return is_identity(T; tol=tol)
end

function is_right_orthogonal(A::ITensor; tol=1e-12)
    site, l, _ = get_site_and_links(A)
    l === nothing && return true   # right boundary

    Ac = dag(A)
    prime!(Ac, l)

    T = A * Ac
    T_arr = Array(T, inds(T))
    row, col = size(T_arr)
    err = norm(T_arr - Matrix(1.0*I, row, col))
    println("err: ", err)
    if err < tol
        println("true") 
    else
        println("false")
    end
    # return is_identity(T; tol=tol)
end

function ortho_properties(M::MPS; tol = 1e-12)
    N = length(M)
    for i in 1:N 
        println("Site $i")
        println("-----------------------------------------------")
        println("left-orthogonal: ")
        is_left_orthogonal(M[i])
        println("right-orthogonal: ")
        is_right_orthogonal(M[i])
        println("-----------------------------------------------")
    end
end

#Creates initial separable state depending on state of qubits (either 0 or 1)
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

function equal_separable(sites)
    N = length(sites)
    M = MPS(N)
    equal_arr = 1/sqrt(2)*ones(2)
    link_size = Int64.(ones(N-1)) 
    link_ind = create_linkinds(N, link_size)
    for i in 1:N 
        if i == 1
            core = zeros(2, 1)
            core[:,1] = equal_arr 
            core_ten = ITensor(core, sites[i], link_ind[i])
        elseif i == N 
            core = zeros(2, 1)
            core[:,1] = equal_arr 
            core_ten = ITensor(core, sites[i], link_ind[i - 1])
        else 
            core = zeros(1, 2, 1)
            core[1,:,1] = equal_arr 
            core_ten = ITensor(core, sites[i], link_ind[i - 1], link_ind[i])
        end
        M[i] = core_ten 
    end
    return M 
end

function applyH_eff(H, M, L, R, site)
    return noprime(L*M*H[site]*R)
end


function applyH2_eff(H, M, L, R, site)
    return noprime(L*H[site]*M*H[site + 1]*R)
end


function applyK_eff(H, C, L, R, site)
    return noprime(L*C*R)
end

function contract_left(H::MPO, M::MPS, termination_site::Int64)
    L_list = []
    L = 1
    push!(L_list, L)
    for i = 1:termination_site
        L = L*M[i]*H[i]*conj(M[i]')
        push!(L_list, L)
    end
    return L_list 
end

function contract_right(H::MPO, M::MPS, termination_site::Int64)
    N = length(M)
    R_list = []
    R = 1
    push!(R_list, R)
    for i = reverse(termination_site:N)
        R = R*M[i]*H[i]*conj(M[i]')
        push!(R_list, R)
    end
    return reverse(R_list) 
end

function TT_fp_2site_new(H, init, L, R, h, site, maxiter, tol, verbose)
    k_init = ITensor(inds(init))
    # count = 0
    for i = 1:maxiter 
        k = -im*applyH2_eff(H, init + 0.5*h*k_init, L, R, site)
        err = norm(k - k_init)
        # count += 1
        if verbose == true 
            println("Iteration $i")
            println("Error: ", err)
        end
        if err < tol 
            break 
        end
        k_init = copy(k)
        
    end
    # println("Converged in $count iterations")
    return k_init 
end

function TT_fp_1site_new(H, init, L, R, h, site, maxiter, tol, verbose)
    k_init = ITensor(inds(init))
    # println("inds k_init: ", inds(k_init))
    for i = 1:maxiter
        k = -im*applyH_eff(H, init + 0.5*h*k_init, L, R, site)
        # println("inds k", inds(k))
        err = norm(k - k_init)
        if verbose == true 
            println("Iteration $i")
            println("Error: ", err)
        end
        if err < tol 
            break 
        end
        k_init = copy(k)
    end
    return k_init 
end

function TT_fp_1site_new_backwards(H, init, L, R, h, site, maxiter, tol, verbose)
    k_init = ITensor(inds(init))
    # count = 0
    for i = 1:maxiter
        k = im*applyH_eff(H, init + 0.5*h*k_init, L, R, site + 1)
        err = norm(k - k_init)
        # count += 1
        if verbose == true 
            println("Iteration $i")
            println("Error: ", err)
        end
        if err < tol 
            break 
        end
        k_init = copy(k)
    end
    # println("Converged in $count iterations")
    return k_init 
end

function TT_fp_0site_new(H, init, L, R, h, site, maxiter, tol, verbose)
    k_init = ITensor(inds(init))
    for i = 1:maxiter 
        k = im*applyK_eff(H, init + 0.5*h*k_init, L, R, site)
        err = norm(k - k_init)
        if verbose == true 
            println("Iteration $i")
            println("Error: ", err)
        end
        if err < tol 
            break 
        end
        k_init = copy(k)
    end
    return k_init 
end

function TT_IMR_2site_new(H, init, L, R, h, site)
    k = TT_fp_2site_new(H, init, L, R, h, site, 100, 1E-13, false)
    update = init + h*k 
    return update 
end

function TT_IMR_1site_new(H, init, L, R, h, site)
    k = TT_fp_1site_new(H, init, L, R, h, site, 100, 1E-13, false)
    update = init + h*k 
    return update 
end

function TT_IMR_1site_new_backwards(H, init, L, R, h, site)
    k = TT_fp_1site_new_backwards(H, init, L, R, h, site, 100, 1E-13, false)
    update = init + h*k 
    return update 
end

function TT_IMR_0site_new(H, init, L, R, h, site)
    k = TT_fp_0site_new(H, init, L, R, h, site, 100, 1E-13, false)
    update = init + h*k 
    return update 
end

function lr_sweep_new_new(H::MPO, M::MPS, R_list::Vector{Any}, t::Float64, h::Float64)
    N = length(M)
    L_list = []
    L = 1
    push!(L_list, L)
    for i = 1:N - 1
        # println("Site $i")
        # println("---------------------------------------------")
        # println("R_list[$i]: ", inds(R_list[i]))
        # println("R_inds: ", length(inds(R_list[i])))
        # println("M[$i]: ", inds(M[i]))
        # println("L: ", inds(L))
        # println("H: ", inds(H[i]))
        M_evolve = TT_IMR_1site_new(H, M[i], L, R_list[i], h, i)
        # println("M")
        # println(M[i])
        # println("M_evolve: ")
        # println(M_evolve)
        if i==1
            # Q, R = qr(M_evolve, inds(M[i])[1])
            Q, R = qr(M_evolve, inds(M[i]; tags = "n = 1"); tags = "Link, l = 1")
        else
            # Q, R = qr(M_evolve, inds(M[i])[1:2])
            # println(inds(M[i]))
            # println(inds(M[i]; tags = "l = $(i-1)"), inds(M[i]; tags = "n = $i"))
            # println(inds(M[i])[1:2])
            # println("This is getting called")
            Q, R = qr(M_evolve, inds(M[i]; tags = "n = $i")[1], inds(M[i]; tags = "l = $(i-1)")[1], ; tags = "Link, l = $i")
        end
        # L = L*(Q*H[i]*conj(Q)')
        L = L*Q*H[i]*conj(Q)'
        push!(L_list, L)
        M[i] = Q 
        # println(Q)
        R_evolve = TT_IMR_0site_new(H, R, L, R_list[i], h, i)
        # println("R_evolve: ")
        # println(R_evolve)
        M[i + 1] = R_evolve*M[i + 1]
    end
    M_N_evolve = TT_IMR_1site_new(H, M[N], L, R_list[N], h, N)
    M[N] = M_N_evolve 
    # println(M)
    return M, L_list
end



function rl_sweep_new_new(H::MPO, M::MPS, L_list::Vector{Any}, t::Float64, h::Float64)
    N = length(M)
    R_list = []
    R_block = 1
    push!(R_list, R_block)
    for i = N:-1:2 
        # println("---------------------------------------------")
        # println("L_list[$i]: ", inds(L_list[i]))
        # println("M[$i]: ", inds(M[i]))
        # println("R: ", inds(R_block))
        # println("H: ", inds(H[i]))
        # println(length(L_list))
        M_evolve = TT_IMR_1site_new(H, M[i], L_list[i], R_block, h, i)
        R, Q = factorize(M_evolve, inds(M[i]; tags = "l = $(i - 1)"); ortho = "right", tags = "Link, l = $(i-1)")

        R_block = R_block*Q*H[i]*conj(Q)'
        push!(R_list, R_block)
        M[i] = Q 
        R_evolve = TT_IMR_0site_new(H, R, L_list[i], R_block, h, i)
        M[i - 1] = R_evolve*M[i-1]
    end
    M_1_evolve = TT_IMR_1site_new(H, M[1], L_list[1], R_block, h, 1)
    M[1] = M_1_evolve 
    return M, reverse(R_list)
end


function lr_sweep_2site_new(H::MPO, M::MPS, R_list::Vector{Any}, h::Float64, cutoff::Union{Float64, Nothing}, maxdim::Union{Int64, Nothing}=nothing; normalize::Bool = false)
    N = length(M)
    L_list = []    
    L = 1
    push!(L_list, L)
    # trunc_err = 0.0
    trunc_err = zeros(N-1)
    for i = 1:N-1
        # println("Site $i")
        two_site = M[i]*M[i + 1]
        two_site_evolve = TT_IMR_2site_new(H, two_site, L, R_list[i + 1], h, i)
        M_inds = inds(two_site_evolve)
        
        if i == 1
            if N > 2
                bd = min(dim(M_inds[1]), dim(M_inds[2])*dim(M_inds[3]))
                # U_trunc, S_trunc, V_trunc = svd(two_site_evolve, M_inds[1], cutoff = cutoff)
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = 1")[1], cutoff = cutoff; lefttags = "Link, l = 1", maxdim = maxdim)
                if normalize == true 
                    S_trunc = S_trunc/norm(S_trunc)
                end
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                # U_trunc, S_trunc, V_trunc = svd(two_site_evolve, M_inds[1], cutoff = cutoff)
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = 1")[1], cutoff = cutoff; lefttags = "Link, l = 1", maxdim = maxdim)
                if normalize == true 
                    S_trunc = S_trunc/norm(S_trunc)
                end
            end
        else
            if i != N - 1
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3])*dim(M_inds[4]))
            else 
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3]))
            end
            # U_trunc, S_trunc, V_trunc = svd(two_site_evolve, M_inds[1:2], cutoff = cutoff)
            U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $i"), inds(two_site_evolve; tags = "l = $(i - 1)"), cutoff = cutoff; lefttags = "Link, l = $(i)", maxdim = maxdim)
            if normalize == true 
                S_trunc = S_trunc/norm(S_trunc)
            end
        end
        # trunc_err += sqrt(spectrum.truncerr)
        # println("Spectrum: ", spectrum)
        trunc_err[i] = sqrt(spectrum.truncerr)
        # println("Method 1")
        # @btime begin 
        #     A = $L*$U_trunc*$H[$i]*conj($U_trunc)'
        # end
        # println("Method 2")
        # @btime begin 
        #     B = $L*($U_trunc*$H[$i]*conj($U_trunc)')
        # end
        # L = L*(U_trunc*H[i]*conj(U_trunc)')
        L = L*U_trunc*H[i]*conj(U_trunc)'
        push!(L_list, L)
        M[i] = U_trunc
        M_n = S_trunc*V_trunc
        if i != N - 1
            
            M_evolve = TT_IMR_1site_new_backwards(H, M_n, L, R_list[i + 1], h, i)
            M[i + 1] = M_evolve 
        elseif i == N - 1
            M[i + 1] = S_trunc*V_trunc

        end 
    end
    return M, L_list, trunc_err
end



function rl_sweep_2site_new(H::MPO, M::MPS, L_list::Vector{Any}, h::Float64, cutoff::Union{Float64, Nothing}, maxdim::Union{Int64, Nothing}=nothing; normalize::Bool = false)
    N = length(M)
    R_list = []
    R_block = 1
    push!(R_list, R_block)
    # trunc_err = 0.0
    trunc_err = zeros(N - 1)
    for i = N:-1:2
        # println("Site $i")
        two_site = M[i]*M[i-1]
        two_site_evolve = TT_IMR_2site_new(H, two_site, L_list[i - 1], R_block, h, i - 1)
        M_inds = inds(two_site_evolve)
        if i == N
            if N > 2
                bd = min(dim(M_inds[1]), dim(M_inds[2])*dim(M_inds[3]))
                # U_trunc, S_trunc, V_trunc = svd(two_site_evolve, M_inds[1], cutoff = cutoff)
                # println("M_inds: ", M_inds)
                # println(inds(two_site_evolve; tags = "n = $N"), inds(two_site_evolve; tags = "l = $(N - 2)"))
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $(N-1)")[1], inds(two_site_evolve; tags = "l = $(N - 2)")[1], cutoff = cutoff; righttags = "l = $(N - 1)", maxdim = maxdim)
                # println("U: ", U_trunc)
                # println("S: ", S_trunc)
                # println("V: ", V_trunc)
                if normalize == true 
                    S_trunc = S_trunc/norm(S_trunc)
                end
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $N")[1], inds(two_site_evolve; tags = "l = $(N - 1)"), cutoff = cutoff; righttags = "l = $(N - 1)", maxdim = maxdim)
                # U_trunc, S_trunc, V_trunc = svd(two_site_evolve, M_inds[1], cutoff = cutoff)
                if normalize == true 
                    S_trunc = S_trunc/norm(S_trunc)
                end
            end
        else
            if i != 2
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3])*dim(M_inds[4]))
            else 
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3]))
            end
            # U_trunc, S_trunc, V_trunc = svd(two_site_evolve, M_inds[1:2], cutoff = cutoff)
            U_trunc, S_trunc, V_trunc, spectrum = svd(two_site_evolve, inds(two_site_evolve; tags = "n = $(i - 1)")[1], inds(two_site_evolve; tags = "l = $(i - 2)"), cutoff = cutoff; righttags = "l = $(i - 1)", maxdim = maxdim)

            if normalize == true 
                S_trunc = S_trunc/norm(S_trunc)
            end
        end
        # trunc_err += sqrt(spectrum.truncerr)
        trunc_err[i - 1] = sqrt(spectrum.truncerr)
        R_block = R_block*(V_trunc*H[i]*conj(V_trunc)')
        push!(R_list, R_block)
        M[i] = V_trunc 
        # println("V_trunc: ", V_trunc)
        M_n = U_trunc*S_trunc
        if i != 2
            
            M_evolve = TT_IMR_1site_new_backwards(H, M_n, L_list[i - 1], R_block, h, i - 2)
            M[i - 1] = M_evolve 
        elseif i == 2
            M[i - 1] = U_trunc*S_trunc 
        end
    end 
    return M, reverse(R_list), trunc_err
end


function tdvp_constant_adjoint(H, init, t0, T, steps, verbose = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    # storage_arr = zeros(ComplexF64, (steps + 1, d))
    # storage_arr[1,:] = vectorize_mps(init_copy; order = "natural")
    # if orthoCenter(init) != 1
        # orthogonalize!(init, 1)
    # end
    L_list_init = Vector{Any}(undef, N)
    R_list_init = contract_right(H, init, 2)
    # L_list_init = contract_left(H, init, N - 1)
    #Run time stepper
    @showprogress 1 "TDVP" for i = 1:steps
        if verbose == true
            println("Step: ", i)
        end
        
        # println("orthoCenter: ", orthoCenter(init_copy))
        
        # init_copy = lr_sweep_new(H, init_copy, t0, h)
        # println("Evolving left to right")
         
        init_copy, L_list_init = lr_sweep_new_new(H, init_copy, R_list_init, t0, h/2)
        
        t0 += h/2
        # println("L_list_init: ")
        # println(L_list_init)
        # println("init_copy1: ")
        # println(init_copy1)
        # println("Evolving right to left")
        
        init_copy, R_list_init = rl_sweep_new_new(H, init_copy, L_list_init, t0, h/2)
        
        # init_copy, L_list_init = lr_sweep_new_new(H, init_copy, R_list_init, t0, h/2)
        t0 += h/2
        # storage_arr[i + 1,:] = vectorize_mps(init_copy; order = "natural")
        # orthogonalize!(init_copy, 1)
        # orthogonalize!(init_copy, N)
        # R_list_init = contract_right(H, init_copy, 2)
        # L_list_init = contract_left(H, init_copy, N - 1)
        # init_copy = init_copy2
    end
    
    #Return evolved MPS, as well as state data at each time step
    return init_copy
    # , storage_arr
end



function tdvp2_constant(H::MPO, init::MPS, t0::Float64, T::Float64, steps::Int64; cutoff::Union{Float64, Nothing}=nothing, maxdim::Union{Int64, Nothing} = nothing, magnet::Bool = false, energy::Bool = false, verbose::Bool = false, normalize::Bool = false, strang::Bool = true)
    N = length(init)
    orthogonalize!(init, 1)
    # println("orthoCenter: ", orthoCenter(init))
    # orthogonalize!(init, N)
    
    sites = siteinds(init)
    init_copy = deepcopy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    # storage_arr = zeros(ComplexF64, (steps + 1, d))
    # storage_arr[1,:] = vectorize_mps(init_copy; order = "natural")
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(init_copy)
    magnet_history = zeros(steps + 1, N)
    if magnet == true 
        magnet_history[1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
        # for j = 1:N 
            # magnet_history[1,j] = expect(init_copy, [1 0; 0 -1]; sites = j)
        # end
    end
    energy_history = zeros(steps + 1)   
    if energy == true 
        energy_history[1] = real(inner(init_copy', H, init_copy))
    end
    L_list_list = []
    R_list_list = []
    if strang == true
        trunc_err = zeros(2*steps, N - 1)
        #Run time stepper
        L_list = Vector{Any}(undef, N)
        R_list = contract_right(H, init, 2)
        @showprogress 1 "TDVP2 Strang Splitting" for i = 1:steps
        # for i = 1:steps
            if verbose == true
                println("Step: ", i)
            end
            # init_copy = lr_sweep_new(H, init_copy, t0, h)
            # println("R_list")
            # println(length(R_list))
            # println("L_list")
            # println(length(L_list))
            # println("Evolving left to right")
            # @time begin 
            init_copy, L_list, trunc1 = lr_sweep_2site_new(H, init_copy, R_list, h/2, cutoff, maxdim; normalize = normalize)
            # println(L_list)
            # end
            # t0 += h
            # println("Evolving right to left")
            # @time begin 
            init_copy, R_list, trunc2 = rl_sweep_2site_new(H, init_copy, L_list, h/2, cutoff, maxdim; normalize = normalize)
            # end
            trunc_err[2*i - 1, :] = trunc1 
            trunc_err[2*i, :] = trunc2
            # t0 += h
            # @time begin 
            # storage_arr[i + 1,:] = vectorize_mps(init_copy; order = "natural")
            # end
            link_dim[i + 1,:] = linkdims(init_copy)
            if magnet == true 
                magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
                # for j = 1:N 
                    # magnet_history[i + 1,j] = expect(init_copy, [1 0; 0 -1]; sites = j)
                # end
            end
            if energy == true 
                energy_history[i + 1] = real(inner(init_copy', H, init_copy))
            end
            # orthogonalize!(init_copy, 1)
            # L_list = contract_left(H, init_copy, 2)
            # R_list = contract_right(H, init_copy, N - 1)
            # init_copy = init_copy2
        end
        
        #Return evolved MPS, as well as state data at each time step
        # return init_copy, storage_arr, link_dim
        
    elseif strang == false 
        trunc_err = zeros(steps, N - 1)
        @showprogress 1 "TDVP2 Lie-Trotter Splitting" for i = 1:steps
            R_list = contract_right(H, init_copy, 2)
            push!(R_list_list, R_list)
            if verbose == true 
                println("Step: ", i)
            end
            init_copy, L_list, trunc = lr_sweep_2site_new(H, init_copy, R_list, h, cutoff, maxdim; normalize = normalize)
            link_dim[i + 1,:] = linkdims(init_copy)
            if magnet == true 
                magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
                # for j = 1:N 
                    # magnet_history[i + 1,j] = expect(init_copy, [1 0; 0 -1]; sites = j)
                # end
            end
            if energy == true 
                energy_history[i + 1] = real(inner(init_copy', H, init_copy))
            end
            trunc_err[i,:] = trunc
            orthogonalize!(init_copy, 1)
        end
    end

    return init_copy, link_dim, magnet_history, energy_history, trunc_err
    # , storage_arr
end

function tdvp2(H::MPO, init::MPS, t0::Float64, T::Float64, steps::Int64, bc_params::bcparams; cutoff::Union{Float64, Nothing}=nothing, maxdim::Union{Int64, Nothing} = nothing, magnet::Bool = false, energy::Bool = false, verbose::Bool = false, normalize::Bool = false, strang::Bool = true)
    N = length(init)
    orthogonalize!(init, 1)
    # println("orthoCenter: ", orthoCenter(init))
    # orthogonalize!(init, N)
    
    sites = siteinds(init)
    init_copy = deepcopy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    # storage_arr = zeros(ComplexF64, (steps + 1, d))
    # storage_arr[1,:] = vectorize_mps(init_copy; order = "natural")
    link_dim = zeros(steps + 1, N - 1)
    link_dim[1,:] = linkdims(init_copy)
    magnet_history = zeros(steps + 1, N)
    if magnet == true 
        magnet_history[1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
        # for j = 1:N 
            # magnet_history[1,j] = expect(init_copy, [1 0; 0 -1]; sites = j)
        # end
    end
    energy_history = zeros(steps + 1)   
    if energy == true 
        energy_history[1] = real(inner(init_copy', H, init_copy))
    end
    # L_list_list = []
    # R_list_list = []
    if strang == true
        t0 = t0
        trunc_err = zeros(2*steps, N - 1)
        #Run time stepper
        L_list = Vector{Any}(undef, N)
        # R_list = contract_right(H, init, 2)
        @showprogress 1 "TDVP2 Strang Splitting" for i = 1:steps
        # for i = 1:steps
            if verbose == true
                println("Step: ", i)
            end
            # init_copy = lr_sweep_new(H, init_copy, t0, h)
            # println("R_list")
            # println(length(R_list))
            # println("L_list")
            # println(length(L_list))
            # println("Evolving left to right")
            # @time begin 
            update_MPO!(H, bc_params, t0 + h/4)
            R_list = contract_right(H, init_copy, 2)
            init_copy, _, trunc1 = lr_sweep_2site_new(H, init_copy, R_list, h/2, cutoff, maxdim; normalize = normalize)
            
            # println(L_list)
            # end
            # t0 += h
            # println("Evolving right to left")
            # @time begin
            t0 += h/2
            update_MPO!(H, bc_params, t0 + h/4)
            L_list = contract_left(H, init_copy, N - 1)
            init_copy, _, trunc2 = rl_sweep_2site_new(H, init_copy, L_list, h/2, cutoff, maxdim; normalize = normalize)
            
            # end
            trunc_err[2*i - 1, :] = trunc1 
            trunc_err[2*i, :] = trunc2
            # t0 += h
            # @time begin 
            # storage_arr[i + 1,:] = vectorize_mps(init_copy; order = "natural")
            # end
            link_dim[i + 1,:] = linkdims(init_copy)
            if magnet == true 
                magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
                # for j = 1:N 
                    # magnet_history[i + 1,j] = expect(init_copy, [1 0; 0 -1]; sites = j)
                # end
            end
            if energy == true 
                energy_history[i + 1] = real(inner(init_copy', H, init_copy))
            end
            # orthogonalize!(init_copy, 1)
            # L_list = contract_left(H, init_copy, 2)
            # R_list = contract_right(H, init_copy, N - 1)
            # init_copy = init_copy2
            t0 += h/2
        end
        
        #Return evolved MPS, as well as state data at each time step
        # return init_copy, storage_arr, link_dim
        
    elseif strang == false 
        t0 = t0
        trunc_err = zeros(steps, N - 1)
        @showprogress 1 "TDVP2 Lie-Trotter Splitting" for i = 1:steps
            update_MPO!(H, bc_params, t0 + h/2)
            R_list = contract_right(H, init_copy, 2)
            if verbose == true 
                println("Step: ", i)
            end
            init_copy, _, trunc = lr_sweep_2site_new(H, init_copy, R_list, h, cutoff, maxdim; normalize = normalize)
            link_dim[i + 1,:] = linkdims(init_copy)
            if magnet == true 
                magnet_history[i + 1,:] = reverse(expect(init_copy, [1 0; 0 -1]))
                # for j = 1:N 
                    # magnet_history[i + 1,j] = expect(init_copy, [1 0; 0 -1]; sites = j)
                # end
            end
            if energy == true 
                energy_history[i + 1] = real(inner(init_copy', H, init_copy))
            end
            t0 += h
            trunc_err[i,:] = trunc
            orthogonalize!(init_copy, 1)
        end
    end

    return init_copy, link_dim, magnet_history, energy_history, trunc_err
    # , storage_arr
end