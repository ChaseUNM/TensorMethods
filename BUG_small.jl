using LinearAlgebra, ITensors, ProgressMeter, BenchmarkTools, Random 

Random.seed!(42)
include("BUG_tucker(8-27).jl")
include("Hamiltonian.jl")
include("Tucker_Matrices.jl")
#Pratice BUG but don't form any large matrices
sz = [1 0; 0 -1]
sx = [0 1; 1 0]
sy = [0 -im; im 0]

function op_itensor(op, sites, ind)
    return ITensor(op, sites[ind], sites[ind]')
end



#Let H the xxx hamiltonian with J = -1, g = 1

# core_inds = inds(core)
# Q1, S1 = qr(core, core_inds[2], core_inds[3])
# Q2, S2 = qr(core, core_inds[1], core_inds[3])
# Q3, S3 = qr(core, core_inds[1], core_inds[2])

# K1 = factors[1]*S1 
# K2 = factors[2]*S2 
# K3 = factors[3]*S3

function applyH_tucker(op_list, Q, factors, site)
    # core_inds = inds(core)
    # not_site = setdiff(core_inds, [core_inds[site]])
    # Q, R = qr(core, not_site)
    new_factors = [] 
    N = length(factors)
    # println("N: $N")
    for i = 1:N
        # println("i: $i")
        if i == site
            # display(op_list[i])
            # display(factors[i])
            # println(op_list[i]*factors[i]) 
            push!(new_factors, op_list[i]*factors[i])
        else 
            # display(conj(factors[i]')*op_list[i]*factors[i])
            push!(new_factors, conj(factors[i]')*op_list[i]*factors[i])
        end
        # println("New factors: ")
        # println(new_factors)
    end
    return new_factors 
end

function applyH_mat(op_list, factors, site)
    # core_inds = inds(core)
    # not_site = setdiff(core_inds, [core_inds[site]])
    # Q, R = qr(core, not_site)
    # Preallocate new_factors with known sizes for efficiency
    new_factors = [zeros(eltype(factors[i]), size(factors[i], 2), size(factors[i], 2)) for i in 1:length(factors)]
    # Adjust size for the selected site, since op_list[site]*factors[site] changes shape
    new_factors[site] = zeros(eltype(factors[site]), size(op_list[site], 1), size(factors[site], 2))
    temp_factors = [zeros(eltype(factors[i]), size(factors[i], 1), size(factors[i], 2)) for i in 1:length(factors)]
    N = length(factors)
    # println("N: $N")
    for i = 1:N
        # println("Site $i")
        # println("i: $i")
        if i == site
            # display(op_list[i])
            # display(factors[i])
            # println(op_list[i]*factors[i]) 
            # new_factors[i] = op_list[i]*factors[i]
            mul!(new_factors[i], op_list[i], factors[i])
        else 
            # display(conj(factors[i]')*op_list[i]*factors[i])
            # new_factors[i] = (op_list[i]*factors[i])'*factors[i]
            mul!(temp_factors[i], op_list[i], factors[i])
            mul!(new_factors[i], (temp_factors[i])', factors[i])
        end
        # println("New factors: ")
        # println(new_factors)
    end
    return new_factors 
end

#This is only for Hamiltonians that act on maximum 2 sites at a time
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

function op_itensor_list(op_mat_list, sites) 
    N = length(op_mat_list)
    ops = Array{Any}(undef, N) 
    for i in 1:N 
        # op = op_itensor(op_mat_list, sites, i)
        op = ITensor(op_mat_list[i], sites[i], sites[i]')
        ops[i] = op 
    end
    return ops 
end

# K1_mat = rand(2,2)
# K1 = ITensor(K1_mat, inds(K1))

# Ident = Matrix(1.0*I, 2, 2)
# op = [sx, sz, sx]
# Q1_inds = inds(Q1)
# op_list = op_itensor_list(op, sites)

# factors_intermediate = [K1, factors[2], factors[3]]
# new_factors = applyH_tucker(op_list, Q1, factors_intermediate, 1)
# println("Answer: ")
# println(reconstruct(Q1, new_factors)*conj(Q1)')

# K1_mat = Array(K1, inds(K1))
# combine_Q = combiner(inds(Q1)[1], inds(Q1)[2])
# Q = Q1*combine_Q
# Q_mat = Array(Q, inds(Q))
# println("Answer 2")
# display(K1_mat*Q_mat*Q_mat') 

# A = zeros(2, 2, 2)
# A[:,:,1] = [1 2; 3 4]
# A[:,:,2] = [5 6; 7 8]
# A1 = ITensor(A, inds(Q1))
# A = A1*combine_Q 
# A_mat = Array(A, inds(A))
# # A_arr = reshape(Array(A1, inds(A1)), 2, 4)


# new_factors = applyH_tucker(op_list, A1, factors_intermediate, 1)
# println("Answer")
# ans1 = reconstruct(Q1, new_factors)*conj(Q1)'
# println(ans1)


# mat_factors = matricize_factors(factors)
# println("Answer 2")
# display(op[1]*K1_mat*Q_mat*(kron(mat_factors[3]'*op[3]'*mat_factors[3], mat_factors[2]*op[2]'*mat_factors[2]))*Q_mat')

# #Now try summing 
# ops1 = [sx, Ident, Ident]
# ops2 = [Ident, sx, Ident]
# ops1_itensor = op_itensor_list(ops1, sites)
# ops2_itensor = op_itensor_list(ops2, sites)


# new_factors1 = applyH_tucker(ops1_itensor, Q1, factors_intermediate, 1)
# new_factors2 = applyH_tucker(ops2_itensor, Q1, factors_intermediate, 1)

# println("Answer: ")
# ans_sum = (reconstruct(Q1, new_factors1) + reconstruct(Q1, new_factors2))*conj(Q1)'
# println(ans_sum)
# println("Answer 2")
# display((ops1[1]*K1_mat*Q_mat*(kron(mat_factors[3]'*ops1[3]'*mat_factors[3], mat_factors[2]*ops1[2]'*mat_factors[2])) + ops2[1]*K1_mat*Q_mat*(kron(mat_factors[3]'*ops2[3]'*mat_factors[3], mat_factors[2]*ops2[2]'*mat_factors[2])))*Q_mat')

function xxx_ops(N,J,g)
    op_list = [] 
    # create zz chain
    for i in 1:N-1
        ops = [ComplexF64.(Ident) for _ in 1:N]
        ops[i] = ComplexF64.(-J*sz) 
        ops[i + 1] = ComplexF64.(sz)
        push!(op_list, ops)
    end
    #create the x chain
    if g != 0.0
        for i in 1:N 
            ops = [ComplexF64.(Ident) for _ in 1:N]
            ops[i] = ComplexF64.(-g*J*sx)
            push!(op_list, ops)
        end
    end
    return op_list
end

function ops_ex_3(a, b, c)
    op_list = []
    for i in 1:1 
        ops = [a, b, c]
        push!(op_list, ops)
    end
    return op_list 
end

function identity_ops(N)
    ops = [Ident for _ in 1:N]
    return ops 
end

function total_H_itensor(op_list_list, sites)
    N = length(op_list_list)
    op_list_list_itensor = Array{Any}(undef, N)
    for i in 1:N 
        op_list_list_itensor[i] = op_itensor_list(op_list_list[i], sites)
    end
    return op_list_list_itensor 
end

function K_evolution_mat(core, factors, site, K, ops_list)
    core_inds = inds(core)
    not_site_inds = setdiff(core_inds, [core_inds[site]])
    Q, S = qr(core, not_site_inds)
    K = factors[site]*S 
    K = Array(K, inds(K))
    C = combiner(not_site_inds;tags = "c")
    Q_arr = Q*C 
    Q_arr = Array(Q_arr, inds(Q_arr))

    N_ops = length(ops_list)
    mat_factors = matricize_factors(factors)
    init = ops_list[1][1]*K*Q_arr*kron(mat_factors[3]'*ops_list[1][3]'*mat_factors[3], mat_factors[2]*ops_list[1][2]'*mat_factors[2])
    for i in 2:N_ops
        init += ops_list[i][1]*K*Q_arr*kron(mat_factors[3]'*ops_list[i][3]'*mat_factors[3], mat_factors[2]*ops_list[i][2]'*mat_factors[2])
    end
    Kdot = init*Q_arr' 
    return Kdot
end
    

function K_evolution_itensor(core,factors,site, K, total_H,sites)
    core_inds = inds(core)
    not_site_inds = setdiff(core_inds, [core_inds[site]])
    Q, S = qr(core, not_site_inds)
    K_ten = factors[site]*S
    # println("K_ten")
    # println(K_ten)
    factors_copy = copy(factors)
    factors_intermediate = factors_copy 
    # K_inds = inds(factors[site]*S)
    # K_ten = ITensor(K, K_inds)
    factors_intermediate[site] = K_ten

    N_ops = length(total_H)
    
    new_factors = applyH_tucker(total_H[1], Q, factors_intermediate, site)
    # println("Q")
    # display(Q)
    # println("factors_intermediate")
    # display(factors_intermediate)
    init = reconstruct(Q, new_factors)
    # display(init)
    # @time begin 
    for i in 2:N_ops 
        new_factors = applyH_tucker(total_H[i], Q, factors_intermediate, site)
        init += reconstruct(Q, new_factors)
    # end
    # display(init)
    K_dot = init*conj(Q')
    end
    return K_dot, K_ten
end

function K_evolution_itensor(core,factors,site, total_H, sites)
    core_inds = inds(core)
    not_site_inds = setdiff(core_inds, [core_inds[site]])
    Q, S = qr(core, not_site_inds)
    K_ten = factors[site]*S
    # println("K_ten")
    # println(K_ten)
    factors_copy = copy(factors)
    factors_intermediate = factors_copy 
    # K_inds = inds(factors[site]*S)
    # K_ten = ITensor(K, K_inds)
    factors_intermediate[site] = K_ten

    N_ops = length(total_H)
    new_factors = applyH_tucker(total_H[1], Q, factors_intermediate, site)
    init = reconstruct(Q, new_factors)

    # @time begin 
    for i in 2:N_ops 
        new_factors = applyH_tucker(total_H[i], Q, factors_intermediate, site)
        init += reconstruct(Q, new_factors)
    end
    # display(init)
    K_dot = noprime(init*conj(Q'))
    # end
    return K_dot, K_ten
end

function K_evolution_itensor2(core,factors,site, total_H, sites)
    core_inds = inds(core)
    not_site_inds = setdiff(core_inds, [core_inds[site]])
    Q, S = qr(core, not_site_inds)
    K_ten = factors[site]*S
    # println("K_ten")
    # println(K_ten)
    factors_copy = copy(factors)
    factors_intermediate = factors_copy 
    # K_inds = inds(factors[site]*S)
    # K_ten = ITensor(K, K_inds)
    factors_intermediate[site] = K_ten

    N_ops = length(total_H)
    new_factors = applyHV_tucker(total_H[1], factors_intermediate, site, sites)
    init = reconstruct(Q, new_factors)

    # @time begin 
    for i in 2:N_ops 
        new_factors = applyHV_tucker(total_H[i], factors_intermediate, site, sites)
        init += reconstruct(Q, new_factors)
    end
    # display(init)
    K_dot = noprime(init*conj(Q'))
    # end
    return K_dot, K_ten
end

function K_evolution_mat(core,factors,site, total_H, M_list, P_list, Y_list)
    # core_inds = inds(core)
    # not_site_inds = setdiff(core_inds, [core_inds[site]])
    # Q, S = qr(core, not_site_inds)
    Q, S = qr(transpose(matricization(core, site)))
    Q = Array(Q)[:,1:size(core, site)]
    Q_ten = refold_mat(Array(transpose(Q)), size(core), site)

    K = factors[site]*transpose(S)
    # K_ten = factors[site]*S
    # println("K_ten")
    # println(K_ten)
    factors_copy = copy(factors)
    factors_intermediate = factors_copy 
    # K_inds = inds(factors[site]*S)
    # K_ten = ITensor(K, K_inds)
    factors_intermediate[site] = K
    # println("Size factors_intermediate: ")
    # println(size(factors_intermediate[1]))
    # println(size(factors_intermediate[2]))
    # println(size(factors_intermediate[3]))
    N_ops = length(total_H)
    # @time begin 
    new_factors = applyH_mat(total_H[1], factors_intermediate, site)
    # end
    # println("Matrix Factors: ")
    # for i in 1:length(factors)
    #     println("Factor $i: ")
    #     display(new_factors[i])
    # end
    # println("Q")
    # display(Q)
    # println("factors_intermediate")
    # display(factors_intermediate)
    # init = zeros(ComplexF64, size(core)...)
    # println(size(init))
    # println("Size Q_ten: ", size(Q_ten))
    # println("Size Factors: ")
    # println(size(new_factors[1]))
    # println(size(new_factors[2]))
    # println(size(new_factors[3]))
    # init = Multi_TTM_allocate_recursive(Q_ten, new_factors, M_list, P_list, Y_list)
    init = Multi_TTM_recursive(Q_ten, new_factors)
    # init .= Multi_TTM(Q_ten, new_factors)
    # println("Q: ")
    # display(Q_ten)
    # println("Init:")
    # display(init)
    # println("Arr Q*factor2: ")
    # display(TTM(Q_ten, new_factors[1], 1))
    # @time begin 
    for i in 2:N_ops
        new_factors = applyH_mat(total_H[i], factors_intermediate, site)
        # init .+= Multi_TTM_allocate_recursive(Q_ten, new_factors, M_list, P_list, Y_list)
        init .+= Multi_TTM_recursive(Q_ten, new_factors)
        # init .+= Multi_TTM(Q_ten, new_factors)
    end
    # display(init)
    # println("Mat_i Init size: ", size(matricization(init, site)))
    # println("Q' size: ", size(Q))
    K_dot = matricization(init, site)*conj(Q)
    # end
    # return K_dot, K, new_factors, Q_ten
    return K_dot, K
end

#Allocated memory version 
function K_evolution_mat2(core,factors,site, total_H, M_list, P_list, Y_list)
    # core_inds = inds(core)
    # not_site_inds = setdiff(core_inds, [core_inds[site]])
    # Q, S = qr(core, not_site_inds)
    Q, S = qr(transpose(matricization(core, site)))
    Q = Array(Q)[:,1:size(core, site)]
    Q_ten = refold_mat(Array(transpose(Q)), size(core), site)

    K = factors[site]*transpose(S)
    # K_ten = factors[site]*S
    # println("K_ten")
    # println(K_ten)
    factors_copy = copy(factors)
    factors_intermediate = factors_copy 
    # K_inds = inds(factors[site]*S)
    # K_ten = ITensor(K, K_inds)
    factors_intermediate[site] = K
    # println("Size factors_intermediate: ")
    # println(size(factors_intermediate[1]))
    # println(size(factors_intermediate[2]))
    # println(size(factors_intermediate[3]))
    N_ops = length(total_H)
    # @time begin 
    new_factors = applyHV(total_H[1], factors_intermediate, site)
    # end
    # println("Matrix Factors: ")
    # for i in 1:length(factors)
    #     println("Factor $i: ")
    #     display(new_factors[i])
    # end
    # println("Q")
    # display(Q)
    # println("factors_intermediate")
    # display(factors_intermediate)
    # init = zeros(ComplexF64, size(core)...)
    # println(size(init))
    # println("Size Q_ten: ", size(Q_ten))
    # println("Size Factors: ")
    # println(size(new_factors[1]))
    # println(size(new_factors[2]))
    # println(size(new_factors[3]))
    init = Multi_TTM_allocate_recursive(Q_ten, new_factors, M_list, P_list, Y_list)
    # init = Multi_TTM_recursive(Q_ten, new_factors)
    # init .= Multi_TTM(Q_ten, new_factors)
    # println("Q: ")
    # display(Q_ten)
    # println("Init:")
    # display(init)
    # println("Arr Q*factor2: ")
    # display(TTM(Q_ten, new_factors[1], 1))
    # @time begin 
    for i in 2:N_ops
        new_factors = applyHV(total_H[i], factors_intermediate, site)
        init .+= Multi_TTM_allocate_recursive(Q_ten, new_factors, M_list, P_list, Y_list)
        # init .+= Multi_TTM_recursive(Q_ten, new_factors)
        # init .+= Multi_TTM(Q_ten, new_factors)
    end
    # display(init)
    # println("Mat_i Init size: ", size(matricization(init, site)))
    # println("Q' size: ", size(Q))
    K_dot = matricization(init, site)*conj(Q)
    # end
    # return K_dot, K, new_factors, Q_ten
    return K_dot, K
end

#Non-allocated memory version
function K_evolution_mat2(core,factors,site, total_H)
    Q, S = qr(transpose(matricization(core, site)))
    K = factors[site]*transpose(S)
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
    for i in 2:N_ops
        new_factors = applyHV(total_H[i], factors_intermediate, site)
        # new_factors = applyH_mat(total_H[i], factors_intermediate, site)
        init .+= Multi_TTM_recursive(Q_ten, new_factors)
    end
    K_dot = matricization(init, site)*conj(Q)
    return K_dot, K
end

function C_dot_itensor(core, factors, total_H, sites)
    N_ops = length(total_H)
    N_factors = length(factors)
    mat_factors = matricize_factors(factors)
    desired_inds = prime(inds(core))
    init = ITensor(desired_inds)
    new_factors = []
    new_factors = Array{Any}(undef, N_factors)
    # factor_multiple_time1 = 0.0
    # factor_multiple_time2 = 0.0
    for i in 1:N_ops 
        for j in 1:N_factors 
            # new_factor = mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j]
            # println(new_factor)
            # new_factors[j] = ITensor(new_factor, inds(core)[j], inds(core)[j]')
            # mult_elapsed = @elapsed begin
            #     new_factors[j] = ITensor(mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j] , inds(core)[j], inds(core)[j]')
            # # new_factors[j] = conj(factors[j]')*total_H[i][j]*factors[j]
            # end
            # @time begin 
            new_factors[j] = conj(factors[j]')*total_H[i][j]*factors[j]
            # end
            # factor_multiple_time1 += mult_elapsed
            # factor_multiple_time2 += mult_elapsed2
            # push!(new_factors, conj(factors[j]')*total_H[i][j]*factors[j])
        end
        # println(core)
        # println(new_factors)
        # time_elapsed = @elapsed begin
        init += reconstruct(core, new_factors)
        # reconstruct_time += time_elapsed
    end
    # display(new_factors)
    return init
end

function C_dot_itensor_im(core, factors, total_H, sites)
    N_ops = length(total_H)
    N_factors = length(factors)
    mat_factors = matricize_factors(factors)
    desired_inds = prime(inds(core))
    init = ITensor(desired_inds)
    new_factors = []
    new_factors = Array{Any}(undef, N_factors)
    # factor_multiple_time1 = 0.0
    # factor_multiple_time2 = 0.0
    for i in 1:N_ops 
        for j in 1:N_factors 
            # new_factor = mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j]
            # println(new_factor)
            # new_factors[j] = ITensor(new_factor, inds(core)[j], inds(core)[j]')
            # mult_elapsed = @elapsed begin
            #     new_factors[j] = ITensor(mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j] , inds(core)[j], inds(core)[j]')
            # # new_factors[j] = conj(factors[j]')*total_H[i][j]*factors[j]
            # end
            # @time begin 
            new_factors[j] = conj(factors[j]')*total_H[i][j]*factors[j]
            # println("Factor $j")
            # println(factors[j]')
            # println(factors[j])
            # println(total_H[i][j])
            # println(new_factors[j])
            # println("LOOK HERE")
            # display(new_factors[j])
            # end
            # factor_multiple_time1 += mult_elapsed
            # factor_multiple_time2 += mult_elapsed2
            # push!(new_factors, conj(factors[j]')*total_H[i][j]*factors[j])
        end
        # println(core)
        # println(new_factors)
        # time_elapsed = @elapsed begin
        # println("LOOK HERE NOW")
        # display(core)
        # @time begin 
        init += -im*reconstruct(core, new_factors)
        # end
        # reconstruct_time += time_elapsed
    end
    return noprime(init)
end

function C_dot_itensor_im(core, factors, alloc_arr, total_H, sites)
    N_ops = length(total_H)
    N_factors = length(factors)
    mat_factors = matricize_factors(factors)
    desired_inds = inds(core)
    init = ITensor(desired_inds)
    new_factors = []
    new_factors = Array{Any}(undef, N_factors)
    # factor_multiple_time1 = 0.0
    # factor_multiple_time2 = 0.0
    for i in 1:N_ops 
        for j in 1:N_factors 
            # new_factor = mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j]
            # println(new_factor)
            # new_factors[j] = ITensor(new_factor, inds(core)[j], inds(core)[j]')
            # mult_elapsed = @elapsed begin
            #     new_factors[j] = ITensor(mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j] , inds(core)[j], inds(core)[j]')
            # # new_factors[j] = conj(factors[j]')*total_H[i][j]*factors[j]
            # end
            # @time begin 
            new_factors[j] = conj(factors[j]')*total_H[i][j]*factors[j]
            # println("Factor $j")
            # println(factors[j]')
            # println(factors[j])
            # println(total_H[i][j])
            # println(new_factors[j])
            # println("LOOK HERE")
            # display(new_factors[j])
            # end
            # factor_multiple_time1 += mult_elapsed
            # factor_multiple_time2 += mult_elapsed2
            # push!(new_factors, conj(factors[j]')*total_H[i][j]*factors[j])
        end
        # println(core)
        # println(new_factors)
        # time_elapsed = @elapsed begin
        # println("LOOK HERE NOW")
        # display(core)
        # @time begin 
        # init += -im*reconstruct(core, new_factors)
        Multi_TTM!(core, new_factors, alloc_arr)
        init += alloc_arr[end]
        # end
        # reconstruct_time += time_elapsed
    end
    return noprime(init)
end

function C_dot_im_mat(core, factors, total_H, Ms, Ps, Ys)
    N_ops = length(total_H)
    # println("N_ops: $N_ops")
    N_factors = length(factors)
    # desired_inds = prime(inds(core))
    # init = ITensor(desired_inds)
    # init = zeros(eltype(core), size(core)...)
    init = rand(eltype(core), size(core)...)
    # new_factors = []
    # new_factors = Vector{AbstractMatrix}(undef, N_factors)
    new_factors = [zeros(eltype(factors[i]), size(factors[i], 2), size(factors[i], 2)) for i in 1:length(factors)]
    temp_factors = [zeros(eltype(factors[i]), size(factors[i], 2), size(factors[i], 2)) for i in 1:length(factors)]
    # factor_multiple_time1 = 0.0
    # factor_multiple_time2 = 0.0
    # a = zeros(eltype(core), size(core)...)
    # b = zeros(eltype(core), size(core)...)
    # c = zeros(eltype(core), size(core)...)
    # factors_copy = copy(factors)
    # core_copy = copy(core)
    for i in 1:N_ops
        println("Operation $i: ") 
        for j in 1:N_factors
            
            # new_factor = mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j]
            # println(new_factor)
            # new_factors[j] = ITensor(new_factor, inds(core)[j], inds(core)[j]')
            # mult_elapsed = @elapsed begin
            #     new_factors[j] = ITensor(mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j] , inds(core)[j], inds(core)[j]')
            # # new_factors[j] = conj(factors[j]')*total_H[i][j]*factors[j]
            # end
            # @time begin 
            # new_factors[j] = factors[j]'*total_H[i][j]*factors[j]
            # new_factors[j] = (total_H[i][j]*factors[j])'*factors[j]
            # @time begin
            # temp_factors = [zeros(eltype(factors[i]), size(factors[i], 2), size(factors[i], 2)) for i in 1:length(factors)] 
            mul!(temp_factors[j], total_H[i][j], factors[j])
            mul!(new_factors[j], temp_factors[j]', factors[j])
            # end
            # display(total_H[i][j])
            # println("Factor $j")
            # println(factors[j]')
            # println(factors[j])
            # println(total_H[i][j])
            # println(new_factors[j])
            # println("LOOK HERE")
            # display(new_factors[j])
            # end
            # factor_multiple_time1 += mult_elapsed
            # factor_multiple_time2 += mult_elapsed2
            # push!(new_factors, conj(factors[j]')*total_H[i][j]*factors[j])
        end
        # println("Ms: $Ms")
        # println("Ps: $Ps")
        # println("Ys: ")
        # display(Ys)
        # init .+= -im*Multi_TTM_recursive(core, new_factors)
        # display(-im*Multi_TTM_allocate_recursive(core, new_factors, Ms, Ps, Ys))
        display(-im*Multi_TTM_allocate(core, new_factors, Ms, Ps, Ys))
        # display(-im*Multi_TTM_recursive(core, new_factors))
        # a = -im*Multi_TTM_allocate(core, new_factors, M_list, P_list, Y_list)
        # println("Incorrect")
        # a .+= -im*Multi_TTM_allocate(core, new_factors, M_list, P_list, Y_list)
        # println("Look here")
        # println("Difference: ", norm(a - b))
        # end
        # init .+= -im*Multi_TTM(core, new_factors)
        # reconstruct_time += time_elapsed
    end
    return init
end



function tensorize_factors(factors, inds)
    arr = []
    for i in 1:length(factors)
        push!(arr, ITensor(factors[i], inds[i], inds[i]'))
    end
    return arr 
end

function C_dot_itensor_eff(core, factors, total_H, total_H_arr, sites)
    N_ops = length(total_H)
    N_factors = length(factors)
    mat_factors = matricize_factors(factors)
    desired_inds = prime(inds(core))
    # println("Desired inds: ")
    # println(desired_inds)
    init = ITensor(desired_inds)
    new_factors = []
    new_factors = Array{Any}(undef, N_factors)
    # factor_multiple_time1 = 0.0
    # factor_multiple_time2 = 0.0
    for i in 1:N_ops 
        for j in 1:N_factors 
            # new_factor = mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j]
            # println(new_factor)
            # new_factors[j] = ITensor(new_factor, inds(core)[j], inds(core)[j]')
            # mult_elapsed = @elapsed begin
            #     new_factors[j] = ITensor(mat_factors[j]'*Array(total_H[i][j], inds(total_H[i][j]))*mat_factors[j] , inds(core)[j], inds(core)[j]')
            # # new_factors[j] = conj(factors[j]')*total_H[i][j]*factors[j]
            # end
            # @time begin 
            # display(conj(factors[j]')*total_H[i][j]*factors[j])
            # end
            # @time begin 
            new_factors[j] = mat_factors[j]'*total_H_arr[i][j]*mat_factors[j]
            # end
            # factor_multiple_time1 += mult_elapsed
            # factor_multiple_time2 += mult_elapsed2
            # push!(new_factors, conj(factors[j]')*total_H[i][j]*factors[j])
        end
        # println(core)
        # println(new_factors)
        # time_elapsed = @elapsed begin
        # display(reconstruct_mat(core, new_factors))
        init += reconstruct_mat(core, new_factors)
        # reconstruct_time += time_elapsed
    end
    # display(new_factors)
    return init
end


#Test Kdot and Cdot versus matrix versions 
#Create the Hamiltonian
# N = 4
# sites = siteinds("Qubit", N)
# cutoff = 0.0

# J = -1
# g = 1.0
# xxx_ops_list = xxx_ops(N, J, g)
# total_H = total_H_itensor(xxx_ops_list, sites)
# ident_ops = [identity_ops(N)]
# total_H_ident = total_H_itensor(ident_ops, sites)

# A = rand(ComplexF64, collect(fill(2, N))...)
# core_ten, factors_ten = tucker_itensor(A, sites; cutoff = 0.0)
# core_arr, factors_arr = tucker(A; cutoff = 0.0)

# M_list, P_list, Y_list = pre_allocate(core_arr, factors_arr)
# a1 = @ballocated K_evolution_mat(core_arr, factors_arr, 1, xxx_ops_list)
# a2 = @ballocations K_evolution_itensor(core_ten, factors_ten, 1, total_H, sites)

# alloc_1 = @btimed begin 
# Kdot1_arr,_, _,_ = K_evolution_mat(core_arr, factors_arr, 1, xxx_ops_list)
# end 

# N_last = 15

# Kdot_mat_time = zeros(N_last - 1)
# Cdot_mat_time = zeros(N_last - 1)
# Kdot_ten_time = zeros(N_last - 1)
# Cdot_ten_time = zeros(N_last - 1)
# Kdot_mat_mem = zeros(N_last - 1)
# Cdot_mat_mem = zeros(N_last - 1)
# Kdot_ten_mem = zeros(N_last - 1)
# Cdot_ten_mem = zeros(N_last - 1)
# Kdot_mat_alloc = zeros(N_last - 1)
# Cdot_mat_alloc = zeros(N_last - 1)
# Kdot_ten_alloc = zeros(N_last - 1)
# Cdot_ten_alloc = zeros(N_last - 1)

# Cdot_full_time = zeros(N_last - 1)
# Cdot_full_mem = zeros(N_last - 1)
# Cdot_full_alloc = zeros(N_last - 1)

# p1 = plot(collect(2:N_last), [Kdot_mat_time, Kdot_ten_time], labels = ["Array" "ITensor"], xlabel = "# of qubits", ylabel = "time(s)")
# p2 = plot(collect(2:N_last), [Kdot_mat_mem, Kdot_ten_mem], labels = ["Array" "ITensor"], xlabel = "# of qubits", ylabel = "memory (bytes)")
# p3 = plot(collect(2:N_last), [Kdot_mat_alloc, Kdot_ten_alloc], labels = ["Array" "ITensor"], xlabel = "# of qubits", ylabel = "# of allocations")
# p4 = plot(collect(2:N_last), [Cdot_mat_time, Cdot_ten_time], labels = ["Array" "ITensor"], xlabel = "# of qubits", ylabel = "time(s)", legend =:topleft)
# plot!(collect(2:13), Cdot_full_time[1:12], label = "ITensor Full", yscale =:log10, legend =:topleft)
# p5 = plot(collect(2:N_last), [Cdot_mat_mem, Cdot_ten_mem], labels = ["Array" "ITensor"], xlabel = "# of qubits", ylabel = "memory (bytes)", legend =:topleft)
# plot!(collect(2:13), Cdot_full_mem[1:12], label = "ITensor Full", yscale =:log10, legend =:topleft)
# p6 = plot(collect(2:N_last), [Cdot_mat_alloc, Cdot_ten_alloc], labels = ["Array" "ITensor"], xlabel = "# of qubits", ylabel = "# of allocations", legend =:topleft)
# plot!(collect(2:13), Cdot_full_alloc[1:12], label = "ITensor Full", yscale =:log10)

# for N = 2:N_last
# println("# of sites: $N") 
# N = 12
# sites = siteinds("Qubit", N)
# J = -1
# g = 1.0

# xxx_ops_list = xxx_ops(N, J, g)
# total_H = total_H_itensor(xxx_ops_list, sites)

# A = rand(ComplexF64, collect(fill(2, N))...)
# core_ten, factors_ten = tucker_itensor(A, sites; cutoff = 0.0)
# core_arr, factors_arr = tucker(A; cutoff = 0.0)

# M_list, P_list, Y_list = pre_allocate(core_arr, factors_arr)

# alloc1 = @btimed begin 
# Kdot1_arr,_, _,_ = K_evolution_mat(core_arr, factors_arr, 1, $xxx_ops_list)
# end 
# alloc1 = @btimed begin 
#     K_evolution_mat(core_arr, factors_arr, 1, $xxx_ops_list)
# end
# Kdot_mat_time[N - 1] = alloc1[2]
# Kdot_mat_mem[N - 1] = alloc1[3]
# Kdot_mat_alloc[N - 1] = alloc1[4]

# alloc2 = @btimed begin 
# Kdot1_ten,_, _,_ = K_evolution_itensor(core_ten, factors_ten, 1, total_H, sites)
# end
# Kdot_ten_time[N - 1] = alloc2[2]
# Kdot_ten_mem[N - 1] = alloc2[3]
# Kdot_ten_alloc[N - 1] = alloc2[4]

# alloc3 = @btimed begin
# Cdot_arr = C_dot_im_mat(core_arr, factors_arr, xxx_ops_list)
# end
# Cdot_mat_time[N - 1] = alloc3[2]
# Cdot_mat_mem[N - 1] = alloc3[3]
# Cdot_mat_alloc[N - 1] = alloc3[4]

# alloc4 = @btimed begin 
# Cdot_ten = C_dot_itensor_im(core_ten, factors_ten, total_H, sites)
# end
# Cdot_ten_time[N - 1] = alloc4[2]
# Cdot_ten_mem[N - 1] = alloc4[3]
# Cdot_ten_alloc[N - 1] = alloc4[4]

    # if N < 11
# H_mat = xxx(N, J, g)
# H_ten = ITensor(H_mat, sites, sites')
# alloc5 = @btimed begin
# inter_C = $H_ten*reconstruct(core_ten,factors_ten)
# Cdot = reconstruct(inter_C, conj_factors2(factors_ten))
# end
# Cdot_full_time[N - 1] = alloc5[2]
# Cdot_full_mem[N - 1] = alloc5[3]
# Cdot_full_alloc[N - 1] = alloc5[4]
    # end
    # println("N: $N")
    # println("Mat Alloc: ", alloc_1.memory + alloc3.memory)
    # println("Ten Alloc: ", alloc2.memory + alloc4.memory)
    # println("Mat Time: ", alloc_1.time + alloc3.time)
    # println("Ten Time: ", alloc2.time + alloc4.time)

# timing(3)

# alloc_1 = @belapsed begin 
# Kdot1_arr,_, _,_ = K_evolution_mat(core_arr, factors_arr, 1, xxx_ops_list)
# end 

# alloc2 = @belapsed begin 
# Kdot1_ten,_, _,_ = K_evolution_itensor(core_ten, factors_ten, 1, total_H, sites)
# end

# # @btime begin 
# # 
# # end 

# alloc3 = @belapsed begin
# Cdot_arr = C_dot_im_mat(core_arr, factors_arr, xxx_ops_list)
# end

# alloc4 = @belapsed begin 
# Cdot_ten = C_dot_itensor_im(core_ten, factors_ten, total_H, sites)
# end
#Display Kdot 
# println("Kdot ITensor: ")
# display(Array(Kdot1_ten, inds(Kdot1_ten)))
# println("Kdot Mat: ")
# display(Kdot1_arr)

# println("Norm difference Kdot: ", norm(Array(Kdot1_ten, inds(Kdot1_ten)) - Kdot1_arr))
# println("Norm difference Cdot: ", norm(Array(Cdot_ten, inds(Cdot_ten)) - Cdot_arr))
# # ops1 = [sx, Ident, Ident]
# ops2 = [Ident, sx, Ident]
# ops1_itensor = op_itensor_list(ops1, sites)
# ops2_itensor = op_itensor_list(ops2, sites)

# total_H = [ops1_itensor, ops2_itensor]
# ops_list = [ops1, ops2]

# Kdot = K_evolution_mat(core, factors, 1, K1_mat, ops_list)
# # Kdot_ten = K_evolution_itensor(core, factors, 1, K1_mat, total_H, sites)

# #Compare result to code in bug_step
# xxx_ops_list = xxx_ops(N, 1, 1)
# total_H_xxx = total_H_itensor(xxx_ops_list, sites)

# H_mat = xxx(N, 1, 1)
# H_ten = ITensor(H_mat, sites, sites')
# @btime begin
# core_f,factors_f = bug_step_itensor(H_ten, core, factors, 0.01, sites)
# end

# core_f, factors_f = bug_step_itensor(H_ten, core, factors, 0.01, sites)

# Kdot2 = K_evolution_itensor(core, factors, 1, K2, total_H_xxx, sites)

# @btime begin 
# Kdot1 = K_evolution_itensor(core, factors, 1, K1, total_H_xxx, sites)
# Kdot2 = K_evolution_itensor(core, factors, 2, K2, total_H_xxx, sites)
# Kdot3 = K_evolution_itensor(core, factors, 3, K3, total_H_xxx, sites)
# end

#Now test the Cdot function 
#First create the "inefficient"? way to do it 
# @btime begin 
# inter_C = H_ten*reconstruct(core,factors)
# Cdot = reconstruct(inter_C, conj_factors2(factors))
# end
#Now test the "efficient"? way 
# @btime begin
# @btime begin
# Cdot_eff= C_dot_itensor_eff(core, factors, total_H_xxx, xxx_ops_list,sites)
# Cdot_eff = C_dot_itensor(core, factors, total_H_xxx, sites)
# end
# println("reconstruct_time: ", reconstruct_time)
# println("factor_multiple time: ", mult_time)
# println("factor_multiple time2: ", mult_time2)
# end
# println("Norm difference: ", norm(Cdot - Cdot_eff))

# println("'Slow' Runtime")
# @btime begin  
# core_f, factors_f = bug_step_itensor(H_ten, core, factors, 0.01, sites)
# end 
 
# println("'Fast' Runtime")
# @btime begin
# core_eff, factors_eff = bug_step_eff(total_H_xxx, core, factors, 0.01, sites)
# end

# println("End")
# for i in 1:N 
#     println("Norm Difference Factors: ", norm(Array(factors_f[i], inds(factors_f[i])) - Array(factors_eff[i], inds(factors_eff[i]))))
# end
# println("Norm Difference Core: ", norm(Array(core_f, inds(core_f)) - Array(core_eff, inds(core_eff))))

# q_state = [0,0,1]
# init_core, init_factors = tucker_separable(sites, q_state)
# init_mps = init_separable(sites, q_state)
# init_vec = ComplexF64.(vectorize_mps(init_mps, order = "reverse"))
# t0 = 0.0
# T = 1.0
# steps = 100
# h = (T - t0)/steps
# init_update = exp(-im*H_mat*(T - t0))*init_vec
# final_core, final_factors, _ = bug_integrator_itensor_ra(H_ten, init_core, init_factors, t0, T, steps, sites, 0.0)

# println("Norm difference: ", norm(init_update - vec_itensor(reconstruct(final_core, final_factors); order = "reverse")))

