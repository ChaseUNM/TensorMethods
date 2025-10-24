using LinearAlgebra, ITensors 

include("hamiltonian(8-4).jl")
include("Hamiltonian.jl")
include("Tucker_Matrices.jl")
include("BUG_tucker(8-27).jl")
include("BUG_small.jl")

function ladder(n)
    A = zeros(n, n)
    for i = 1:n-1  
            A[i,i + 1] = sqrt(i)
    end
    return ComplexF64.(A)
end

sz = [1 0; 0 -1]
sx = [0 1; 1 0]
function quandary_h2(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz)
    op_list = []
    #Create the single site operators
    for i in 1:N - 1
        a = ladder(N_levels[i]) 
        ops = [ComplexF64.(Ident) for _ in 1:N]
        ops[i] = (transition_freq[i] - rot_freq[i])a'*a - 0.5*self_kerr[i]*a'*a'*a*a
        push!(op_list, ops)
    end
    #Create the two-site operators
    for i in 1:N - 1
        for j in 1:N - 1
        end
    end
end


function quandary_h(N, N_levels, transition_freq, self_kerr, dipole, zz)

    H_single_site = Vector{Tuple{AbstractMatrix, Int64}}()
    H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    # H_dip = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    # H_single_site = Vector{Tuple{AbstractMatrix, Int64}}(undef, max(sum(iszero.(transition_freq)), sum(iszero.(self_kerr))))
    # H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, sum(iszero.(dipole)) + sum(iszero.(zz)))
    # H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    for i in 1:N
        a = ladder(N_levels[i]) 
        tup = (transition_freq[i]*a'*a - 0.5*self_kerr[i]*a'*a'*a*a, i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        a_i = ladder(N_levels[i])
        for j = i + 1: N 
            a_j = ladder(N_levels[j])
            tup1 = (dipole[i,j]*a_i', i, a_j, j)
            tup2 = (dipole[i,j]*a_i, i, a_j', j)
            # tup_dip = (dipole[i,j]*(a_i + a_i'), i, a_j + a_j', j)
            tup_zz = (-zz[i,j]*a_i'*a_i, i, a_j'*a_j, j)
            push!(H_dipole_1, tup1)
            push!(H_dipole_2, tup2)
            push!(H_zz, tup_zz)
            # push!(H_dip, tup_dip)
        end
    end

    # for i in 1:N - 1
    #     a_i = ladder(N_levels[i])
    #     for j = i + 1:N 
    #         a_j = ladders(N_levels[j])
    #         tup = (zz[i][j]*a_i'*a_i, i, a_j'*a_j)
    #         push!(H_zz, tup)
    #     end
    # end
    H_total = vcat(H_single_site, H_dipole_1, H_dipole_2, H_zz)
    return H_total 
end

function ops_xxx_new(N, J, g)
    H_single_site = Vector{Tuple{AbstractMatrix, Int64}}()
    H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    for i in 1:N 
        tup = (-g*J*sx, i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        tup = (-J*sz, i, sz, i + 1)
        push!(H_two_site, tup)
    end
    return vcat(H_single_site, H_two_site)
end
        

function quandary_h(N, N_levels, sites, transition_freq, self_kerr, dipole, zz)

    H_single_site = Vector{Tuple{AbstractMatrix, Int64}}()
    H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    # H_single_site = Vector{Tuple{AbstractMatrix, Int64}}(undef, max(sum(iszero.(transition_freq)), sum(iszero.(self_kerr))))
    # H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, sum(iszero.(dipole)) + sum(iszero.(zz)))
    # H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    for i in 1:N
        a = ladder(N_levels[i])
        # display(transition_freq[i]*a'*a - 0.5*self_kerr[i]*a'*a'*a*a)
        tup = (ITensor(transition_freq[i]*a'*a - 0.5*self_kerr[i]*a'*a'*a*a, sites[i], sites[i]'), i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        a_i = ladder(N_levels[i])
        for j = i + 1: N 
            a_j = ladder(N_levels[j])
            tup1 = (dipole[i,j]*ITensor(a_i', sites[i], sites[i]'), i, ITensor(a_j,sites[j], sites[j]'), j)
            tup2 = (dipole[i,j]*ITensor(a_i, sites[i], sites[i]'), i, ITensor(a_j', sites[j], sites[j]'), j)
            tup_zz = (-zz[i,j]*ITensor(a_i'*a_i, sites[i], sites[i]'), i, ITensor(a_j'*a_j, sites[j], sites[j]'), j)
            push!(H_dipole_1, tup1)
            push!(H_dipole_2, tup2)
            push!(H_zz, tup_zz)
        end
    end

    # for i in 1:N - 1
    #     a_i = ladder(N_levels[i])
    #     for j = i + 1:N 
    #         a_j = ladders(N_levels[j])
    #         tup = (zz[i][j]*a_i'*a_i, i, a_j'*a_j)
    #         push!(H_zz, tup)
    #     end
    # end
    H_total = vcat(H_single_site, H_dipole_1, H_dipole_2, H_zz)
    return H_total 
end

#Assume rotation is an average of transition frequencies

function H_sys_rot(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz)
    H_single_site = Vector{Tuple{AbstractMatrix, Int64}}()
    H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    # H_single_site = Vector{Tuple{AbstractMatrix, Int64}}(undef, max(sum(iszero.(transition_freq)), sum(iszero.(self_kerr))))
    # H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, sum(iszero.(dipole)) + sum(iszero.(zz)))
    # H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    for i in 1:N
        a = ladder(N_levels[i]) 
        tup = ((transition_freq[i] - rot_freq[i])a'*a - 0.5*self_kerr[i]*a'*a'*a*a, i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        a_i = ladder(N_levels[i])
        for j = i + 1: N 
            a_j = ladder(N_levels[j])
            tup1 = (dipole[i,j]*a_i', i, a_j, j)
            tup2 = (dipole[i,j]*a_i, i, a_j', j)
            tup_zz = (-zz[i,j]*a_i'*a_i, i, a_j'*a_j, j)
            push!(H_dipole_1, tup1)
            push!(H_dipole_2, tup2)
            push!(H_zz, tup_zz)
        end
    end

    # for i in 1:N - 1
    #     a_i = ladder(N_levels[i])
    #     for j = i + 1:N 
    #         a_j = ladders(N_levels[j])
    #         tup = (zz[i][j]*a_i'*a_i, i, a_j'*a_j)
    #         push!(H_zz, tup)
    #     end
    # end
    H_total = vcat(H_single_site, H_dipole_1, H_dipole_2, H_zz)
    return H_total 
end

function H_sys_rot_ten(N, N_levels, sites, transition_freq, rot_freq, self_kerr, dipole, zz)
    H_single_site = Vector{Tuple{ITensor, Int64}}()
    H_dipole_1 = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    H_dipole_2 = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    H_zz = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    # H_single_site = Vector{Tuple{AbstractMatrix, Int64}}(undef, max(sum(iszero.(transition_freq)), sum(iszero.(self_kerr))))
    # H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, sum(iszero.(dipole)) + sum(iszero.(zz)))
    # H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    for i in 1:N
        a = ladder(N_levels[i]) 
        tup = (ITensor((transition_freq[i] - rot_freq[i])a'*a - 0.5*self_kerr[i]*a'*a'*a*a, sites[i], sites[i]'), i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        a_i = ladder(N_levels[i])
        for j = i + 1: N 
            a_j = ladder(N_levels[j])
            tup1 = (ITensor(dipole[i,j]*a_i', sites[i], sites[i]'), i, ITensor(a_j, sites[j], sites[j]'), j)
            tup2 = (ITensor(dipole[i,j]*a_i, sites[i], sites[i]'), i, ITensor(a_j', sites[j], sites[j]'), j)
            tup_zz = (ITensor(zz[i,j]*a_i'*a_i, sites[i], sites[i]'), i, ITensor(a_j'*a_j, sites[j], sites[j]'), j)
            push!(H_dipole_1, tup1)
            push!(H_dipole_2, tup2)
            push!(H_zz, tup_zz)
        end
    end

    # for i in 1:N - 1
    #     a_i = ladder(N_levels[i])
    #     for j = i + 1:N 
    #         a_j = ladders(N_levels[j])
    #         tup = (zz[i][j]*a_i'*a_i, i, a_j'*a_j)
    #         push!(H_zz, tup)
    #     end
    # end
    H_total = vcat(H_single_site, H_dipole_1, H_dipole_2, H_zz)
    return H_total 
end

function updateH(H_s, bcparams, t)
    N_ops = length(H_s)
    H_s_copy = deepcopy(H_s)
    for i in 1:N_ops 
        if length(H_s_copy[i]) == 2
            a = ladder(size(H_s_copy[i][1], 1))
            H_s_copy[i][1] .+= bcarrier2(t, bcparams, 2*(i - 1))*(a + a') + im*bcarrier2(t, bcparams, (2*(i - 1) + 1))*(a - a')
        end
    end
    return H_s_copy
end

function updateH_ten(H_s, bcparams, t)
    N_ops = length(H_s)
    H_s_copy = deepcopy(H_s)
    for i in 1:N_ops 
        if length(H_s_copy[i]) == 2
            a = ladder(size(H_s_copy[i][1], 1))
            # println(inds(H_s_copy[i][1]))
            H_s_copy[i][1] .+= ITensor(bcarrier2(t, bcparams, 2*(i - 1))*(a + a') + im*bcarrier2(t, bcparams, (2*(i - 1) + 1))*(a - a'), inds(H_s_copy[i][1])[1], inds(H_s_copy[i][1])[2])
        end
    end
    return H_s_copy 
end

function quandary_h_rot(N, N_levels, sites, transition_freq, rot_freq, self_kerr, dipole, zz)

    H_single_site = Vector{Tuple{ITensor, Int64}}()
    H_dipole_1 = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    H_dipole_2 = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    H_zz = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    # H_single_site = Vector{Tuple{AbstractMatrix, Int64}}(undef, max(sum(iszero.(transition_freq)), sum(iszero.(self_kerr))))
    # H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, sum(iszero.(dipole)) + sum(iszero.(zz)))
    # H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    for i in 1:N
        a = ladder(N_levels[i]) 
        tup = ((transition_freq[i] - rot_freq[i])a'*a - 0.5*self_kerr[i]*a'*a'*a*a, i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        a_i = ladder(N_levels[i])
        for j = i + 1: N 
            a_j = ladder(N_levels[j])
            tup1 = (dipole[i,j]*a_i', i, a_j, j)
            tup2 = (dipole[i,j]*a_i, i, a_j', j)
            tup_zz = (zz[i,j]*a_i'*a_i, i, a_j'*a_j, j)
            push!(H_dipole_1, tup1)
            push!(H_dipole_2, tup2)
            push!(H_zz, tup_zz)
        end
    end

    # for i in 1:N - 1
    #     a_i = ladder(N_levels[i])
    #     for j = i + 1:N 
    #         a_j = ladders(N_levels[j])
    #         tup = (zz[i][j]*a_i'*a_i, i, a_j'*a_j)
    #         push!(H_zz, tup)
    #     end
    # end
    H_total = vcat(H_single_site, H_dipole_1, H_dipole_2, H_zz)
    return H_total 
end

function control_h_rot(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz, control_p, control_q, t)
    H_single_site = Vector{Tuple{ITensor, Int64}}()
    H_dipole_1 = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    H_dipole_2 = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    H_zz = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    # H_single_site = Vector{Tuple{AbstractMatrix, Int64}}(undef, max(sum(iszero.(transition_freq)), sum(iszero.(self_kerr))))
    # H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, sum(iszero.(dipole)) + sum(iszero.(zz)))
    # H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    for i in 1:N
        a = ladder(N_levels[i]) 
        tup = ((transition_freq[i] - rot_freq[i])a'*a - 0.5*self_kerr[i]*a'*a'*a*a + control_p[i](t)*(a + a')im*control_q[i](t)*(a - a'), i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        a_i = ladder(N_levels[i])
        for j = i + 1: N 
            a_j = ladder(N_levels[j])
            tup1 = (dipole[i,j]*a_i', i, a_j, j)
            tup2 = (dipole[i,j]*a_i, i, a_j', j)
            tup_zz = (zz[i,j]*a_i'*a_i, i, a_j'*a_j, j)
            push!(H_dipole_1, tup1)
            push!(H_dipole_2, tup2)
            push!(H_zz, tup_zz)
        end
    end

    # for i in 1:N - 1
    #     a_i = ladder(N_levels[i])
    #     for j = i + 1:N 
    #         a_j = ladders(N_levels[j])
    #         tup = (zz[i][j]*a_i'*a_i, i, a_j'*a_j)
    #         push!(H_zz, tup)
    #     end
    # end
    H_total = vcat(H_single_site, H_dipole_1, H_dipole_2, H_zz)
    return H_total 
end

function control_h_rot(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz, bcparams, t)
    H_single_site = Vector{Tuple{AbstractMatrix, Int64}}()
    H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}()
    # H_single_site = Vector{Tuple{AbstractMatrix, Int64}}(undef, max(sum(iszero.(transition_freq)), sum(iszero.(self_kerr))))
    # H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, sum(iszero.(dipole)) + sum(iszero.(zz)))
    # H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    for i in 1:N
        a = ladder(N_levels[i]) 
        tup = ((transition_freq[i] - rot_freq[i])a'*a - 0.5*self_kerr[i]*a'*a'*a*a + bcarrier2(t, bcparams, 2*(i - 1))*(a + a') + im*bcarrier2(t, bcparams, (2*(i - 1) + 1))*(a - a'), i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        a_i = ladder(N_levels[i])
        for j = i + 1: N 
            a_j = ladder(N_levels[j])
            tup1 = (dipole[i,j]*a_i', i, a_j, j)
            tup2 = (dipole[i,j]*a_i, i, a_j', j)
            tup_zz = (zz[i,j]*a_i'*a_i, i, a_j'*a_j, j)
            push!(H_dipole_1, tup1)
            push!(H_dipole_2, tup2)
            push!(H_zz, tup_zz)
        end
    end

    # for i in 1:N - 1
    #     a_i = ladder(N_levels[i])
    #     for j = i + 1:N 
    #         a_j = ladders(N_levels[j])
    #         tup = (zz[i][j]*a_i'*a_i, i, a_j'*a_j)
    #         push!(H_zz, tup)
    #     end
    # end
    H_total = vcat(H_single_site, H_dipole_1, H_dipole_2, H_zz)
    return H_total 
end

function quandary_h(N, N_levels, sites, transition_freq, self_kerr, dipole, zz)
    H_single_site = Vector{Tuple{ITensor, Int64}}()
    H_dipole_1 = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    H_dipole_2 = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    H_zz = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    # H_single_site = Vector{Tuple{AbstractMatrix, Int64}}(undef, max(sum(iszero.(transition_freq)), sum(iszero.(self_kerr))))
    # H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, sum(iszero.(dipole)) + sum(iszero.(zz)))
    # H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    for i in 1:N
        a = ladder(N_levels[i]) 
        tup = (ITensor(transition_freq[i]*a'*a - 0.5*self_kerr[i]*a'*a'*a*a, sites[i], sites[i]'), i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        a_i = ladder(N_levels[i])
        for j = i + 1: N 
            a_j = ladder(N_levels[j])
            tup1 = (dipole[i,j]*ITensor(a_i', sites[i], sites[i]'), i, ITensor(a_j,sites[j], sites[j]'), j)
            tup2 = (dipole[i,j]*ITensor(a_i, sites[i], sites[i]'), i, ITensor(a_j', sites[j], sites[j]'), j)
            tup_zz = (-zz[i,j]*ITensor(a_i'*a_i, sites[i], sites[i]'), i, ITensor(a_j'*a_j, sites[j], sites[j]'), j)
            push!(H_dipole_1, tup1)
            push!(H_dipole_2, tup2)
            push!(H_zz, tup_zz)
        end
    end

    # for i in 1:N - 1
    #     a_i = ladder(N_levels[i])
    #     for j = i + 1:N 
    #         a_j = ladders(N_levels[j])
    #         tup = (zz[i][j]*a_i'*a_i, i, a_j'*a_j)
    #         push!(H_zz, tup)
    #     end
    # end
    H_total = vcat(H_single_site, H_dipole_1, H_dipole_2, H_zz)
    return H_total 
end

function quandary_h_rot(N, N_levels, sites, transition_freq, self_kerr, dipole, zz)

    H_single_site = Vector{Tuple{ITensor, Int64}}()
    H_dipole_1 = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    H_dipole_2 = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    H_zz = Vector{Tuple{ITensor, Int64, ITensor, Int64}}()
    # H_single_site = Vector{Tuple{AbstractMatrix, Int64}}(undef, max(sum(iszero.(transition_freq)), sum(iszero.(self_kerr))))
    # H_two_site = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, sum(iszero.(dipole)) + sum(iszero.(zz)))
    # H_dipole_1 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_dipole_2 = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    # H_zz = Vector{Tuple{AbstractMatrix, Int64, AbstractMatrix, Int64}}(undef, Int64(N/2*(N - 1)))
    for i in 1:N
        tup = ((transition_freq[i] - rot_freq[i])*ITensor(a'*a - 0.5*self_kerr[i]*a'*a'*a*a, sites[i]', sites[i]), i)
        push!(H_single_site, tup)
    end
    for i in 1:N - 1
        a_i = ladder(N_levels[i])
        for j = i + 1: N 
            a_j = ladder(N_levels[j])
            tup1 = (dipole[i,j]*ITensor(a_i', sites[i], sites[i]'), i, ITensor(a_j,sites[j]', sites[j]), j)
            tup2 = (dipole[i,j]*ITensor(a_i, sites[i], sites[i]'), i, ITensor(a_j', sites[j]', sites[j]), j)
            tup_zz = (zz[i,j]*ITensor(a_i'*a_i, sites[i], sites[i]'), i, ITensor(a_j'*a_j, sites[j]', sites[j]), j)
            push!(H_dipole_1, tup1)
            push!(H_dipole_2, tup2)
            push!(H_zz, tup_zz)
        end
    end

    # for i in 1:N - 1
    #     a_i = ladder(N_levels[i])
    #     for j = i + 1:N 
    #         a_j = ladders(N_levels[j])
    #         tup = (zz[i][j]*a_i'*a_i, i, a_j'*a_j)
    #         push!(H_zz, tup)
    #     end
    # end
    H_total = vcat(H_single_site, H_dipole_1, H_dipole_2, H_zz)
    return H_total 
end

function qudit_siteinds(N::Int64, N_levels::Vector{Int64})
    ind = Vector{Index{Int64}}(undef, N)
    for i in 1:N 
        ind[i] = Index(N_levels[i]; tags = "Qudit, Site, n = $i")
    end
    return ind 
end

function applyH_mat(op, factors)
    # core_inds = inds(core)
    # not_site = setdiff(core_inds, [core_inds[site]])
    # Q, R = qr(core, not_site)
    # Preallocate new_factors with known sizes for efficiency
    # new_factors = [zeros(eltype(factors[i]), size(factors[i], 2), size(factors[i], 2)) for i in 1:length(factors)]
    new_factors = copy(factors)
    # Adjust size for the selected site, since op_list[site]*factors[site] changes shape
    # new_factors[site] = zeros(eltype(factors[site]), size(op_list[site], 1), size(factors[site], 2))
    # temp_factors = [zeros(eltype(factors[i]), size(factors[i], 1), size(factors[i], 2)) for i in 1:length(factors)]
    N = length(factors)
    if length(op) == 2
        new_factors[op[2]] = op[1]*new_factors[op[2]]
    elseif length(op) == 4 
        new_factors[op[2]] = op[1]*new_factors[op[2]]
        new_factors[op[4]] = op[2]*new_factors[op[4]]
    end
    return new_factors 
end

function applyH_mat(op, factors)
    # core_inds = inds(core)
    # not_site = setdiff(core_inds, [core_inds[site]])
    # Q, R = qr(core, not_site)
    # Preallocate new_factors with known sizes for efficiency
    # new_factors = [zeros(eltype(factors[i]), size(factors[i], 2), size(factors[i], 2)) for i in 1:length(factors)]
    new_factors = copy(factors)
    # Adjust size for the selected site, since op_list[site]*factors[site] changes shape
    # new_factors[site] = zeros(eltype(factors[site]), size(op_list[site], 1), size(factors[site], 2))
    # temp_factors = [zeros(eltype(factors[i]), size(factors[i], 1), size(factors[i], 2)) for i in 1:length(factors)]
    if length(op) == 2
        new_factors[op[2]] = op[1]*new_factors[op[2]]
    elseif length(op) == 4 
        new_factors[op[2]] = op[1]*new_factors[op[2]]
        new_factors[op[4]] = op[3]*new_factors[op[4]]
    end
    return new_factors 
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

function applyHV_tucker(op, factors, site, site_indices)
    N = length(factors)
    new_factors = [ITensor(ComplexF64.(Matrix(1.0*I, size(factors[i], 2), size(factors[i], 2))), inds(factors[i])[2], inds(factors[i])[2]') for i = 1:N]
    
    if length(op) == 2
        op_site = op[2]
        if op_site == site 
            new_factors[op[2]] = op[1]*prime(factors[op[2]], "Site")
        elseif op_site != site
            new_factors[op[2]] = noprime(conj(op[1]*prime(factors[op[2]], "Site"))', "Site")*factors[op[2]]
            new_factors[site] = factors[site]
        end
    # new_factors[op[4]] = op[3]*new_factors[op[4]]
    elseif length(op) == 4
        op_site1 = op[2]
        op_site2 = op[4]
        if op_site1 == site
            new_factors[op[2]] = op[1]*prime(factors[op[2]], "Site")
            new_factors[op[4]] = noprime(conj(op[3]*prime(factors[op[4]], "Site"))', "Site")*factors[op[4]]
        elseif op_site2 == site 
            new_factors[op[2]] = noprime(conj(op[1]*prime(factors[op[2]], "Site"))', "Site")*factors[op[2]]
            new_factors[op[4]] = op[3]*prime(factors[op[4]], "Site")
        else 
            new_factors[op[2]] = noprime(conj(op[1]*prime(factors[op[2]], "Site"))', "Site")*factors[op[2]]
            new_factors[op[4]] = noprime(conj(op[3]*prime(factors[op[4]], "Site"))', "Site")*factors[op[4]]
            new_factors[site] = factors[site]
        end
    end
    return new_factors 
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

function energy_tucker(total_H, core::ITensor, factors)
    energy = 0.0
    vec_core = vec_itensor(core)
    N_ops = length(total_H)
    for i = 1:N_ops 
        new_factors = [ITensor(ComplexF64.(Matrix(1.0*I, size(factors[i], 2), size(factors[i], 2))), inds(factors[i])[2], inds(factors[i])[2]') for i in 1:length(factors)]
        if length(total_H[i]) == 2 
            new_factors[total_H[i][2]] = conj(factors[total_H[i][2]]')*total_H[i][1]*factors[total_H[i][2]]
        elseif length(total_H[i]) == 4
            new_factors[total_H[i][2]] = conj(factors[total_H[i][2]]')*total_H[i][1]*factors[total_H[i][2]]
            new_factors[total_H[i][4]] = conj(factors[total_H[i][4]]')*total_H[i][3]*factors[total_H[i][4]]
        end 
        energy += vec_core'*vec_itensor(reconstruct(core, new_factors))
    end
    return real(energy)
end

function inner(core1, factors1, core2, factors2)
    N = length(factors1)
    vec_core = vec(core1)
    new_factors = [zeros(ComplexF64, size(factors1[i], 2), size(factors2[i], 2)) for i in 1:N]
    for i in 1:N 
        new_factors[i] = factors1[i]'*factors2[i]
        println(size(new_factors[i]))
    end
    println("Size: ", size(Multi_TTM_recursive(core2, new_factors)))
    return vec_core'*vec(Multi_TTM_recursive(core2, new_factors))
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
# om[1,:]= [0.17817603064445744, -0.06158197214672839, -0.11659405849772948, -0.23190739860436582, 0.12279645755956273, -0.35582688473631774].*2*pi
# om[2,:] = [0.17817603064445744, -0.06158197214672839, -0.11659405849772948, -0.23190739860436582, 0.12279645755956273, -0.35582688473631774].*2*pi
# om[3,:] = [0.17817603064445744, -0.06158197214672839, -0.11659405849772948, -0.23190739860436582, 0.12279645755956273, -0.35582688473631774].*2*pi
# t0 = 0.0
# T = 300.0
# nsplines = 92
# pars = vec(readdlm("params_N3.dat"))
# bc_params = bcparams((T - t0),nsplines, om, pars)
N = 9
N_levels = fill(4, N)
sites = qudit_siteinds(N, N_levels)
transition_freq = rand(N)
rot_freq = zeros(N)
self_kerr = rand(N)
dipole = rand(N,N)
zz = rand(N,N)
# dipole = [0.0 2.0 1.0; 0.0 0.0 4.0; 0.0 0.0 0.0]
# zz = [0.0 0.4 0.5; 0.0 0.0 0.6; 0.0 0.0 0.0]
H_ops = H_sys_rot(N, N_levels, transition_freq, rot_freq,  self_kerr, dipole, zz)

# H_ops_ten = H_sys_rot_ten(N, N_levels, sites, transition_freq, rot_freq, self_kerr, dipole, zz)
# H_mat = H_sys(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz)

A = rand(ComplexF64, N_levels...)
A = A/norm(A)
B = rand(ComplexF64, N_levels...)
B = B/norm(B)
A_vec = vec(A)
B_vec = vec(B)
core, factors = tucker(A; cutoff = 0.0)
core2, factors2 = tucker(B; cutoff = 0.0)
println("Inner 1: ", A_vec'*B_vec)
println("Inner 2: ", inner(core, factors, core2, factors2))
# core_trunc, factors_trunc = truncate_tucker(core, factors; cutoff = 0.1)
# println("Old Size: ", size(core))
# println("New Size: ", size(core_trunc))
# println("Truncation Error: ", norm(Multi_TTM_recursive(core, factors) - Multi_TTM_recursive(core_trunc, factors_trunc)))
# core_ten, factors_ten = tucker_itensor(A, sites; cutoff = 0.0)

# h = 0.01
# H_ops2 = updateH(H_ops, bc_params, 10.0)
# H_ops2_ten = updateH_ten(H_ops_ten, bc_params, 10.0)
# for i in 1:length(H_ops2)
#     if length(H_ops2[i]) == 2
#         println("Diff: ", norm(H_ops2[i][1] - Array(H_ops2_ten[i][1], inds(H_ops2_ten[i][1]))))
#     elseif length(H_ops2[i]) == 4
#         println("Diff: ", norm(H_ops2[i][1] - Array(H_ops2_ten[i][1], inds(H_ops2_ten[i][1]))) + norm(H_ops2[i][3] - Array(H_ops2_ten[i][3], inds(H_ops2_ten[i][3]))))
#     end
# end
# @btime begin 
# @time begin 
# core_u, factors_u = bug_step_mat(H_ops, core, factors, h)
# end
# println("end")
# end 
# @btime begin 
# core_u_t, factors_u_t = bug_step_eff(H_ops2_ten, core_ten, factors_ten, h, sites)
# end
# println("Core err: ", norm(core_u - Array(core_u_t, inds(core_u_t))))
# for i in 1:N 
#     println("Factors $i err: ", norm(factors_u[i] - Array(factors_u_t[i], inds(factors_u_t[i]))))
# end

# for i in 1:N
#     Kdot_ten, _ = K_evolution_itensor2(core_ten, factors_ten, i, H_ops_ten, sites)
#     Kdot, _ = K_evolution_mat2(core, factors, i, H_ops)
#     println("Kdot difference $i: ", norm(Kdot - Array(Kdot_ten, inds(Kdot_ten))))
# end

# energy_mat = energy_tucker(H_ops, core, factors)
# energy_ten = energy_tucker(H_ops_ten, core_ten, factors_ten)
# println("Energy Difference: ", norm(energy_mat - energy_ten))

# for i in 1:N 
#     println("Norm difference: ", norm(Array(new_factors_ten[i], inds(new_factors_ten[i])) - new_factors[i]))
# end

# for k in 1:N
#     for j in 1:length(H_ops)
#         new_factors_ten = applyHV_tucker(H_ops_ten[j], factors_ten, k, sites)
#         new_factors = applyHV(H_ops[j], factors, k)
#         for i in 1:N 
#             println("Norm difference (op:$j, site:$k, factor:$i): ", norm(Array(new_factors_ten[i], inds(new_factors_ten[i])) - new_factors[i]))
#         end
#     end
# end
# h = 0.01
# Cdot = C_dot_test2(core, factors, H_ops)
# Cdot_tucker = C_dot_test2_tucker(core_ten, factors_ten, H_ops_ten)
# println("Cdot Difference: ", norm(Cdot - Array(Cdot_tucker, inds(Cdot_tucker))))

# #IMR with core 
# C_update = IMR_core_mat(H_ops, core, h, factors, 100, 1E-14, false)
# C_update_ten = IMR_core_ten(H_ops_ten, core_ten, h, factors_ten, 100, 1E-14, false)
# println("Iterative Method difference: ", norm(C_update - Array(C_update_ten, inds(C_update_ten))))

# H_s = H_sys_rot(N, N_levels, transition_freq, rot_freq, self_kerr, dipole, zz)
# H_s_ten = H_sys_rot_ten(N, N_levels, sites, transition_freq, rot_freq, self_kerr, dipole, zz)


# om[1,:]= [0.17817603064445744, -0.06158197214672839, -0.11659405849772948, -0.23190739860436582, 0.12279645755956273, -0.35582688473631774].*2*pi
# om[2,:] = [0.17817603064445744, -0.06158197214672839, -0.11659405849772948, -0.23190739860436582, 0.12279645755956273, -0.35582688473631774].*2*pi
# om[3,:] = [0.17817603064445744, -0.06158197214672839, -0.11659405849772948, -0.23190739860436582, 0.12279645755956273, -0.35582688473631774].*2*pi
# t0 = 0.0
# T = 300.0
# nsplines = 92
# pars = vec(readdlm("params_N3.dat"))
# bc_params = bcparams((T - t0),nsplines, om, pars)

# H_update = updateH(H_s, bc_params, 10.0)
# H_update_ten = updateH_ten(H_s_ten, bc_params, 10.0)

# for i in 1:length(H_s)
#     if length(H_update[i]) == 2
#         println("Diff: ", norm(H_update[i][1] - Array(H_update_ten[i][1], inds(H_update_ten[i][1]))))
#     elseif length(H_s[i]) == 4
#         println("Diff: ", norm(H_update[i][1] - Array(H_update_ten[i][1], inds(H_update_ten[i][1]))) + norm(H_update[i][3] - Array(H_update_ten[i][3], inds(H_update_ten[i][3]))))
#     end
# end
# println("Mat_vec: ")
# display(H_mat*A_vec)
# new_factors = applyH_mat(H_ops[1], factors)
# A_re = Multi_TTM_recursive(core, new_factors)
# for i in 2:length(H_ops)
#     new_factors = applyH_mat(H_ops[i], factors)
#     global A_re += Multi_TTM_recursive(core, new_factors)
# end
# new_factors_ten = applyH_mat(H_ops_ten[1], factors_ten)
# A_re_ten = noprime(reconstruct(core_ten, new_factors_ten))
# display(A_re_ten)
# for i in 2:length(H_ops)
#     # println("OP $i")
#     new_factors_ten = applyH_mat(H_ops_ten[i], factors_ten)
#     # display(reconstruct(core_ten, new_factors_ten))
#     global A_re_ten += noprime(reconstruct(core_ten, new_factors_ten))
# end


# println("Tucker: ")
# display(vec(A_re))
# println("Norm difference: ", norm(H_mat*A_vec - vec(A_re)))
# display("Tucker ITensor")
# display(vec_itensor(A_re_ten))
# println("Norm difference: ", norm(H_mat*A_vec - vec_itensor(A_re_ten)))

# J = 1.0
# g = 1.0
# xxx_ops1 = xxx_ops(N, J, g)
# xxx_ops2 = ops_xxx_new(N, J, g)

# @btime begin 
# new_factors = applyH_mat(xxx_ops1[2], factors, 3)
# end 
# @btime begin 
# new_factors2 = applyHV(xxx_ops2[5], factors, 3)
# end
# println("Check factors are correct: ")
# for i in 1:N 
#     println("Factor $i Difference: ", norm(new_factors[i] - new_factors2[i]))
# end