using LinearAlgebra
using ITensors, ITensorMPS
using SparseArrays

# include("vectorization(5-5).jl")



# Create different operators for use in ITensor opsum operation
ITensors.op(::OpName"Sx2", ::SiteType"Qubit") = [0 1; 1 0]
ITensors.op(::OpName"Sy2", ::SiteType"Qubit") = [0 -im; im 0]
ITensors.op(::OpName"Sz2", ::SiteType"Qubit") = [1 0; 0 -1]
ITensors.op(::OpName"S+2", ::SiteType"Qubit") = [0 2; 0 0]
ITensors.op(::OpName"a+a", ::SiteType"Qubit") = [0 0; 0 1]
ITensors.op(::OpName"aa+", ::SiteType"Qubit") = [1 0; 0 0]
ITensors.op(::OpName"a", ::SiteType"Qubit") = [0 1; 0 0]
ITensors.op(::OpName"a+", ::SiteType"Qubit") = [0 0; 1 0]
ITensors.op(::OpName"-Sy2", ::SiteType"Qubit") = [0 im; -im 0]

sx = [0 1; 1 0]
sy = [0 -im; im 0]
sz = [1 0; 0 -1]
splus = sx + im.*sy
sminus = sx - im.*sy
a = [0 0; 0 1]

# a = Array(Bidiagonal(zeros(d), sqrt.(collect(1: d - 1)), :U))
# ITensors.op(::OpName"a", ::SiteType"Qudit") = a
# ITensors.op(::OpName"adag", ::SiteType"Qudit") = a'
# ITensors.op(::OpName"a'a'", ::SiteType"Qudit") = a'*a'
# ITensors.op(::OpName"aa", ::SiteType"Qudit") = a*a
# ITensors.op(::OpName"a'a", ::SiteType"Qudit") = a'*a 
# ITensors.op(::OpName"a+a'", ::SiteType"Qudit") = a + a'
# ITensors.op(::OpName"a-a'", ::SiteType"Qudit") = a - a'


function annihilation_operator(N::Int)
    a = zeros(ComplexF64, N, N)
    for n in 2:N
        a[n-1, n] = sqrt(n - 1)
    end
    return a
end

struct bcparams
    T ::Float64
    D1::Int64 # number of B-spline coefficients per control function
    om::Array{Float64,2} #Carrier wave frequencies [rad/s], size Nfreq
    tcenter::Array{Float64,1}
    dtknot::Float64
    pcof::Array{Float64,1} # coefficients for all 2*Ncoupled splines, size Ncoupled*D1*Nfreq*2 (*2 because of sin/cos)
    Nfreq::Int64 # Number of frequencies
    Ncoeff:: Int64 # Total number of coefficients
    Ncoupled::Int64 # Number of B-splines functions for the coupled ctrl Hamiltonians
    Nunc::Int64 # Number of B-spline functions  for the UNcoupled ctrl Hamiltonians

    # New constructor to allow defining number of symmetric Hamiltonian terms
    function bcparams(T::Float64, D1::Int64, Ncoupled::Int64, Nunc::Int64, omega::Array{Float64,2}, pcof::Array{Float64,1})
        dtknot = T/(D1 -2)
        tcenter = dtknot.*(collect(1:D1) .- 1.5)
        Nfreq = size(omega,2)
        nCoeff = Nfreq*D1*2*(Ncoupled + Nunc)
        if nCoeff != length(pcof)
            println("nCoeff = ", nCoeff, " Nfreq = ", Nfreq, " D1 = ", D1, " Ncoupled = ", Ncoupled, " Nunc = ", Nunc, " len(pcof) = ", length(pcof))
            throw(DimensionMismatch("Inconsistent number of coefficients and size of parameter vector (nCoeff ≠ length(pcof)."))
        end
        new(T, D1, omega, tcenter, dtknot, pcof, Nfreq, nCoeff, Ncoupled, Nunc)
    end

end

# simplified constructor (assumes no uncoupled terms)
function bcparams(T::Float64, D1::Int64, omega::Array{Float64,2}, pcof::Array{Float64,1})
  dtknot = T/(D1 -2)
  tcenter = dtknot.*(collect(1:D1) .- 1.5)
  Ncoupled = size(omega,1) # should check that Ncoupled >=1
  Nfreq = size(omega,2)
  Nunc = 0
  nCoeff = Nfreq*D1*2*Ncoupled
  if nCoeff != length(pcof)
    throw(DimensionMismatch("Inconsistent number of coefficients and size of parameter vector (nCoeff ≠ length(pcof)."))
  end
  bcparams(T, D1, Ncoupled, Nunc, omega, pcof)
end

struct bcparams2
    T ::Float64
    D1::Int64 # number of B-spline coefficients per control function
    om::Vector{Vector{Float64}} #Carrier wave frequencies [rad/s], size Nfreq
    tcenter::Array{Float64,1}
    dtknot::Float64
    pcof::Array{Float64,1} # coefficients for all 2*Ncoupled splines, size Ncoupled*D1*Nfreq*2 (*2 because of sin/cos)
    Nfreq::Vector{Int64} # Number of frequencies
    Ncoeff:: Int64 # Total number of coefficients
    Ncoupled::Int64 # Number of B-splines functions for the coupled ctrl Hamiltonians
    Nunc::Int64 # Number of B-spline functions  for the UNcoupled ctrl Hamiltonians

    # New constructor to allow defining number of symmetric Hamiltonian terms
    function bcparams2(T::Float64, D1::Int64, Ncoupled::Int64, Nunc::Int64, omega::Vector{Vector{Float64}}, pcof::Array{Float64,1})
        dtknot = T/(D1 -2)
        tcenter = dtknot.*(collect(1:D1) .- 1.5)
        # Nfreq = size(omega,2)
        Nfreq = [length(omega[i]) for i in 1:length(omega)]
        # nCoeff = Nfreq*D1*2*(Ncoupled + Nunc)
        nCoeff = 2*sum([D1*Nfreq[i] for i in 1:length(Nfreq)])
        if nCoeff != length(pcof)
            println("nCoeff = ", nCoeff, " Nfreq = ", Nfreq, " D1 = ", D1, " Ncoupled = ", Ncoupled, " Nunc = ", Nunc, " len(pcof) = ", length(pcof))
            throw(DimensionMismatch("Inconsistent number of coefficients and size of parameter vector (nCoeff ≠ length(pcof)."))
        end
        new(T, D1, omega, tcenter, dtknot, pcof, Nfreq, nCoeff, Ncoupled, Nunc)
    end
end

#Simplified constructor (assumes no uncoupled terms)
function bcparams2(T::Float64, D1::Int64, omega::Vector{Vector{Float64}}, pcof::Array{Float64,1})
    dtknot = T/(D1 -2)
    tcenter = dtknot.*(collect(1:D1) .- 1.5)
    Ncoupled = size(omega,1) # should check that Ncoupled >=1
    Nfreq = [length(omega[i]) for i in 1:length(omega)]
    Nunc = 0
    nCoeff = 2*sum([D1*Nfreq[i] for i in 1:length(Nfreq)])
    # println(nCoeff)
    #nCoeff = Nfreq*D1*2*Ncoupled
    if nCoeff != length(pcof)
        throw(DimensionMismatch("Inconsistent number of coefficients and size of parameter vector (nCoeff ≠ length(pcof)."))
    end
    bcparams2(T, D1, Ncoupled, Nunc, omega, pcof)
end

"""
    f = bcarrier2(t, params, func)

Evaluate a B-spline function with carrier waves. See also the `bcparams` constructor.

# Arguments
- `t::Float64`: Evaluate spline at parameter t ∈ [0, param.T]
- `param::params`: Parameters for the spline
- `func::Int64`: Spline function index ∈ [0, param.Nseg-1]
"""
@inline function bcarrier2(t::Float64, params::bcparams, func::Int64)
    # for a single oscillator, func=0 corresponds to p(t) and func=1 to q(t)
    # in general, 0 <= func < 2*Ncoupled + Nunc

    # compute basic offset: func 0 and 1 use the same spline coefficients, but combined in a different way
    osc = div(func, 2) # osc is base 0; 0<= osc < Ncoupled
    q_func = func % 2 # q_func = 0 for p and q_func=1 for q
    
    f = 0.0 # initialize
    
    dtknot = params.dtknot
    width = 3*dtknot
    
    k = max.(3, ceil.(Int64,t./dtknot + 2)) # pick out the index of the last basis function corresponding to t
    k = min.(k, params.D1) #  Make sure we don't access outside the array
    
    if func < 2*(params.Ncoupled + params.Nunc)
        # Coupled and uncoupled controls
        @fastmath @inbounds @simd for freq in 1:params.Nfreq
            fbs1 = 0.0 # initialize
            fbs2 = 0.0 # initialize
            # offset in parameter array (osc = 0,1,2,...
            # Vary freq first, then osc
            offset1 = 2*osc*params.Nfreq*params.D1 + (freq-1)*2*params.D1
            offset2 = 2*osc*params.Nfreq*params.D1 + (freq-1)*2*params.D1 + params.D1

            # 1st segment of nurb k
            tc = params.tcenter[k]
            tau = (t .- tc)./width
            fbs1 += params.pcof[offset1+k] * (9/8 .+ 4.5*tau + 4.5*tau^2)
            fbs2 += params.pcof[offset2+k] * (9/8 .+ 4.5*tau + 4.5*tau^2)
            
            # 2nd segment of nurb k-1
            tc = params.tcenter[k-1]
            tau = (t - tc)./width
            fbs1 += params.pcof[offset1+k.-1] .* (0.75 - 9 *tau^2)
            fbs2 += params.pcof[offset2+k.-1] .* (0.75 - 9 *tau^2)
            
            # 3rd segment of nurb k-2
            tc = params.tcenter[k.-2]
            tau = (t .- tc)./width
            fbs1 += params.pcof[offset1+k-2] * (9/8 - 4.5*tau + 4.5*tau.^2)
            fbs2 += params.pcof[offset2+k-2] * (9/8 - 4.5*tau + 4.5*tau.^2)

            #    end # for carrier phase
            # p(t)
            if q_func==1
                f += fbs1 * sin(params.om[osc+1,freq]*t) + fbs2 * cos(params.om[osc+1,freq]*t) # q-func
                
            else
                f += fbs1 * cos(params.om[osc+1,freq]*t) - fbs2 * sin(params.om[osc+1,freq]*t) # p-func
            end
        end # for freq
    end # if
    return f
end

@inline function bcarrier2(t::Float64, params::bcparams2, func::Int64)
    # for a single oscillator, func=0 corresponds to p(t) and func=1 to q(t)
    # in general, 0 <= func < 2*Ncoupled + Nunc
    # compute basic offset: func 0 and 1 use the same spline coefficients, but combined in a different way
    osc = div(func, 2) # osc is base 0; 0<= osc < Ncoupled
    q_func = func % 2 # q_func = 0 for p and q_func=1 for q
    
    f = 0.0 # initialize
    
    dtknot = params.dtknot
    # println(dtknot)
    width = 3*dtknot
    
    k = max.(3, ceil.(Int64,t./dtknot + 2)) # pick out the index of the last basis function corresponding to t
    k = min.(k, params.D1) #  Make sure we don't access outside the array
    # println("k: $k")
    if func < 2*(params.Ncoupled + params.Nunc)
        # Coupled and uncoupled controls
        # for nfreq in params.Nfreq
        @fastmath @inbounds @simd for freq in 1:params.Nfreq[osc+1]
            fbs1 = 0.0 # initialize
            fbs2 = 0.0 # initialize
            # offset in parameter array (osc = 0,1,2,...
            # Vary freq first, then osc

            offset1 = 2*params.D1*(sum(params.Nfreq[1:osc])) + (freq - 1)*2*params.D1
            offset2 = 2*params.D1*(sum(params.Nfreq[1:osc])) + (freq - 1)*2*params.D1 + params.D1
            # println("offset3: ", offset3)
            # println("offset4: ", offset4)
            # offset1 = 2*osc*params.Nfreq[osc + 1]*params.D1 + (freq-1)*2*params.D1
            # offset2 = 2*osc*params.Nfreq[osc + 1]*params.D1 + (freq-1)*2*params.D1 + params.D1
            # println("offset1: ", offset1)
            # println("offset2: ", offset2)
            # 1st segment of nurb k
            tc = params.tcenter[k]
            tau = (t .- tc)./width
            # println("tau: ", tau)
            # println("offset1+k: ", offset1+k)
            # println("offset2+k: ", offset2+k)
            
            # println("params.pcof[offset1+k]: ", params.pcof[offset1+k])
            # println("(9/8 .+ 4.5*tau + 4.5*tau^2): ", (9/8 .+ 4.5*tau + 4.5*tau^2))
            fbs1 += params.pcof[offset1+k] * (9/8 .+ 4.5*tau + 4.5*tau^2)
            fbs2 += params.pcof[offset2+k] * (9/8 .+ 4.5*tau + 4.5*tau^2)
            
            # 2nd segment of nurb k-1
            tc = params.tcenter[k-1]    
            tau = (t - tc)./width
            # println("tau: ", tau)
            # println("offset1+k.-1: ", offset1+k.-1)
            # println("offset2+k.-1: ", offset2+k.-1)
            # println("params.pcof[offset1+k.-1]: ", params.pcof[offset1+k.-1])
            # println("(0.75 - 9 *tau^2): ", (0.75 - 9 *tau^2))
            fbs1 += params.pcof[offset1+k.-1] .* (0.75 - 9 *tau^2)
            fbs2 += params.pcof[offset2+k.-1] .* (0.75 - 9 *tau^2)
            
            # 3rd segment of nurb k-2
            tc = params.tcenter[k-2]
            tau = (t .- tc)./width
            # println("tau: ", tau)
            # println("offset1+k-2: ", offset1+k-2)
            # println("offset2+k.-2: ", offset2+k.-2)
            # println("(9/8 - 4.5*tau + 4.5*tau.^2): ", (9/8 - 4.5*tau + 4.5*tau.^2))
            # println("params.pcof[offset1+k-2]: ", params.pcof[offset1+k-2])
            fbs1 += params.pcof[offset1+k-2] * (9/8 - 4.5*tau + 4.5*tau.^2)
            fbs2 += params.pcof[offset2+k-2] * (9/8 - 4.5*tau + 4.5*tau.^2)
            # println("fbs1: $fbs1")
            # println("fbs2: $fbs2")
            #    end # for carrier phase
            # p(t)
            if q_func==1
                # println("Sin: ", sin(params.om[osc + 1][freq]*t))
                # println("Cos: ", cos(params.om[osc+1][freq]*t))
                f += fbs1 * sin(params.om[osc + 1][freq]*t) + fbs2 * cos(params.om[osc+1][freq]*t) # q-func
            else
                # println("Sin: ", sin(params.om[osc + 1][freq]*t))
                # println("Cos: ", cos(params.om[osc+1][freq]*t))
                f += fbs1 * cos(params.om[osc+1][freq]*t) - fbs2 * sin(params.om[osc+1][freq]*t) # p-func
            end
            # println("f: $f")
        end
        # end # for freq
    end # if
    return f
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

function reverse_vec_itensor(tensor::ITensor)
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

function vec_itensor(tensor::ITensor)
    tensor_array = Array(tensor, inds(tensor))
    N = length(tensor_array)
    vec_tensor = zeros(eltype(tensor_array), N)
    size_tuple = size(tensor_array)
    size_vec = collect(size_tuple)
    for i = 1:N 
        index = tuple_index_natural(i, size_vec)
        vec_tensor[i] = tensor_array[index...,]
    end
    return vec_tensor 
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

function vectorize_mps(M::MPS; order = "natural")
    N = length(M)
    total_ten = M[1]
    for i in 2:N 
        total_ten = total_ten*M[i]
    end
    if order == "natural"
        total_ten_vec = vec_itensor(total_ten)
    elseif order == "reverse"
        total_ten_vec = reverse_vec_itensor(total_ten)
    end
    return total_ten_vec 
end

function vec_mps2(M::MPS; order = "natural")
    N = length(M)
    total_ten = M[1]
    for i in 2:N 
        total_ten = total_ten*M[i]
    end
    if order == "natural"
        total_ten_vec = vec_itensor(total_ten)
    elseif order == "reverse"
        total_ten_vec = reverse_vec_itensor(total_ten)
    end
    return total_ten_vec 
end
    
    

function matrix_form(H::MPO, sites)
    N = length(H)
    # sites = siteinds(H)
    d = dim(sites[1])
    Matrix_Form = zeros(ComplexF64, (d^N, d^N))
    for i = 1:d^N
        # println("Col: $i")
        vec = zeros(d^N)
        vec[i] = 1.0
        vec_mps = MPS(vec, sites)
        mpo_col = noprime(H*vec_mps)
        mpo_col = vectorize_mps(mpo_col; order = "natural")
        Matrix_Form[:,i] = mpo_col
    end
    return Matrix_Form
end



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

function s_op_general(op, j, N, d)
    if j == 1 || j == N + 1
        Ident = Matrix(I, d^(N - 1), d^(N - 1))
        return kron(op, Ident)
    elseif j == N
        Ident = Matrix(I, d^(j - 1), d^(j - 1))
        return kron(Ident, op)
    else 
        I1 = Matrix(I, d^(j - 1), d^(j - 1))
        I2 = Matrix(I, d^(N - j), d^(N - j))
        return kron(I1, op, I2)
    end
end

function s_op_general_reverse(op, j, N, d)
    if j == 1 || j == N + 1
        Ident = Matrix(I, d^(N - 1), d^(N - 1))
        return kron(Ident, op)
    elseif j == N
        Ident = Matrix(I, d^(j - 1), d^(j - 1))
        return kron(op, Ident)
    else 
        I1 = Matrix(I, d^(N - j), d^(N - j))
        I2 = Matrix(I, d^(j - 1), d^(j - 1))
        return kron(I1, op, I2)
    end
end

function s_op_sparse(op, j, N)
    op = sparse(op)
    if j == 1 ||j == N + 1
        Ident = sparse(I, 2^(N - 1), 2^(N - 1))
        return kron(op, Ident)
    elseif j == 1
        Ident = sparse(I, 2^(j - 1), 2^(j - 1))
        return kron(Ident, op)
    else
        I1 = Matrix(I, 2^(j - 1), 2^(j - 1))
        I2 = Matrix(I, 2^(N - j), 2^(N - j))
        return kron(I1, op, I2)
    end
end

#All of the below functions are for different Hamiltonian models
#Hamiltonian model without rotational frequency change and with time dependent term
function time_MPO(t, p, ground_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N 
        os += ground_freq[N - i + 1], "a+a", i
        os += p(t, N - i + 1), "Sx2", i
        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "a+a", i, "a+a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "a+", i, "a", j 
                os += dipole[i,j], "a", i, "a+", j
            end
        end
    end
    H = MPO(os, sites)
    return H 
end

#Same as above, exc ept the time dependent part is in terms of a vector instead of a function
function piecewise_H_MPO(step, pt0, qt0, ground_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N 
        os += ground_freq[N - i + 1], "a+a", i
        os += pt0[i,step], "Sx2", N - i + 1
        os += qt0[i,step], "-Sy2", N - i + 1
        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "a+a", i, "a+a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "a+", i, "a", j 
                os += dipole[i,j], "a", i, "a+", j
            end
        end
    end
    H = MPO(os, sites)
    return H 
end

function piecewise_H_MPO(step, pt0, qt0, ground_freq, rot_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N
        
        freq = ground_freq[N - i + 1] - rot_freq[N - i + 1]
        os += freq, "a'a", i
        os -= 0.5*self_kerr[N - i + 1], "a'a'", i, "aa", i

        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "a'a", i, "a'a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "adag", i, "a", j 
                os += dipole[i,j], "a", i, "adag", j
            end
        end
        os += pt0[N - i + 1,step], "a + a'", i
        os += im*qt0[N - i + 1,step], "a - a'", i
    end
    H = MPO(os, sites)
    return H 
end

function piecewise_H_MPO_v2(step, pt0, qt0, ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N
        
        freq = ground_freq[N - i + 1] - rot_freq[N - i + 1]
        os += freq, "adag", i, "a", i
        os -= 0.5*self_kerr[N - i + 1], "adag", i, "adag", i, "a", i, "a", i

        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "adag", i, "a", i, "adag", j, "a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "adag", i, "a", j 
                os += dipole[i,j], "a", i, "adag", j
            end
        end
        os += pt0[N - i + 1,step], "a", i
        os += pt0[N - i + 1,step], "adag", i
        os += im*qt0[N - i + 1,step], "a", i 
        os -= im*qt0[N - i + 1,step], "adag", i
    end
    H = MPO(os, sites)
    return H 
end

function H_MPO_v2(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N
        
        freq = ground_freq[N -  i+ 1] - rot_freq[N - i + i]
        os += freq, "adag", i, "a", i
        os -= 0.5*self_kerr[N - i + 1], "adag", i, "adag", i, "a", i, "a", i

        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "adag", i, "a", i, "adag", j, "a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "adag", i, "a", j 
                os += dipole[i,j], "a", i, "adag", j
            end
        end
    end
    H = MPO(os, sites)
    return H 
end

function piecewise_H_MPO_no_rot(step, pt, qt, ground_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N 
        os += ground_freq[N - i + 1], "a+a", i
        os += f[i,step], "Sx2", N - i + 1

        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "a+a", i, "a+a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "a+", i, "a", j 
                os += dipole[i,j], "a", i, "a+", j
            end
        end
    end
    H = MPO(os, sites)
    return H 
end

function piecewise_H(step, f, ground_freq, cross_kerr, dipole, N)
    H = zeros(ComplexF64, (2^N, 2^N))
    for i = 1:N 
        H .+= ground_freq[i]*s_op(a, i, N)
        H .+= pt0[i, step]*s_op([0 1; 1 0], i, N)
        H .+= im*qt0[i, step]*s_op([0 1; -1 0], i, N)
        if i != N 
            for j = i + 1: N 
                #zz-coupling interaction
                H .-= cross_kerr[i,j]*s_op(a, i, N)*s_op(a, j, N)
                #dipole-dipole interaction
                
                H .+= dipole[i,j]*s_op([0 0; 1 0], i, N)*s_op([0 1; 0 0], j, N)
                H .+= dipole[i,j]*s_op([0 1; 0 0], i, N)*s_op([0 0; 1 0], j, N)
            end
        end
    end
    return H 
end

function piecewise_H_no_rot(step, f, ground_freq, cross_kerr, dipole, N)
    H = zeros(ComplexF64, (2^N, 2^N))
    for i = 1:N 
        H .+= ground_freq[i]*s_op(a, i, N)
        H .+= f[i, step]*s_op([0 1; 1 0], i, N)
        if i != N 
            for j = i + 1: N 
                #zz-coupling interaction
                H .-= cross_kerr[i,j]*s_op(a, i, N)*s_op(a, j, N)
                #dipole-dipole interaction
                
                H .+= dipole[i,j]*s_op([0 0; 1 0], i, N)*s_op([0 1; 0 0], j, N)
                H .+= dipole[i,j]*s_op([0 1; 0 0], i, N)*s_op([0 0; 1 0], j, N)
            end
        end
    end
    return H 
end

# function H_sys(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, d)
#     H = zeros(ComplexF64, (d^N, d^N))
#     a = Array(Bidiagonal(zeros(d), sqrt.(collect(1: d - 1)), :U))
#     for i = 1:N 
#         H .+= (ground_freq[i] - rot_freq[i])*s_op_general(a'*a, i, N, d)
#         H .-= 0.5*self_kerr[i]*s_op_general(a'*a'*a*a, i, N, d)
#         if i != N 
#             for j = i + 1: N
#                 #zz-coupling interaction
#                 H .-= cross_kerr[i,j]*s_op_general(a'*a, i, N, d)*s_op_general(a'*a, j, N, d)
#                 #dipole-dipole interaction
#                 H .+= dipole[i,j]*s_op_general(a', i, N, d)*s_op_general(a, j, N, d)
#                 H .+= dipole[i,j]*s_op_general(a, i, N, d)*s_op_general(a', j, N, d)
#             end
#         end
#     end
#     return H 
# end

function H_sys(N::Int64, d::Vector{Int64}, ground_freq::Vector{Float64}, rot_freq::Vector{Float64}, self_kerr::Vector{Float64}, dipole::AbstractMatrix, cross_kerr::AbstractMatrix)
    H = zeros(ComplexF64, (prod(d), prod(d)))
    
    for i = 1:N
        a = Array(Bidiagonal(zeros(d[i]), sqrt.(collect(1: d[i] - 1)), :U))
        H .+= (ground_freq[i] - rot_freq[i])*s_op_general_reverse(a'*a, i, N, d[i])
        H .-= 0.5*self_kerr[i]*s_op_general_reverse(a'*a'*a*a, i, N, d[i])
        if i != N 
            for j = i + 1: N
                #zz-coupling interaction
                H .-= cross_kerr[i,j]*s_op_general_reverse(a'*a, i, N, d[i])*s_op_general_reverse(a'*a, j, N, d[j])
                #dipole-dipole interaction
                H .+= dipole[i,j]*s_op_general_reverse(a', i, N, d[i])*s_op_general_reverse(a, j, N, d[j])
                H .+= dipole[i,j]*s_op_general_reverse(a, i, N, d[i])*s_op_general_reverse(a', j, N, d[j])
            end
        end
    end
    return H 
end

# function H_sys(N::Int64, d::Vector{Int64}, ground_freq::Vector{Float64}, rot_freq::Vector{Float64}, self_kerr::Vector{Float64}, dipole::AbstractMatrix, cross_kerr::AbstractMatrix)
#     H = zeros(ComplexF64, (prod(d), prod(d)))
    
#     for i = 1:N
#         a = Array(Bidiagonal(zeros(d[i]), sqrt.(collect(1: d[i] - 1)), :U))
#         H .+= (ground_freq[i] - rot_freq[i])*s_op_general(a'*a, i, N, d[i])
#         H .-= 0.5*self_kerr[i]*s_op_general(a'*a'*a*a, i, N, d[i])
#         if i != N 
#             for j = i + 1: N
#                 #zz-coupling interaction
#                 H .-= cross_kerr[i,j]*s_op_general(a'*a, i, N, d[i])*s_op_general(a'*a, j, N, d[j])
#                 #dipole-dipole interaction
#                 H .+= dipole[i,j]*s_op_general(a', i, N, d[i])*s_op_general(a, j, N, d[j])
#                 H .+= dipole[i,j]*s_op_general(a, i, N, d[i])*s_op_general(a', j, N, d[j])
#             end
#         end
#     end
#     return H 
# end

function H_ctrl(H_s, N, d, bc_params, t)
    H_c = zeros(ComplexF64, size(H_s))
    for i = 1:N 
        a = Array(Bidiagonal(zeros(d[i]), sqrt.(collect(1: d[i] - 1)), :U))
        H_c .+= bcarrier2(t, bc_params, 2*(i - 1))*s_op_general_reverse(a + a', i, N, d[i]) + im*bcarrier2(t, bc_params, 2*(i - 1) + 1)*s_op_general_reverse(a - a', i, N, d[i])
    end
    return H_c 
end

# function H_ctrl(H_s, N, d, bc_params, t)
#     H_c = zeros(ComplexF64, size(H_s))
#     for i = 1:N 
#         a = Array(Bidiagonal(zeros(d[i]), sqrt.(collect(1: d[i] - 1)), :U))
#         H_c .+= bcarrier2(t, bc_params, 2*(i - 1))*s_op_general(a + a', i, N, d[i]) + im*bcarrier2(t, bc_params, 2*(i - 1) + 1)*s_op_general(a - a', i, N, d[i])
#     end
#     return H_c 
# end

# function H_ctrl(step, p, q, N, d)
#     H = zeros(ComplexF64, (d^N, d^N))
#     a = Array(Bidiagonal(zeros(d), sqrt.(collect(1: d - 1)), :U))
#     for i = 1:N 
#         H .+= p[i,step]*s_op_general(a + a', i, N, d)
#         H .+= im*q[i, step]*s_op_general(a - a', i, N, d)
#     end
#     return H 
# end


function system_MPO(ground_freq, cross_kerr, dipole, N, sites)
    os = OpSum() 
    for i = 1:N 
        os += ground_freq[N - i + 1], "a+a", i
        #Don't need to worry about self-kerr with qubits, the self kerr process just becomes 0
        if  i != N
            for j = i + 1:N
                #zz-coupling interactions
                os -= cross_kerr[i,j], "a+a", i, "a+a", j 
                #dipole-dipole interactions
                os += dipole[i,j], "a+", i, "a", j 
                os += dipole[i,j], "a", i, "a+", j
            end
        end
    end
    H = MPO(os, sites)
    return H 
end

#Simple Ising Model (https://en.wikipedia.org/wiki/Ising_model#Definition) with J_ij = 0, mu = 0 as an MPO
function ising_mpo(N, sites)
    # Make N S=1/2 spin indices
    # sites = siteinds("S=1/2",N)
    # Input the operator terms
    os = OpSum()
    for i=1:N-1
        os += "Sz2",i,"Sz2",i+1
    end
    # Convert these terms to an MPO
    H = MPO(os,sites)
    return H
end

function downsample_pulse(pt, qt, nsplines, nsteps)
    if length(pt) == nsteps & length(qt) == nsteps 
        return pt, qt 
    else
        pt_n = zeros(size(pt)[1], nsteps)
        qt_n = zeros(size(qt)[1], nsteps)
        if nsteps % nsplines == 0
            
            for j in 1:size(pt)[1]
                for i in 1:nsplines
                    spline_len = Int64(nsteps/nsplines) 
                    pt_n[j, (i - 1)*spline_len + 1:i*spline_len] .= pt[j, i]
                    qt_n[j, (i - 1)*spline_len + 1:i*spline_len] .= qt[j, i]
                end
            end 
            
        elseif nsteps % nsplines != 0
            println("Number of steps is not divisible by the number of splines") 
            spline_len = Int64(floor(nsteps/nsplines))
            spline_remainder = nsteps % nsplines
            for j in 1:size(pt)[1]
                for i in 1:nsplines
                    if i == nsplines 
                        pt_n[j, (i - 1)*spline_len + 1: i*spline_len + spline_remainder] .= pt[j, i]
                        qt_n[j, (i - 1)*spline_len + 1: i*spline_len + spline_remainder] .= qt[j, i]
                    else
                        pt_n[j, (i - 1)*spline_len + 1:i*spline_len] .= pt[j, i]
                        qt_n[j, (i - 1)*spline_len + 1:i*spline_len] .= qt[j, i]
                    end
                end 
            end
        end 
    end
    return pt_n, qt_n
end


#xxx hamiltonain model as MPO
function xxx_mpo(N, sites, J, g)
    os = OpSum()
    for i = 1:N - 1
        os -= J, "Sz2",i, "Sz2", i + 1
    end
    for i = 1:N
        os -= g*J, "Sx2", i
    end
    H = MPO(os, sites)
    return H 
end

#Ising Model with J_ij = 0, mu = 0 as matrix
function Ising(N)
    H = zeros(ComplexF64, (2^N, 2^N))
    for j in 1:N - 1
        H .+= s_op(sz, j, N)*s_op(sz, j + 1, N)
    end
    return H
end

#xxx hamiltonian as a matrix
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

function xxx_v2(N, J, g)
    H = zeros(ComplexF64, (2^N, 2^N))
    for j in 1:N - 1
        H .+= -g*J*s_op(sz, j, N)*s_op(sz, j + 1, N)
    end
    for j in 1:N 
        H .-= J*s_op(sx, j, N)
    end
    return H 
end 

#Sparse version of xxx hamiltonian matrix
function xxx_sparse(N, J, g)
    H = spzeros(ComplexF64, (2^N, 2^N))
    for j in 1:N - 1
        H += -J*s_op_sparse(sz, j, N)*s_op_sparse(sz, j + 1, N)
    end
    for j in 1:N 
        H -= g*J*s_op_sparse(sx, j, N)
    end
    return H 
end

function H_MPO_manual(ground_freq, rot_freq, self_kerr, cross_kerr, dipole, N, sites)
    #Construct Hamiltonian Manually with no control 

    H = MPO(ComplexF64, sites)
    
    s1 = dim(sites[1])
    s2 = dim(sites[2])
    a1 = annihilation_operator(s1)
    a2 = annihilation_operator(s2)
    l1 = 4 
    H1 = zeros(ComplexF64, s1, s1, l1)
    H2 = zeros(ComplexF64, s2, s2, l1)
    H1[1,:,:] = (ground_freq[2] - rot_freq[2])*(a1'*a1) - 0.5*self_kerr[2]*(a1'*a1'*a1*a1)
    H1[2,:,:] = dipole[1,2]*a1'
    H1[3,:,:] = dipole[1,2]*a1
    H1[4,:,:] = Matrix(1.0*I, s1, s1)
    
    H2[1,:,:] = Matrix(1.0*I,  s2, s2)
    H2[2,:,:] = a2
    H2[3,:,:] = a2'
    H2[4,:,:] = (ground_freq[1] - rot_freq[1])*(a2'*a2) - 0.5*self_kerr[1]*(a2'*a2'*a2*a2)
    s1 = sites[1]
    s2 = sites[2]
    l1 = Index(l1, "Link, l = 1")
    
    H[1] = ITensor(H1, l1, s1', s1)
    H[2] = ITensor(H2, l1, s2', s2)

    return H 
end

function update_H(H_MPO::MPO, bcparams, t)
    pt_1 = bcarrier2(t, bcparams, 0)
    qt_1 = bcarrier2(t, bcparams, 1)
    pt_2 = bcarrier2(t, bcparams, 2)
    qt_2 = bcarrier2(t, bcparams, 3)

    # println(H_MPO)
    
    links = linkinds(H_MPO)
    site1_inds = siteinds(H_MPO)[1]
    site2_inds = siteinds(H_MPO)[2]

    site1 = H_MPO[1]
    site2 = H_MPO[2]
    # println(site2)
    # display(Array(H_MPO[1], inds(H_MPO[1]))[1,:,:])
    # display(Array(site2, inds(site2))[4,:,:])
    for i = 1:4
        if i < 4
            H_MPO[1][links[1] => 1, site1_inds[1] => i, site1_inds[2] => i + 1] = sqrt(i)*(pt_2 + im*qt_2)
            H_MPO[2][links[1] => 4, site2_inds[1] => i, site2_inds[2] => i + 1] = sqrt(i)*(pt_1 + im*qt_1)
        end
        if i > 1
            H_MPO[1][links[1] => 1, site1_inds[1] => i, site1_inds[2] => i - 1] = sqrt(i - 1)*(pt_2 - im*qt_2)
            H_MPO[2][links[1] => 4, site2_inds[1] => i, site2_inds[2] => i - 1] = sqrt(i - 1)*(pt_1 - im*qt_1)
        end
    end
    # site1[links[1] => 1, site1_inds[1] => 1, site1_inds[2] => 2] = 1
    # site1[links[1] => 1, site1_inds[1] => 2, site1_inds[2] => 1] = 1
    # site1[links[1] => 1, site1_inds[1] => 2, site1_inds[2] => 3] = sqrt(2)
    # site1[links[1] => 1, site1_inds[1] => 3, site1_inds[2] => 2] = sqrt(2)
    # site1[links[1] => 1, site1_inds[1] => 3, site1_inds[2] => 4] = sqrt(3)
    # site1[links[1] => 1, site1_inds[1] => 4, site1_inds[2] => 3] = sqrt(3)
    # display(Array(H_MPO[1], inds(H_MPO[1]))[1,:,:])

    # display(Array(site2, inds(site2))[4,:,:])
    # site2[links[1] => 4, site2_inds[1] => 1, site2_inds[2] => 2] = 1
    # site2[links[1] => 4, site2_inds[1] => 2, site2_inds[2] => 1] = 1
    # site2[links[1] => 4, site2_inds[1] => 2, site2_inds[2] => 3] = sqrt(2)
    # site2[links[1] => 4, site2_inds[1] => 3, site2_inds[2] => 2] = sqrt(2)
    # site2[links[1] => 4, site2_inds[1] => 3, site2_inds[2] => 4] = sqrt(3)
    # site2[links[1] => 4, site2_inds[1] => 4, site2_inds[2] => 3] = sqrt(3)
    return H_MPO
end

function graziani_H(N, J, U, hj_list, hp_list)
    H = zeros(2^N, 2^N)
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

function graziani_H_MPO(N, sites, J, U, hj_list, hp_list)
    os = OpSum()
    for i = 1: N
        if i != N  
            os += -4*J, "Sx", i, "Sx", i + 1
            os += -4*J, "Sy", i, "Sy", i + 1
            os += 4*U, "Sz", i, "Sz", i + 1 
        end 
        os += 2*hj_list[i], "Sz", i 
        os += 2*hp_list[i], "Sx", i
    end
    H = MPO(os, sites)
    return H 
end



