using ITensors
using LinearAlgebra, ProgressMeter



function exp_solver(A, y, h)
    y_n = exp(A*h)*y
    return y_n 
end

function IMR_solver(A, y, h)
    LHS = I - h/2*A 
    RHS = (I + h/2*A)*y 
    y_n = LHS\RHS
    return y_n 
end

function euler(A, y, h)
    return y + h*A*y
end

function constant_midpoint(A, y, h)
    return y + h*A*y + 0.5*(h*A)^2 * y 
end
# function Midpoint_solver(A, y, t, h)
#     y_n = y - im*h*A

function backwards_euler(A, y, h)
    LHS = (I - h*A)
    y_n = LHS\y 
    return y_n 
end

function create_linkinds(N::Int64, link_size::Vector{Int64})
    ind_vec = Index{Int64}[]
    for i in 1:N-1
        ind = Index(link_size[i];tags="Link, $i \u2194 $(i + 1)")
        push!(ind_vec, ind)
    end
    return ind_vec
end

#Helper function that matches indices between an MPO and MPS
function match_index(M, R)
    M_inds = inds(M)
    R_inds = inds(R)
    ind_match = Int64[]
    for i in 1:length(R_inds) 
        for j in 1:length(M_inds)

            if R_inds[i] == M_inds[j]
                # println("It's a match at: ", [i,j])
                push!(ind_match, j)
            end
        end
    end
    ind = collect(1:length(M_inds))
    ind_no_match = setdiff(ind, ind_match)
    return ind_match, ind_no_match 
end

#Converts a tensor to a vector
function tensor_to_vec(T::ITensor)
    T_inds = inds(T)
    T_arr = Array(T, T_inds)
    # println("T_arr")
    # println("----------------------------")
    # display(T_arr)
    T_vec = reshape(T_arr, dim(T))
    return T_vec 
end

#Converts both an effective Hamiltonian and tensor to a Matrix and vector, respectively
#Then checks if the conversion was correct by doing a multiplcation with this matrix and vector and checks against the tensor multiplication
function conversion(H, M)
    H_contract = contract(H)
    #Gets indices of both tensors and determines which indices are matching and not matching
    H_inds = inds(H_contract)
    M_inds = inds(M)
    match_ind, ind_no_match = match_index(H_inds, M_inds)

    #Gets dimensions for row and columns of matrix
    row_H = 1 
    col_H = 1
    # println("H_inds: ", H_inds)
    for i = 1:length(H_inds) 
        if plev(H_inds[i]) == 1
            row_H *= dim(H_inds[i])
        else
            col_H *= dim(H_inds[i])
        end 
    end
    
    #When working the tdvp and tdvp2 there will only be three different situations for the number of the indices
    #below we convert the tensor object into an array in order to convert into a matrix
    if length(H_inds) == 4
        H_arr = Array(H_contract, H_inds[ind_no_match[1]], H_inds[ind_no_match[2]], H_inds[match_ind[1]], H_inds[match_ind[2]])
    elseif length(H_inds) == 6
        H_arr = Array(H_contract, H_inds[ind_no_match[1]], H_inds[ind_no_match[2]], H_inds[ind_no_match[3]], H_inds[match_ind[1]]
        , H_inds[match_ind[2]], H_inds[match_ind[3]])
    elseif length(H_inds) == 8
        H_arr = Array(H_contract, H_inds[ind_no_match[1]], H_inds[ind_no_match[2]], H_inds[ind_no_match[3]], H_inds[ind_no_match[4]], H_inds[match_ind[1]]
        , H_inds[match_ind[2]], H_inds[match_ind[3]], H_inds[match_ind[4]])
    end
    #Convert H_arr into a matrix and the M tensor into a vector
    H_mat = reshape(H_arr, (row_H, col_H))
    M_arr = Array(M, M_inds)
    M_vec = reshape(M_arr, dim(M))

    #Calculate both possible multiplcations in order to test if conversion was correct 
    mult1 = H_mat*M_vec
    mult2 = tensor_to_vec(H_contract*M)

    # println("Size of matrix: $(size(H_mat))")

    #Returns H_mat and M_vec if conversion was successful, otherwise doesn't return and gives the error between the multiplcations
    if norm(mult1 - mult2) < 1E-12
        # println("Multiplication difference: ", norm(mult1 - mult2))
        return H_mat, M_vec 
    else 
        println("Tensors not correctly converted to matrix/vector")
        println("Norm error: ", norm(mult1 - mult2))
        # return H_mat, M_vec
    end
end

#Creates effective Hamiltonian
function effective_Hamiltonian(H, M, i)
    N = length(M)
    H_eff = MPO(N)
    qubit_range = setdiff(collect(1:N), i)
    # println("Qubit range: ", qubit_range)
    # println("Length: $N")
    # println(H[1])
    for j in qubit_range
        # println("Qubit $j")
        H_eff[j] = H[j]*M[j]*conj(M[j])'
    end
    # println(H[i])
    H_eff[i] = H[i]
    # println("Contracted inds: ", inds(contract(H_eff)))
    return H_eff
end

#Creates effective 2-site Hamiltonian
function effective_Hamiltonian_2site(H, M, i)
    N = length(M)
    H_eff = MPO(N)
    qubit_range = setdiff(collect(1:N), [i, i + 1])
    for j in qubit_range 
        H_eff[j] = H[j]*M[j]*conj(M[j])'
    end
    H_eff[i] = H[i]
    H_eff[i + 1] = H[i + 1]
    return H_eff
end

#Creates effective 0-site Hamiltonian, which is referred to as K so I'm referring to it as the Kamiltonian
function effective_Kamiltonian(H, M)
    N = length(M)
    K_eff = MPO(N)
    for j in 1:N 
        K_eff[j] = H[j]*M[j]*conj(M[j])'
    end
    # println("K_eff: ", K_eff)
    return K_eff 
end

#Performs a single left-to-right sweep of an MPS using the tdvp, evolving forward one time step.
function lr_sweep(H, M, t, h, method)
    
    #Ensures orthogonality center is the first site
    orthogonalize!(M, 1)
    # if return_projectors == true
    #     P_L = []
    #     P_R = []
    # end
    N = length(M)
    for i in 1:N - 1 
        # println("Site: ", i)

        #Creates effective Hamiltonian matrices and converts the i-th site to a vector
        H_eff = effective_Hamiltonian(H, M, i)
        H_mat, M_vec = conversion(H_eff, M[i])
        # println("H_Mat: ")
        # display(H_mat)
        # println("M_vec: ")
        # display(M_vec)
        #Evolves M_vec with H_mat with step size 'h'

        # M_evolve = exp(-im*H_mat*h)*M_vec
        if method == 1
            M_evolve = IMR_solver(-im*H_mat, M_vec, h)
        else
            M_evolve = exp_solver(-im*H_mat, M_vec, h)
        end
        # M_evolve = backwards_euler(-im*H_mat, M_vec, h)
        # M_evolve = euler(-im*H_mat, M_vec, h)
        # M_evolve = constant_midpoint(-im*H_mat, M_vec, h)
        # println("Difference: ")
        # display(norm(constant_midpoint(-im*H_mat, M_vec, h) - exp(-im*H_mat*h)*M_vec))
        # display(norm(IMR_solver(-im*H_mat, M_vec, h) - exp(-im*H_mat*h)*M_vec))
        # println("M evolve: ")
        # display(M_evolve)

        #Converts back into a tensor
        M_inds = inds(M[i]) 
        M_evolve = ITensor(M_evolve, M_inds)

        #Performs QR decomposition in order to get left-orthogonal tensor
        if i==1
            Q, R = qr(M_evolve, M_inds[1])
        else
            Q, R = qr(M_evolve, M_inds[1:2])
        end

        #Set left-orthogonal tensor as new tensor in MPS
        M[i] = Q

        #Creates effective Kamiltonian matrix and converts the upper triangular part from the QR into a vector
        K_eff = effective_Kamiltonian(H, M)
        K_mat, R_vec = conversion(K_eff, R)
        # println("K_mat")
        # display(K_mat)
        #Evolves R_vec with K_mat and step size h

        # R_evolve = exp(im*K_mat*h)*R_vec
        if method == 1
            R_evolve = IMR_solver(im*K_mat, R_vec, h)
        else
            R_evolve = exp_solver(im*K_mat, R_vec, h)
        end
        # R_evolve = backwards_euler(im*K_mat, R_vec, h)
        # R_evolve = euler(im*K_mat, R_vec, h)
        # R_evolve = constant_midpoint(im*K_mat, R_vec, h)
        # println("R_evolve")
        # display(R_evolve)
        #Convert R into tensor and multiply it with next tensor in the MPS and then replace
        R_inds = inds(R) 
        R_evolve = ITensor(R_evolve, R_inds)
        M[i + 1] = R_evolve*M[i + 1]
    end
    
    #Performs evolution on last site but without an QR decomposition as the MPS will be completely left-orthogonal.
    H_eff_N = effective_Hamiltonian(H, M, N)
    H_N_mat, M_N_vec = conversion(H_eff_N, M[N])
    # println("M2_vec pre-evolve")
    # display(M_N_vec)
    # println("H_mat 2")
    # display(H_N_mat)

    # M_N_evolve = exp(-im*H_N_mat*h)*M_N_vec
    if method == 1
        M_N_evolve = IMR_solver(-im*H_N_mat, M_N_vec, h)
    else
        M_N_evolve = exp_solver(-im*H_N_mat, M_N_vec, h)
    end
    # M_N_evolve = backwards_euler(-im*H_N_mat, M_N_vec, h)
    # M_N_evolve = euler(-im*H_N_mat, M_N_vec, h)
    # M_N_evolve = constant_midpoint(-im*H_N_mat, M_N_vec, h)
    # println("M_2 vec")
    # display(M_N_evolve) 
    M_N_inds = inds(M[N])
    M_N_evolve = ITensor(M_N_evolve, M_N_inds)
    M[N] .= M_N_evolve
    
    #Return completely evolved MPS
    return M 
end

function rl_sweep(H, M, t, h, method)    
    #Ensures orthogonality center is the last site
    N = length(M)
    orthogonalize!(M, N)
    # if return_projectors == true
    #     P_L = []
    #     P_R = []
    # end
    
    for i in N:-1:2 
        # println("Site: ", i)

        #Creates effective Hamiltonian matrices and converts the i-th site to a vector
        H_eff = effective_Hamiltonian(H, M, i)
        H_mat, M_vec = conversion(H_eff, M[i])
        # println("H_Mat: ")
        # display(H_mat)
        # println("M_vec: ")
        # display(M_vec)
        #Evolves M_vec with H_mat with step size 'h'

        # M_evolve = exp(-im*H_mat*h)*M_vec
        if method == 1
            M_evolve = IMR_solver(-im*H_mat, M_vec, h)
        else
            M_evolve = exp_solver(-im*H_mat, M_vec, h)
        end
        # M_evolve = backwards_euler(-im*H_mat, M_vec, h)
        # M_evolve = euler(-im*H_mat, M_vec, h)
        # M_evolve = constant_midpoint(-im*H_mat, M_vec, h)
        # println("Difference: ")
        # display(norm(constant_midpoint(-im*H_mat, M_vec, h) - exp(-im*H_mat*h)*M_vec))
        # display(norm(IMR_solver(-im*H_mat, M_vec, h) - exp(-im*H_mat*h)*M_vec))
        # println("M evolve: ")
        # display(M_evolve)

        #Converts back into a tensor
        M_inds = inds(M[i])
        M_evolve = ITensor(M_evolve, M_inds)

        #Performs QR decomposition in order to get left-orthogonal tensor
        if i==1
            R, Q = factorize(M_evolve, commonind(inds(M[1]), inds(M[2])))
        else
            R, Q = factorize(M_evolve, commonind(inds(M[i]), M_inds[i - 1]))
        end

        #Set left-orthogonal tensor as new tensor in MPS
        M[i] = Q

        #Creates effective Kamiltonian matrix and converts the upper triangular part from the QR into a vector
        K_eff = effective_Kamiltonian(H, M)
        K_mat, R_vec = conversion(K_eff, R)
        # println("K_mat")
        # display(K_mat)
        #Evolves R_vec with K_mat and step size h

        # R_evolve = exp(im*K_mat*h)*R_vec
        if method == 1
            R_evolve = IMR_solver(im*K_mat, R_vec, h)
        else
            R_evolve = exp_solver(im*K_mat, R_vec, h)
        end
        # R_evolve = backwards_euler(im*K_mat, R_vec, h)
        # R_evolve = euler(im*K_mat, R_vec, h)
        # R_evolve = constant_midpoint(im*K_mat, R_vec, h)
        # println("R_evolve")
        # display(R_evolve)
        #Convert R into tensor and multiply it with next tensor in the MPS and then replace
        R_inds = inds(R) 
        R_evolve = ITensor(R_evolve, R_inds)
        M[i - 1] = R_evolve*M[i - 1]
    end
    
    #Performs evolution on last site but without an QR decomposition as the MPS will be completely left-orthogonal.
    H_eff_1 = effective_Hamiltonian(H, M, 1)
    H_1_mat, M_1_vec = conversion(H_eff_1, M[1])
    # println("M2_vec pre-evolve")
    # display(M_N_vec)
    # println("H_mat 2")
    # display(H_N_mat)

    # M_N_evolve = exp(-im*H_N_mat*h)*M_N_vec
    if method == 1
        M_1_evolve = IMR_solver(-im*H_1_mat, M_1_vec, h)
    else
        M_1_evolve = exp_solver(-im*H_1_mat, M_1_vec, h)
    end
    # M_N_evolve = backwards_euler(-im*H_N_mat, M_N_vec, h)
    # M_N_evolve = euler(-im*H_N_mat, M_N_vec, h)
    # M_N_evolve = constant_midpoint(-im*H_N_mat, M_N_vec, h)
    # println("M_2 vec")
    # display(M_N_evolve) 
    M_1_inds = inds(M[1])
    M_1_evolve = ITensor(M_1_evolve, M_1_inds)
    M[1] .= M_1_evolve
    
    #Return completely evolved MPS
    return M 
end

# function fixed_point(H, M, h, maxiter = 100, tol = 1E-14, verbose = false)
#     M_inds = inds(M)
#     K_init = ITensor(eltype(M), M_inds)
#     for i in 1:maxiter 
#         K = H*(M + 0.5*h*K_init)
#         err = norm(K - K_init)
#         if verbose == true
#             println("Iteration: ", i)
#             println("Error: ", err)
#         end
#         K_init .= K
#         if err < tol 
#             break 
#         end
#     end
#     return K_init 
# end

# function IMR_MPS(H, M, h)
#     K = fixed_point(H, M, h, 100,1E-14, false)
#     M_update = M + 0.5*h*K 
#     return M_update 
# end


#Performs a single left-to-right sweep of an MPS using the 2 site tdvp, evolving forward one time step.
function lr_sweep_2site_err(H, M, h, cutoff, verbose)
    
    #Ensures orthogonalityu center is 1
    orthogonalize!(M, 1)
    N = length(M)
    error = 0.0
    for i in 1:N - 1 
        # println("Site $i")
        #Creates the 2-site Hamiltonian matrix and converts the 2 site M block (M[i]*M[i + 1]) to a vector
        H_eff_2 = effective_Hamiltonian_2site(H, M, i)
        M_block = M[i]*M[i + 1]
        H_mat_2, M_vec = conversion(H_eff_2, M_block)
        
        M_inds = inds(M_block)

        #Evolves the M block forward with the effective Hamiltonian and convert back into a tensor
        M_evolve = exp(-im*H_mat_2*h)*M_vec
        M_evolve = ITensor(M_evolve, M_inds)
        #println("(Row, Col): ", size(H_mat_2))
        

        #Performs SVD on the M block to get new left-orthogonal tensor
        if i == 1
            if N > 2
                bd = min(dim(M_inds[1]), dim(M_inds[2])*dim(M_inds[3]))

                U, S, V = svd(M_evolve, M_inds[1])
                S_diag = diag(Array(S, inds(S)))
                U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1], cutoff = cutoff)
                #Renormalize S_trunc 
                # S_trunc = S_trunc/norm(S_trunc)
                S_trunc_diag = diag(Array(S_trunc, inds(S_trunc)))
                if verbose == true
                    println("Singular Values: ", S_diag)
                    println(-1*sum(abs.(S_diag).^2 .* log.(S_diag.^2)))
                end
                # println("Max # of Singular Values: $bd ||| Removed Singular Values: ", setdiff(S_diag, S_trunc_diag))
                error += sqrt(sum(setdiff(S_diag, S_trunc_diag).^2))
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                U, S, V = svd(M_evolve, M_inds[1])
                S_diag = diag(Array(S, inds(S)))
                U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1], cutoff = cutoff)
                #Renormalize S_trunc 
                # S_trunc = S_trunc/norm(S_trunc)
                S_trunc_diag = diag(Array(S_trunc, inds(S_trunc)))
                if verbose == true
                    println("Singular Values: ", S_diag)
                    println(-1*sum(abs.(S_diag).^2 .* log.(S_diag.^2)))
                end
                # println("Max # of Singular Values: $bd ||| Removed Singular Values: ", setdiff(S_diag, S_trunc_diag))
                error += sqrt(sum(setdiff(S_diag, S_trunc_diag).^2))
            end
        else
            if i != N - 1
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3])*dim(M_inds[4]))
            else 
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3]))
            end
            U, S, V = svd(M_evolve, M_inds[1:2])
            S_diag = diag(Array(S, inds(S)))
            U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1:2], cutoff = cutoff)
            #Renormalize S_trunc 
            # S_trunc = S_trunc/norm(S_trunc)
            S_trunc_diag = diag(Array(S_trunc, inds(S_trunc)))
            if verbose == true
                println("Singular Values: ", S_diag)
                println(-1*sum(abs.(S_diag).^2 .* log.(S_diag.^2)))
            end
            # println("Max # of Singular Values: $bd ||| Removed Singular Values: ", setdiff(S_diag, S_trunc_diag))
            error += sqrt(sum(setdiff(S_diag, S_trunc_diag).^2))
            
        end

        #Set the i-th tensor in MPS to be U which is left-orthogonal
        M[i] = U_trunc
        M_n = S_trunc*V_trunc

        #If we're not on the last M block then evolve the (S*V) tensor with the effective Hamiltonian
        if i != N - 1
            M_n_inds = inds(M_n)
            H_eff = effective_Hamiltonian(H, M, i + 1)
            H_mat, M2_vec = conversion(H_eff, M_n)

            M2_evolve = exp(im*H_mat*h)*M2_vec
            M2_evolve = ITensor(M2_evolve, M_n_inds)
            #Set next tensor to evolved (S*V) tensor
            M[i + 1] = M2_evolve
        elseif i == N - 1
            #If on last site no evolution takes places
            M[i + 1] = S_trunc*V_trunc
        end
        
    end
    if verbose == true
        println("Error: ", error)
    end
    return M, error
end

function lr_sweep_2site_normal_err(H, M, h, cutoff, verbose)
    
    #Ensures orthogonalityu center is 1
    orthogonalize!(M, 1)
    N = length(M)
    error = 0.0
    for i in 1:N - 1 
        # println("Site $i")
        #Creates the 2-site Hamiltonian matrix and converts the 2 site M block (M[i]*M[i + 1]) to a vector
        H_eff_2 = effective_Hamiltonian_2site(H, M, i)
        M_block = M[i]*M[i + 1]
        H_mat_2, M_vec = conversion(H_eff_2, M_block)
        
        M_inds = inds(M_block)

        #Evolves the M block forward with the effective Hamiltonian and convert back into a tensor
        M_evolve = exp(-im*H_mat_2*h)*M_vec
        M_evolve = ITensor(M_evolve, M_inds)
        #println("(Row, Col): ", size(H_mat_2))
        

        #Performs SVD on the M block to get new left-orthogonal tensor
        if i == 1
            if N > 2
                bd = min(dim(M_inds[1]), dim(M_inds[2])*dim(M_inds[3]))

                U, S, V = svd(M_evolve, M_inds[1])
                S_diag = diag(Array(S, inds(S)))
                U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1], cutoff = cutoff)
                #Renormalize S_trunc 
                S_trunc = S_trunc/norm(S_trunc)
                S_trunc_diag = diag(Array(S_trunc, inds(S_trunc)))
                if verbose == true
                    println("Singular Values: ", S_diag)
                    println(-1*sum(abs.(S_diag).^2 .* log.(S_diag.^2)))
                end
                # println("Max # of Singular Values: $bd ||| Removed Singular Values: ", setdiff(S_diag, S_trunc_diag))
                error += sqrt(sum(setdiff(S_diag, S_trunc_diag).^2))
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                U, S, V = svd(M_evolve, M_inds[1])
                S_diag = diag(Array(S, inds(S)))
                U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1], cutoff = cutoff)
                #Renormalize S_trunc 
                S_trunc = S_trunc/norm(S_trunc)
                S_trunc_diag = diag(Array(S_trunc, inds(S_trunc)))
                if verbose == true
                    println("Singular Values: ", S_diag)
                    println(-1*sum(abs.(S_diag).^2 .* log.(S_diag.^2)))
                end
                # println("Max # of Singular Values: $bd ||| Removed Singular Values: ", setdiff(S_diag, S_trunc_diag))
                error += sqrt(sum(setdiff(S_diag, S_trunc_diag).^2))
            end
        else
            if i != N - 1
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3])*dim(M_inds[4]))
            else 
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3]))
            end
            U, S, V = svd(M_evolve, M_inds[1:2])
            S_diag = diag(Array(S, inds(S)))
            U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1:2], cutoff = cutoff)
            #Renormalize S_trunc 
            S_trunc = S_trunc/norm(S_trunc)
            S_trunc_diag = diag(Array(S_trunc, inds(S_trunc)))
            if verbose == true
                println("Singular Values: ", S_diag)
                println(-1*sum(abs.(S_diag).^2 .* log.(S_diag.^2)))
            end
            # println("Max # of Singular Values: $bd ||| Removed Singular Values: ", setdiff(S_diag, S_trunc_diag))
            error += sqrt(sum(setdiff(S_diag, S_trunc_diag).^2))
            
        end

        #Set the i-th tensor in MPS to be U which is left-orthogonal
        M[i] = U_trunc
        M_n = S_trunc*V_trunc

        #If we're not on the last M block then evolve the (S*V) tensor with the effective Hamiltonian
        if i != N - 1
            M_n_inds = inds(M_n)
            H_eff = effective_Hamiltonian(H, M, i + 1)
            H_mat, M2_vec = conversion(H_eff, M_n)

            M2_evolve = exp(im*H_mat*h)*M2_vec
            M2_evolve = ITensor(M2_evolve, M_n_inds)
            #Set next tensor to evolved (S*V) tensor
            M[i + 1] = M2_evolve
        elseif i == N - 1
            #If on last site no evolution takes places
            M[i + 1] = S_trunc*V_trunc
        end
        
    end
    if verbose == true
        println("Error: ", error)
    end
    return M, error
end

#Performs a single left-to-right sweep of an MPS using the 2 site tdvp, evolving forward one time step.
function lr_sweep_2site(H, M, h, cutoff)
    
    #Ensures orthogonalityu center is 1
    orthogonalize!(M, 1)
    N = length(M)
    for i in 1:N - 1 
        # println("Site $i")
        #Creates the 2-site Hamiltonian matrix and converts the 2 site M block (M[i]*M[i + 1]) to a vector
        H_eff_2 = effective_Hamiltonian_2site(H, M, i)
        M_block = M[i]*M[i + 1]
        H_mat_2, M_vec = conversion(H_eff_2, M_block)
        M_inds = inds(M_block)

        #Evolves the M block forward with the effective Hamiltonian and convert back into a tensor
        M_evolve = exp(-im*H_mat_2*h)*M_vec
        M_evolve = ITensor(M_evolve, M_inds)
        #println("(Row, Col): ", size(H_mat_2))
                

        #Performs SVD on the M block to get new left-orthogonal tensor
        if i == 1
            if N > 2
                bd = min(dim(M_inds[1]), dim(M_inds[2])*dim(M_inds[3]))
                U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1], cutoff = cutoff)
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1], cutoff = cutoff)
            end
        else
            if i != N - 1
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3])*dim(M_inds[4]))
            else 
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3]))
            end
            U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1:2], cutoff = cutoff)
        end
        #Set the i-th tensor in MPS to be U which is left-orthogonal
        M[i] = U_trunc
        M_n = S_trunc*V_trunc

        #If we're not on the last M block then evolve the (S*V) tensor with the effective Hamiltonian
        if i != N - 1
            M_n_inds = inds(M_n)
            H_eff = effective_Hamiltonian(H, M, i + 1)
            H_mat, M2_vec = conversion(H_eff, M_n)

            M2_evolve = exp(im*H_mat*h)*M2_vec
            M2_evolve = ITensor(M2_evolve, M_n_inds)
            # M2_evolve = IMR_MPS(H_eff, M[i+1], h)
            #Set next tensor to evolved (S*V) tensor
            M[i + 1] = M2_evolve
        elseif i == N - 1
            #If on last site no evolution takes places
            M[i + 1] = S_trunc*V_trunc
        end
    end
    return M
end

function lr_sweep_2site_normal(H, M, h, cutoff)
    
    #Ensures orthogonalityu center is 1
    orthogonalize!(M, 1)
    N = length(M)
    error = 0.0
    for i in 1:N - 1 
        # println("Site $i")
        #Creates the 2-site Hamiltonian matrix and converts the 2 site M block (M[i]*M[i + 1]) to a vector
        H_eff_2 = effective_Hamiltonian_2site(H, M, i)
        M_block = M[i]*M[i + 1]
        H_mat_2, M_vec = conversion(H_eff_2, M_block)
        
        M_inds = inds(M_block)

        #Evolves the M block forward with the effective Hamiltonian and convert back into a tensor
        M_evolve = exp(-im*H_mat_2*h)*M_vec
        M_evolve = ITensor(M_evolve, M_inds)
        #println("(Row, Col): ", size(H_mat_2))
        

        #Performs SVD on the M block to get new left-orthogonal tensor
        if i == 1
            if N > 2
                bd = min(dim(M_inds[1]), dim(M_inds[2])*dim(M_inds[3]))
                U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1], cutoff = cutoff)
                #Renormalize S_trunc 
                S_trunc = S_trunc/norm(S_trunc)
                S_trunc_diag = diag(Array(S_trunc, inds(S_trunc)))
            elseif N == 2 
                bd = min(dim(M_inds[1]), dim(M_inds[2]))
                U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1], cutoff = cutoff)
                #Renormalize S_trunc 
                S_trunc = S_trunc/norm(S_trunc)
            end
        else
            if i != N - 1
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3])*dim(M_inds[4]))
            else 
                bd = min(dim(M_inds[1])*dim(M_inds[2]),dim(M_inds[3]))
            end
            S_diag = diag(Array(S, inds(S)))
            U_trunc, S_trunc, V_trunc = svd(M_evolve, M_inds[1:2], cutoff = cutoff)
            #Renormalize S_trunc 
            S_trunc = S_trunc/norm(S_trunc)
            
        end

        #Set the i-th tensor in MPS to be U which is left-orthogonal
        M[i] = U_trunc
        M_n = S_trunc*V_trunc

        #If we're not on the last M block then evolve the (S*V) tensor with the effective Hamiltonian
        if i != N - 1
            M_n_inds = inds(M_n)
            H_eff = effective_Hamiltonian(H, M, i + 1)
            H_mat, M2_vec = conversion(H_eff, M_n)

            M2_evolve = exp(im*H_mat*h)*M2_vec
            M2_evolve = ITensor(M2_evolve, M_n_inds)
            #Set next tensor to evolved (S*V) tensor
            M[i + 1] = M2_evolve
        elseif i == N - 1
            #If on last site no evolution takes places
            M[i + 1] = S_trunc*V_trunc
        end
        
    end
    return M
end

function tdvp_constant(H, init, t0, T, steps, method, verbose = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    storage_arr = zeros(ComplexF64, (d, steps + 1))
    storage_arr[:,1] = reconstruct_arr_v2(init_copy)
    
    #Run time stepper
    for i = 1:steps
        
        if verbose == true
            println("Step: ", i)
        end
        init_copy = lr_sweep(H, init_copy, t0, h, method)
        t0 += h
        storage_arr[:,i + 1] = reconstruct_arr_v2(init_copy)
    end
    
    #Return evolved MPS, as well as state data at each time step
    return init_copy, storage_arr
    # , storage_arr
end

function tdvp_constant2(H, init, t0, T, steps, method, verbose = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    storage_arr = zeros(ComplexF64, (steps + 1, d))
    storage_arr[1,:] = reconstruct_arr_v2(init_copy)
    
    #Run time stepper
    for i = 1:steps
        
        if verbose == true
            println("Step: ", i)
        end
        init_copy = rl_sweep(H, init_copy, t0, h, method)
        t0 += h
        storage_arr[i + 1,:] = reconstruct_arr_v2(init_copy)
    end
    
    #Return evolved MPS, as well as state data at each time step
    return init_copy, storage_arr
end

function tdvp_constant3(H, init, t0, T, steps, method, verbose = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    storage_arr = zeros(ComplexF64, (steps + 1, d))
    storage_arr[1,:] = reconstruct_arr_v2(init_copy)
    
    #Run time stepper
    for i = 1:steps
        
        if verbose == true
            println("Step: ", i)
        end
        init_copy = lr_sweep(H, init_copy, t0, h/2, method)
        init_copy_step_2 = rl_sweep(H, init_copy, t0, h/2, method) 
        t0 += h
        storage_arr[i + 1,:] = reconstruct_arr_v2(init_copy)
    end
    
    #Return evolved MPS, as well as state data at each time step
    return init_copy, storage_arr
end

function tdvp2_constant(H, init, t0, T, steps, cutoff, normalize = false, verbose = false)
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create arry to store evolved state
    storage_arr = zeros(ComplexF64, (d, steps + 1))
    storage_arr[:,1] = vectorize_mps(init_copy; order = "reverse")
    # truncation_err = zeros(steps + 1)
    # truncation_err[1] = 0.0
    link_dim = zeros(N - 1, steps + 1)
    link_dim[:,1] = linkdims(init_copy)
    # push!(link_dim, prod(linkdims(init_copy)))
    #Run time stepper
    @showprogress 1 "TDVP2" for i = 1:steps
        if normalize == false
            init_copy = lr_sweep_2site(H, init_copy, h, cutoff)
        elseif normalize == true 
            init_copy = lr_sweep_2site_normal(H, init_copy, h, cutoff)
        end
        # println(length(init_copy))
        # truncation_err[i + 1] = err
        # if verbose == true
        #     println("Step: ", i)
        #     println("Link Dimensions at step $i: ", linkdims(init_copy))
        # end
        # display(reconstruct_arr_v2(init_copy))
        storage_arr[:,i + 1] = vectorize_mps(init_copy; order = "reverse")
        # push!(link_dim, prod(linkdims(init_copy)))
        link_dim[:,i + 1] = linkdims(init_copy)
    end
    return init_copy, storage_arr, link_dim
    # return init_copy, truncation_err, link_dim
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

function tdvp2_constant_magnet(H, init, t0, T, steps, cutoff, magnet_site, verbose = false)
    
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    magnet_history = zeros(steps + 1)
    # truncation_err = zeros(steps + 1)
    # truncation_err[1] = 0.0
    link_dim = zeros(N - 1, steps + 1)
    link_dim[:,1] = linkdims(init_copy)
    m_mpo = magnetization_MPO(sites, magnet_site)
    magnet_history[1] = expect(init, [1.0 0.0; 0.0 -1.0];sites = magnet_site)
    #Run time stepper
    @showprogress 1 "TDVP2" for i = 1:steps
        
        init_copy = lr_sweep_2site(H, init_copy, h, cutoff)
        # println(length(init_copy))
        # truncation_err[i + 1] = err
        if verbose == true
            println("Step: ", i)
            println("Link Dimensions at step $i: ", linkdims(init_copy))
        end
        magnet_history[i + 1] = expect(init_copy, [1.0 0.0; 0.0 -1.0];sites = magnet_site)
        link_dim[:,i + 1] = linkdims(init_copy)
    end
    return init_copy, magnet_history, link_dim
end

function tdvp_time(H, init, t0, T, steps, bcparams, h_list = [], verbose = false)
    if length(h_list) > 0
        steps = length(h_list)
    end
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    storage_arr = zeros(ComplexF64, (steps + 1, d))
    storage_arr[1,:] = reconstruct_arr_v2(init_copy)

    #Run time stepper
    for i = 1:steps
        if verbose == true
            println("Step: ", i)
        end
        if length(h_list) > 0
            t0 += h_list[i]
            H = update_H(H, bcparams, t0)
            init_copy = lr_sweep(H, init_copy, t0, h_list[i])
        elseif length(h_list) == 0
            t0 += h
            H= update_H(H, bcparams, t0)
            init_copy = lr_sweep(H, init_copy, t0, h)
        end
        storage_arr[i + 1,:] = reconstruct_arr_v2(init_copy)
    end
    
    #Return evolved MPS, as well as state data at each time step
    return init_copy, storage_arr
end

function tdvp_time_IMR(H, init, t0, T, steps, bcparams, h_list = [], verbose = false)
    if length(h_list) > 0
        steps = length(h_list)
    end
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    storage_arr = zeros(ComplexF64, (steps + 1, d))
    storage_arr[1,:] = reconstruct_arr_v2(init_copy)

    #Run time stepper
    for i = 1:steps
        if verbose == true
            println("Step: ", i)
        end
        if length(h_list) > 0
            init_copy = lr_sweep(H, init_copy, t0, h_list[i])
            t0 += h_list[i]
            H = update_H(H, bcparams, t0)
        elseif length(h_list) == 0
            println("t0: ", t0)
            println("t0 + h/2: ", t0 + h/2)
            init_copy = lr_sweep(H, init_copy, t0, h)
            t0 += h
            H = update_H(H, bcparams, t0)
        end
        storage_arr[i + 1,:] = reconstruct_arr_v2(init_copy)
    end
    
    #Return evolved MPS, as well as state data at each time step
    return init_copy, storage_arr
end

function tdvp_time(H, init, t0, T, steps, bcparams, method, h_list = [], verbose = false)
    if length(h_list) > 0
        steps = length(h_list)
    end
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    storage_arr = zeros(ComplexF64, (steps + 1, d))
    storage_arr[1,:] = reconstruct_arr_v2(init_copy)

    #Run time stepper
    for i = 1:steps
        if verbose == true
            println("Step: ", i)
        end
        if length(h_list) > 0
            init_copy = lr_sweep(H, init_copy, t0, h_list[i])
            t0 += h_list[i]
            H = update_H(H, bcparams, t0)
        elseif length(h_list) == 0
            if verbose == true 
                println("t0: ", t0)
                println("t0 + h/2: ", t0 + h/2)
            end
            # H = update_H(H, bcparams, t0)
            init_copy = lr_sweep(H, init_copy, t0, h, method)
            if method == 1
                H = update_H(H, bcparams, t0 + h/2)
            else
                H = update_H(H, bcparams, t0)
            end
            t0 += h
            
        end
        storage_arr[i + 1,:] = reconstruct_arr_v2(init_copy)
    end
    
    #Return evolved MPS, as well as state data at each time step
    return init_copy, storage_arr
end

function tdvp2_time(H, init, t0, T, steps, bcparams, cutoff, h_list = [], verbose = false)
    if length(h_list) > 0
        steps = length(h_list)
    end
    N = length(init)
    orthogonalize!(init, 1)
    sites = siteinds(init)
    init_copy = copy(init)
    d = prod(dim(sites))
    println("# of steps: $steps")
    #Get step size
    h = (T - t0)/steps
    #Create array to store evolved state
    storage_arr = zeros(ComplexF64, (steps + 1, d))
    storage_arr[1,:] = reconstruct_arr_v2(init_copy)
    bond_dim_list = []
    push!(bond_dim_list, prod(linkdims(init_copy)))
    #Run time stepper
    for i = 1:steps

        if length(h_list) > 0
            t0 += h_list[i]
            H = update_H(H, bcparams, t0)
            init_copy, _ = lr_sweep_2site(H, init_copy, h_list[i], cutoff, verbose)
        elseif length(h_list) == 0
            init_copy, _ = lr_sweep_2site(H, init_copy, h, cutoff, verbose)
            H = update_H(H, bcparams, t0)
            t0 += h
            
        end
        if verbose == true
            println("Step: ", i)
            println("Linkdim: ", linkdims(init_copy))
        end
        
        storage_arr[i + 1,:] = reconstruct_arr_v2(init_copy)
        # storage_arr[i + 1,:] = zeros(ComplexF64, d)
        push!(bond_dim_list, prod(linkdims(init_copy)))
        
    end
    #Return evolved MPS, as well as state data at each time step
    return init_copy, storage_arr, bond_dim_list
end