using ITensors, LinearAlgebra, Random

Random.seed!(42)
#contract in place
i = Index(5;tags = "i")
j = Index(5;tags = "j")
k = Index(5;tags = "k")
l = Index(5;tags = "l")

u1 = Index(10;tags = "u1")
u2 = Index(10;tags = "u2")
u3 = Index(10;tags = "u3")

core = random_itensor(i,j,k)
A = random_itensor(i,u1)
B = random_itensor(j,u2)
C = random_itensor(k,u3)

C1 = ITensor(u1, j, k)

println("Single TTM: ")
println("--------------------------------------")
println("Using *")
@btime begin 
    A1 = core*A
end
println()
println("Using contract!")
@btime begin 
    ITensors.contract!(C1, core, A)
end
alloc1 = ITensor(u1,j,k)
alloc2 = ITensor(u1,u2,k)
alloc3 = ITensor(u1,u2,u3)




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

# function Multi_TTM!(core::ITensor, factors::Vector{ITensor}, alloc_tensors::Vector{ITensor}, i::Int=1)
#     N = length(factors)
#     if i > N
#         return
#     elseif i == 1
#         TTM_alloc!(core, factors[i], alloc_tensors[i])
#     else
#         TTM_alloc!(alloc_tensors[i - 1], factors[i], alloc_tensors[i])
#     end
#     Multi_TTM!(core, factors, alloc_tensors, i + 1)
# end



factors = [A, B, C]
alloc_tensors = preallocate_itensor(core, factors)
println()
println("Multi-TTM")
println("-------------------------------------------")
println("Using *")
@btime begin 
ans_1 = reconstruct(core, factors)
end
println()
println("Using contract!")
@btime begin
Multi_TTM!(core, factors, alloc_tensors)
end

# @btime begin
# ans1 = core*A
# end

# @btime begin 
# TTM_alloc!(core, A, alloc1)
# end


