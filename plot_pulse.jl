using LinearAlgebra, Plots, DelimitedFiles

include("hamiltonian(8-4).jl")

N = 3
control0 = readdlm("./pulse_data/control0.dat")
control1 = readdlm("./pulse_data/control1.dat")
control2 = readdlm("./pulse_data/control2.dat")

t = control0[2:end,1]
p0 = control0[2:end,2]*(500/pi)
q0 = control0[2:end,3]*(500/pi)
p1 = control1[2:end,2]*(500/pi)
q1 = control1[2:end,3]*(500/pi)
p2 = control2[2:end,2]*(500/pi)
q2 = control2[2:end,3]*(500/pi)

pulse_1 = plot(t, [p0, q0], labels = ["p(t)" "q(t)"], title = "Qudit 1")
pulse_2 = plot(t, [p1, q1], labels = ["p(t)" "q(t)"], title = "Qudit 2")
pulse_3 = plot(t, [p2, q2], labels = ["p(t)" "q(t)"], title = "Qudit 3")
total_pulse = plot(pulse_1, pulse_2, pulse_3, layout = (3, 1))

#Now attempt to recreate using bcparams
#Test first if answer remains the same across bcparams and bcparams2
t0 = 0.0
T = 300.0
nsplines = 32 
pars = vec(readdlm("params_toffoli.dat"))
# pars = rand(2*2*nsplines*N)
om = Vector{Vector{Float64}}(undef, N)
# om2 = zeros(3, 2)
# om[1] = [1.0,2.0]
# om[2] = [3.0,4.0]
# om[3] = [5.0,6.0]
# om2[1,:] = [1.0,2.0]
# om2[2,:] = [3.0,4.0]
# om2[3,:] = [5.0,6.0]
om[1] = [0.0, -0.20012496]*2*pi
om[2] = [0.20012496, 0.0, -0.20012496]*2*pi
om[3] = [0.20012496, 0.0]*2*pi
# om[1] = [0.0, -0.20012496]
# om[2] = [0.20012496, 0.0, -0.20012496]
# om[3] = [0.20012496, 0.0]

bc_params2 = bcparams2((T - t0), nsplines, om, pars)
ans1 = bcarrier2(0.0, bc_params2, 0)
# println(ans1)
# println(p0[1])


# for i in 1:5
#     println("New")
#     t1 = bcarrier2(1.0, bc_params2, i)
#     println(t1)
#     println("Old")
#     t2 = bcarrier2(1.0, bc_params1, i)
#     println(t2)
#     println("Ratio: t1/t2: ", t1/t2)
# end


