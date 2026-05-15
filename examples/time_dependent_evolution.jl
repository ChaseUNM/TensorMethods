using ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles, Plots, BenchmarkTools
using TensorMethods

# -------------------------
# System setup
# -------------------------

# Number of qudits / qubits
N = 10

# Local Hilbert space dimension for each subsystem
nlevels = fill(2, N)

# Standard qubit site indices (currently unused)
# sites = siteinds("Qubit", N)

# Build qudit site indices with specified local dimensions
sites = qudit_siteinds(N, nlevels)

# Bare 0-1 transition frequencies for each qubit [GHz], converted to angular frequency
freq01_all = [5.18, 5.12, 5.06, 5.0, 4.94, 4.88, 4.82, 4.76, 4.7, 4.74] .* 2pi

# Self-Kerr nonlinearities (set to zero here)
self_kerr = zeros(N)

# ZZ coupling matrix (set to zero here)
zz = zeros(N, N)

# Nearest-neighbor coupling strength
J = 5E-3 * 2pi

# Coupling matrix Jkl
Jkl = zeros(N, N)
for i = 2:N 
    Jkl[i - 1, i] = J 
end

# -------------------------
# Time evolution parameters
# -------------------------

t0 = 0.0
T = 40.0 				# [ns] Pulse duration

# Number of spline segments used for control pulse parameterization
splines = 6

# Number of time steps used in evolution
steps = 1000

# Reverse frequency ordering and truncate to N qubits
freq01 = reverse(freq01_all)[1:N]

# Average frequency used for rotating frame
favg = sum(freq01) / N 

# Rotating frame frequencies (all set equal to the average)
rot_freq = ones(N) .* favg 

# -------------------------
# Load pulse parameters
# -------------------------

# Path to optimized pulse spline coefficients
datafile = joinpath(@__DIR__, "spline_params", "params_10_coupled.dat")

# Read spline coefficient data from file, stored in same order as quandary output
pcof = vec(readdlm(datafile))

# -------------------------
# Carrier frequency setup
# -------------------------

# Each qubit has a list of carrier frequencies used in the pulse expansion
carrier_frequency_list = Vector{Vector{Float64}}(undef, N)

# These carrier frequencies are calculated in quandary and just copied over here
carrier_frequency_list[1] = [-0.17999999999999972, -0.21999999999999975] .* 2pi
carrier_frequency_list[2] = [-0.17999999999999972, -0.21999999999999975, -0.16000000000000014] .* 2pi
carrier_frequency_list[3] = [-0.21999999999999975, -0.16000000000000014, -0.09999999999999964] .* 2pi
carrier_frequency_list[4] = [-0.16000000000000014, -0.09999999999999964, -0.040000000000000036] .* 2pi
carrier_frequency_list[5] = [-0.09999999999999964, -0.040000000000000036, 0.020000000000000462] .* 2pi
carrier_frequency_list[6] = [-0.040000000000000036, 0.020000000000000462, 0.08000000000000007] .* 2pi
carrier_frequency_list[7] = [0.020000000000000462, 0.08000000000000007, 0.13999999999999968] .* 2pi
carrier_frequency_list[8] = [0.08000000000000007, 0.13999999999999968, 0.20000000000000018] .* 2pi 
carrier_frequency_list[9] = [0.13999999999999968, 0.20000000000000018, 0.2599999999999998] .* 2pi 
carrier_frequency_list[10] = [0.20000000000000018, 0.2599999999999998] .* 2pi

# Build pulse boundary/carrier parameter object
bc_params = bcparams(T, splines, carrier_frequency_list, pcof)

# -------------------------
# Initial state and Hamiltonians
# -------------------------

# Initial product state |0,0,...,0⟩
q_state = fill(0, N)

# Construct initial MPS
init_MPS = init_separable(sites, q_state)

# Construct drift Hamiltonian as MPO
H_s = drift_MPO(N, sites, freq01, rot_freq, self_kerr, zz, Jkl)

#simulate with TDVP2 for MPS
"""
Evolve an MPS under a Hamiltonian MPO using a two-site TDVP integrator.

Performs time evolution of the matrix product state `init` under the Hamiltonian `H`
from time `t0` to `T` using `steps` discrete time steps. The integrator works on
two-site updates and supports optional SVD truncation and bond-dimension control.
A symmetric second-order (Strang) splitting is used by default.

Arguments
- H::MPO
    The Hamiltonian as an MPO that generates the time evolution (may be time-independent).
- init::MPS
    The initial MPS to be evolved. This state is modified or copied depending on implementation.
- t0::Real
    Initial time of the evolution.
- T::Real
    Final time of the evolution.
- steps::Int64
    Number of time steps. The step size used is dt = (T - t0) / steps.
- bc_params::bcparams
    Boundary-condition parameters (type depends on implementation) controlling edge terms/closures.

Keyword arguments
- cutoff::Union{Float64, Nothing}=nothing
    SVD truncation tolerance: singular values smaller than `cutoff` are discarded.
    If `nothing`, no truncation by tolerance is performed.
- maxdim::Union{Int64, Nothing}=nothing
    Maximum allowed bond dimension during truncation. If `nothing`, bond dimensions are
    not explicitly limited (only controlled by `cutoff`).
- magnet::Bool=false
    If true, compute and record site magnetizations (or a user-defined local observable)
    at each saved time point.
- energy::Bool=false
    If true, compute and record the energy ⟨H⟩ at each saved time point.
- verbose::Bool=false
    Print progress and diagnostic information during the evolution.
- normalize::Bool=false
    If true, renormalize the MPS (to unit norm) after each time step/update.
- strang::Bool=true
    Use Strang (second-order symmetric) splitting for the integrator when true.
    If false, a first-order integrator is used.

Returns
A tuple typically containing:
- evolved MPS at final time,
- bond-dimension history,
- magnetization history (if requested),
- energy history (if requested),
- truncation error(s) or other diagnostics.

- verbose::Bool = false
  If true, print progress information and diagnostics during the time evolution to assist with monitoring and debugging.

"""
ans_mps_tdvp,bd_history_tdvp,magnet_history,energy_history,trunc_err = tdvp2(H_s, init_MPS, t0, T, Int64(steps/2), bc_params; cutoff = 1E-15^2, magnet = true, verbose = false, strang = true) 


#simulate with BUG for MPS
"""Evolve an MPS under a Hamiltonian MPO using the BUG-MPS integrator.

mps_bug(H, bc_params, M, t0, T, steps; center=nothing, cutoff=nothing, maxdim=nothing, magnet=false, energy=false, verbose=false)

Evolve a matrix product state (MPS) in time under a matrix product operator (MPO) Hamiltonian,
sampling diagnostics along the way.

Arguments
- H::MPO
    The Hamiltonian expressed as an MPO that generates the time evolution.
- bc_params::bcparams
    Boundary-condition parameters or other auxiliary data required by the evolution routine.
- M::MPS
    The initial state provided as an MPS. This state will be evolved from time `t0` to `T`.
- t0::Real
    Initial time of the evolution.
- T::Real
    Final time of the evolution.
- steps::Int64
    Number of time steps to take between `t0` and `T`. The times at which diagnostics are
    sampled are determined by this parameter.

Keyword arguments
- center::Union{Nothing, Int64} = nothing
    Optional site index to enforce or use as the orthogonality center of the MPS. When `nothing`,
    the routine may choose or preserve the current center.
- cutoff::Union{Nothing, Float64} = nothing
    Singular value truncation threshold. If provided, singular values below `cutoff` are discarded
    when truncating bond dimensions during the update steps. A `nothing` value disables truncation
    by threshold.
- maxdim::Union{Nothing, Float64} = nothing
    Maximum allowed bond dimension. If provided, bond dimensions are capped at `maxdim` during
    truncation. A `nothing` value disables a strict cap.
- magnet::Bool = false
    If true, compute and record magnetization (or other local observable specified by the
    implementation) at each sampled time.
- energy::Bool = false
    If true, compute and record the expectation value of the Hamiltonian (energy) at each sampled time.
- verbose::Bool = false
    Toggle verbose logging or progress output to aid debugging or monitor the evolution.

Returns
A tuple typically containing:
- evolved MPS at final time,
- bond-dimension history,
- magnetization history (if requested),
- energy history (if requested),lementation-specific scalars

"""
ans_mps_bug, bd_history_bug, magnet_bug, energy_bug = mps_bug(H_s, bc_params, init_MPS, t0, T, steps; cutoff = 1E-15^2, magnet = true, verbose = false)