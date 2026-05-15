using ITensors, ITensorMPS, LinearAlgebra, DelimitedFiles, Plots, BenchmarkTools
using TensorMethods

# time grid
t0 = 0.0           # initial time
T = 5.0            # final time
steps = 250        # number of equal time steps (dt = (T - t0)/steps)

# system definition
N = 10                             # number of lattice sites / qubits
sites = siteinds("Qubit", N)       # site index objects for ITensors/ITensorMPS

# Hamiltonian parameters
J = 1.0
g = 0.5                            # transverse-field or coupling parameter (example)
H = xxx_mpo_scaled(N, sites, J, g) # construct an XXX-model MPO; args: N, sites, coupling, field

# initial product state
q_state = Int64.(fill(0, N))       # local quantum numbers (all spins in |0⟩ state)
init_MPS = init_separable(sites, q_state) # build separable/product MPS from local states


#simulate with TDVP2 for MPS
"""
tdvp2_constant(H::MPO, init::MPS, t0::Real, T::Real, steps::Int64;
               cutoff::Union{Float64,Nothing}=nothing,
               maxdim::Union{Int64,Nothing}=nothing,
               magnet::Bool=false,
               energy::Bool=false,
               verbose::Bool=false,
               normalize::Bool=false,
               strang::Bool=true)

Perform two-site TDVP time evolution with a constant time-step.

Arguments
- H::MPO
  Hamiltonian (MPO) driving the evolution. Must be compatible with `init` (same sites/ordering).

- init::MPS
  Initial matrix product state to be evolved.

- t0::Real
  Initial time.

- T::Real
  Final time.

- steps::Int64
  Number of equal time steps between `t0` and `T` (dt = (T - t0) / steps).

Keyword arguments
- cutoff::Union{Float64,Nothing} = nothing
  Truncation tolerance for SVD. If `nothing`, no explicit cutoff is applied (implementation-dependent default).
  Typical call sites may pass a squared tolerance; follow the convention used by the implementation.

- maxdim::Union{Int64,Nothing} = nothing
  Maximum allowed bond dimension during truncation. If `nothing`, no explicit cap beyond algorithmic limits.

- magnet::Bool = false
  If true, compute and return local magnetization history during the evolution.

- energy::Bool = false
  If true, compute and return energy expectation values during the evolution.

- verbose::Bool = false
  If true, emit progress and diagnostic information.

- normalize::Bool = false
  If true, re-normalize the MPS at appropriate steps to control norm drift.

- strang::Bool = true
  If true, use Strang (symmetric) splitting ordering for two-site updates; otherwise use a non-symmetric (Lie-Trotter) ordering.

Returns
A tuple typically containing:
- evolved MPS at final time,
- bond-dimension history,
- magnetization history (if requested),
- energy history (if requested),
- truncation error(s) or other diagnostics.
"""
ans_mps_tdvp,bd_history_tdvp,magnet_history,energy_history,trunc_err = tdvp2_constant(H, init_MPS, t0, T, Int64(steps/2); cutoff = 1E-15^2, magnet = true, energy = false, verbose = false, strang = true) 

# simulate with BUG for MPS
"""
mps_bug_constant(H::MPO, M::MPS, t0::Real, T::Real, steps::Int64;
                 center::Union{Nothing,Int64}=nothing,
                 cutoff::Union{Nothing,Float64}=nothing,
                 maxdim::Union{Nothing,Int64}=nothing,
                 magnet::Bool=false,
                 energy::Bool=false,
                 verbose::Bool=false)

Evolve an MPS in time under an MPO using the mps_bug_constant routine.

Arguments
- H::MPO
  The Hamiltonian (or more generally the time-evolution generator) represented as a Matrix Product Operator.
  Must be compatible with the lattice and local physical dimensions of M.

- M::MPS
  The initial Matrix Product State to be evolved. Should match H in number of sites and physical dimensions.

- t0::Real
  The initial time of the evolution.

- T::Real
  The final time to which the state should be evolved.

- steps::Int64
  The number of equal time steps between t0 and T. The time step used is (T - t0) / steps.

Keyword arguments
- center::Union{Nothing,Int64} = nothing
  Optional orthogonality-center site index for the MPS. If given, the algorithm will treat this site as the canonical center.
  If nothing, the routine will set the midpoint as the center by default.

- cutoff::Union{Nothing,Float64} = nothing
  Singular-value cutoff used when truncating bond dimensions during evolution.
  Singular values below this threshold are discarded. If nothing, truncation is disabled or a library default is applied.
  Typical values are very small (e.g. 1e-15) when high accuracy is required.

- maxdim::Union{Nothing,Int64} = nothing
  Maximum allowed bond dimension during truncation. If nothing, no explicit cap is imposed beyond algorithmic or memory limits.
  Use this to limit memory/compute when truncation alone is insufficient.

- magnet::Bool = false
  If true, compute and return magnetization (local spin/particle expectation values) at each time step as part of diagnostics.

- energy::Bool = false
  If true, compute and return the energy expectation value at each time step.

- verbose::Bool = false
  If true, print progress information and diagnostics during the time evolution to assist with monitoring and debugging.

"""
ans_mps_bug, bd_history_bug, magnet_bug, energy_bug = mps_bug_constant(H, init_MPS, t0, T, steps; cutoff = 1E-15^2, magnet = true, energy = false, verbose = false)