module TensorMethods

# ============================================================================
# Standard Library
# ============================================================================

using LinearAlgebra
using SparseArrays

# ============================================================================
# External Packages
# ============================================================================

using CPUTime
using ITensors
using ITensorMPS
using LaTeXStrings
using Plots
using ProgressMeter

# ============================================================================
# Source Files
# ============================================================================

include("hamiltonian_constructors.jl")
include("tdvp_algorithms.jl")
include("BUG_MPS.jl")
include("BUG_tucker.jl")
# include("...")

# ============================================================================
# Exports
# ============================================================================

# Hamiltonian construction
export H_drift_mat,
       H_sys_rot,
       H_total_mat,
       drift_MPO,
       long_range_dissipative_ising,
       s_op_general,
       s_op_reverse,
       updateH_mat!,
       update_MPO!,
       create_dipole_matrix,
       dipole_value,
       Drift_Hamiltonian,
       make_couplings_QEC,
       initial_states, 
       create_CNOT

# TDVP algorithms
export tdvp,
       tdvp2,
       tdvp2_constant,
       tdvp_constant_adjoint,
       tdvp2_changing_dipole,
       TDVP1_style_truncation_in_move_orthogonal,
       TDVP1_style_truncation_out_move_orthogonal

# BUG algorithms
export bug_integrator_mat,
       bug_integrator_mat_ra,
       mps_bug,
       mps_bug_constant

# Tensor utilities
export contract_left,
       contract_right,
       equal_separable,
       exp_solver,
       get_site_and_links,
       init_separable,
       is_left_orthogonal,
       is_right_orthogonal,
       LLSV,
       matricization,
       max_bond_dimension,
       Multi_TTM_recursive,
       ortho_properties,
       qudit_siteinds,
       RLSV,
       tucker,
       tucker_separable,
       TT_IMR_1site_new,
       vectorize_mps,
       siteinds_tensor, 
       linkinds_tensor,
       MPS_subset,
       build_qec_groups,
       contains,
       remove_dim1_links

# Pulse utilities
export bcparams,
       bcarrier2

# Operators
export ops_xxx,
       ops_xxx_scaled,
       xxx,
       xxx_mpo,
       xxx_mpo_scaled,
       xxx_scaled

# Diagnostics
export count_MPS,
       count_MPS_history,
       count_tucker,
       count_tucker_history

# Quantum Error Correction
export create_initial,
       mps_element,
       QEC_circuit,
       QEC_initial_states,
       QEC_operator

end