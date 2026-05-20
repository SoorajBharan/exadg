/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  ______________________________________________________________________
 */

#include "coupler.h"
#include <exadg/operators/wall_modelling/coupler.h>

namespace ExaDG
{
template<int dim, typename Number>
WallDGCoupler<dim, Number>::WallDGCoupler()
  : matrix_free_global(nullptr)
  , dof_index_global_u(0)
{}

template <int dim, typename Number>
void
WallDGCoupler<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &          matrix_free_global_in,
  unsigned int                                     dof_index_global_u_in,
  std::shared_ptr<FunctionEnrichment<dim, Number>> function_enrichment_in)
{
  matrix_free_global  = &matrix_free_global_in;
  dof_index_global_u  = dof_index_global_u_in;
  function_enrichment = function_enrichment_in;

  active_fe_index = function_enrichment->get_active_fe_index();
}

/*
 * Precompute Schur Complement Matrices
 */
template <int dim, typename Number>
void
WallDGCoupler<dim, Number>::precompute_schur_matrices()
{
  function_enrichment->wall_distance.update_ghost_values();
  function_enrichment->friction_velocity.update_ghost_values();

  auto const & mf_en = *(function_enrichment->get_matrix_free());
  const unsigned int n_batches = mf_en.n_cell_batches();
  cell_schur_data.resize(n_batches);

  unsigned int idx_shadow = function_enrichment->get_dof_index_shadow_vector();
  unsigned int idx_en     = function_enrichment->get_dof_index_en();
  unsigned int idx_cg     = function_enrichment->get_dof_index_cg_scalar();
  unsigned int idx_en_scalar = function_enrichment->get_dof_index_en_scalar();
  unsigned int quad_idx   = function_enrichment->get_quad_index();

  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_dg(mf_en, idx_shadow, quad_idx, 0, active_fe_index);
  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_en(mf_en, idx_en, quad_idx, 0, active_fe_index);
  dealii::FEEvaluation<dim, -1, 0, 1, Number>   y_eval(mf_en, idx_en_scalar, quad_idx, 0, active_fe_index);
  dealii::FEEvaluation<dim, -1, 0, 1, Number>   utau_eval(mf_en, idx_cg, quad_idx, 0, active_fe_index);

  const unsigned int n_dg = phi_dg.dofs_per_component;
  const unsigned int n_en = phi_en.dofs_per_component;

  // The shape functions are vectors!
  using TensorType = dealii::Tensor<1, dim, scalar>;

  for (unsigned int cell = 0; cell < n_batches; ++cell)
  {
    if (mf_en.get_cell_iterator(cell, 0)->active_fe_index() == 0)
    {
      continue;
    }

    phi_dg.reinit(cell); phi_en.reinit(cell);
    y_eval.reinit(cell); utau_eval.reinit(cell);

    y_eval.read_dof_values_plain(function_enrichment->wall_distance);
    utau_eval.read_dof_values_plain(function_enrichment->friction_velocity);
    y_eval.evaluate(dealii::EvaluationFlags::values);
    utau_eval.evaluate(dealii::EvaluationFlags::values);

    // The shape functions are vectors!
    using TensorType = dealii::Tensor<1, dim, scalar>;
    
    // Create our filter tensors: (1, 0, 0) and (0, 0, 0)
    TensorType unit_tensor;
    TensorType zero_tensor;
    for (unsigned int d = 0; d < dim; ++d) 
    {
      unit_tensor[d] = dealii::make_vectorized_array<Number>(d == 0 ? 1.0 : 0.0);
      zero_tensor[d] = dealii::make_vectorized_array<Number>(0.0);
    }

    // Compute N_dg Reference (Vector Basis Functions)
    std::vector<std::vector<TensorType>> N_dg(n_dg, std::vector<TensorType>(phi_dg.n_q_points));
    for(unsigned int i = 0; i < n_dg; ++i) 
    {
      // Isolate the i-th shape function using the unit tensor
      for(unsigned int j = 0; j < n_dg; ++j) 
      {
        phi_dg.submit_dof_value(j == i ? unit_tensor : zero_tensor, j);
      }
      phi_dg.evaluate(dealii::EvaluationFlags::values);
      for(unsigned int q = 0; q < phi_dg.n_q_points; ++q)
      {
        N_dg[i][q] = phi_dg.get_value(q);
      }
    }

    // Compute N_en Reference (Vector Basis Functions)
    std::vector<std::vector<TensorType>> N_en(n_en, std::vector<TensorType>(phi_en.n_q_points));
    for(unsigned int i = 0; i < n_en; ++i) 
    {
      // Isolate the i-th shape function using the unit tensor
      for(unsigned int j = 0; j < n_en; ++j) 
      {
        phi_en.submit_dof_value(j == i ? unit_tensor : zero_tensor, j);
      }
      phi_en.evaluate(dealii::EvaluationFlags::values);
      for(unsigned int q = 0; q < phi_en.n_q_points; ++q)
      {
        N_en[i][q] = phi_en.get_value(q);
      }
    }

    SchurData data;
    auto zero = dealii::make_vectorized_array<Number>(0.0);
    auto M_bar_bar     = std::vector<std::vector<scalar>>(n_dg, std::vector<scalar>(n_dg, zero));
    data.M_tilde_tilde = std::vector<std::vector<scalar>>(n_en, std::vector<scalar>(n_en, zero));
    data.M_bar_tilde   = std::vector<std::vector<scalar>>(n_dg, std::vector<scalar>(n_en, zero));

    // Fill Mass Matrices with Tensor Dot Products
    for (unsigned int q = 0; q < phi_dg.n_q_points; ++q) 
    {
      auto psi = function_enrichment->get_value(utau_eval.get_value(q), y_eval.get_value(q)); 
      auto JxW = phi_dg.JxW(q); 

      for (unsigned int i = 0; i < n_en; ++i)
      {
        for (unsigned int j = 0; j < n_dg; ++j)
        {
          data.M_bar_tilde[j][i] += (N_dg[j][q] * N_en[i][q]) * psi * JxW;
        }
      }

      for (unsigned int i = 0; i < n_en; ++i)
      {
        for (unsigned int j = 0; j < n_en; ++j)
        {
          data.M_tilde_tilde[i][j] += (N_en[i][q] * N_en[j][q]) * (psi * psi) * JxW;
        }
      }

      for (unsigned int i = 0; i < n_dg; ++i)
      {
        for (unsigned int j = 0; j < n_dg; ++j)
        {
          M_bar_bar[i][j] += (N_dg[i][q] * N_dg[j][q]) * JxW;
        }
      }
    }

    data.M_bar_bar_inverse     = invert_matrix_simd(M_bar_bar, n_dg);
    data.M_tilde_tilde_inverse = invert_matrix_simd(data.M_tilde_tilde, n_en);

    auto S = data.M_tilde_tilde; 
    for(unsigned int i = 0; i < n_en; ++i) 
    {
      for(unsigned int j = 0; j < n_en; ++j) 
      {
        scalar sum = zero;
        for(unsigned int k = 0; k < n_dg; ++k) 
        {
          for(unsigned int l = 0; l < n_dg; ++l) 
          {
            sum += data.M_bar_tilde[k][i] * data.M_bar_bar_inverse[k][l] * data.M_bar_tilde[l][j];
          }
        }
        
        S[i][j] -= sum;
      }
    }

    /*
     * Tikhonov Regularization
     */
    for(unsigned int i = 0; i < n_en; ++i) 
    {
      // Add a scale-invariant threshold to the diagonal (1e-10)
      S[i][i] += dealii::make_vectorized_array<Number>(1e-10) * data.M_tilde_tilde[i][i]; 
    }

    data.schur_inverse = invert_matrix_simd(S, n_en);
    cell_schur_data[cell] = data; 
  }
}

// Helper lambda for general SIMD matrix inversion (Gauss Jordan Elimination)
template<int dim, typename Number>
std::vector<std::vector<dealii::VectorizedArray<Number>>>
WallDGCoupler<dim, Number>::invert_matrix_simd(std::vector<std::vector<scalar>> M, unsigned int n) 
{
  std::vector<std::vector<scalar>> M_copy = M;
  std::vector<std::vector<scalar>> inv(n, std::vector<scalar>(n, dealii::make_vectorized_array<Number>(0.0)));
  
  for (unsigned int i = 0; i < n; ++i)
  {
    inv[i][i] = dealii::make_vectorized_array<Number>(1.0);
  }
  
  for (unsigned int i = 0; i < n; ++i) 
  {
    auto pivot = dealii::make_vectorized_array<Number>(1.0) / M_copy[i][i];
    for (unsigned int j = 0; j < n; ++j) 
    {
      M_copy[i][j] *= pivot; 
      inv[i][j] *= pivot; 
    }
    for (unsigned int k = 0; k < n; ++k) 
    {
      if (k != i) 
      {
        auto factor = M_copy[k][i];
        for (unsigned int j = 0; j < n; ++j) 
        {
          M_copy[k][j] -= factor * M_copy[i][j];
          inv[k][j] -= factor * inv[i][j];
        }
      }
    }
  }
  return inv;
}

template <int dim, typename Number>
void
WallDGCoupler<dim, Number>::apply_schur_complement(
  VectorType & dst_global,
  VectorType const & src_global,
  VectorType & dst_enrichment,
  VectorType const & src_enrichment) const
{
  function_enrichment->copy_global_dg_to_wall_layout(src_global, function_enrichment->shadow_velocity_residual);

  apply_schur_inverse_mass(function_enrichment->shadow_velocity,
                           function_enrichment->shadow_velocity_residual,
                           dst_enrichment,
                           src_enrichment);

  function_enrichment->copy_wall_layout_to_global_dg(function_enrichment->shadow_velocity, dst_global);
}

template<int dim, typename Number>
void 
WallDGCoupler<dim, Number>::apply_schur_inverse_mass(
                                VectorType & dst_bar,
                                VectorType const & src_bar,
                                VectorType & dst_tilde,
                                VectorType const & src_tilde) const
{
  auto const & mf_en = *(function_enrichment->get_matrix_free());
  unsigned int idx_shadow = function_enrichment->get_dof_index_shadow_vector();
  unsigned int idx_en     = function_enrichment->get_dof_index_en();
  unsigned int quad_idx   = function_enrichment->get_quad_index();

  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_dg(mf_en, idx_shadow, quad_idx, 0, active_fe_index);
  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_en(mf_en, idx_en, quad_idx, 0, active_fe_index);

  const unsigned int n_dg = phi_dg.dofs_per_component;
  const unsigned int n_en = phi_en.dofs_per_component;
  const unsigned int n_batches = mf_en.n_cell_batches();

  using TensorType = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  TensorType zero_tensor;
  for (unsigned int d = 0; d < dim; ++d) {
    zero_tensor[d] = dealii::make_vectorized_array<Number>(0.0);
  }

  for (unsigned int cell = 0; cell < n_batches; ++cell)
  {
    if (mf_en.get_cell_iterator(cell, 0)->active_fe_index() == 0) continue;

    phi_dg.reinit(cell); phi_en.reinit(cell);

    phi_dg.read_dof_values_plain(src_bar);
    phi_en.read_dof_values_plain(src_tilde);

    std::vector<TensorType> R_bar(n_dg);
    for(unsigned int i=0; i<n_dg; ++i)
    {
      R_bar[i] = phi_dg.get_dof_value(i);
    }

    std::vector<TensorType> R_tilde(n_en);
    for(unsigned int i=0; i<n_en; ++i)
    {
      R_tilde[i] = phi_en.get_dof_value(i);
    }

    auto const & M_bar_tilde       = cell_schur_data[cell].M_bar_tilde;
    auto const & M_bar_bar_inverse = cell_schur_data[cell].M_bar_bar_inverse;
    auto const & schur_inverse     = cell_schur_data[cell].schur_inverse;

    // --- STEP 1: U_tilde ---
    std::vector<TensorType> M_inv_R_bar(n_dg, zero_tensor);
    for(unsigned int i=0; i<n_dg; ++i)
    {
      for(unsigned int j=0; j<n_dg; ++j)
      {
        M_inv_R_bar[i] += M_bar_bar_inverse[i][j] * R_bar[j];
      }
    }

    std::vector<TensorType> coupling(n_en, zero_tensor);
    for(unsigned int i=0; i<n_en; ++i) 
    {
      for(unsigned int j=0; j<n_dg; ++j)
      {
        coupling[i] += M_bar_tilde[j][i] * M_inv_R_bar[j];
      }
    }

    std::vector<TensorType> U_tilde(n_en, zero_tensor);
    for(unsigned int i=0; i<n_en; ++i) 
    {
      auto diff = R_tilde[i] - coupling[i];
      for(unsigned int j=0; j<n_en; ++j)
      {
        U_tilde[i] += schur_inverse[i][j] * diff;
      }
    }

    // --- STEP 2: U_bar ---
    std::vector<TensorType> diff_bar(n_dg, zero_tensor);
    for(unsigned int i=0; i<n_dg; ++i) 
    {
      diff_bar[i] = R_bar[i];
      for(unsigned int j=0; j<n_en; ++j)
      {
        diff_bar[i] -= M_bar_tilde[i][j] * U_tilde[j];
      }
    }

    std::vector<TensorType> U_bar(n_dg, zero_tensor);
    for(unsigned int i=0; i<n_dg; ++i) 
    {
      for(unsigned int j=0; j<n_dg; ++j)
      {
        U_bar[i] += M_bar_bar_inverse[i][j] * diff_bar[j];
      }
    }

    // --- STEP 3: Write back to memory ---
    for(unsigned int i=0; i<n_dg; ++i)
    {
      phi_dg.submit_dof_value(U_bar[i], i);
    }
    phi_dg.set_dof_values_plain(dst_bar);

    for(unsigned int i=0; i<n_en; ++i)
    {
      phi_en.submit_dof_value(U_tilde[i], i);
    }
    phi_en.set_dof_values_plain(dst_tilde);
  }
}


/*  ______________________________________________________________________
 *
 * Compute Enrichment Velocity
 *  ______________________________________________________________________
 */
template<int dim, typename Number>
void
WallDGCoupler<dim, Number>::compute_enrichment_velocity_from_residual(
  VectorType const & global_residual,
  double scaling_factor)
{
  function_enrichment->copy_global_dg_to_wall_layout(global_residual, function_enrichment->shadow_velocity_residual);

  auto const & mf_en = *(function_enrichment->get_matrix_free());
  unsigned int idx_shadow = function_enrichment->get_dof_index_shadow_vector();
  unsigned int idx_en     = function_enrichment->get_dof_index_en();
  unsigned int quad_idx   = function_enrichment->get_quad_index();

  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_dg(mf_en, idx_shadow, quad_idx, 0, active_fe_index);
  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_en(mf_en, idx_en, quad_idx, 0, active_fe_index);

  const unsigned int n_dg = phi_dg.dofs_per_component;
  const unsigned int n_en = phi_en.dofs_per_component;
  const unsigned int n_batches = mf_en.n_cell_batches();

  using TensorType = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  TensorType zero_tensor;
  for (unsigned int d = 0; d < dim; ++d) 
  {
    zero_tensor[d] = dealii::make_vectorized_array<Number>(0.0);
  }

  dealii::VectorizedArray<Number> vec_scaling = dealii::make_vectorized_array<Number>(scaling_factor);

  for (unsigned int cell = 0; cell < n_batches; ++cell)
  {
    if (mf_en.get_cell_iterator(cell, 0)->active_fe_index() == 0)
    {
      continue;
    }

    phi_dg.reinit(cell);
    phi_en.reinit(cell);

    // Read safely from the isolated wall model vectors
    phi_dg.read_dof_values_plain(function_enrichment->shadow_velocity_residual);
    phi_en.read_dof_values_plain(function_enrichment->enrichment_residual);

    std::vector<TensorType> R_bar(n_dg);
    for(unsigned int i = 0; i < n_dg; ++i)
    {
      R_bar[i] = phi_dg.get_dof_value(i);
    }

    std::vector<TensorType> R_tilde(n_en);
    for(unsigned int i = 0; i < n_en; ++i)
    {
      R_tilde[i] = phi_en.get_dof_value(i);
    }

    auto const & M_bar_bar_inverse = cell_schur_data[cell].M_bar_bar_inverse;
    auto const & M_bar_tilde       = cell_schur_data[cell].M_bar_tilde; 
    auto const & schur_inverse     = cell_schur_data[cell].schur_inverse;

    // M_bar_inv_R_bar = M_{\bar{U}\bar{U}}^{-1} * \bar{R}
    std::vector<TensorType> M_inv_R_bar(n_dg, zero_tensor);
    for(unsigned int i = 0; i < n_dg; ++i) 
    {
      for(unsigned int j = 0; j < n_dg; ++j) 
      {
        M_inv_R_bar[i] += M_bar_bar_inverse[i][j] * R_bar[j];
      }
    }

    // MM_inv_R = M_bar_tilde^T * M_bar_inv_R_bar
    std::vector<TensorType> MM_inv_R(n_en, zero_tensor);
    for(unsigned int i = 0; i < n_en; ++i) 
    {
      for(unsigned int j = 0; j < n_dg; ++j) 
      {
        MM_inv_R[i] += M_bar_tilde[j][i] * M_inv_R_bar[j];
      }
    }

    // \tilde{U} = S^{-1} * (\tilde{R} - MM_inv_R)
    std::vector<TensorType> U_tilde(n_en, zero_tensor);
    for(unsigned int i = 0; i < n_en; ++i) 
    {
      auto diff = R_tilde[i] - MM_inv_R[i];
      for(unsigned int j = 0; j < n_en; ++j) 
      {
        U_tilde[i] += schur_inverse[i][j] * diff;
      }
    }

    // Scale and write back to the enrichment velocity vector
    for(unsigned int i = 0; i < n_en; ++i) 
    {
      U_tilde[i] *= vec_scaling;
      phi_en.submit_dof_value(U_tilde[i], i);
    }

    phi_en.set_dof_values_plain(function_enrichment->enrichment_velocity);
  }
}

template<int dim, typename Number>
void
WallDGCoupler<dim, Number>::compute_enrichment_velocity(
  VectorType const & inter_velocity,
  double scaling_factor)
{
  function_enrichment->copy_global_dg_to_wall_layout(inter_velocity, function_enrichment->shadow_velocity);

  auto const & mf_en = *(function_enrichment->get_matrix_free());

 this->current_scaling_factor = scaling_factor; 

  mf_en.cell_loop(&WallDGCoupler::local_compute_enrichment_velocity,
                  this,
                  function_enrichment->enrichment_velocity,
                  function_enrichment->shadow_velocity);
}

template<int dim, typename Number>
void
WallDGCoupler<dim, Number>::local_compute_enrichment_velocity(
  dealii::MatrixFree<dim, Number> const & mf_en,
  VectorType & dst,
  VectorType const & src,
  std::pair<unsigned int, unsigned int> const & range) const
{
  unsigned int idx_shadow = function_enrichment->get_dof_index_shadow_vector();
  unsigned int idx_en     = function_enrichment->get_dof_index_en();
  unsigned int quad_idx   = function_enrichment->get_quad_index();
  unsigned int fe_idx     = active_fe_index; 

  dealii::VectorizedArray<Number> vec_scaling = dealii::make_vectorized_array<Number>(this->current_scaling_factor);

  // Initialize FEEvaluation
  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_dg(mf_en, idx_shadow, quad_idx, 0, fe_idx);
  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_en(mf_en, idx_en, quad_idx, 0, fe_idx);

  const unsigned int n_dg = phi_dg.dofs_per_component;
  const unsigned int n_en = phi_en.dofs_per_component;

  using TensorType = dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>;
  TensorType zero_tensor;
  for (unsigned int d = 0; d < dim; ++d) 
  {
    zero_tensor[d] = dealii::make_vectorized_array<Number>(0.0);
  }

  // Process the batch chunk assigned to this thread
  for (unsigned int cell = range.first; cell < range.second; ++cell)
  {
    // Safely skip any cell that isn't actively part of the wall layer
    if (mf_en.get_cell_iterator(cell, 0)->active_fe_index() != fe_idx)
    {
      continue;
    }

    phi_dg.reinit(cell);
    phi_en.reinit(cell);

    // Read from the source vectors
    phi_dg.read_dof_values_plain(src); 
    phi_en.read_dof_values_plain(function_enrichment->enrichment_residual);

    std::vector<TensorType> U_bar_star(n_dg);
    for(unsigned int i = 0; i < n_dg; ++i)
    {
      U_bar_star[i] = phi_dg.get_dof_value(i);
    }

    std::vector<TensorType> R_tilde(n_en);
    for(unsigned int i = 0; i < n_en; ++i)
    {
      R_tilde[i] = phi_en.get_dof_value(i);
    }

    // Precomputed local Schur Complement Data
    auto const & M_tilde_tilde_inverse = cell_schur_data[cell].M_tilde_tilde_inverse;
    auto const & M_bar_tilde           = cell_schur_data[cell].M_bar_tilde; 

    AssertThrow(M_bar_tilde.size() == n_dg, 
                dealii::ExcMessage("M_bar_tilde outer size is wrong! Matrix was not precomputed."));
    if (n_dg > 0) {
      AssertThrow(M_bar_tilde[0].size() == n_en, 
                  dealii::ExcMessage("M_bar_tilde inner size is wrong!"));
    }

    // coupling = M_bar_tilde^T * U_bar_star
    std::vector<TensorType> coupling(n_en, zero_tensor);
    for(unsigned int i = 0; i < n_en; ++i) 
    {
      for(unsigned int j = 0; j < n_dg; ++j) 
      {
        coupling[i] += M_bar_tilde[j][i] * U_bar_star[j];
      }
    }

    // \tilde{U} = M_{\tilde{V}\tilde{U}}^{-1} \left( \tilde{R} - coupling \right)
    std::vector<TensorType> U_tilde(n_en, zero_tensor);
    for(unsigned int i = 0; i < n_en; ++i) 
    {
      auto diff = R_tilde[i] - coupling[i];
      for(unsigned int j = 0; j < n_en; ++j) 
      {
        U_tilde[i] += M_tilde_tilde_inverse[i][j] * diff;
      }
    }

    // Scale and push back to MatrixFree
    for(unsigned int i = 0; i < n_en; ++i) 
    {
      U_tilde[i] *= vec_scaling;
      phi_en.submit_dof_value(U_tilde[i], i);
    }

    // Write directly to the destination vector
    phi_en.set_dof_values_plain(dst);
  }
}

template<int dim, typename Number>
void
WallDGCoupler<dim, Number>::update_wall_enrichment_vectors(VectorType const & global_velocity) const
{
  function_enrichment->copy_global_dg_to_wall_layout(global_velocity, function_enrichment->shadow_velocity);
  function_enrichment->evaluate_friction_velocity(global_velocity);

  function_enrichment->shadow_velocity.update_ghost_values();
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
WallDGCoupler<dim, Number>::get_shadow_velocity() const
{
  return function_enrichment->shadow_velocity;
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
WallDGCoupler<dim, Number>::get_enrichment_velocity() const
{
  return function_enrichment->enrichment_velocity;
}

template class WallDGCoupler<2, float>;
template class WallDGCoupler<2, double>;
template class WallDGCoupler<3, float>;
template class WallDGCoupler<3, double>;

} // namespace ExaDG
