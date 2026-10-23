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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <exadg/operators/wall_modelling/coupler.h>
#include <deal.II/lac/full_matrix.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <memory>

namespace ExaDG
{
template<int dim, typename Number>
WallDGCoupler<dim, Number>::WallDGCoupler()
  : matrix_free_global(nullptr),
  dof_index_global_u(0),
  active_fe_index(1),
  current_scaling_factor(1.0),
  rank_tolerance(1.e-8),
  n_modes_dropped_last(dealii::numbers::invalid_unsigned_int)
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

template<int dim, typename Number>
void
WallDGCoupler<dim, Number>::set_rank_tolerance(double const tolerance)
{
  AssertThrow(tolerance >= 0.0 and tolerance < 1.0,
              dealii::ExcMessage("The rank tolerance must be in [0, 1)."));
  rank_tolerance = tolerance;
}

/*
 * Precompute Schur Complement Matrices
 */
template <int dim, typename Number>
void
WallDGCoupler<dim, Number>::precompute_schur_matrices()
{
  auto const & mf_en = *(function_enrichment->get_matrix_free());
  const unsigned int n_batches = mf_en.n_cell_batches();

  // keep M_bar_bar(_inverse) from the first call (fixed mesh); psi-dependent parts are overwritten
  if(cell_schur_data.size() != n_batches)
  {
    cell_schur_data.clear();
    cell_schur_data.resize(n_batches);
  }

  unsigned int idx_shadow = function_enrichment->get_dof_index_shadow_vector();
  unsigned int idx_en     = function_enrichment->get_dof_index_en();

  unsigned int quad_idx   = function_enrichment->get_quad_index();

  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_dg(mf_en, idx_shadow, quad_idx, 0, active_fe_index);
  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi_en(mf_en, idx_en, quad_idx, 0, active_fe_index);

  const unsigned int n_dg = phi_dg.dofs_per_component;
  const unsigned int n_en = phi_en.dofs_per_component;

  // The shape functions are vectors!
  using TensorType = dealii::Tensor<1, dim, scalar>;

  unsigned int n_modes_total = 0;
  unsigned int n_modes_dropped = 0;
  double lambda_min_kept = std::numeric_limits<double>::max();

  using SimdMatrix = std::vector<std::vector<scalar>>;

  // distinct M_bar_bar computed in this call (first call only), each stored once
  struct StoredMassMatrix
  { 
    std::shared_ptr<SimdMatrix const> M_bar_bar;
    std::shared_ptr<SimdMatrix const> M_bar_bar_inverse;
    Number                            max_entry;
  };
  std::vector<StoredMassMatrix> unique_M_bar_bar;

  // max |M_ij| over all entries and SIMD lanes
  auto const max_abs_entry = [](SimdMatrix const & M) {
    Number m = 0;
    for(auto const & row : M)
    for(auto const & a : row) 
    for(unsigned int v = 0; v < scalar::size(); ++v)
      m = std::max(m, std::abs(a[v]));
    return m;
  };
    
  // A = B up to round-off: |A_ij - B_ij| <= 10^3 eps max|A| for all entries and   SIMD lanes.
  // Cells of the same geometry give the same matrix up to round-off; a different  geometry
  // differs by O(1) relative and fails at once (usually in the first entry).
  Number const tolerance = 1.e3 * std::numeric_limits<Number>::epsilon();
  auto const   equal_up_to_roundoff =
    [&](SimdMatrix const & A, Number const max_A, SimdMatrix const & B) {
      for(unsigned int i = 0; i < A.size(); ++i)
        for(unsigned int j = 0; j < A[i].size(); ++j)
          for(unsigned int v = 0; v < scalar::size(); ++v)
            if(std::abs(A[i][j][v] - B[i][j][v]) > tolerance * max_A)
              return false;
      return true;
    };

  for (unsigned int cell = 0; cell < n_batches; ++cell)
  {
    if (mf_en.get_cell_iterator(cell, 0)->active_fe_index() != 1)
    {
      continue;
    }

    phi_dg.reinit(cell);
    phi_en.reinit(cell);

    // Create our filter tensors: (1, 0, 0) and (0, 0, 0)
    TensorType unit_tensor;
    TensorType zero_tensor;
    for (unsigned int d = 0; d < dim; ++d) 
    {
      unit_tensor[d] = dealii::make_vectorized_array<Number>(d == 0 ? 1.0 : 0.0);
      zero_tensor[d] = dealii::make_vectorized_array<Number>(0.0);
    }

    SchurData & data = cell_schur_data[cell];
    auto zero = dealii::make_vectorized_array<Number>(0.0);


    // M_bar_bar depends on the cell geometry only: computed and inverted on the   first call.
    // A batch whose M_bar_bar agrees with an earlier one up to round-off shares   that copy of
    // M_bar_bar and M_bar_bar^{-1} (the same matrices, stored once); any other   batch gets its own.
    if(data.M_bar_bar_inverse == nullptr)
    { 
      SimdMatrix M_bar_bar(n_dg, std::vector<scalar>(n_dg, zero));
      for(unsigned int i = 0; i < n_dg; ++i)
      { 
        for(unsigned int j = 0; j < n_dg; ++j)
        {
          phi_dg.submit_dof_value(j == i ? unit_tensor : zero_tensor, j);
        }
        phi_dg.evaluate(dealii::EvaluationFlags::values);
        for(unsigned int q = 0; q < phi_dg.n_q_points; ++q)
        { 
          phi_dg.submit_value(phi_dg.get_value(q), q); // JxW applied by submit_value
        }
        phi_dg.integrate(dealii::EvaluationFlags::values);
        for(unsigned int j = 0; j < n_dg; ++j)
        {
          M_bar_bar[j][i] = phi_dg.get_dof_value(j)[0];
        }
      }
      Number const max_entry = max_abs_entry(M_bar_bar);

      for(auto const & stored : unique_M_bar_bar)
      {
        if(equal_up_to_roundoff(*stored.M_bar_bar, stored.max_entry, M_bar_bar))
        {
          data.M_bar_bar         = stored.M_bar_bar;
          data.M_bar_bar_inverse = stored.M_bar_bar_inverse;
          break;
        }
      }

      if(data.M_bar_bar_inverse == nullptr)
      {
        data.M_bar_bar_inverse = std::make_shared<SimdMatrix const>(invert_matrix_simd(M_bar_bar, n_dg));
        data.M_bar_bar = std::make_shared<SimdMatrix
          const>(std::move(M_bar_bar));
        unique_M_bar_bar.push_back({data.M_bar_bar, data.M_bar_bar_inverse,max_entry});
      }
    } 

    // psi-dependent blocks, column i = enrichment shape function i:
    //   M_bar_tilde(j,i)   = (N_dg_j, psi N_en_i),   M_tilde_tilde(j,i) = (N_en_j, psi^2 N_en_i)
    data.M_bar_tilde   = std::vector<std::vector<scalar>>(n_dg, std::vector<scalar>(n_en, zero));
    data.M_tilde_tilde = std::vector<std::vector<scalar>>(n_en, std::vector<scalar>(n_en, zero));
    for(unsigned int i = 0; i < n_en; ++i)
    {
      for(unsigned int j = 0; j < n_en; ++j)
      {
        phi_en.submit_dof_value(j == i ? unit_tensor : zero_tensor, j);
      }
      phi_en.evaluate(dealii::EvaluationFlags::values);

      for(unsigned int q = 0; q < phi_en.n_q_points; ++q)
      {
        scalar const     psi   = function_enrichment->get_psi_cell(cell, q);
        TensorType const psi_N = psi * phi_en.get_value(q);
        phi_dg.submit_value(psi_N, q);
        phi_en.submit_value(psi * psi_N, q);
      }
      phi_dg.integrate(dealii::EvaluationFlags::values);
      phi_en.integrate(dealii::EvaluationFlags::values);

      for(unsigned int j = 0; j < n_dg; ++j)
      {
        data.M_bar_tilde[j][i] = phi_dg.get_dof_value(j)[0];
      }
      for(unsigned int j = 0; j < n_en; ++j)
      {
        data.M_tilde_tilde[j][i] = phi_en.get_dof_value(j)[0];
      }
    }

    data.M_tilde_tilde_inverse = invert_matrix_simd(data.M_tilde_tilde, n_en);

    // S = M_tilde_tilde - M_bar_tilde^T X,   X = M_bar_bar^{-1} M_bar_tilde
    std::vector<std::vector<scalar>> X(n_dg, std::vector<scalar>(n_en, zero));
    for(unsigned int k = 0; k < n_dg; ++k)
    {
      for(unsigned int l = 0; l < n_dg; ++l)
      {
        scalar const a = (*data.M_bar_bar_inverse)[k][l];
        for(unsigned int j = 0; j < n_en; ++j)
        {
          X[k][j] += a * data.M_bar_tilde[l][j];
        }
      }
    }

    auto S = data.M_tilde_tilde;
    for(unsigned int i = 0; i < n_en; ++i)
    {
      for(unsigned int j = 0; j < n_en; ++j)
      {
        scalar sum = zero;
        for(unsigned int k = 0; k < n_dg; ++k)
        {
          sum += data.M_bar_tilde[k][i] * X[k][j];
        }
        S[i][j] -= sum;
      }
    }
    // exact inverse of the Schur complement on the well-posed enrichment modes;
    // (numerically) linearly dependent modes are removed
    data.schur_inverse = compute_schur_pseudo_inverse(S,
                                                      data.M_tilde_tilde,
                                                      data.projector,
                                                      mf_en.n_active_entries_per_cell_batch(cell),
                                                      n_modes_total,
                                                      n_modes_dropped,
                                                      lambda_min_kept);
  }

  /****************************************DEBUG******************************************/
    MPI_Comm const comm = function_enrichment->get_dof_handler_shadow_vector().get_triangulation().get_communicator();

    n_modes_total   = dealii::Utilities::MPI::sum(n_modes_total, comm);
    n_modes_dropped = dealii::Utilities::MPI::sum(n_modes_dropped, comm);
    lambda_min_kept = dealii::Utilities::MPI::min(lambda_min_kept, comm);

#ifdef EXADG_PROFILE
  // once (first call, where M_bar_bar is computed): distinct M_bar_bar per   rank
  static bool reported = false;
  if(not reported)
  {
    unsigned int n_enriched_batches = 0;
    for(auto const & d : cell_schur_data)
    if(d.M_bar_bar_inverse != nullptr)
      ++n_enriched_batches;

    unsigned int const n_distinct_max =
      dealii::Utilities::MPI::max(static_cast<unsigned
                                  int>(unique_M_bar_bar.size()), comm);
    unsigned int const n_batches_max =
      dealii::Utilities::MPI::max(n_enriched_batches, comm);

    dealii::ConditionalOStream pcout(std::cout,

                                     dealii::Utilities::MPI::this_mpi_process(comm) == 0);
    pcout << "  [enrichment] M_bar_bar stored per rank (max over ranks): " << n_distinct_max << " distinct matrices for " << n_batches_max << " enriched cell batches" << std::endl;

    reported = true;
  }
#endif

    if(n_modes_dropped != n_modes_dropped_last)
    {
#ifdef EXADG_DEBUG
    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(comm) == 0);
    pcout << std::scientific << std::setprecision(3) << "  [enrichment] removed "
      << n_modes_dropped << " of " << n_modes_total
      << " enrichment modes, smallest kept eigenvalue of D^{-1/2} S D^{-1/2} = "
      << lambda_min_kept << std::endl;

#endif
      n_modes_dropped_last = n_modes_dropped;
    }
  /***************************************************************************************/
}

/*
 * Inverse of the Schur complement of the block mass matrix of an enriched cell, and the projector
 * onto the well-posed enrichment space.
 *
 * Mass matrix per velocity component, with N_i \in P_k(K), \tilde N_j \in P_l(K):
 *
 *   M = [ M_bb  M_bt ],  (M_bb)_{ij} = (N_i, N_j)_K,
 *       [ M_tb  M_tt ]   (M_bt)_{ij} = (N_i, \psi \tilde N_j)_K,
 *                        (M_tt)_{ij} = (\psi \tilde N_i, \psi \tilde N_j)_K.
 *
 * Block elimination = exact LU factorization of M (Krank, thesis Eqs. (6.16), (6.17)):
 *
 *   S \tilde U = \tilde R - M_tb M_bb^{-1} \bar R,   \bar U = M_bb^{-1} (\bar R - M_bt \tilde U),
 *   S = M_tt - M_tb M_bb^{-1} M_bt,   \tilde U^T S \tilde U = \| (I - \Pi_k)(\psi \tilde u_h) \|_K^2 .
 *
 * With D = diag(M_tt), the eigenvalues \lambda_m of \hat S = D^{-1/2} S D^{-1/2} are the fraction
 * of an enrichment mode that P_k(K) cannot represent. Modes with \lambda_m <= \epsilon are
 * (numerically) linearly dependent and removed from the enrichment space:
 *
 *   W   = D^{-1/2} [ v_m ]_{\lambda_m > \epsilon}                (kept coefficient directions)
 *   S^+ = W \Lambda^{-1} W^T = W (W^T S W)^{-1} W^T            (Galerkin inverse on range(W))
 *   P   = W (W^T W)^{-1} W^T                                     (Euclidean projector onto range(W))
 *
 * All block systems are solved on \tilde U \in range(P) (EnrichmentConstrainedOperator,
 * EnrichedBlockJacobiPreconditioner), consistent with S^+. Nothing removed: S^+ = S^{-1}, P = I.
 *
 * \epsilon = max(rank_tolerance, 10^3 \epsilon_{machine}(Number)).
 * Unfilled SIMD lanes (v >= n_active_lanes): S^+ = 0, P = 0, not counted.
 */
template<int dim, typename Number>
std::vector<std::vector<dealii::VectorizedArray<Number>>>
WallDGCoupler<dim, Number>::compute_schur_pseudo_inverse(
  std::vector<std::vector<scalar>> const & S,
  std::vector<std::vector<scalar>> const & M_tilde_tilde,
  std::vector<std::vector<scalar>>       & projector,
  unsigned int const                       n_active_lanes,
  unsigned int &                           n_modes_total,
  unsigned int &                           n_modes_dropped,
  double &                                 lambda_min_kept) const
{
  unsigned int const n = S.size();

  double const epsilon =
    std::max(rank_tolerance, 1.e3 * double(std::numeric_limits<Number>::epsilon()));

  scalar const zero = dealii::make_vectorized_array<Number>(0.0);
  std::vector<std::vector<scalar>> S_inverse(n, std::vector<scalar>(n, zero));
  projector.assign(n, std::vector<scalar>(n, zero));

  for(unsigned int v = 0; v < n_active_lanes; ++v)
  {
    // d_i = (M_tt)_{ii}^{-1/2}
    std::vector<double> d(n);
    for(unsigned int i = 0; i < n; ++i)
    {
      double const m_ii = M_tilde_tilde[i][i][v];
      AssertThrow(m_ii > 0.0,
                  dealii::ExcMessage("Enrichment mass matrix with non-positive diagonal entry."));
      d[i] = 1.0 / std::sqrt(m_ii);
    }


    // \hat S = D^{-1/2} S D^{-1/2}, symmetrized (S is symmetric up to round-off)
    dealii::LAPACKFullMatrix<double> S_hat(n, n);
    for(unsigned int i = 0; i < n; ++i)
      for(unsigned int j = 0; j < n; ++j)
      {
        S_hat(i, j) = 0.5 * d[i] * (double(S[i][j][v]) + double(S[j][i][v])) * d[j];
          AssertThrow(std::isfinite(S_hat(i, j)),
                      dealii::ExcMessage("Schur complement of an enriched cell is not finite."));
      }


    // \hat S is symmetric positive semi-definite: \hat S = V \Lambda V^T. The symmetric
    // eigensolver (Xsyevx) is robust for the clustered near-zero modes, unlike the 
    // divide-and-conquer SVD (Xgesdd), which failed to converge once in ~10^7 calls.
    S_hat.set_property(dealii::LAPACKSupport::symmetric);
    dealii::Vector<double>     eigenvalues;
    dealii::FullMatrix<double> eigenvectors;
    S_hat.compute_eigenvalues_symmetric(-1.0, 1.0e10, 0.0, eigenvalues,
                                          eigenvectors);

    // eigenvalues outside (-1, 1e10] cannot occur for \hat S; count them as removed if they do
    n_modes_total += n;
    n_modes_dropped += n - eigenvalues.size();

    // kept directions w_m = D^{-1/2} v_m (columns of W)
    std::vector<std::vector<double>> W;
    for(unsigned int m = 0; m < eigenvalues.size(); ++m)
    {
      double const lambda = eigenvalues(m);

      if(lambda <= epsilon)
      {
        ++n_modes_dropped;
        continue;
      }
      lambda_min_kept = std::min(lambda_min_kept, lambda);

      std::vector<double> w(n);
      for(unsigned int i = 0; i < n; ++i)
        w[i] = d[i] * eigenvectors(i, m);
      W.push_back(w); 

      // (S^+)_{ij} += d_i (v_m)_i \lambda_m^{-1} (v_m)_j d_j
      for(unsigned int i = 0; i < n; ++i)
        for(unsigned int j = 0; j < n; ++j)
          S_inverse[i][j][v] += Number(d[i] * eigenvectors(i, m) / lambda *
                                       eigenvectors(j, m) * d[j]);
    }

    // P = W (W^T W)^{-1} W^T; exactly I if nothing is removed, 0 if everything is removed
    unsigned int const r = W.size();
    if(r == n)
    {
      for(unsigned int i = 0; i < n; ++i)
        projector[i][i][v] = 1.0;
    }
    else if(r > 0)
    {
      dealii::FullMatrix<double> G(r, r);
      for(unsigned int a = 0; a < r; ++a)
        for(unsigned int b = 0; b < r; ++b)
          for(unsigned int i = 0; i < n; ++i)
            G(a, b) += W[a][i] * W[b][i];
      G.gauss_jordan(); // (W^T W)^{-1}, SPD

      for(unsigned int i = 0; i < n; ++i)
        for(unsigned int j = 0; j < n; ++j)
        {
          double p_ij = 0.0;
          for(unsigned int a = 0; a < r; ++a)
            for(unsigned int b = 0; b < r; ++b)
              p_ij += W[a][i] * G(a, b) * W[b][j];
          projector[i][j][v] = Number(p_ij);
        }
    }
  }

  return S_inverse;
}

/*
 * Projection onto the well-posed enrichment space (see compute_schur_pseudo_inverse):
 *   complement = false:  \tilde U <- P \tilde U
 *   complement = true:   \tilde U <- (I - P) \tilde U
 * P acts on the coefficients of each velocity component (same P for all components).
 */
template<int dim, typename Number>
void
WallDGCoupler<dim, Number>::apply_enrichment_projector(VectorType &       dst,
                                                       VectorType const & src,
                                                       bool const         complement) const
{
  auto const & mf_en = *(function_enrichment->get_matrix_free());

  dealii::FEEvaluation<dim, -1, 0, dim, Number> phi(mf_en,
                                                    function_enrichment->get_dof_index_en(),
                                                    function_enrichment->get_quad_index(),
                                                    0,
                                                    active_fe_index);
  unsigned int const n_en = phi.dofs_per_component;

  using TensorType = dealii::Tensor<1, dim, scalar>;
  TensorType zero_tensor;
  for(unsigned int d = 0; d < dim; ++d)
    zero_tensor[d] = dealii::make_vectorized_array<Number>(0.0);

  std::vector<TensorType> u(n_en);

  for(unsigned int cell = 0; cell < mf_en.n_cell_batches(); ++cell)
  {
    if(mf_en.get_cell_iterator(cell, 0)->active_fe_index() != active_fe_index)
      continue;

    phi.reinit(cell);
    phi.read_dof_values_plain(src);
    for(unsigned int i = 0; i < n_en; ++i)
      u[i] = phi.get_dof_value(i);

    auto const & P = cell_schur_data[cell].projector;
    for(unsigned int i = 0; i < n_en; ++i)
    {
      TensorType Pu = zero_tensor;
      for(unsigned int j = 0; j < n_en; ++j)
        Pu += P[i][j] * u[j];
      phi.submit_dof_value(complement ? u[i] - Pu : Pu, i);
    }
    phi.set_dof_values_plain(dst); // cell-local: dst may alias src
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
  for (unsigned int d = 0; d < dim; ++d) 
  {
    zero_tensor[d] = dealii::make_vectorized_array<Number>(0.0);
  }

  std::vector<TensorType> R_bar(n_dg), M_inv_R_bar(n_dg), diff_bar(n_dg);
  std::vector<TensorType> R_tilde(n_en), coupling(n_en), U_tilde(n_en);

  for(unsigned int cell = 0; cell < n_batches; ++cell)
  {
    if(mf_en.get_cell_iterator(cell, 0)->active_fe_index() != 1)
      continue;

    phi_dg.reinit(cell);
    phi_en.reinit(cell);

    phi_dg.read_dof_values_plain(src_bar);
    phi_en.read_dof_values_plain(src_tilde);

    for(unsigned int i = 0; i < n_dg; ++i)
      R_bar[i] = phi_dg.get_dof_value(i);
    for(unsigned int i = 0; i < n_en; ++i)
      R_tilde[i] = phi_en.get_dof_value(i);

    auto const & M_bar_tilde       = cell_schur_data[cell].M_bar_tilde;
    auto const & M_bar_bar_inverse = *cell_schur_data[cell].M_bar_bar_inverse;
    auto const & schur_inverse     = cell_schur_data[cell].schur_inverse;

    // --- STEP 1, Eq. (6.16): S U_tilde = R_tilde - M_tilde_bar M_bar_bar^{-1} R_bar ---

    // M_bar_bar^{-1} R_bar
    for(unsigned int i = 0; i < n_dg; ++i)
    {
      TensorType sum = zero_tensor;
      for(unsigned int j = 0; j < n_dg; ++j)
        sum += M_bar_bar_inverse[i][j] * R_bar[j];
      M_inv_R_bar[i] = sum;
    }

    // M_tilde_bar (M_bar_bar^{-1} R_bar),  M_tilde_bar = M_bar_tilde^T: the rows of the stored
    // M_bar_tilde are traversed contiguously
    for(unsigned int i = 0; i < n_en; ++i)
      coupling[i] = zero_tensor;
    for(unsigned int j = 0; j < n_dg; ++j)
      for(unsigned int i = 0; i < n_en; ++i)
        coupling[i] += M_bar_tilde[j][i] * M_inv_R_bar[j];

    // U_tilde = S^{-1} (R_tilde - M_tilde_bar M_bar_bar^{-1} R_bar)
    for(unsigned int i = 0; i < n_en; ++i)
    {
      TensorType sum = zero_tensor;
      for(unsigned int j = 0; j < n_en; ++j)
        sum += schur_inverse[i][j] * (R_tilde[j] - coupling[j]);
      U_tilde[i] = sum;
    }
    // --- STEP 2, Eq. (6.17): U_bar = M_bar_bar^{-1} (R_bar - M_bar_tilde  U_tilde) ---

    // R_bar - M_bar_tilde U_tilde
    for(unsigned int i = 0; i < n_dg; ++i)
    {
      TensorType sum = R_bar[i];
      for(unsigned int j = 0; j < n_en; ++j)
        sum -= M_bar_tilde[i][j] * U_tilde[j];
      diff_bar[i] = sum;
    }

    // U_bar = M_bar_bar^{-1} (R_bar - M_bar_tilde U_tilde)
    for(unsigned int i = 0; i < n_dg; ++i)
    {
      TensorType sum = zero_tensor;
      for(unsigned int j = 0; j < n_dg; ++j)
        sum += M_bar_bar_inverse[i][j] * diff_bar[j];
      phi_dg.submit_dof_value(sum, i); 
    }

    // --- STEP 3: write back ---
    for(unsigned int i = 0; i < n_en; ++i)
      phi_en.submit_dof_value(U_tilde[i], i);

    phi_dg.set_dof_values_plain(dst_bar);
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
    if (mf_en.get_cell_iterator(cell, 0)->active_fe_index() != 1)
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

    auto const & M_bar_bar_inverse = *cell_schur_data[cell].M_bar_bar_inverse;
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

    std::vector<TensorType> diff_tilde(n_en, zero_tensor);
    for(unsigned int j = 0; j < n_en; ++j) 
    {
      diff_tilde[j] = R_tilde[j] - MM_inv_R[j];
    }

    // \tilde{U} = S^{-1} * (\tilde{R} - MM_inv_R)
    std::vector<TensorType> U_tilde(n_en, zero_tensor);
    for(unsigned int i = 0; i < n_en; ++i) 
    {
      for(unsigned int j = 0; j < n_en; ++j) 
      {
        U_tilde[i] += schur_inverse[i][j] * diff_tilde[j];
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
      for(unsigned int j = 0; j < n_en; ++j) 
      {
        U_tilde[i] += M_tilde_tilde_inverse[i][j] * (R_tilde[j] - coupling[j]);
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
