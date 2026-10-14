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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_ENRICHED_BLOCK_JACOBI_PRECONDITIONER_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_ENRICHED_BLOCK_JACOBI_PRECONDITIONER_H_

#include <deal.II/base/aligned_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <exadg/operators/wall_modelling/coupler.h>
#include <exadg/operators/wall_modelling/fe_enriched_evaluation.h>
#include <exadg/operators/wall_modelling/function_enrichment.h>

#include <vector>

namespace ExaDG
{
namespace IncNS
{
// forward declaration: SpatialOperatorBase owns this preconditioner (spatial_operator_base.h
// includes this header), it is only accessed through a pointer in the member functions below
template<int dim, typename Number>
class SpatialOperatorBase;

/*
 * Block-Jacobi preconditioner for the wall-enriched velocity (block(0) = U_bar, block(1) = U_tilde).
 *
 *   non-enriched cells: (1/s) M^{-1}
 *   enriched cells:     exact inverse of the local cell operator
 *                         A_K = s M_K + nu (grad u_h, grad v)_K + tau_K (div u_h, div v)_K
 *                       on the local (U_bar, U_tilde) DoFs (face terms not included).
 *
 * Captures the stiff terms with grad(psi) cell by cell. SPD -> usable with CG.
 */
template<int dim, typename Number>
class EnrichedBlockJacobiPreconditioner
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;
  typedef dealii::VectorizedArray<Number>                         scalar;
  typedef dealii::Tensor<1, dim, scalar>                          vector;
  typedef dealii::Tensor<2, dim, scalar>                          tensor;

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number>      CellIntegratorU;
  typedef FEEnrichedEvaluation<dim, Number, CellIntegratorU> EnrichedCellIntegrator;

  EnrichedBlockJacobiPreconditioner()
    : spatial_operator(nullptr), mass_factor(1.0), viscosity(0.0), n_local(0)
  {
  }

  void
  initialize(SpatialOperatorBase<dim, Number> const & spatial_operator_in)
  {
    spatial_operator = &spatial_operator_in;
  }

  // Assemble and invert the local operators. Call before every solve (psi changes every step).
  //   mass_factor_in    : s
  //   viscosity_in      : nu (0 for the penalty step)
  //   tau_divergence_in : per cell batch of matrix_free_en, ALREADY multiplied by dt (nullptr: none)
  void
  update(double const                          mass_factor_in,
         double const                          viscosity_in,
         dealii::AlignedVector<scalar> const * tau_divergence_in,
         VectorType const *                    free_bar = nullptr)
  {
    AssertThrow(spatial_operator != nullptr,
                dealii::ExcMessage("EnrichedBlockJacobiPreconditioner not initialized."));
    AssertThrow(mass_factor_in > 0.0, dealii::ExcMessage("Mass factor must be positive."));

    mass_factor = mass_factor_in;
    viscosity   = viscosity_in;

    auto const & function_enrichment = spatial_operator->get_function_enrichment();
    auto const & mf                  = *function_enrichment->get_matrix_free();
    auto const & coupler             = *spatial_operator->get_wall_dg_coupler();

    // zero vectors: only used so that y and u_tau (-> psi) are evaluated
    VectorType zero_bar, zero_tilde;
    mf.initialize_dof_vector(zero_bar, function_enrichment->get_dof_index_shadow_vector());
    mf.initialize_dof_vector(zero_tilde, function_enrichment->get_dof_index_en());
    zero_bar.update_ghost_values();
    zero_tilde.update_ghost_values();

    EnrichedCellIntegrator phi(mf, *function_enrichment);

    unsigned int const n_bar   = phi.phi_dg.dofs_per_cell;
    unsigned int const n_tilde = phi.phi_en.dofs_per_cell;
    n_local                    = n_bar + n_tilde;

    scalar const s  = dealii::make_vectorized_array<Number>(mass_factor);
    scalar const nu = dealii::make_vectorized_array<Number>(viscosity);
    scalar const zero = dealii::make_vectorized_array<Number>(0.0);
    scalar const one  = dealii::make_vectorized_array<Number>(1.0);


    // strong no-slip: E_bar in shadow layout, read with the same FEEvaluation numbering as phi_dg
    VectorType free_bar_shadow;
    if(free_bar != nullptr)
    {
      mf.initialize_dof_vector(free_bar_shadow, function_enrichment->get_dof_index_shadow_vector());
      function_enrichment->copy_global_dg_to_wall_layout(*free_bar, free_bar_shadow);
    }
    CellIntegratorU phi_free(mf,
                             function_enrichment->get_dof_index_shadow_vector(),
                             function_enrichment->get_quad_index(),
                             0,
                             1);

    local_inverse.clear();
    local_inverse.resize(mf.n_cell_batches());

    for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
    {
      if(mf.get_cell_iterator(cell, 0)->active_fe_index() != 1)
        continue;

      phi.reinit(cell);
      phi.gather_evaluate(zero_bar, zero_tilde,
                          dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

      scalar const tau = (tau_divergence_in != nullptr) ? (*tau_divergence_in)[cell] : zero;

      std::vector<scalar> A(n_local * n_local, zero);

      // column j = local operator applied to unit DoF j
      for(unsigned int j = 0; j < n_local; ++j)
      {
        for(unsigned int i = 0; i < n_bar; ++i)
        {
          phi.phi_dg.begin_dof_values()[i] = (i == j) ? one : zero;
        }
        for(unsigned int i = 0; i < n_tilde; ++i)
        {
          phi.phi_en.begin_dof_values()[i] = (n_bar + i == j) ? one : zero;
        }

        phi.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

        for(unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          vector const value_flux    = s * phi.get_value(q);
          tensor       gradient_flux = nu * phi.get_gradient(q);
          if(tau_divergence_in != nullptr)
          {
            scalar const divergence = tau * phi.get_divergence(q);
            for(unsigned int d = 0; d < dim; ++d)
            {
              gradient_flux[d][d] += divergence;
            }
          }
          phi.submit_value_and_gradient(value_flux, gradient_flux, q);
        }

        phi.integrate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

        for(unsigned int i = 0; i < n_bar; ++i)
        {
          A[i * n_local + j] = phi.phi_dg.begin_dof_values()[i];
        }
        for(unsigned int i = 0; i < n_tilde; ++i)
        {
          A[(n_bar + i) * n_local + j] = phi.phi_en.begin_dof_values()[i];
        }
      }

        /*
       * Step 5b: local operator on the well-posed enrichment space (same restriction as
       * EnrichmentConstrainedOperator):
       *
       *   A_c = E A E + [ 0    0   ],   E = [ I  0 ]
       *                 [ 0  I - P ]        [ 0  P ]
       *
       * P (n_en x n_en, WallDGCoupler) acts on each velocity component. Local U_tilde DoF
       * a = c * n_en + j (component c, shape function j), so the full projector is
       * block-diagonal: P_full(a, b) = P(a % n_en, b % n_en) if a / n_en == b / n_en, else 0.
       * P = I (no mode removed) gives A_c = A.
       */
        auto const &       P    = coupler.cell_schur_data[cell].projector;
        unsigned int const n_en = P.size();

        auto const P_full = [&](unsigned int const a, unsigned int const b) -> scalar {
        return (a / n_en == b / n_en) ? P[a % n_en][b % n_en] : zero;
        };

        // (A E): U_tilde columns multiplied by P from the right
        std::vector<scalar> AE(A);
        for(unsigned int i = 0; i < n_local; ++i)
          for(unsigned int b = 0; b < n_tilde; ++b)
          {
            scalar sum = zero;
            for(unsigned int a = 0; a < n_tilde; ++a)
              sum += A[i * n_local + n_bar + a] * P_full(a, b);
            AE[i * n_local + n_bar + b] = sum;
          }

        // E (A E): U_tilde rows multiplied by P from the left
        A = AE;
        for(unsigned int a = 0; a < n_tilde; ++a)
          for(unsigned int j = 0; j < n_local; ++j)
          {
            scalar sum = zero;
            for(unsigned int b = 0; b < n_tilde; ++b)
              sum += P_full(a, b) * AE[(n_bar + b) * n_local + j];
            A[(n_bar + a) * n_local + j] = sum;
          }

        // + (I - P) in the U_tilde block: identity on the removed modes
        for(unsigned int a = 0; a < n_tilde; ++a)
          for(unsigned int b = 0; b < n_tilde; ++b)
            A[(n_bar + a) * n_local + n_bar + b] += (a == b ? one : zero) - P_full(a, b);

      /*
       * Step 7c, strong no-slip: same restriction for the fixed U_bar DoFs,
       *   A_c(i,j) = f_i f_j A(i,j) + (1 - f_i) delta_ij,
       * f_i = E_bar(i,i) for U_bar DoFs (0: fixed), f_i = 1 for U_tilde DoFs.
       */
      if(free_bar != nullptr)
      {
        phi_free.reinit(cell);
        phi_free.read_dof_values_plain(free_bar_shadow);

        std::vector<scalar> f(n_local, one);
        for(unsigned int i = 0; i < n_bar; ++i)
          f[i] = phi_free.begin_dof_values()[i];

        for(unsigned int i = 0; i < n_local; ++i)
          for(unsigned int k = 0; k < n_local; ++k)
            A[i * n_local + k] = f[i] * f[k] * A[i * n_local + k] + (i == k ? one - f[i] : zero);
      }

      // unused SIMD lanes: identity (keeps the inversion finite)
      for(unsigned int v = mf.n_active_entries_per_cell_batch(cell); v < scalar::size(); ++v)
        for(unsigned int i = 0; i < n_local; ++i)
          for(unsigned int k = 0; k < n_local; ++k)
            A[i * n_local + k][v] = (i == k) ? 1.0 : 0.0;

      local_inverse[cell] = invert(A, n_local);
    }
  }

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    auto const & function_enrichment = spatial_operator->get_function_enrichment();
    auto const & mf                  = *function_enrichment->get_matrix_free();

    dst.block(0).zero_out_ghost_values();
    dst.block(1).zero_out_ghost_values();

    // (1) all cells: (1/s) M^{-1}  (standard DG inverse mass)
    spatial_operator->apply_inverse_mass_operator(dst.block(0), src.block(0));
    dst.block(0) *= 1.0 / mass_factor;

    // (2) enriched cells: exact inverse of the local operator
    VectorType r_bar, z_bar;
    mf.initialize_dof_vector(r_bar, function_enrichment->get_dof_index_shadow_vector());
    mf.initialize_dof_vector(z_bar, function_enrichment->get_dof_index_shadow_vector());
    function_enrichment->copy_global_dg_to_wall_layout(src.block(0), r_bar);

    dst.block(1) = 0.0;

    unsigned int const quad_index = function_enrichment->get_quad_index();
    CellIntegratorU    phi_bar(mf, function_enrichment->get_dof_index_shadow_vector(), quad_index, 0, 1);
    CellIntegratorU    phi_tilde(mf, function_enrichment->get_dof_index_en(), quad_index, 0, 1);

    unsigned int const n_bar   = phi_bar.dofs_per_cell;
    unsigned int const n_tilde = phi_tilde.dofs_per_cell;

    std::vector<scalar> r(n_local), z(n_local);

    for(unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell)
    {
      if(mf.get_cell_iterator(cell, 0)->active_fe_index() != 1)
        continue;

      phi_bar.reinit(cell);
      phi_tilde.reinit(cell);
      phi_bar.read_dof_values_plain(r_bar);
      phi_tilde.read_dof_values_plain(src.block(1));

      for(unsigned int i = 0; i < n_bar; ++i)
        r[i] = phi_bar.begin_dof_values()[i];
      for(unsigned int i = 0; i < n_tilde; ++i)
        r[n_bar + i] = phi_tilde.begin_dof_values()[i];

      std::vector<scalar> const & A_inverse = local_inverse[cell];
      for(unsigned int i = 0; i < n_local; ++i)
      {
        z[i] = dealii::make_vectorized_array<Number>(0.0);
        for(unsigned int k = 0; k < n_local; ++k)
          z[i] += A_inverse[i * n_local + k] * r[k];
      }

      for(unsigned int i = 0; i < n_bar; ++i)
        phi_bar.begin_dof_values()[i] = z[i];
      for(unsigned int i = 0; i < n_tilde; ++i)
        phi_tilde.begin_dof_values()[i] = z[n_bar + i];

      phi_bar.set_dof_values_plain(z_bar);
      phi_tilde.set_dof_values_plain(dst.block(1));
    }

    // overwrite the enriched cells of block(0)
    function_enrichment->copy_wall_layout_to_global_dg(z_bar, dst.block(0));
  }

private:
  // Gauss-Jordan inversion of a dense n x n matrix (row-major), SIMD over the lanes.
  // The local operator is SPD -> no pivoting needed.
  static std::vector<scalar>
  invert(std::vector<scalar> A, unsigned int const n)
  {
    std::vector<scalar> inverse(n * n, dealii::make_vectorized_array<Number>(0.0));
    for(unsigned int i = 0; i < n; ++i)
      inverse[i * n + i] = dealii::make_vectorized_array<Number>(1.0);

    for(unsigned int i = 0; i < n; ++i)
    {
      scalar const pivot = dealii::make_vectorized_array<Number>(1.0) / A[i * n + i];
      for(unsigned int j = 0; j < n; ++j)
      {
        A[i * n + j] *= pivot;
        inverse[i * n + j] *= pivot;
      }
      for(unsigned int k = 0; k < n; ++k)
      {
        if(k == i)
          continue;
        scalar const factor = A[k * n + i];
        for(unsigned int j = 0; j < n; ++j)
        {
          A[k * n + j] -= factor * A[i * n + j];
          inverse[k * n + j] -= factor * inverse[i * n + j];
        }
      }
    }
    return inverse;
  }

  SpatialOperatorBase<dim, Number> const * spatial_operator;
  double                                   mass_factor;
  double                                   viscosity;
  unsigned int                             n_local;

  // per cell batch of matrix_free_en (empty for non-enriched batches), row-major n_local x n_local
  std::vector<std::vector<scalar>> local_inverse;
};

} // namespace IncNS
} // namespace ExaDG

#endif
