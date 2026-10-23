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

#ifndef INCLUDE_OPERATORS_WALL_MODELLING_FE_ENRICHED_EVALUATION_H_
#define INCLUDE_OPERATORS_WALL_MODELLING_FE_ENRICHED_EVALUATION_H_

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/evaluation_flags.h>

#include <exadg/operators/wall_modelling/function_enrichment.h>

namespace ExaDG
{

template<int dim, typename Number, typename BaseIntegrator>
class FEEnrichedEvaluation
{
public:
  typedef typename BaseIntegrator::value_type value_type;
  typedef typename BaseIntegrator::gradient_type gradient_type;

  using scalar = dealii::VectorizedArray<Number>;
  using vector = dealii::Tensor<1, dim, scalar>;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  // polynomial part \bar{u}_h (shadow layout of the global DG velocity)
  BaseIntegrator phi_dg;
  BaseIntegrator phi_en;

  FunctionEnrichment<dim, Number> const * enrichment;
  unsigned int n_q_points;

  FEEnrichedEvaluation(dealii::MatrixFree<dim, Number> const & mf,
                       FunctionEnrichment<dim, Number> const & enrichment_in)
    : phi_dg(mf,
             enrichment_in.get_dof_index_shadow_vector(),
             enrichment_in.get_quad_index(),
             0,
             enrichment_in.get_active_fe_index()),
      phi_en(mf,
             enrichment_in.get_dof_index_en(),
             enrichment_in.get_quad_index(),
             0,
             enrichment_in.get_active_fe_index()),
      enrichment(&enrichment_in),
      n_q_points(phi_dg.n_q_points),
      cell(dealii::numbers::invalid_unsigned_int),
      en_gradient_submitted(false)
  {
  }

  // Reinit for both spaces
  void reinit(unsigned int const cell_batch)
  {
    phi_dg.reinit(cell_batch);
    phi_en.reinit(cell_batch);

    cell = cell_batch;
    en_gradient_submitted = false;
  }

  void
  evaluate_enrichment_function()
  {}

  // Read DOFs for all required fields simultaneously
  void read_dof_values(VectorType const & dg_sol, VectorType const & en_sol)
  {
    phi_dg.read_dof_values(dg_sol);

    phi_en.read_dof_values(en_sol);
  }

  void evaluate(dealii::EvaluationFlags::EvaluationFlags flags)
  {
    phi_dg.evaluate(flags);
    phi_en.evaluate(flags);
  }

  void gather_evaluate(
    VectorType const & dg_sol,
    VectorType const & en_sol,
    dealii::EvaluationFlags::EvaluationFlags flags)
  {
    read_dof_values(dg_sol, en_sol);
    evaluate(flags);
  }

  void integrate(dealii::EvaluationFlags::EvaluationFlags flags)
  {
    phi_dg.integrate(flags);

    dealii::EvaluationFlags::EvaluationFlags enrichment_flags = dealii::EvaluationFlags::values;
    if (en_gradient_submitted)
    {
      enrichment_flags |= dealii::EvaluationFlags::gradients;
    }

    phi_en.integrate(enrichment_flags);
  }

  value_type get_value(unsigned int q) const
  {
    return phi_dg.get_value(q) + psi(q) * phi_en.get_value(q);
  }

  gradient_type get_gradient(unsigned int q) const
  {
    return phi_dg.get_gradient(q) + psi(q) * phi_en.get_gradient(q) + dealii::outer_product(phi_en.get_value(q), grad_psi(q));
  }

  scalar
  get_divergence(unsigned int q) const
  {
    return phi_dg.get_divergence(q) + (psi(q) * phi_en.get_divergence(q)) + (phi_en.get_value(q) * grad_psi(q));
  }

  // enrichment part psi U_tilde only (polynomial part U_bar excluded)
  value_type 
  get_value_enrichment(unsigned int q) const
  {
    return psi(q) * phi_en.get_value(q);
  }

  gradient_type 
  get_gradient_enrichment(unsigned int q) const
  {
    return psi(q) * phi_en.get_gradient(q) +
    dealii::outer_product(phi_en.get_value(q), grad_psi(q));
  }

  scalar
  get_divergence_enrichment(unsigned int q) const
  {
    return (psi(q) * phi_en.get_divergence(q)) + (phi_en.get_value(q) *
    grad_psi(q));
  }

  // different fluxes for the two test spaces: (value_bar, gradient_bar) tested  with v_bar,
  // (value_tilde, gradient_tilde) tested with psi v_tilde
  void submit_value_and_gradient_split(value_type const &    value_bar,
                                       gradient_type const & gradient_bar,
                                       value_type const &    value_tilde,
                                       gradient_type const & gradient_tilde,
                                       unsigned int          q)
  {
    phi_dg.submit_value(value_bar, q); 
    phi_dg.submit_gradient(gradient_bar, q);

    phi_en.submit_value(psi(q) * value_tilde + gradient_tilde * grad_psi(q), q);
    phi_en.submit_gradient(psi(q) * gradient_tilde, q);

    en_gradient_submitted = true;
  }

  // Submit values to BOTH test functions (Galerkin requirement)
  void submit_value(value_type const & val, unsigned int q)
  {
    phi_dg.submit_value(val, q);
    phi_en.submit_value(psi(q) * val, q);
  }

  void submit_gradient(gradient_type const & gradient, unsigned int q) 
  {
    phi_dg.submit_gradient(gradient, q);
    
    phi_en.submit_gradient(psi(q) * gradient, q);

    phi_en.submit_value(gradient * grad_psi(q), q); 

    en_gradient_submitted = true;
  }

  // When submit_value and submit_graadient are called together the second call will overwrite the value slot submitted in the first call. This function submits both value and gradient to both test functions safely.
  void submit_value_and_gradient(value_type const &    value,
                                 gradient_type const & gradient,
                                 unsigned int          q)
  {
    phi_dg.submit_value(value, q);
    phi_dg.submit_gradient(gradient, q);

    phi_en.submit_value(psi(q) * value + gradient * grad_psi(q), q);
    phi_en.submit_gradient(psi(q) * gradient, q);

    en_gradient_submitted = true;
  }

  void
  submit_divergence(scalar const & divergence, unsigned int q)
  {
    phi_dg.submit_divergence(divergence, q);
    
    phi_en.submit_divergence(psi(q) * divergence, q);

    phi_en.submit_value(divergence * grad_psi(q), q);

    en_gradient_submitted = true;
  }

  // Integrate and Scatter to BOTH global matrices safely
  void integrate_scatter(dealii::EvaluationFlags::EvaluationFlags flags,
                         VectorType & dst_dg,
                         VectorType & dst_en)
  {
    integrate(flags);
    distribute_local_to_global(dst_dg, dst_en);
  }

  void distribute_local_to_global(VectorType & dst_dg, VectorType & dst_en)
  {
    phi_dg.distribute_local_to_global(dst_dg);
    phi_en.distribute_local_to_global(dst_en);
  }

  vector get_normal_vector(unsigned int q) const
  {
    return phi_dg.get_normal_vector(q);
  }

private:
  /**
   * Cached \f$\psi = \kappa\,u^+(y^+)\f$ at quadrature point q of the current cell batch.
   */
  scalar const &
  psi(unsigned int const q) const
  {
    return enrichment->get_psi_cell(cell, q);
  }

  /**
   * Cached \f$\nabla\psi = \kappa\,\frac{\mathrm{d}u^+}{\mathrm{d}y^+}\,\nabla y^+\f$ at
   * quadrature point q of the current cell batch.
   */
  vector const &
  grad_psi(unsigned int const q) const
  {
    return enrichment->get_grad_psi_cell(cell, q);
  }

  unsigned int cell;
  bool en_gradient_submitted;

}; // FEEnrichedEvaluation declaration

} // namespace ExaDG

#endif
