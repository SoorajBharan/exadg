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
#include <utility>

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

  BaseIntegrator phi_dg;
  BaseIntegrator phi_en;

  dealii::FEEvaluation<dim, -1, 0, 1, Number> y_eval;
  dealii::FEEvaluation<dim, -1, 0, 1, Number> utau_eval;

  FunctionEnrichment<dim, Number> const * enrichment;
  unsigned int n_q_points;

  // Constructor for CellIntegrator
  FEEnrichedEvaluation(dealii::MatrixFree<dim, Number> const & mf,
                       FunctionEnrichment<dim, Number> const & enrichment_in)
    : phi_dg(mf, enrichment_in.get_dof_index_shadow_vector(), enrichment_in.get_quad_index()),
    phi_en(mf, enrichment_in.get_dof_index_en(), enrichment_in.get_quad_index()),
    y_eval(mf, enrichment_in.get_dof_index_cg_scalar(), enrichment_in.get_quad_index()),
    utau_eval(mf, enrichment_in.get_dof_index_cg_scalar(), enrichment_in.get_quad_index()),
    enrichment(&enrichment_in),
    n_q_points(phi_dg.n_q_points) {}

  // Constructor for FaceIntegrator (needs the inner_face boolean)
  FEEnrichedEvaluation(dealii::MatrixFree<dim, Number> const & mf,
                       FunctionEnrichment<dim, Number> const & enrichment_in,
                       bool inner_face)
    : phi_dg(mf, inner_face, enrichment_in.get_dof_index_shadow_vector(), enrichment_in.get_quad_index()),
    phi_en(mf, inner_face, enrichment_in.get_dof_index_en(), enrichment_in.get_quad_index()),
    y_eval(mf, enrichment_in.get_dof_index_cg_scalar(), enrichment_in.get_quad_index()),
    utau_eval(mf, enrichment_in.get_dof_index_cg_scalar(), enrichment_in.get_quad_index()),
    enrichment(&enrichment_in),
    n_q_points(phi_dg.n_q_points) {}

  // Reinit for both spaces
  template<typename Iterator>
  void reinit(Iterator const & iter)
  {
    phi_dg.reinit(iter);
    phi_en.reinit(iter);

    y_eval.reinit(iter);
    utau_eval.reinit(iter);
  }

  // Read DOFs for all required fields simultaneously
  void read_dof_values(VectorType const & dg_sol, VectorType const & cg_sol)
  {
    phi_dg.read_dof_values(dg_sol);

    phi_en.read_dof_values(cg_sol);

    y_eval.read_dof_values(enrichment->wall_distance);

    utau_eval.read_dof_values(enrichment->friction_velocity);
  }

  void evaluate(dealii::EvaluationFlags::EvaluationFlags flags)
  {
    phi_dg.evaluate(flags);
    phi_en.evaluate(flags);

    y_eval.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
    utau_eval.evaluate(dealii::EvaluationFlags::values);
  }

  void gather_evaluate(
    VectorType const & dg_sol,
    VectorType const & cg_sol,
    dealii::EvaluationFlags::EvaluationFlags flags)
  {
    phi_dg.read_dof_values(dg_sol);
    phi_en.read_dof_values(cg_sol);

    y_eval.read_dof_values(enrichment->wall_distance);
    utau_eval.read_dof_values(enrichment->friction_velocity);

    phi_dg.evaluate(flags);
    phi_en.evaluate(flags);
    y_eval.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);
    utau_eval.evaluate(dealii::EvaluationFlags::values);
  }

  void integrate(dealii::EvaluationFlags::EvaluationFlags flags)
  {
    phi_dg.integrate(flags);
    phi_en.integrate(dealii::EvaluationFlags::values |
                     dealii::EvaluationFlags::gradients);
  }

  value_type get_value(unsigned int q) const
  {
    value_type u_dg = phi_dg.get_value(q);
    value_type u_cg = phi_en.get_value(q); // The Continuous Galerkin component

    scalar y    = y_eval.get_value(q);
    scalar utau = utau_eval.get_value(q);

    // Get the scalar Spalding evaluation (psi)
    scalar psi = enrichment->get_value(utau, y);

    // Total Velocity = DG + (enrichment function * CG)
    return u_dg + (psi * u_cg);
  }

  gradient_type get_gradient(unsigned int q) const
  {
    gradient_type grad_u_dg = phi_dg.get_gradient(q);

    // CG properties
    value_type u_cg      = phi_en.get_value(q);
    gradient_type grad_u_cg = phi_en.get_gradient(q);

    // Wall physics
    scalar y    = y_eval.get_value(q);
    scalar utau = utau_eval.get_value(q);

    // Wall normal vector
    vector grad_y = y_eval.get_gradient(q);
    scalar epsilon = dealii::make_vectorized_array<Number>(1e-14);
    vector normal = grad_y / std::max(grad_y.norm(), epsilon);

    scalar psi = enrichment->get_value(utau, y);
    scalar grad_psi = enrichment->get_gradient(utau, y);

    return grad_u_dg +  (psi * grad_u_cg) + dealii::outer_product(u_cg, grad_psi * normal);
  }

  scalar
  get_divergence(unsigned int q) const
  {
    scalar y      = y_eval.get_value(q);
    scalar utau   = utau_eval.get_value(q);
    vector grad_y = y_eval.get_gradient(q);

    scalar epsilon = dealii::make_vectorized_array<Number>(1e-14);
    vector normal = grad_y / std::max(grad_y.norm(), epsilon);

    scalar psi       = enrichment->get_value(utau, y);
    scalar grad_psi  = enrichment->get_gradient(utau, y);
    vector nabla_psi = grad_psi * normal;

    return phi_dg.get_divergence(q) + (psi * phi_en.get_divergence(q)) + (phi_en.get_value(q) * nabla_psi);
  }

  value_type get_normal_derivative(unsigned int q) const
  {
    value_type du_dn_dg = phi_dg.get_normal_derivative(q);
    value_type du_dn_cg = phi_en.get_normal_derivative(q);
    value_type u_cg     = phi_en.get_value(q);

    scalar y    = y_eval.get_value(q);
    scalar utau = utau_eval.get_value(q);

    vector grad_y = y_eval.get_gradient(q);
    scalar epsilon = dealii::make_vectorized_array<Number>(1e-14);
    vector n_wall = grad_y / std::max(grad_y.norm(), epsilon);

    scalar psi        = enrichment->get_value(utau, y);
    scalar grad_psi_y = enrichment->get_gradient(utau, y);

    vector nabla_psi = grad_psi_y * n_wall;

    vector n_face = phi_dg.get_normal_vector(q);

    scalar dpsi_dn_face = nabla_psi * n_face;

    return du_dn_dg + (psi * du_dn_cg) + (dpsi_dn_face * u_cg);
  }

  // Submit values to BOTH test functions (Galerkin requirement)
  void submit_value(value_type const & val, unsigned int q)
  {
    scalar y    = y_eval.get_value(q);
    scalar utau = utau_eval.get_value(q);
    scalar psi  = enrichment->get_value(utau, y);

    phi_dg.submit_value(val, q);
    phi_en.submit_value(psi * val, q);
  }

  void submit_gradient(gradient_type const & val, unsigned int q) 
  {
    scalar y    = y_eval.get_value(q);
    scalar utau = utau_eval.get_value(q);
    
    vector grad_y = y_eval.get_gradient(q);
    scalar epsilon = dealii::make_vectorized_array<Number>(1e-14);
    vector normal = grad_y / std::max(grad_y.norm(), epsilon);

    scalar psi      = enrichment->get_value(utau, y);
    scalar grad_psi = enrichment->get_gradient(utau, y);
    vector nabla_psi = grad_psi * normal;

    phi_dg.submit_gradient(val, q);
    
    phi_en.submit_gradient(psi * val, q);

    phi_en.submit_value(val * nabla_psi, q); 
  }

  void
  submit_divergence(scalar const & val, unsigned int q)
  {
    scalar y    = y_eval.get_value(q);
    scalar utau = utau_eval.get_value(q);
    
    vector grad_y = y_eval.get_gradient(q);
    scalar epsilon = dealii::make_vectorized_array<Number>(1e-14);
    vector normal = grad_y / std::max(grad_y.norm(), epsilon);

    scalar psi      = enrichment->get_value(utau, y);
    scalar grad_psi = enrichment->get_gradient(utau, y);
    vector nabla_psi = grad_psi * normal;

    phi_dg.submit_divergence(val, q);
    
    phi_en.submit_divergence(psi * val, q);

    phi_en.submit_value(val * nabla_psi, q);
  }

  void submit_normal_derivative(value_type const & val, unsigned int q)
  {
    scalar y    = y_eval.get_value(q);
    scalar utau = utau_eval.get_value(q);

    vector grad_y = y_eval.get_gradient(q);
    scalar epsilon = dealii::make_vectorized_array<Number>(1e-14);
    vector n_wall = grad_y / std::max(grad_y.norm(), epsilon);

    scalar psi        = enrichment->get_value(utau, y);
    scalar grad_psi_y = enrichment->get_gradient(utau, y);
    vector nabla_psi  = grad_psi_y * n_wall;

    vector n_face = phi_dg.get_normal_vector(q);
    scalar dpsi_dn_face = nabla_psi * n_face;

    phi_dg.submit_normal_derivative(val, q);
    
    phi_en.submit_normal_derivative(psi * val, q);

    phi_en.submit_value(dpsi_dn_face * val, q);
  }

  // Integrate and Scatter to BOTH global matrices safely
  void integrate_scatter(dealii::EvaluationFlags::EvaluationFlags flags,
                         VectorType & dst_dg,
                         VectorType & dst_en)
  {
    phi_dg.integrate_scatter(flags, dst_dg);
    phi_en.integrate_scatter(flags, dst_en);
  }

  void distribute_local_to_global(VectorType & dst_dg, VectorType & dst_en)
  {
    phi_dg.distribute_local_to_global(dst_dg);
    phi_en.distribute_local_to_global(dst_en);
  }

}; // FEEnrichedEvaluation declaration

} // namespace ExaDG

#endif
