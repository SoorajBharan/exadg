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

#ifndef INCLUDE_EXADG_OPERATORS_WALL_MODELLING_FE_OMEGA_ENRICHED_EVALUATION_H_
#define INCLUDE_EXADG_OPERATORS_WALL_MODELLING_FE_OMEGA_ENRICHED_EVALUATION_H_

#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace ExaDG
{
/*
 * Enrichment of omega in the wall cells, as for the velocity (U_bar + psi U_tilde):
 *
 *   omega:     omega_h = omega_bar + psi_omega omega_tilde          psi = psi_omega
 *   ln omega:  W_h     = W_bar     + ln(psi_omega) W_tilde          psi = ln(psi_omega)
 *
 * psi_omega: capped blended function sqrt(omega_vis+^2 + omega_log+^2) of y+.
 * The class only sees psi and grad psi of one variable (cached by FunctionEnrichmentRANS).
 * Test functions: v_bar and psi v_tilde (Galerkin).
 */
template<int dim, typename Number>
class FunctionEnrichmentRANS;

/*
 * Cell integrator. BaseIntegrator: scalar FEEvaluation (CellIntegrator<dim, 1, Number>).
 */
template<int dim, typename Number, typename BaseIntegrator>
class FEOmegaEnrichedEvaluation
{
public:
  using scalar     = dealii::VectorizedArray<Number>;
  using vector     = dealii::Tensor<1, dim, scalar>;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  // polynomial part omega_bar (W_bar)
  BaseIntegrator phi_dg;
  // enrichment unknown omega_tilde (W_tilde)
  BaseIntegrator phi_en;

  FunctionEnrichmentRANS<dim, Number> const * enrichment;
  unsigned int                         n_q_points;

  FEOmegaEnrichedEvaluation(dealii::MatrixFree<dim, Number> const & mf,
                            FunctionEnrichmentRANS<dim, Number> const & enrichment_in,
                            unsigned int const                          variable_in)
    : phi_dg(mf,
             enrichment_in.get_dof_index_dg(),
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
      variable(variable_in),
      en_gradient_submitted(false)
  {
  }

  void
  reinit(unsigned int const cell_batch)
  {
    phi_dg.reinit(cell_batch);
    phi_en.reinit(cell_batch);

    cell                  = cell_batch;
    en_gradient_submitted = false;
  }

  void
  read_dof_values(VectorType const & dg_sol, VectorType const & en_sol)
  {
    phi_dg.read_dof_values(dg_sol);
    phi_en.read_dof_values(en_sol);
  }

  void
  evaluate(dealii::EvaluationFlags::EvaluationFlags const flags)
  {
    phi_dg.evaluate(flags);
    phi_en.evaluate(flags);
  }

  void
  gather_evaluate(VectorType const &                             dg_sol,
                  VectorType const &                             en_sol,
                  dealii::EvaluationFlags::EvaluationFlags const flags)
  {
    read_dof_values(dg_sol, en_sol);
    evaluate(flags);
  }

  void
  integrate(dealii::EvaluationFlags::EvaluationFlags const flags)
  {
    phi_dg.integrate(flags);

    dealii::EvaluationFlags::EvaluationFlags enrichment_flags = dealii::EvaluationFlags::values;
    if(en_gradient_submitted)
      enrichment_flags |= dealii::EvaluationFlags::gradients;

    phi_en.integrate(enrichment_flags);
  }

  // omega_h = omega_bar + psi omega_tilde
  scalar
  get_value(unsigned int const q) const
  {
    return phi_dg.get_value(q) + psi(q) * phi_en.get_value(q);
  }

  // grad omega_h = grad omega_bar + psi grad omega_tilde + omega_tilde grad psi
  vector
  get_gradient(unsigned int const q) const
  {
    return phi_dg.get_gradient(q) + psi(q) * phi_en.get_gradient(q) +
           phi_en.get_value(q) * grad_psi(q);
  }

  // enrichment part psi omega_tilde only (omega_bar excluded)
  scalar
  get_value_enrichment(unsigned int const q) const
  {
    return psi(q) * phi_en.get_value(q);
  }

  vector
  get_gradient_enrichment(unsigned int const q) const
  {
    return psi(q) * phi_en.get_gradient(q) + phi_en.get_value(q) * grad_psi(q);
  }

  // test with v_bar and psi v_tilde
  void
  submit_value(scalar const & value, unsigned int const q)
  {
    phi_dg.submit_value(value, q);
    phi_en.submit_value(psi(q) * value, q);
  }

  // grad(psi v_tilde) = psi grad v_tilde + v_tilde grad psi
  void
  submit_gradient(vector const & gradient, unsigned int const q)
  {
    phi_dg.submit_gradient(gradient, q);
    phi_en.submit_gradient(psi(q) * gradient, q);
    phi_en.submit_value(gradient * grad_psi(q), q);

    en_gradient_submitted = true;
  }

  // value and gradient together: submit_gradient alone would overwrite the value slot of phi_en
  void
  submit_value_and_gradient(scalar const & value, vector const & gradient, unsigned int const q)
  {
    phi_dg.submit_value(value, q);
    phi_dg.submit_gradient(gradient, q);

    phi_en.submit_value(psi(q) * value + gradient * grad_psi(q), q);
    phi_en.submit_gradient(psi(q) * gradient, q);

    en_gradient_submitted = true;
  }

  void
  integrate_scatter(dealii::EvaluationFlags::EvaluationFlags const flags,
                    VectorType &                                   dst_dg,
                    VectorType &                                   dst_en)
  {
    integrate(flags);
    distribute_local_to_global(dst_dg, dst_en);
  }

  void
  distribute_local_to_global(VectorType & dst_dg, VectorType & dst_en)
  {
    phi_dg.distribute_local_to_global(dst_dg);
    phi_en.distribute_local_to_global(dst_en);
  }

private:
  // psi_omega (omega) or ln(psi_omega) (ln omega) at quadrature point q of the current cell batch
  scalar const &
  psi(unsigned int const q) const
  {
    return enrichment->get_psi_cell(variable, cell, q);
  }

  vector const &
  grad_psi(unsigned int const q) const
  {
    return enrichment->get_grad_psi_cell(variable, cell, q);
  }

  unsigned int cell;
  unsigned int variable;
  bool         en_gradient_submitted;
};

/*
 * Face integrator. BaseIntegrator: scalar FEFaceEvaluation (FaceIntegrator<dim, 1, Number>).
 * psi per face batch and side: it jumps at the top of the wall layer.
 */
template<int dim, typename Number, typename BaseIntegrator>
class FEFaceOmegaEnrichedEvaluation
{
public:
  using scalar     = dealii::VectorizedArray<Number>;
  using vector     = dealii::Tensor<1, dim, scalar>;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  BaseIntegrator phi_dg;
  BaseIntegrator phi_en;

  FunctionEnrichmentRANS<dim, Number> const * enrichment;
  unsigned int                         n_q_points;

  FEFaceOmegaEnrichedEvaluation(dealii::MatrixFree<dim, Number> const & mf,
                                FunctionEnrichmentRANS<dim, Number> const & enrichment_in,
                                bool const                              is_interior_face,
                                unsigned int const                      active_fe_index,
                                unsigned int const                      variable_in)
    : phi_dg(mf,
             is_interior_face,
             enrichment_in.get_dof_index_dg(),
             enrichment_in.get_quad_index(),
             0,
             active_fe_index),
      phi_en(mf,
             is_interior_face,
             enrichment_in.get_dof_index_en(),
             enrichment_in.get_quad_index(),
             0,
             active_fe_index),
      enrichment(&enrichment_in),
      n_q_points(phi_dg.n_q_points),
      face(dealii::numbers::invalid_unsigned_int),
      interior(is_interior_face),
      variable(variable_in),
      en_gradient_submitted(false)
  {
  }

  void
  reinit(unsigned int const face_batch)
  {
    phi_dg.reinit(face_batch);
    phi_en.reinit(face_batch);

    face                  = face_batch;
    en_gradient_submitted = false;
  }

  void
  read_dof_values(VectorType const & dg_sol, VectorType const & en_sol)
  {
    phi_dg.read_dof_values(dg_sol);
    phi_en.read_dof_values(en_sol);
  }

  void
  evaluate(dealii::EvaluationFlags::EvaluationFlags const flags)
  {
    phi_dg.evaluate(flags);
    phi_en.evaluate(flags);
  }

  void
  gather_evaluate(VectorType const &                             dg_sol,
                  VectorType const &                             en_sol,
                  dealii::EvaluationFlags::EvaluationFlags const flags)
  {
    read_dof_values(dg_sol, en_sol);
    evaluate(flags);
  }

  void
  integrate(dealii::EvaluationFlags::EvaluationFlags const flags)
  {
    phi_dg.integrate(flags);

    dealii::EvaluationFlags::EvaluationFlags enrichment_flags = dealii::EvaluationFlags::values;
    if(en_gradient_submitted)
      enrichment_flags |= dealii::EvaluationFlags::gradients;

    phi_en.integrate(enrichment_flags);
  }

  scalar
  get_value(unsigned int const q) const
  {
    return phi_dg.get_value(q) + psi(q) * phi_en.get_value(q);
  }

  vector
  get_gradient(unsigned int const q) const
  {
    return phi_dg.get_gradient(q) + psi(q) * phi_en.get_gradient(q) +
           phi_en.get_value(q) * grad_psi(q);
  }

  // grad omega_h . n
  scalar
  get_normal_derivative(unsigned int const q) const
  {
    return phi_dg.get_normal_derivative(q) + psi(q) * phi_en.get_normal_derivative(q) +
           (grad_psi(q) * get_normal_vector(q)) * phi_en.get_value(q);
  }

  scalar
  get_value_enrichment(unsigned int const q) const
  {
    return psi(q) * phi_en.get_value(q);
  }

  vector
  get_gradient_enrichment(unsigned int const q) const
  {
    return psi(q) * phi_en.get_gradient(q) + phi_en.get_value(q) * grad_psi(q);
  }

  void
  submit_value(scalar const & value, unsigned int const q)
  {
    phi_dg.submit_value(value, q);
    phi_en.submit_value(psi(q) * value, q);
  }

  void
  submit_gradient(vector const & gradient, unsigned int const q)
  {
    phi_dg.submit_gradient(gradient, q);
    phi_en.submit_gradient(psi(q) * gradient, q);
    phi_en.submit_value(gradient * grad_psi(q), q);

    en_gradient_submitted = true;
  }

  void
  submit_value_and_gradient(scalar const & value, vector const & gradient, unsigned int const q)
  {
    phi_dg.submit_value(value, q);
    phi_dg.submit_gradient(gradient, q);

    phi_en.submit_value(psi(q) * value + gradient * grad_psi(q), q);
    phi_en.submit_gradient(psi(q) * gradient, q);

    en_gradient_submitted = true;
  }

  // d(psi v_tilde)/dn = psi dv_tilde/dn + v_tilde (grad psi . n)
  void
  submit_normal_derivative(scalar const & value, unsigned int const q)
  {
    phi_dg.submit_normal_derivative(value, q);
    phi_en.submit_normal_derivative(psi(q) * value, q);
    phi_en.submit_value((grad_psi(q) * get_normal_vector(q)) * value, q);

    en_gradient_submitted = true;
  }

  // value and normal derivative together (SIPG): submit_normal_derivative alone would overwrite
  // the value slot of phi_en
  void
  submit_value_and_normal_derivative(scalar const &     value,
                                     scalar const &     normal_derivative,
                                     unsigned int const q)
  {
    phi_dg.submit_value(value, q);
    phi_dg.submit_normal_derivative(normal_derivative, q);

    phi_en.submit_value(psi(q) * value + (grad_psi(q) * get_normal_vector(q)) * normal_derivative,
                        q);
    phi_en.submit_normal_derivative(psi(q) * normal_derivative, q);

    en_gradient_submitted = true;
  }

  void
  integrate_scatter(dealii::EvaluationFlags::EvaluationFlags const flags,
                    VectorType &                                   dst_dg,
                    VectorType &                                   dst_en)
  {
    integrate(flags);
    distribute_local_to_global(dst_dg, dst_en);
  }

  void
  distribute_local_to_global(VectorType & dst_dg, VectorType & dst_en)
  {
    phi_dg.distribute_local_to_global(dst_dg);
    phi_en.distribute_local_to_global(dst_en);
  }

  vector
  get_normal_vector(unsigned int const q) const
  {
    return phi_dg.get_normal_vector(q);
  }

private:
  // psi_omega (omega) or ln(psi_omega) (ln omega) at quadrature point q of the current face and side
  scalar const &
  psi(unsigned int const q) const
  {
    return enrichment->get_psi_face(variable, face, interior, q);
  }

  vector const &
  grad_psi(unsigned int const q) const
  {
    return enrichment->get_grad_psi_face(variable, face, interior, q);
  }

  unsigned int face;
  bool         interior;
  unsigned int variable;
  bool         en_gradient_submitted;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_WALL_MODELLING_FE_OMEGA_ENRICHED_EVALUATION_H_ */
