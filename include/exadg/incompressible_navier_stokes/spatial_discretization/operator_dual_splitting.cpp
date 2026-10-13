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
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include "operator_dual_splitting.h"
#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_momentum.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/inverse_mass_preconditioner.h>
#include <exadg/solvers_and_preconditioners/preconditioners/jacobi_preconditioner.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
OperatorDualSplitting<dim, Number>::OperatorDualSplitting(
  std::shared_ptr<Grid<dim> const>                      grid_in,
  std::shared_ptr<dealii::Mapping<dim> const>           mapping_in,
  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings_in,
  std::shared_ptr<BoundaryDescriptor<dim> const>        boundary_descriptor_in,
  std::shared_ptr<FieldFunctions<dim> const>            field_functions_in,
  Parameters const &                                    parameters_in,
  std::string const &                                   field_in,
  MPI_Comm const &                                      mpi_comm_in)
  : ProjectionBase(grid_in,
                   mapping_in,
                   multigrid_mappings_in,
                   boundary_descriptor_in,
                   field_functions_in,
                   parameters_in,
                   field_in,
                   mpi_comm_in)
{
}

template<int dim, typename Number>
OperatorDualSplitting<dim, Number>::~OperatorDualSplitting()
{
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::apply_velocity_divergence_term(VectorType &       dst,
                                                                   VectorType const & src) const
{
  this->divergence_operator.apply(dst, src);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::rhs_ppe_div_term_body_forces_add(VectorType &   dst,
                                                                     double const & time) const
{
  this->evaluation_time = time;

  VectorType src_dummy;
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_div_term_body_forces_boundary_face,
                               this,
                               dst,
                               src_dummy);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::local_rhs_ppe_div_term_body_forces_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FaceIntegratorP integrator(matrix_free, true, dof_index_pressure, quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);

    BoundaryTypeU boundary_type =
      this->boundary_descriptor->velocity->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet or
         boundary_type == BoundaryTypeU::DirichletCached or
         boundary_type == BoundaryTypeU::WallEnrichment)
      {
        dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

        // evaluate right-hand side
        vector rhs =
          FunctionEvaluator<1, dim, Number>::value(*(this->field_functions->right_hand_side),
                                                   q_points,
                                                   this->evaluation_time);

        scalar flux_times_normal = rhs * integrator.get_normal_vector(q);
        // minus sign is introduced here which allows to call a function of type ...add()
        // and avoids a scaling of the resulting vector by the factor -1.0
        integrator.submit_value(-flux_times_normal, q);
      }
      else if(boundary_type == BoundaryTypeU::Neumann or boundary_type == BoundaryTypeU::Symmetry)
      {
        // Do nothing on Neumann and symmetry boundaries.
        // Remark: On symmetry boundaries it follows from g_u * n = 0 that also g_{u_hat} * n = 0.
        // Hence, a symmetry boundary for u is also a symmetry boundary for u_hat. Hence, there
        // are no inhomogeneous contributions on symmetry boundaries.
        scalar zero = dealii::make_vectorized_array<Number>(0.0);
        integrator.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    integrator.integrate(dealii::EvaluationFlags::values);
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::rhs_velocity_divergence_term_dirichlet_bc_from_dof_vector(
  VectorType &       dst,
  VectorType const & velocity) const
{
  this->divergence_operator.rhs_bc_from_dof_vector(dst, velocity);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::rhs_ppe_div_term_convective_term_add(
  VectorType &       dst,
  VectorType const & src) const
{
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_div_term_convective_term_boundary_face,
                               this,
                               dst,
                               src);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::local_rhs_ppe_div_term_convective_term_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  unsigned int const dof_index_velocity = this->get_dof_index_velocity();
  unsigned int const dof_index_pressure = this->get_dof_index_pressure();
  unsigned int const quad_index         = this->get_quad_index_velocity_overintegration();

  FaceIntegratorU velocity(matrix_free, true, dof_index_velocity, quad_index);
  FaceIntegratorP pressure(matrix_free, true, dof_index_pressure, quad_index);

  FaceIntegratorU grid_velocity(matrix_free, true, dof_index_velocity, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    velocity.reinit(face);
    velocity.gather_evaluate(src,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    if(this->param.ale_formulation)
    {
      grid_velocity.reinit(face);
      grid_velocity.gather_evaluate(this->convective_kernel->get_grid_velocity(),
                                    dealii::EvaluationFlags::values);
    }

    pressure.reinit(face);

    BoundaryTypeU boundary_type =
      this->boundary_descriptor->velocity->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeU::Dirichlet or
         boundary_type == BoundaryTypeU::DirichletCached or
         boundary_type == BoundaryTypeU::WallEnrichment)
      {
        vector normal = pressure.get_normal_vector(q);

        vector u      = velocity.get_value(q);
        tensor grad_u = velocity.get_gradient(q);

        vector flux;
        if(this->param.formulation_convective_term_bc ==
           FormulationConvectiveTerm::DivergenceFormulation)
        {
          scalar div_u = velocity.get_divergence(q);
          flux         = grad_u * u + div_u * u;
        }
        else if(this->param.formulation_convective_term_bc ==
                FormulationConvectiveTerm::ConvectiveFormulation)
        {
          flux = grad_u * u;
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented."));
        }

        if(this->param.ale_formulation)
        {
          flux -= grad_u * grid_velocity.get_value(q);
        }

        scalar flux_times_normal = flux * normal;

        pressure.submit_value(flux_times_normal, q);
      }
      else if(boundary_type == BoundaryTypeU::Neumann or boundary_type == BoundaryTypeU::Symmetry)
      {
        // Do nothing on Neumann and symmetry boundaries.
        // Remark: On symmetry boundaries it follows from g_u * n = 0 that also g_{u_hat} * n = 0.
        // Hence, a symmetry boundary for u is also a symmetry boundary for u_hat. Hence, there
        // are no inhomogeneous contributions on symmetry boundaries.
        scalar zero = dealii::make_vectorized_array<Number>(0.0);
        pressure.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    pressure.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::rhs_ppe_nbc_numerical_time_derivative_add(
  VectorType &       dst,
  VectorType const & acceleration) const
{
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_nbc_numerical_time_derivative_add_boundary_face,
                               this,
                               dst,
                               acceleration);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::local_rhs_ppe_nbc_numerical_time_derivative_add_boundary_face(
  dealii::MatrixFree<dim, Number> const & data,
  VectorType &                            dst,
  VectorType const &                      acceleration,
  Range const &                           face_range) const
{
  unsigned int dof_index_velocity  = this->get_dof_index_velocity();
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_velocity = this->get_quad_index_velocity_standard();

  FaceIntegratorU integrator_velocity(data, true, dof_index_velocity, quad_index_velocity);
  FaceIntegratorP integrator_pressure(data, true, dof_index_pressure, quad_index_velocity);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator_velocity.reinit(face);
    integrator_velocity.gather_evaluate(acceleration, dealii::EvaluationFlags::values);

    integrator_pressure.reinit(face);

    dealii::types::boundary_id boundary_id = data.get_boundary_id(face);
    BoundaryTypeP              boundary_type =
      this->boundary_descriptor->pressure->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator_pressure.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        vector normal = integrator_velocity.get_normal_vector(q);
        vector dudt   = integrator_velocity.get_value(q);
        scalar h      = -normal * dudt;

        integrator_pressure.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        scalar zero = dealii::make_vectorized_array<Number>(0.0);
        integrator_pressure.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }

    integrator_pressure.integrate(dealii::EvaluationFlags::values);
    integrator_pressure.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::rhs_ppe_nbc_body_force_term_add(VectorType &   dst,
                                                                    double const & time) const
{
  this->evaluation_time = time;

  VectorType src_dummy;
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_nbc_body_force_term_add_boundary_face,
                               this,
                               dst,
                               src_dummy);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::local_rhs_ppe_nbc_body_force_term_add_boundary_face(
  dealii::MatrixFree<dim, Number> const & data,
  VectorType &                            dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int dof_index_pressure  = this->get_dof_index_pressure();
  unsigned int quad_index_pressure = this->get_quad_index_pressure();

  FaceIntegratorP integrator(data, true, dof_index_pressure, quad_index_pressure);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    integrator.reinit(face);

    dealii::types::boundary_id boundary_id = data.get_boundary_id(face);
    BoundaryTypeP              boundary_type =
      this->boundary_descriptor->pressure->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        dealii::Point<dim, scalar> q_points = integrator.quadrature_point(q);

        // evaluate right-hand side
        vector rhs =
          FunctionEvaluator<1, dim, Number>::value(*(this->field_functions->right_hand_side),
                                                   q_points,
                                                   this->evaluation_time);

        vector normal = integrator.get_normal_vector(q);

        scalar h = normal * rhs;

        integrator.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        scalar zero = dealii::make_vectorized_array<Number>(0.0);
        integrator.submit_value(zero, q);
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    integrator.integrate(dealii::EvaluationFlags::values);
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::rhs_ppe_nbc_convective_add(VectorType &       dst,
                                                               VectorType const & src) const
{
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_nbc_convective_add_boundary_face,
                               this,
                               dst,
                               src);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::local_rhs_ppe_nbc_convective_add_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  unsigned int const dof_index_velocity = this->get_dof_index_velocity();
  unsigned int const dof_index_pressure = this->get_dof_index_pressure();
  unsigned int const quad_index         = this->get_quad_index_velocity_overintegration();

  FaceIntegratorU velocity(matrix_free, true, dof_index_velocity, quad_index);
  FaceIntegratorP pressure(matrix_free, true, dof_index_pressure, quad_index);
  FaceIntegratorU grid_velocity(matrix_free, true, dof_index_velocity, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    velocity.reinit(face);
    velocity.gather_evaluate(src,
                             dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    if(this->param.ale_formulation)
    {
      grid_velocity.reinit(face);
      grid_velocity.gather_evaluate(this->convective_kernel->get_grid_velocity(),
                                    dealii::EvaluationFlags::values);
    }

    pressure.reinit(face);

    BoundaryTypeP boundary_type =
      this->boundary_descriptor->pressure->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      if(boundary_type == BoundaryTypeP::Neumann)
      {
        vector normal = pressure.get_normal_vector(q);

        vector u      = velocity.get_value(q);
        tensor grad_u = velocity.get_gradient(q);

        vector flux;
        if(this->param.formulation_convective_term_bc ==
           FormulationConvectiveTerm::DivergenceFormulation)
        {
          scalar div_u = velocity.get_divergence(q);
          flux         = grad_u * u + div_u * u;
        }
        else if(this->param.formulation_convective_term_bc ==
                FormulationConvectiveTerm::ConvectiveFormulation)
        {
          flux = grad_u * u;
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented."));
        }

        if(this->param.ale_formulation)
        {
          flux -= grad_u * grid_velocity.get_value(q);
        }

        pressure.submit_value(-normal * flux, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        pressure.submit_value(dealii::make_vectorized_array<Number>(0.0), q);
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }

    pressure.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::rhs_ppe_nbc_viscous_add(VectorType &       dst,
                                                            VectorType const & src) const
{
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_rhs_ppe_nbc_viscous_add_boundary_face,
                               this,
                               dst,
                               src);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::local_rhs_ppe_nbc_viscous_add_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           face_range) const
{
  unsigned int const dof_index_velocity = this->get_dof_index_velocity();
  unsigned int const dof_index_pressure = this->get_quad_index_pressure();
  unsigned int const quad_index         = this->get_quad_index_velocity_standard();

  FaceIntegratorU omega(matrix_free, true, dof_index_velocity, quad_index);

  FaceIntegratorP pressure(matrix_free, true, dof_index_pressure, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    pressure.reinit(face);

    omega.reinit(face);
    omega.gather_evaluate(src, dealii::EvaluationFlags::gradients);

    BoundaryTypeP boundary_type =
      this->boundary_descriptor->pressure->get_boundary_type(matrix_free.get_boundary_id(face));

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      scalar viscosity = this->get_viscosity_boundary_face(face, q);

      if(boundary_type == BoundaryTypeP::Neumann)
      {
        scalar h = dealii::make_vectorized_array<Number>(0.0);

        vector normal = pressure.get_normal_vector(q);

        vector curl_omega = CurlCompute<dim, FaceIntegratorU>::compute(omega, q);

        h = -normal * (viscosity * curl_omega);

        pressure.submit_value(h, q);
      }
      else if(boundary_type == BoundaryTypeP::Dirichlet)
      {
        pressure.submit_value(dealii::make_vectorized_array<Number>(0.0), q);
      }
      else
      {
        AssertThrow(false,
                    dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
      }
    }
    pressure.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::rhs_ppe_laplace_add(VectorType &   dst,
                                                        double const & evaluation_time) const
{
  ProjectionBase::do_rhs_ppe_laplace_add(dst, evaluation_time);
}

template<int dim, typename Number>
unsigned int
OperatorDualSplitting<dim, Number>::solve_pressure(VectorType &       dst,
                                                   VectorType const & src,
                                                   bool const         update_preconditioner) const
{
  return ProjectionBase::do_solve_pressure(dst, src, update_preconditioner);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::interpolate_velocity_dirichlet_bc(VectorType &   dst,
                                                                      double const & time) const
{
  this->evaluation_time = time;

  dst = 0.0;

  VectorType src_dummy;
  this->get_matrix_free().loop(&This::cell_loop_empty,
                               &This::face_loop_empty,
                               &This::local_interpolate_velocity_dirichlet_bc_boundary_face,
                               this,
                               dst,
                               src_dummy);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::local_interpolate_velocity_dirichlet_bc_boundary_face(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &,
  Range const & face_range) const
{
  unsigned int const dof_index = this->get_dof_index_velocity();
  AssertThrow(
    matrix_free.get_dof_handler(dof_index).get_triangulation().all_reference_cells_are_hyper_cube(),
    dealii::ExcMessage("This function is only implemented for hypercube elements."));
  unsigned int const quad_index = this->get_quad_index_velocity_gauss_lobatto();

  FaceIntegratorU integrator(matrix_free, true, dof_index, quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; face++)
  {
    dealii::types::boundary_id const boundary_id = matrix_free.get_boundary_id(face);

    BoundaryTypeU const boundary_type =
      this->boundary_descriptor->velocity->get_boundary_type(boundary_id);

    if(boundary_type == BoundaryTypeU::Dirichlet or boundary_type == BoundaryTypeU::DirichletCached or boundary_type == BoundaryTypeU::WallEnrichment)
    {
      integrator.reinit(face);
      integrator.read_dof_values(dst);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        unsigned int const local_face_number = matrix_free.get_face_info(face).interior_face_no;

        unsigned int const index = matrix_free.get_shape_info(dof_index, quad_index)
                                     .face_to_cell_index_nodal[local_face_number][q];

        vector g = vector();

        if(boundary_type == BoundaryTypeU::Dirichlet)
        {
          auto bc = this->boundary_descriptor->velocity->dirichlet_bc.find(boundary_id)->second;
          auto q_points = integrator.quadrature_point(q);

          g = FunctionEvaluator<1, dim, Number>::value(*bc, q_points, this->evaluation_time);
        }
        else if(boundary_type == BoundaryTypeU::WallEnrichment)
        {
          auto bc = this->boundary_descriptor->velocity->wall_enrichment_bc.find(boundary_id)->second;
          auto q_points = integrator.quadrature_point(q);

          g = FunctionEvaluator<1, dim, Number>::value(*bc, q_points, this->evaluation_time);
        }
        else if(boundary_type == BoundaryTypeU::DirichletCached)
        {
          auto bc = this->boundary_descriptor->velocity->get_dirichlet_cached_data();

          g = FunctionEvaluator<1, dim, Number>::value(*bc, face, q, quad_index);
        }
        else
        {
          AssertThrow(false, dealii::ExcMessage("Not implemented."));
        }

        integrator.submit_dof_value(g, index);
      }

      integrator.set_dof_values(dst);
    }
    else
    {
      AssertThrow(boundary_type == BoundaryTypeU::Neumann or
                    boundary_type == BoundaryTypeU::Symmetry,
                  dealii::ExcMessage("BoundaryTypeU not implemented."));
    }
  }
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::apply_helmholtz_operator(VectorType &       dst,
                                                             VectorType const & src) const
{
  this->momentum_operator.vmult(dst, src);
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::apply_velocity_divergence_term(
  VectorType &            dst,
  BlockVectorType const & src) const
{
  // (1) D(q, U_bar) on the whole domain: existing ExaDG operator
  this->divergence_operator.apply(dst, src.block(0));

  AssertThrow(this->param.divu_integrated_by_parts and
                this->param.divu_formulation == FormulationVelocityDivergenceTerm::Weak,
              dealii::ExcMessage("Enriched divergence: only weak formulation with "
                                 "integration by parts is implemented."));

  // (2) D(q, psi*U_tilde) on enriched cells and faces touching them
  auto const & fe_en = this->get_function_enrichment();
  auto const & mf_en = *fe_en->get_matrix_free();

  unsigned int const dof_p = fe_en->get_dof_index_shadow_pressure();
  unsigned int const quad  = fe_en->get_quad_index();

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number>     CellIntU;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, dim, Number> FaceIntU;
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number>       CellIntP;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number>   FaceIntP;

  // U_bar = 0, so the enriched evaluators return exactly psi * U_tilde
  VectorType zero_bar;
  mf_en.initialize_dof_vector(zero_bar, fe_en->get_dof_index_shadow_vector());
  zero_bar.update_ghost_values();
  src.block(1).update_ghost_values();

  VectorType correction_p;
  mf_en.initialize_dof_vector(correction_p, dof_p);

  // --- cell term: -int grad(q) . (psi U_tilde) ------------------------------
  {
    FEEnrichedEvaluation<dim, Number, CellIntU> phi_u(mf_en, *fe_en);
    CellIntP                                    phi_p(mf_en, dof_p, quad, 0, 1);

    for(unsigned int cell = 0; cell < mf_en.n_cell_batches(); ++cell)
    {
      if(mf_en.get_cell_iterator(cell, 0)->active_fe_index() != 1)
        continue;

      phi_u.reinit(cell);
      phi_u.gather_evaluate(zero_bar, src.block(1), dealii::EvaluationFlags::values);

      phi_p.reinit(cell);
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q)
        phi_p.submit_gradient(-phi_u.get_value(q), q);
      phi_p.integrate(dealii::EvaluationFlags::gradients);
      phi_p.distribute_local_to_global(correction_p);
    }
  }

  // psi*U_tilde on one side of a face; zero if that side is not enriched (halo)
  auto const eval_side = [&](bool const interior, unsigned int const fe_index, unsigned int const face,
                             unsigned int const n_q, std::vector<dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>> & vals) {
    vals.assign(n_q, dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>());
    if(fe_index != 1)
      return;
    FEFaceEnrichedEvaluation<dim, Number, FaceIntU> phi(mf_en, *fe_en, interior, 1);
    phi.reinit(face);
    phi.gather_evaluate(zero_bar, src.block(1), dealii::EvaluationFlags::values);
    for(unsigned int q = 0; q < n_q; ++q)
      vals[q] = phi.get_value(q);
  };

  std::vector<dealii::Tensor<1, dim, dealii::VectorizedArray<Number>>> u_m, u_p;

  // --- interior faces: int [q] * avg(psi U_tilde) . n ------------------------
  for(unsigned int face = 0; face < mf_en.n_inner_face_batches(); ++face)
  {
    auto const         range = std::make_pair(face, face + 1);
    unsigned int const fe_m  = mf_en.get_face_active_fe_index(range, true);
    unsigned int const fe_p  = mf_en.get_face_active_fe_index(range, false);

    if(fe_m != 1 and fe_p != 1)
      continue;

    AssertThrow(fe_m != 0 and fe_p != 0,
                dealii::ExcMessage("Enriched cell has a neighbour outside the halo layer."));

    FaceIntP p_m(mf_en, true, dof_p, quad, 0, fe_m);
    FaceIntP p_p(mf_en, false, dof_p, quad, 0, fe_p);
    p_m.reinit(face);
    p_p.reinit(face);

    eval_side(true, fe_m, face, p_m.n_q_points, u_m);
    eval_side(false, fe_p, face, p_m.n_q_points, u_p);

    for(unsigned int q = 0; q < p_m.n_q_points; ++q)
    {
      auto const flux = (0.5 * (u_m[q] + u_p[q])) * p_m.get_normal_vector(q);
      p_m.submit_value(flux, q);
      p_p.submit_value(-flux, q);
    }

    p_m.integrate(dealii::EvaluationFlags::values);
    p_m.distribute_local_to_global(correction_p);
    p_p.integrate(dealii::EvaluationFlags::values);
    p_p.distribute_local_to_global(correction_p);
  }

  // --- boundary faces (homogeneous part) -------------------------------------
  //   wall: psi = 0 -> nothing;  Dirichlet: avg = 0 -> nothing;  Neumann: u_m . n
  unsigned int const b_begin = mf_en.n_inner_face_batches();
  unsigned int const b_end   = b_begin + mf_en.n_boundary_face_batches();

  for(unsigned int face = b_begin; face < b_end; ++face)
  {
    if(mf_en.get_face_active_fe_index(std::make_pair(face, face + 1), true) != 1)
      continue;

    BoundaryTypeU const type =
      this->boundary_descriptor->velocity->get_boundary_type(mf_en.get_boundary_id(face));

    if(type != BoundaryTypeU::Neumann)
      continue; // WallEnrichment, Dirichlet, ...: homogeneous contribution of psi*U_tilde is zero

    FaceIntP p_m(mf_en, true, dof_p, quad, 0, 1);
    p_m.reinit(face);
    eval_side(true, 1, face, p_m.n_q_points, u_m);

    for(unsigned int q = 0; q < p_m.n_q_points; ++q)
      p_m.submit_value(u_m[q] * p_m.get_normal_vector(q), q);

    p_m.integrate(dealii::EvaluationFlags::values);
    p_m.distribute_local_to_global(correction_p);
  }

  correction_p.compress(dealii::VectorOperation::add);

  // (3) add onto the global pressure-space result
  fe_en->add_wall_layout_to_global_pressure(correction_p, dst);

  src.block(1).zero_out_ghost_values();
}

template<int dim, typename Number>
void
OperatorDualSplitting<dim, Number>::evaluate_pressure_gradient_term_enriched(
  BlockVectorType &  dst,
  VectorType const & src,
  double const       time) const
{
  dst.block(0).zero_out_ghost_values();
  dst.block(1).zero_out_ghost_values();

  // (1) U_bar rows: existing ExaDG operator, full (incl. boundary data)
  this->evaluate_pressure_gradient_term(dst.block(0), src, time);

  AssertThrow(this->param.gradp_integrated_by_parts and
                this->param.gradp_formulation == FormulationPressureGradientTerm::Weak and
                this->param.gradp_use_boundary_data,
              dealii::ExcMessage("Enriched pressure gradient: only weak formulation with "
                                 "integration by parts and boundary data is implemented."));

  // (2) U_tilde rows: test functions psi * v_tilde
  auto const & fe_en = this->get_function_enrichment();
  auto const & mf_en = *fe_en->get_matrix_free();

  unsigned int const dof_p = fe_en->get_dof_index_shadow_pressure();
  unsigned int const quad  = fe_en->get_quad_index();

  typedef dealii::FEEvaluation<dim, -1, 0, dim, Number>     CellIntU;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, dim, Number> FaceIntU;
  typedef dealii::FEEvaluation<dim, -1, 0, 1, Number>       CellIntP;
  typedef dealii::FEFaceEvaluation<dim, -1, 0, 1, Number>   FaceIntP;
  typedef dealii::VectorizedArray<Number>                   scalar;

  // pressure in shadow layout (enriched + halo cells)
  VectorType & p_shadow = fe_en->shadow_pressure;
  fe_en->copy_global_pressure_to_wall_layout(src, p_shadow);
  p_shadow.update_ghost_values();

  // zero velocity vectors: the enriched evaluators are only used as test functions,
  // gather_evaluate is needed so that y and u_tau (-> psi) are evaluated
  VectorType zero_bar, zero_tilde;
  mf_en.initialize_dof_vector(zero_bar, fe_en->get_dof_index_shadow_vector());
  mf_en.initialize_dof_vector(zero_tilde, fe_en->get_dof_index_en());
  zero_bar.update_ghost_values();
  zero_tilde.update_ghost_values();

  dst.block(1) = 0.0;

  // --- cell term: -int p div(psi v_tilde) ------------------------------------
  {
    FEEnrichedEvaluation<dim, Number, CellIntU> phi_u(mf_en, *fe_en);
    CellIntP                                    phi_p(mf_en, dof_p, quad, 0, 1);

    for(unsigned int cell = 0; cell < mf_en.n_cell_batches(); ++cell)
    {
      if(mf_en.get_cell_iterator(cell, 0)->active_fe_index() != 1)
        continue;

      phi_u.reinit(cell);
      phi_u.gather_evaluate(zero_bar, zero_tilde, dealii::EvaluationFlags::values);

      phi_p.reinit(cell);
      phi_p.gather_evaluate(p_shadow, dealii::EvaluationFlags::gradients);

      // strong form: int psi v_tilde . grad(p)   (no grad(psi), no large p values)
      for(unsigned int q = 0; q < phi_u.n_q_points; ++q)
        phi_u.submit_value(phi_p.get_gradient(q), q); // submit_value tests with v_bar and psi v_tilde

      phi_u.integrate(dealii::EvaluationFlags::values);

      phi_u.phi_en.distribute_local_to_global(dst.block(1)); // U_tilde rows only
    }
  }

  // test one side of a face with psi*v_tilde: submits value_n[q] (a vector) and scatters
  auto const test_side = [&](bool const interior, unsigned int const face,
                             std::vector<dealii::Tensor<1, dim, scalar>> const & value_n) {
    FEFaceEnrichedEvaluation<dim, Number, FaceIntU> phi(mf_en, *fe_en, interior, 1);
    phi.reinit(face);
    phi.gather_evaluate(zero_bar, zero_tilde, dealii::EvaluationFlags::values);
    for(unsigned int q = 0; q < phi.n_q_points; ++q)
      phi.submit_value(value_n[q], q);
    phi.integrate(dealii::EvaluationFlags::values);
    phi.phi_en.distribute_local_to_global(dst.block(1));
  };

  std::vector<dealii::Tensor<1, dim, scalar>> flux_n_m, flux_n_p;

  // --- interior faces: int {p} [psi v_tilde] . n -----------------------------
  for(unsigned int face = 0; face < mf_en.n_inner_face_batches(); ++face)
  {
    auto const         range = std::make_pair(face, face + 1);
    unsigned int const fe_m  = mf_en.get_face_active_fe_index(range, true);
    unsigned int const fe_p  = mf_en.get_face_active_fe_index(range, false);

    if(fe_m != 1 and fe_p != 1)
      continue;

    AssertThrow(fe_m != 0 and fe_p != 0,
                dealii::ExcMessage("Enriched cell has a neighbour outside the halo layer."));

    FaceIntP pm(mf_en, true, dof_p, quad, 0, fe_m);
    FaceIntP pp(mf_en, false, dof_p, quad, 0, fe_p);
    pm.reinit(face);
    pp.reinit(face);
    pm.gather_evaluate(p_shadow, dealii::EvaluationFlags::values);
    pp.gather_evaluate(p_shadow, dealii::EvaluationFlags::values);

    flux_n_m.resize(pm.n_q_points);
    flux_n_p.resize(pm.n_q_points);
    // strong form: - int psi v_tilde . n (p_side - {p}); for both sides this is
    //   - 0.5 (p_m - p_p) psi v_tilde . n_m   (only the JUMP of p enters)
    for(unsigned int q = 0; q < pm.n_q_points; ++q)
    {
      scalar const jump = pm.get_value(q) - pp.get_value(q);
      flux_n_m[q]       = -0.5 * jump * pm.get_normal_vector(q);
      flux_n_p[q]       = flux_n_m[q];
    }

    if(fe_m == 1)
      test_side(true, face, flux_n_m);
    if(fe_p == 1)
      test_side(false, face, flux_n_p);
  }

  // --- boundary faces -----------------------------------------------------------
  //   WallEnrichment: psi = 0 -> nothing
  //   velocity Dirichlet (inflow)  = pressure Neumann   -> p* = p_m
  //   velocity Neumann  (outflow)  = pressure Dirichlet -> p* = g_p(x,t)
  unsigned int const b_begin = mf_en.n_inner_face_batches();
  unsigned int const b_end   = b_begin + mf_en.n_boundary_face_batches();

  for(unsigned int face = b_begin; face < b_end; ++face)
  {
    if(mf_en.get_face_active_fe_index(std::make_pair(face, face + 1), true) != 1)
      continue;

    auto const          boundary_id = mf_en.get_boundary_id(face);
    BoundaryTypeU const type        = this->boundary_descriptor->velocity->get_boundary_type(boundary_id);

    if(type == BoundaryTypeU::WallEnrichment)
      continue;

    FaceIntP pm(mf_en, true, dof_p, quad, 0, 1);
    pm.reinit(face);
    pm.gather_evaluate(p_shadow, dealii::EvaluationFlags::values);

    flux_n_m.resize(pm.n_q_points);

    if(type == BoundaryTypeU::Dirichlet)
    {
      // inflow = pressure Neumann: p* = p_m -> no contribution in the strong form
      continue;
    }
    else if(type == BoundaryTypeU::Neumann)
    {
      // outflow = pressure Dirichlet: p* = g_p -> - int psi v_tilde . n (p_m - g_p)
      auto const & g_p = this->boundary_descriptor->pressure->dirichlet_bc.find(boundary_id)->second;
      g_p->set_time(time);

      for(unsigned int q = 0; q < pm.n_q_points; ++q)
      {
        auto const xq = pm.quadrature_point(q);
        scalar     g;
        for(unsigned int l = 0; l < scalar::size(); ++l)
        {
          dealii::Point<dim> pt;
          for(unsigned int d = 0; d < dim; ++d)
            pt[d] = xq[d][l];
          g[l] = g_p->value(pt);
        }
        flux_n_m[q] = -(pm.get_value(q) - g) * pm.get_normal_vector(q);
      }
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage("Enriched pressure gradient: boundary type not supported "
                                     "on enriched cells."));
    }

    test_side(true, face, flux_n_m);
  }

  dst.block(1).compress(dealii::VectorOperation::add);
  p_shadow.zero_out_ghost_values();
}

template class OperatorDualSplitting<2, float>;
template class OperatorDualSplitting<2, double>;

template class OperatorDualSplitting<3, float>;
template class OperatorDualSplitting<3, double>;

} // namespace IncNS
} // namespace ExaDG
