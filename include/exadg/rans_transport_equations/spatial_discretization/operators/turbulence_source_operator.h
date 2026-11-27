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

#ifndef INCLUDE_EXADG_RANS_TRANSPORT_EQUATIONS_SPATIAL_DISCRETIZATION_OPERATORS_TURBULENCE_SOURCE_OPERATOR
#define INCLUDE_EXADG_RANS_TRANSPORT_EQUATIONS_SPATIAL_DISCRETIZATION_OPERATORS_TURBULENCE_SOURCE_OPERATOR

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/array_view.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <algorithm>
#include <array>

#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/integrator_flags.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/utilities/lazy_ptr.h>
#include <exadg/grid/grid_data.h>
#include <exadg/operators/interior_penalty_parameter.h>

#include <exadg/rans_transport_equations/spatial_discretization/turbulence_model.h>
#include <exadg/convection_diffusion/user_interface/enum_types.h>
#include <exadg/convection_diffusion/spatial_discretization/operators/weak_boundary_conditions.h>
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/operators/wall_modelling/fe_omega_enriched_evaluation.h>
#include <exadg/operators/wall_modelling/function_enrichment_rans.h>

namespace ExaDG
{
namespace RANS
{
using ConvDiff::BoundaryDescriptor;
using ConvDiff::BoundaryType;
namespace Operators
{
template<int dim>
struct TurbulenceSourceKernelData
{
  TurbulenceSourceKernelData()
    : dof_index(0),
    dof_index_velocity(0),
    dof_index_eddy_viscosity(dealii::numbers::invalid_unsigned_int),
    diffusivity(0.0),
    IP_factor(1.0),
    time_step_size(0.0),
    time_step_number(0.0),
    current_time(0.0),
    end_time(0.0),
    production_implicit(false),
    dissipation_implicit(false),
    square_gradient_implicit(false),
    cross_diffusion_implicit(false),
    viscosity_implicit(false),
    k_log(false),
    omega_log(false)
  {
  }

  unsigned int dof_index;
  unsigned int dof_index_velocity;
  // DoF index of the eddy viscosity field (wall-law bound, set_eddy_viscosity_wall_law_ptr)
  unsigned int dof_index_eddy_viscosity;

  // molecular diffusivity nu; the diffusion coefficient of transport variable i is
  // nu + eddy_viscosity_factor[i] * nu_t (one factor per block, same as in DiffusiveKernelData)
  double              diffusivity;
  std::vector<double> eddy_viscosity_factor;

  // interior penalty factor of the diffusive term (same as DiffusiveKernelData::IP_factor)
  double IP_factor;

  double time_step_size;
  double time_step_number;
  double current_time;
  double end_time;

  // implicit (linearized) treatment, identical for all transport variables
  bool production_implicit;
  bool dissipation_implicit;
  bool square_gradient_implicit;
  bool cross_diffusion_implicit;

  // turbulent diffusion -div(sigma_i nu_t grad Q_i): Implicit (Newton linearization) or Explicit
  bool viscosity_implicit;

  bool k_log;
  bool omega_log;

  // boundary conditions of all transport variables (one per block), used by the turbulent diffusion
  std::vector<std::shared_ptr<BoundaryDescriptor<dim> const>> bc;

  TurbulenceModelData turbulence_model_data;
};

template<int dim, typename Number>
class TurbulenceSourceKernel
{
private:
  typedef CellIntegrator<dim, 1, Number>   IntegratorCell;
  typedef FaceIntegrator<dim, 1, Number>   IntegratorFace;
  typedef CellIntegrator<dim, dim, Number> IntegratorCellVelocity;
  typedef FaceIntegrator<dim, dim, Number> IntegratorFaceVelocity;

  typedef dealii::VectorizedArray<Number>   scalar;
  typedef dealii::Tensor<1, dim, scalar> vector;
  typedef dealii::Tensor<2, dim, scalar> tensor;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

public:
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free_in,
         TurbulenceSourceKernelData<dim> const & data_in,
         unsigned int const                      quad_index)
  {
    matrix_free = &matrix_free_in;
    data = data_in;

    AssertThrow(turbulence_model_ptr != nullptr,
                dealii::ExcMessage("Set turbulence_model_ptr before calling reinit()."));

    n_variables = turbulence_model_ptr->get_n_variables();

    // the RANS model: nu_t, nu_D, production, dissipation at the quadrature points
    rans_model = &turbulence_model_ptr->get_eddy_viscosity_evaluator().get_model();

    AssertThrow(rans_model->get_n_variables() == n_variables,
                dealii::ExcMessage("Number of variables of the RANS model and of the turbulence "
                                   "model do not match."));

    AssertThrow(data.eddy_viscosity_factor.size() == n_variables,
                dealii::ExcMessage(
                "One eddy viscosity factor per transport variable has to be specified."));

    integrator_velocity =
      std::make_shared<IntegratorCellVelocity>(matrix_free_in, data.dof_index_velocity, quad_index);

    integrator_solution.resize(n_variables);
    for(auto & integrator : integrator_solution)
    {
      integrator = std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index, quad_index);
    }

    if(data.dof_index_eddy_viscosity != dealii::numbers::invalid_unsigned_int)
    {
      integrator_wall_law =
        std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index_eddy_viscosity, quad_index);

      integrator_tke_production_from_wall_law =
        std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index_eddy_viscosity, quad_index);
      integrator_wall_cell_indicator =
        std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index_eddy_viscosity, quad_index);

      // exact velocity gradient of the wall model (set_velocity_gradient_enriched_ptr)
      integrator_gradient_enriched.resize(dim * dim);
      for(auto & integrator : integrator_gradient_enriched)
        integrator =
          std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index_eddy_viscosity, quad_index);
      integrator_gradient_indicator =
        std::make_shared<IntegratorCell>(matrix_free_in, data.dof_index_eddy_viscosity, quad_index);
    }

    reinit_points(integrator_velocity->n_q_points);

    point_input.reinit(n_variables);
    point_eddy_viscosity.reinit(n_variables);
    point_diffusion_viscosity.reinit(n_variables);
    point_production.reinit(n_variables);
    point_dissipation.reinit(n_variables);
    point_cross_diffusion.reinit(n_variables);
    point_square_gradient.reinit(n_variables);

    // face integrators of the turbulent diffusion (SIPG face terms)
    AssertThrow(data.bc.size() == n_variables,
                dealii::ExcMessage(
                "One boundary descriptor per transport variable has to be specified."));

    for(unsigned int side = 0; side < 2; ++side)
    {
      bool const is_interior_face = (side == 0);

      face_integrator_velocity[side] = std::make_shared<IntegratorFaceVelocity>(
        matrix_free_in, is_interior_face, data.dof_index_velocity, quad_index);

      face_integrator_solution[side].resize(n_variables);
      for(auto & integrator : face_integrator_solution[side])
      {
        integrator =
          std::make_shared<IntegratorFace>(matrix_free_in, is_interior_face, data.dof_index, quad_index);
      }
    }

    reinit_face_points(face_integrator_velocity[0]->n_q_points);

    face_dirichlet.resize(n_variables);
    face_dirichlet_mask.resize(n_variables);

    // storage of the linearization point (Newton_Implicit), one slot per cell / face batch
    unsigned int const n_cell_batches = matrix_free_in.n_cell_batches();
    unsigned int const n_face_batches =
      matrix_free_in.n_inner_face_batches() + matrix_free_in.n_boundary_face_batches();

    stored_cell_solution_gradient.resize(n_cell_batches * cell_solution_gradient.size());
    stored_cell_viscosity.resize(n_cell_batches * cell_viscosity.size());
    stored_cell_viscosity_derivative.resize(n_cell_batches * cell_viscosity_derivative.size());
    stored_cell_production_jacobian.resize(n_cell_batches * cell_production_jacobian.size());
    stored_cell_dissipation_jacobian.resize(n_cell_batches * cell_dissipation_jacobian.size());
    stored_cell_square_gradient_jacobian.resize(n_cell_batches *
                                                cell_square_gradient_jacobian.size());
    stored_cell_square_gradient_gradient_jacobian.resize(
      n_cell_batches * cell_square_gradient_gradient_jacobian.size());
    stored_cell_cross_diffusion_jacobian.resize(n_cell_batches * cell_cross_diffusion_jacobian.size());
    stored_cell_cross_diffusion_gradient_jacobian.resize(n_cell_batches * cell_cross_diffusion_gradient_jacobian.size());

    for(unsigned int side = 0; side < 2; ++side)
    {
      stored_face_solution[side].resize(n_face_batches * face_solution[side].size());
      stored_face_normal_gradient[side].resize(n_face_batches * face_normal_gradient[side].size());
      stored_face_viscosity[side].resize(n_face_batches * face_viscosity[side].size());
      stored_face_viscosity_derivative[side].resize(n_face_batches *
                                                    face_viscosity_derivative[side].size());
    }
    stored_tau.resize(n_face_batches);

    degree = matrix_free_in.get_dof_handler(data.dof_index).get_fe().degree;
    element_type =
      get_element_type(matrix_free_in.get_dof_handler(data.dof_index).get_triangulation());

    calculate_penalty_parameter();
  }

  void
  calculate_penalty_parameter()
  {
    IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter,
                                                 *matrix_free,
                                                 data.dof_index);
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells       = dealii::update_JxW_values | dealii::update_gradients;
    flags.inner_faces = dealii::update_JxW_values | dealii::update_gradients |
      dealii::update_normal_vectors;
    flags.boundary_faces = dealii::update_JxW_values | dealii::update_gradients |
      dealii::update_normal_vectors | dealii::update_quadrature_points;

    return flags;
  }

  TurbulenceSourceKernelData<dim> const &
  get_data() const
  {
    return data;
  }

  // replaces the kernel data (diagnostics: treatment flags switched on / off temporarily)
  void
  set_data(TurbulenceSourceKernelData<dim> const & data_in) const
  {
    data = data_in;
  }

  unsigned int
  get_n_variables() const
  {
    return n_variables;
  }

  // number of points of the cell data (reinit_points)
  unsigned int
  get_n_q_points_cell() const
  {
    return n_q_points_cell;
  }

  // solution set by set_solution_ptr()
  BlockVectorType const &
  get_solution() const
  {
    return *solution;
  }

  // wall-law fields on the eddy viscosity DoFHandler (nullptr: not set)
  VectorType const *
  get_eddy_viscosity_wall_law_ptr() const
  {
    return eddy_viscosity_wall_law;
  }

  VectorType const *
  get_tke_production_from_wall_law_ptr() const
  {
    return tke_production_from_wall_law;
  }

  VectorType const *
  get_wall_cell_indicator_ptr() const
  {
    return wall_cell_indicator;
  }

  /*
     * Function for taking value of velocity from NS solver
     */
  void
  set_velocity_ptr(VectorType const & velocity_in)
  {
    velocity.own() = velocity_in;
    velocity->update_ghost_values();
  }

  /*
     * Linearization point Q* (one block per transport variable)
     */
  void
  set_solution_ptr(BlockVectorType const & solution_in)
  {
    AssertThrow(solution_in.n_blocks() >= n_variables,
                dealii::ExcMessage("Number of solution blocks does not match the turbulence model."));

    solution.own() = solution_in;
    solution->update_ghost_values();
  }

  /*
   * Wall model: upper bound nu_t^psi of the eddy viscosity (DG field on the eddy viscosity
   * DoFHandler, 1e20 = no bound). The vector must stay alive (driver member).
   */
  void
  set_eddy_viscosity_wall_law_ptr(VectorType const & eddy_viscosity_wall_law_in)
  {
    AssertThrow(integrator_wall_law != nullptr,
                dealii::ExcMessage("Set dof_index_eddy_viscosity in TurbulenceSourceKernelData."));

    eddy_viscosity_wall_law = &eddy_viscosity_wall_law_in;
  }

  void
  set_tke_production_from_wall_law_ptr(VectorType const & tke_production_from_wall_law_in,
                                       VectorType const & wall_cell_indicator_in)
  {
    AssertThrow(integrator_tke_production_from_wall_law != nullptr,
                dealii::ExcMessage("Set dof_index_eddy_viscosity in TurbulenceSourceKernelData."));
    tke_production_from_wall_law = &tke_production_from_wall_law_in;
    wall_cell_indicator = &wall_cell_indicator_in;
  }

  /*
   * Wall model: exact velocity gradient of the enriched cells and its indicator (DG fields on the
   * eddy viscosity DoFHandler, FunctionEnrichment::compute_velocity_gradient_rans). Used in the
   * cell terms (production, dissipation, cell nu_t) instead of the gradient of the velocity field
   * where indicator = 1. The vectors must stay alive (driver members).
   */
  void
  set_velocity_gradient_enriched_ptr(std::vector<VectorType> const & velocity_gradient_in,
                                     VectorType const &              indicator_in)
  {
    AssertThrow(integrator_gradient_indicator != nullptr,
                dealii::ExcMessage("Set dof_index_eddy_viscosity in TurbulenceSourceKernelData."));
    AssertDimension(velocity_gradient_in.size(), dim * dim);

    velocity_gradient_enriched  = &velocity_gradient_in;
    velocity_gradient_indicator = &indicator_in;
  }

  void
  reinit_cell(unsigned int const cell) const
  {
    integrator_velocity->reinit(cell);
    integrator_velocity->gather_evaluate(*velocity, dealii::EvaluationFlags::gradients);

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator_solution[i]->reinit(cell);
      integrator_solution[i]->gather_evaluate(solution->block(i),
                                              dealii::EvaluationFlags::values |
                                              dealii::EvaluationFlags::gradients);
    }

    if(eddy_viscosity_wall_law != nullptr)
    {
      integrator_wall_law->reinit(cell);
      integrator_wall_law->gather_evaluate(*eddy_viscosity_wall_law, dealii::EvaluationFlags::values);
    }

    if(tke_production_from_wall_law != nullptr)
    {
      integrator_tke_production_from_wall_law->reinit(cell);
      integrator_tke_production_from_wall_law->gather_evaluate(*tke_production_from_wall_law,
                                                     dealii::EvaluationFlags::values);
      integrator_wall_cell_indicator->reinit(cell);
      integrator_wall_cell_indicator->gather_evaluate(*wall_cell_indicator,
                                                      dealii::EvaluationFlags::values);
    }

    // wall model: exact velocity gradient of the enriched cells
    if(velocity_gradient_enriched != nullptr)
    {
      for(unsigned int c = 0; c < dim * dim; ++c)
      {
        integrator_gradient_enriched[c]->reinit(cell);
        integrator_gradient_enriched[c]->gather_evaluate((*velocity_gradient_enriched)[c],
                                                         dealii::EvaluationFlags::values);
      }
      integrator_gradient_indicator->reinit(cell);
      integrator_gradient_indicator->gather_evaluate(*velocity_gradient_indicator,
                                                     dealii::EvaluationFlags::values);
    }

    for(unsigned int q = 0; q < n_q_points_cell; ++q)
    {
      for(unsigned int i = 0; i < n_variables; ++i)
      {
        cell_solution[index(q, i)]          = integrator_solution[i]->get_value(q);
        cell_solution_gradient[index(q, i)] = integrator_solution[i]->get_gradient(q);
      }

      // gradient of the transferred velocity field; in enriched cells (indicator = 1) the exact
      // gradient grad(U_bar + psi U_tilde) of the wall model instead (the projected field
      // overestimates the production in large wall cells)
      tensor velocity_gradient = integrator_velocity->get_gradient(q);

      if(velocity_gradient_enriched != nullptr)
      {
        scalar const indicator = integrator_gradient_indicator->get_value(q);
        for(unsigned int v = 0; v < scalar::size(); ++v)
          if(indicator[v] > Number(0.5))
            for(unsigned int i = 0; i < dim; ++i)
              for(unsigned int j = 0; j < dim; ++j)
                velocity_gradient[i][j][v] =
                  integrator_gradient_enriched[i * dim + j]->get_value(q)[v];
      }

      // wall-law fields at q (0 if not set; only used if the field is set)
      scalar const zero_q = dealii::make_vectorized_array<Number>(0.0);
      scalar const nu_t_wall_law_q =
        eddy_viscosity_wall_law != nullptr ? integrator_wall_law->get_value(q) : zero_q;
      scalar const tke_production_wall_law_q =
        tke_production_from_wall_law != nullptr ?
          integrator_tke_production_from_wall_law->get_value(q) :
          zero_q;
      scalar const wall_cell_indicator_q =
        tke_production_from_wall_law != nullptr ? integrator_wall_cell_indicator->get_value(q) :
                                                  zero_q;

      evaluate_point(q,
                     velocity_gradient,
                     nu_t_wall_law_q,
                     tke_production_wall_law_q,
                     wall_cell_indicator_q);
    }
  }

  /*
   * Number of quadrature points of the cell data (cell_solution, cell_production, ...): set by
   * reinit() for the cells of the kernel's MatrixFree; another number of points for a copy of the
   * kernel that evaluates the terms at other points (enriched wall cells of FunctionEnrichmentRANS)
   */
  void
  reinit_points(unsigned int const n_points) const
  {
    n_q_points_cell = n_points;

    cell_solution.resize(n_q_points_cell * n_variables);
    cell_solution_gradient.resize(n_q_points_cell * n_variables);
    cell_viscosity.resize(n_q_points_cell);
#ifdef EXADG_DEBUG
    debug_viscosity_unbounded.resize(n_q_points_cell);
    debug_nu_t_wall_law.resize(n_q_points_cell);
#endif
    cell_viscosity_derivative.resize(n_q_points_cell * n_variables);
    cell_production.resize(n_q_points_cell * n_variables);
    cell_dissipation.resize(n_q_points_cell * n_variables);
    cell_square_gradient.resize(n_q_points_cell * n_variables);
    cell_production_jacobian.resize(n_q_points_cell * n_variables * n_variables);
    cell_dissipation_jacobian.resize(n_q_points_cell * n_variables * n_variables);
    cell_square_gradient_jacobian.resize(n_q_points_cell * n_variables * n_variables);
    cell_square_gradient_gradient_jacobian.resize(n_q_points_cell * n_variables);
    cell_cross_diffusion.resize(n_q_points_cell * n_variables);
    cell_cross_diffusion_jacobian.resize(n_q_points_cell * n_variables * n_variables);
    cell_cross_diffusion_gradient_jacobian.resize(n_q_points_cell * n_variables * n_variables);
  }

  // solution of variable i at point q (physical or transported form as stored by reinit_cell)
  void
  set_point_solution(unsigned int const q,
                     unsigned int const i,
                     scalar const &     value,
                     vector const &     gradient) const
  {
    cell_solution[index(q, i)]          = value;
    cell_solution_gradient[index(q, i)] = gradient;
  }

  /*
   * All terms of the RANS model at point q from cell_solution / cell_solution_gradient at q
   * (set_point_solution or reinit_cell), the velocity gradient and the wall-law fields at q
   * (used only if the corresponding field pointer is set): nu_D, P, D, CD, SG and their Jacobians
   */
  void
  evaluate_point(unsigned int const q,
                 tensor const &     velocity_gradient,
                 scalar const &     nu_t_wall_law,
                 scalar const &     tke_production_wall_law,
                 scalar const &     wall_cell_indicator) const
  {
    // input of the RANS model: physical variables (k, omega), their gradients and grad u
    rans_model->set_physical_variables(
      dealii::ArrayView<scalar const>(cell_solution.data() + index(q, 0), n_variables),
      dealii::ArrayView<vector const>(cell_solution_gradient.data() + index(q, 0), n_variables),
      point_input);
    point_input.velocity_gradient   = velocity_gradient;
    point_input.compute_derivatives = true;

    // nu_t and the diffusion viscosity nu_D with d nu / d Q_j, used by all terms (explicit and
    // implicit treatment); cell_viscosity is nu_D (Wilcox 1988: nu_D = nu_t)
    rans_model->evaluate_viscosities(point_input, point_eddy_viscosity, point_diffusion_viscosity);
    rans_model->convert_to_transported_form(point_input, point_diffusion_viscosity);

    cell_viscosity[q] = point_diffusion_viscosity.value;
    for(unsigned int j = 0; j < n_variables; ++j)
      cell_viscosity_derivative[index(q, j)] = point_diffusion_viscosity.derivative[j];

    // wall model: upper bound of nu_t in the turbulent diffusion
    if(eddy_viscosity_wall_law != nullptr)
      apply_wall_law_viscosity_bound(q, nu_t_wall_law);

    // production and dissipation of the RANS model, physical -> transported form (ln k, ln omega)
    // (cross-diffusion: zero for Wilcox 1988, Wilcox 2006 only)
    rans_model->evaluate_sources(point_input,
                                 point_eddy_viscosity,
                                 point_production,
                                 point_dissipation,
                                 point_cross_diffusion);
    if(not rans_model->sources_in_transported_form())
    {
      rans_model->convert_to_transported_form(point_input, point_production);
      rans_model->convert_to_transported_form(point_input, point_dissipation);
      rans_model->convert_to_transported_form(point_input, point_cross_diffusion);
    }

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      cell_production[index(q, i)]  = point_production.value[i];
      cell_dissipation[index(q, i)] = point_dissipation.value[i];
      cell_cross_diffusion[index(q, i)] = point_cross_diffusion.value[i];

      for(unsigned int j = 0; j < n_variables; ++j)
      {
        cell_production_jacobian[index(q, i, j)] =
          point_production.jacobian[point_production.index(i, j)];
        cell_dissipation_jacobian[index(q, i, j)] =
          point_dissipation.jacobian[point_dissipation.index(i, j)];
        cell_cross_diffusion_jacobian[index(q, i, j)] =
          point_cross_diffusion.jacobian[point_cross_diffusion.index(i, j)];
        cell_cross_diffusion_gradient_jacobian[index(q, i, j)] =
          point_cross_diffusion.gradient_jacobian[point_cross_diffusion.index(i, j)];
      }
    }

    // wall model: upper bound of the k production
    if(eddy_viscosity_wall_law != nullptr)
      apply_wall_law_production_bound(q, 
                                      nu_t_wall_law, 
                                      velocity_gradient,
                                      point_input.values[0]);

    if(tke_production_from_wall_law != nullptr)
      apply_tke_production_from_wall_law(q,
                                         tke_production_wall_law,
                                         wall_cell_indicator,
                                         point_input.values[0]);

    if(rans_model->sources_in_transported_form())
    {
      // nu_D after the wall-law bound, with d nu_D / d Q_j
      point_diffusion_viscosity.value = cell_viscosity[q];
      for(unsigned int j = 0; j < n_variables; ++j)
        point_diffusion_viscosity.derivative[j] = cell_viscosity_derivative[index(q, j)];

      rans_model->evaluate_square_gradient(point_input,
                                           dealii::make_vectorized_array<Number>(
                                           data.diffusivity),
                                           point_diffusion_viscosity,
                                           point_square_gradient);

      for(unsigned int i = 0; i < n_variables; ++i)
      {
        cell_square_gradient[index(q, i)] = point_square_gradient.value[i];
        cell_square_gradient_gradient_jacobian[index(q, i)] =
          point_square_gradient.gradient_jacobian[point_square_gradient.index(i, i)];
        for(unsigned int j = 0; j < n_variables; ++j)
          cell_square_gradient_jacobian[index(q, i, j)] =
            point_square_gradient.jacobian[point_square_gradient.index(i, j)];
      }
    }
    else
    compute_square_gradient(q);
  }

  /*
   * Interior face: solution, normal gradient, nu_t and d nu_t / d Q_j on both sides (side 0: e-,
   * side 1: e+) at the solution set by set_solution_ptr(), and the penalty factor tau.
   */
  void
  reinit_face(unsigned int const face) const
  {
    boundary_face = false;

    for(unsigned int side = 0; side < 2; ++side)
    {
      face_integrator_velocity[side]->reinit(face);
      face_integrator_velocity[side]->gather_evaluate(*velocity, dealii::EvaluationFlags::gradients);

      for(unsigned int i = 0; i < n_variables; ++i)
      {
        face_integrator_solution[side][i]->reinit(face);
        face_integrator_solution[side][i]->gather_evaluate(solution->block(i),
                                                           dealii::EvaluationFlags::values |
                                                             dealii::EvaluationFlags::gradients);
      }

      for(unsigned int q = 0; q < n_q_points_face; ++q)
      {
        for(unsigned int i = 0; i < n_variables; ++i)
          set_face_point_solution(side,
                                  q,
                                  i,
                                  face_integrator_solution[side][i]->get_value(q),
                                  face_integrator_solution[side][i]->get_normal_derivative(q));

        evaluate_face_point(side, q, face_integrator_velocity[side]->get_gradient(q));
      }
    }

    tau = std::max(face_integrator_solution[0][0]->read_cell_data(array_penalty_parameter),
                   face_integrator_solution[1][0]->read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<dim, Number>(degree, element_type, data.IP_factor);
  }

  /*
   * Boundary face: interior side as on interior faces. The exterior side (side 1) is set by the
   * boundary condition of each variable, as in weak_boundary_conditions.h:
   *
   *   Dirichlet: Q+ = 2 g(evaluation_time) - Q-, grad Q+ . n = grad Q- . n
   *   Neumann (zero flux only): Q+ = Q-, grad Q+ . n = - grad Q- . n
   *
   * nu_t on the exterior side is the interior nu_t (as in DiffusiveKernel).
   */
  void
  reinit_boundary_face(unsigned int const face) const
  {
    boundary_face = true;

    face_integrator_velocity[0]->reinit(face);
    face_integrator_velocity[0]->gather_evaluate(*velocity, dealii::EvaluationFlags::gradients);

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      face_integrator_solution[0][i]->reinit(face);
      face_integrator_solution[0][i]->gather_evaluate(solution->block(i),
                                                      dealii::EvaluationFlags::values |
                                                        dealii::EvaluationFlags::gradients);
    }

    dealii::types::boundary_id const boundary_id = matrix_free->get_boundary_id(face);

    set_face_dirichlet(boundary_id);

    for(unsigned int q = 0; q < n_q_points_face; ++q)
    {
      for(unsigned int i = 0; i < n_variables; ++i)
      {
        set_face_point_solution(0,
                                q,
                                i,
                                face_integrator_solution[0][i]->get_value(q),
                                face_integrator_solution[0][i]->get_normal_derivative(q));

        scalar g = dealii::make_vectorized_array<Number>(0.0);
        if(face_dirichlet[i])
          g = FunctionEvaluator<0, dim, Number>::value(
            *(data.bc[i]->dirichlet_bc.find(boundary_id)->second),
            face_integrator_solution[0][i]->quadrature_point(q),
            evaluation_time);

        set_boundary_exterior_point(q, i, face_dirichlet[i], g);
      }

      evaluate_face_point(0, q, face_integrator_velocity[0]->get_gradient(q));
      copy_boundary_viscosity(q);
    }

    tau = face_integrator_solution[0][0]->read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<dim, Number>(degree, element_type, data.IP_factor);
  }

  /*
   * Face data at single points (as reinit_cell / evaluate_point for cells): used by reinit_face(),
   * reinit_boundary_face() and by a copy of the kernel evaluated at the faces of the enriched wall
   * cells (FunctionEnrichmentRANS).
   */

  // number of points of the face data
  void
  reinit_face_points(unsigned int const n_points) const
  {
    n_q_points_face = n_points;

    for(unsigned int side = 0; side < 2; ++side)
    {
      face_solution[side].resize(n_q_points_face * n_variables);
      face_normal_gradient[side].resize(n_q_points_face * n_variables);
      face_viscosity[side].resize(n_q_points_face);
      face_viscosity_derivative[side].resize(n_q_points_face * n_variables);
    }
  }

  unsigned int
  get_n_q_points_face() const
  {
    return n_q_points_face;
  }

  // interior face (false) or boundary face (true) for the following face points
  void
  set_boundary_face(bool const is_boundary_face) const
  {
    boundary_face = is_boundary_face;
  }

  // transported value and normal derivative of variable i at point q of side (0: e-, 1: e+)
  void
  set_face_point_solution(unsigned int const side,
                          unsigned int const q,
                          unsigned int const i,
                          scalar const &     value,
                          scalar const &     normal_gradient) const
  {
    face_solution[side][index(q, i)]        = value;
    face_normal_gradient[side][index(q, i)] = normal_gradient;
  }

  // nu_D and d nu_D / d Q_j at point q of side from face_solution and the velocity gradient
  void
  evaluate_face_point(unsigned int const side,
                      unsigned int const q,
                      tensor const &     velocity_gradient) const
  {
    compute_face_viscosity(side, q, velocity_gradient);
  }

  /*
   * Boundary face, point q, variable i: exterior side (1) from the interior side (0), as in
   * weak_boundary_conditions.h:
   *   Dirichlet (value g): Q+ = 2 g - Q-, grad Q+ . n = grad Q- . n
   *   zero flux:           Q+ = Q-,       grad Q+ . n = - grad Q- . n
   */
  void
  set_boundary_exterior_point(unsigned int const q,
                              unsigned int const i,
                              bool const         dirichlet,
                              scalar const &     g) const
  {
    scalar const value_m           = face_solution[0][index(q, i)];
    scalar const normal_gradient_m = face_normal_gradient[0][index(q, i)];

    if(dirichlet)
    {
      face_solution[1][index(q, i)]        = -value_m + g * 2.0;
      face_normal_gradient[1][index(q, i)] = normal_gradient_m;
    }
    else
    {
      face_solution[1][index(q, i)]        = value_m;
      face_normal_gradient[1][index(q, i)] = -normal_gradient_m;
    }
  }

  // boundary face, point q: exterior nu_D = interior nu_D (as in DiffusiveKernel)
  void
  copy_boundary_viscosity(unsigned int const q) const
  {
    face_viscosity[1][q] = face_viscosity[0][q];
    for(unsigned int j = 0; j < n_variables; ++j)
      face_viscosity_derivative[1][index(q, j)] = face_viscosity_derivative[0][index(q, j)];
  }

  /*
   * Boundary face, point q, variable i, lane by lane: Dirichlet (value g) where dirichlet_mask = 1,
   * zero flux where dirichlet_mask = 0 (as set_boundary_exterior_point)
   */
  void
  set_boundary_exterior_point_lanes(unsigned int const q,
                                    unsigned int const i,
                                    scalar const &     dirichlet_mask,
                                    scalar const &     g) const
  {
    scalar const one               = dealii::make_vectorized_array<Number>(1.0);
    scalar const value_m           = face_solution[0][index(q, i)];
    scalar const normal_gradient_m = face_normal_gradient[0][index(q, i)];

    face_solution[1][index(q, i)] =
      dirichlet_mask * (-value_m + g * 2.0) + (one - dirichlet_mask) * value_m;
    face_normal_gradient[1][index(q, i)] =
      dirichlet_mask * normal_gradient_m - (one - dirichlet_mask) * normal_gradient_m;
  }

  // true if variable i has a Dirichlet condition on boundary_id (CellDirichlet: zero flux face)
  bool
  is_dirichlet(unsigned int const i, dealii::types::boundary_id const boundary_id) const
  {
    return data.bc[i]->get_boundary_type(boundary_id) == BoundaryType::Dirichlet;
  }

  // Dirichlet value of variable i on boundary_id at a point, evaluation time of the kernel
  scalar
  get_dirichlet_value(unsigned int const               i,
                      dealii::types::boundary_id const boundary_id,
                      dealii::Point<dim, scalar> const & point) const
  {
    return FunctionEvaluator<0, dim, Number>::value(
      *(data.bc[i]->dirichlet_bc.find(boundary_id)->second), point, evaluation_time);
  }

  // face data at point q of side (after set_face_point_solution / set_boundary_exterior_point)
  scalar
  get_face_point_value(unsigned int const side, unsigned int const q, unsigned int const i) const
  {
    return face_solution[side][index(q, i)];
  }

  scalar
  get_face_point_normal_gradient(unsigned int const side,
                                 unsigned int const q,
                                 unsigned int const i) const
  {
    return face_normal_gradient[side][index(q, i)];
  }

  // penalty factor tau of the current face (set_face_penalty / reinit_face)
  scalar
  get_face_penalty() const
  {
    return tau;
  }

  // interior penalty parameter per cell batch of the kernel's MatrixFree
  dealii::AlignedVector<scalar> const &
  get_penalty_parameter_array() const
  {
    return array_penalty_parameter;
  }

  // penalty factor tau of the following face fluxes
  void
  set_face_penalty(scalar const & tau_in) const
  {
    tau = tau_in;
  }

  // IP factor of the penalty: tau = (cell penalty parameter) * get_penalty_factor()
  Number
  get_penalty_factor() const
  {
    return IP::get_penalty_factor<dim, Number>(degree, element_type, data.IP_factor);
  }

  /*
     * Storage of the linearization point. The Newton_Explicit right-hand side loops compute the data
     * of a cell / face at the linearization point (reinit_cell(), reinit_face(),
     * reinit_boundary_face()) and keep them with store_cell() / store_face(). The Newton_Implicit
     * loops, called in every iteration of the linear solver, read them with reinit_cell_stored(),
     * reinit_face_stored(), reinit_boundary_face_stored() instead of computing them again. Only the
     * data used by the Newton_Implicit fluxes are stored.
     */
  void
  store_cell(unsigned int const cell) const
  {
    copy_to_storage(cell_solution_gradient, stored_cell_solution_gradient, cell);
    copy_to_storage(cell_viscosity, stored_cell_viscosity, cell);
    copy_to_storage(cell_viscosity_derivative, stored_cell_viscosity_derivative, cell);
    copy_to_storage(cell_production_jacobian, stored_cell_production_jacobian, cell);
    copy_to_storage(cell_dissipation_jacobian, stored_cell_dissipation_jacobian, cell);
    copy_to_storage(cell_square_gradient_jacobian, stored_cell_square_gradient_jacobian, cell);
    copy_to_storage(cell_square_gradient_gradient_jacobian,
                    stored_cell_square_gradient_gradient_jacobian,
                    cell);
    copy_to_storage(cell_cross_diffusion_jacobian, stored_cell_cross_diffusion_jacobian, cell);
    copy_to_storage(cell_cross_diffusion_gradient_jacobian,
                    stored_cell_cross_diffusion_gradient_jacobian,
                    cell);
  }

  void
  reinit_cell_stored(unsigned int const cell) const
  {
    copy_from_storage(stored_cell_solution_gradient, cell_solution_gradient, cell);
    copy_from_storage(stored_cell_viscosity, cell_viscosity, cell);
    copy_from_storage(stored_cell_viscosity_derivative, cell_viscosity_derivative, cell);
    copy_from_storage(stored_cell_production_jacobian, cell_production_jacobian, cell);
    copy_from_storage(stored_cell_dissipation_jacobian, cell_dissipation_jacobian, cell);
    copy_from_storage(stored_cell_square_gradient_jacobian, cell_square_gradient_jacobian, cell);
    copy_from_storage(stored_cell_square_gradient_gradient_jacobian,
                      cell_square_gradient_gradient_jacobian,
                      cell);
    copy_from_storage(stored_cell_cross_diffusion_jacobian, cell_cross_diffusion_jacobian, cell);
    copy_from_storage(stored_cell_cross_diffusion_gradient_jacobian,
                      cell_cross_diffusion_gradient_jacobian,
                      cell);
  }

  // Dirichlet mask (1: Dirichlet, 0: zero flux) of variable i on the current boundary face, lane
  // by lane (instead of set_face_dirichlet())
  void
  set_face_dirichlet_mask(unsigned int const i, scalar const & mask) const
  {
    face_dirichlet_mask[i] = mask;
  }

  scalar
  get_face_dirichlet_mask(unsigned int const i) const
  {
    return face_dirichlet_mask[i];
  }

  /*
   * Linearization point outside the kernel's own storage, e.g. at the points of the enriched wall
   * cells of FunctionEnrichmentRANS (another MatrixFree, other numbers of points). Sized with
   * resize_storage() after reinit_points() / reinit_face_points(); slot = cell / face batch.
   */
  struct LinearizationStorage
  {
    dealii::AlignedVector<vector> cell_solution_gradient;
    dealii::AlignedVector<scalar> cell_viscosity;
    dealii::AlignedVector<scalar> cell_viscosity_derivative;
    dealii::AlignedVector<scalar> cell_production_jacobian;
    dealii::AlignedVector<scalar> cell_dissipation_jacobian;
    dealii::AlignedVector<scalar> cell_square_gradient_jacobian;
    dealii::AlignedVector<vector> cell_square_gradient_gradient_jacobian;
    dealii::AlignedVector<scalar> cell_cross_diffusion_jacobian;
    dealii::AlignedVector<vector> cell_cross_diffusion_gradient_jacobian;

    std::array<dealii::AlignedVector<scalar>, 2> face_solution;
    std::array<dealii::AlignedVector<scalar>, 2> face_normal_gradient;
    std::array<dealii::AlignedVector<scalar>, 2> face_viscosity;
    std::array<dealii::AlignedVector<scalar>, 2> face_viscosity_derivative;
    dealii::AlignedVector<scalar>                tau;
    dealii::AlignedVector<scalar>                face_dirichlet_mask; // [face * n_variables + i]
  };

  void
  resize_storage(LinearizationStorage & storage,
                 unsigned int const     n_cell_batches,
                 unsigned int const     n_face_batches) const
  {
    storage.cell_solution_gradient.resize(n_cell_batches * cell_solution_gradient.size());
    storage.cell_viscosity.resize(n_cell_batches * cell_viscosity.size());
    storage.cell_viscosity_derivative.resize(n_cell_batches * cell_viscosity_derivative.size());
    storage.cell_production_jacobian.resize(n_cell_batches * cell_production_jacobian.size());
    storage.cell_dissipation_jacobian.resize(n_cell_batches * cell_dissipation_jacobian.size());
    storage.cell_square_gradient_jacobian.resize(n_cell_batches *
                                                 cell_square_gradient_jacobian.size());
    storage.cell_square_gradient_gradient_jacobian.resize(
      n_cell_batches * cell_square_gradient_gradient_jacobian.size());
    storage.cell_cross_diffusion_jacobian.resize(n_cell_batches *
                                                 cell_cross_diffusion_jacobian.size());
    storage.cell_cross_diffusion_gradient_jacobian.resize(
      n_cell_batches * cell_cross_diffusion_gradient_jacobian.size());

    for(unsigned int side = 0; side < 2; ++side)
    {
      storage.face_solution[side].resize(n_face_batches * face_solution[side].size());
      storage.face_normal_gradient[side].resize(n_face_batches * face_normal_gradient[side].size());
      storage.face_viscosity[side].resize(n_face_batches * face_viscosity[side].size());
      storage.face_viscosity_derivative[side].resize(n_face_batches *
                                                     face_viscosity_derivative[side].size());
    }
    storage.tau.resize(n_face_batches);
    storage.face_dirichlet_mask.resize(n_face_batches * n_variables);
  }

  void
  save_cell_data(LinearizationStorage & storage, unsigned int const cell) const
  {
    copy_to_storage(cell_solution_gradient, storage.cell_solution_gradient, cell);
    copy_to_storage(cell_viscosity, storage.cell_viscosity, cell);
    copy_to_storage(cell_viscosity_derivative, storage.cell_viscosity_derivative, cell);
    copy_to_storage(cell_production_jacobian, storage.cell_production_jacobian, cell);
    copy_to_storage(cell_dissipation_jacobian, storage.cell_dissipation_jacobian, cell);
    copy_to_storage(cell_square_gradient_jacobian, storage.cell_square_gradient_jacobian, cell);
    copy_to_storage(cell_square_gradient_gradient_jacobian,
                    storage.cell_square_gradient_gradient_jacobian,
                    cell);
    copy_to_storage(cell_cross_diffusion_jacobian, storage.cell_cross_diffusion_jacobian, cell);
    copy_to_storage(cell_cross_diffusion_gradient_jacobian,
                    storage.cell_cross_diffusion_gradient_jacobian,
                    cell);
  }

  void
  load_cell_data(LinearizationStorage const & storage, unsigned int const cell) const
  {
    copy_from_storage(storage.cell_solution_gradient, cell_solution_gradient, cell);
    copy_from_storage(storage.cell_viscosity, cell_viscosity, cell);
    copy_from_storage(storage.cell_viscosity_derivative, cell_viscosity_derivative, cell);
    copy_from_storage(storage.cell_production_jacobian, cell_production_jacobian, cell);
    copy_from_storage(storage.cell_dissipation_jacobian, cell_dissipation_jacobian, cell);
    copy_from_storage(storage.cell_square_gradient_jacobian, cell_square_gradient_jacobian, cell);
    copy_from_storage(storage.cell_square_gradient_gradient_jacobian,
                      cell_square_gradient_gradient_jacobian,
                      cell);
    copy_from_storage(storage.cell_cross_diffusion_jacobian, cell_cross_diffusion_jacobian, cell);
    copy_from_storage(storage.cell_cross_diffusion_gradient_jacobian,
                      cell_cross_diffusion_gradient_jacobian,
                      cell);
  }

  void
  save_face_data(LinearizationStorage & storage, unsigned int const face) const
  {
    for(unsigned int side = 0; side < 2; ++side)
    {
      copy_to_storage(face_solution[side], storage.face_solution[side], face);
      copy_to_storage(face_normal_gradient[side], storage.face_normal_gradient[side], face);
      copy_to_storage(face_viscosity[side], storage.face_viscosity[side], face);
      copy_to_storage(face_viscosity_derivative[side],
                      storage.face_viscosity_derivative[side],
                      face);
    }
    storage.tau[face] = tau;
    copy_to_storage(face_dirichlet_mask, storage.face_dirichlet_mask, face);
  }

  // boundary: true for boundary faces (exterior increment from face_dirichlet_mask)
  void
  load_face_data(LinearizationStorage const & storage,
                 unsigned int const           face,
                 bool const                   boundary) const
  {
    boundary_face = boundary;
    for(unsigned int side = 0; side < 2; ++side)
    {
      copy_from_storage(storage.face_solution[side], face_solution[side], face);
      copy_from_storage(storage.face_normal_gradient[side], face_normal_gradient[side], face);
      copy_from_storage(storage.face_viscosity[side], face_viscosity[side], face);
      copy_from_storage(storage.face_viscosity_derivative[side],
                        face_viscosity_derivative[side],
                        face);
    }
    tau = storage.tau[face];
    copy_from_storage(storage.face_dirichlet_mask, face_dirichlet_mask, face);
  }

  // interior and boundary faces (one numbering of face batches in matrix_free)
  void
  store_face(unsigned int const face) const
  {
    for(unsigned int side = 0; side < 2; ++side)
    {
      copy_to_storage(face_solution[side], stored_face_solution[side], face);
      copy_to_storage(face_normal_gradient[side], stored_face_normal_gradient[side], face);
      copy_to_storage(face_viscosity[side], stored_face_viscosity[side], face);
      copy_to_storage(face_viscosity_derivative[side],
                      stored_face_viscosity_derivative[side],
                      face);
    }

    stored_tau[face] = tau;
  }

  void
  reinit_face_stored(unsigned int const face) const
  {
    boundary_face = false;

    load_face(face);
  }

  void
  reinit_boundary_face_stored(unsigned int const face) const
  {
    boundary_face = true;

    set_face_dirichlet(matrix_free->get_boundary_id(face));

    load_face(face);
  }

  /*
     * Time at which the boundary conditions are evaluated (boundary faces of the implicit turbulent
     * diffusion)
     */
  void
  set_evaluation_time(double const time) const
  {
    evaluation_time = time;
  }

  void
  set_time_step_size(double const dt,
                     double const time,
                     double const end_time,
                     double const time_step_number) const
  {
    data.time_step_size   = dt;
    data.current_time     = time;
    data.end_time         = end_time;
    data.time_step_number = time_step_number;
  }

  /*
   * Terminology of the time step (unknown: the increment dQ):
   *
   *   Explicit        : term evaluated at the previous time step Q^n, right-hand side only
   *   Implicit        : term linearized at the current solution Q (Q^n or the Newton iterate)
   *     Newton_Explicit : value of the term at Q, right-hand side
   *     Newton_Implicit : derivative of the term at Q applied to dQ, left-hand side
   *
   * All values below are evaluated at the solution set by set_solution_ptr().
   */

  /*
   * Source terms P, D, SG treated Explicit. Right-hand side signs: +P, -D, +SG.
   */
  scalar
  get_source_explicit_value(unsigned int const q, unsigned int const block) const
  {
    scalar value = dealii::make_vectorized_array<Number>(0.0);

    if(not data.production_implicit)
      value += cell_production[index(q, block)];
    if(not data.dissipation_implicit)
      value -= cell_dissipation[index(q, block)];
    if(not data.square_gradient_implicit)
      value += cell_square_gradient[index(q, block)];
    if(not data.cross_diffusion_implicit)
      value += cell_cross_diffusion[index(q, block)];

    return value;
  }

  /*
   * Newton_Explicit part of the source terms P, D, SG treated Implicit. Right-hand side signs:
   * +P, -D, +SG.
   */
  scalar
  get_source_newton_explicit_value(unsigned int const q, unsigned int const block) const
  {
    scalar value = dealii::make_vectorized_array<Number>(0.0);

    if(data.production_implicit)
      value += cell_production[index(q, block)];
    if(data.dissipation_implicit)
      value -= cell_dissipation[index(q, block)];
    if(data.square_gradient_implicit)
      value += cell_square_gradient[index(q, block)];
    if(data.cross_diffusion_implicit)
      value += cell_cross_diffusion[index(q, block)];

    return value;
  }

  /*
     * Newton_Implicit part of the source terms P, D, CD (cross-diffusion), SG treated Implicit, at
     * quadrature point q, row "block" (equation of transport variable "block"), all columns j:
     *
     *   values[j]    : dQ_j at q
     *   gradients[j] : grad dQ_j at q
     * 
     * Left-hand side signs: -P', +D', -CD', -SG'.
     */
  scalar
  get_source_newton_implicit(dealii::ArrayView<scalar const> const & values,
                             dealii::ArrayView<vector const> const & gradients,
                             unsigned int const                      q,
                             unsigned int const                      block) const
  {
    scalar flux = dealii::make_vectorized_array<Number>(0.0);

    for(unsigned int j = 0; j < n_variables; ++j)
    {
      if(data.production_implicit)
        flux -= cell_production_jacobian[index(q, block, j)] * values[j];
      if(data.dissipation_implicit)
        flux += cell_dissipation_jacobian[index(q, block, j)] * values[j];
      if(data.square_gradient_implicit)
        flux -= cell_square_gradient_jacobian[index(q, block, j)] * values[j];

      // CD_i depends on Q_j and on grad Q_j of all variables (omega equation: grad k and
      // grad omega): d CD_i / d Q_j dQ_j + d CD_i / d grad Q_j . grad dQ_j
      if(data.cross_diffusion_implicit)
        flux -= cell_cross_diffusion_jacobian[index(q, block, j)] * values[j] +
          cell_cross_diffusion_gradient_jacobian[index(q, block, j)] * gradients[j];
    }

    // SG_i = (nu + sigma_i nu_t) |grad Q_i|^2 also depends on grad Q_i (own variable only):
    // d SG_i / d grad Q_i * grad dQ_i
    if(data.square_gradient_implicit)
      flux -= cell_square_gradient_gradient_jacobian[index(q, block)] * gradients[block];

    return flux;
  }

  /*
   * Turbulent diffusion -div(sigma_i nu_t grad Q_i), SIPG discretization in the same form as the
   * molecular diffusion in DiffusiveKernel. All fluxes use the left-hand side sign (the operator
   * applied to Q); the right-hand side (Explicit, Newton_Explicit) is the negative.
   *
   * Cell flux (tested with grad v): sigma_i nu_t grad Q_i
   */
#ifdef EXADG_DEBUG
  // diagnostics: transported variable, production, dissipation and square-gradient term of block i
  // at point q (as in the source terms), and the diffusion viscosity nu_D of point q
  // (also nu_D before the wall-law bound and the bound nu_t^psi, unclamped; 0 without wall law)
  std::array<scalar, 7>
  get_point_terms(unsigned int const q, unsigned int const i) const
  {
    scalar const zero = dealii::make_vectorized_array<Number>(0.0);
    bool const   wall_law = eddy_viscosity_wall_law != nullptr;
    return {{cell_solution[index(q, i)],
             cell_production[index(q, i)],
             cell_dissipation[index(q, i)],
             cell_square_gradient[index(q, i)],
             cell_viscosity[q],
             wall_law ? debug_viscosity_unbounded[q] : cell_viscosity[q],
             wall_law ? debug_nu_t_wall_law[q] : zero}};
  }
#endif

  vector
  get_diffusion_cell_flux(unsigned int const q, unsigned int const block) const
  {
    scalar const sigma = dealii::make_vectorized_array<Number>(data.eddy_viscosity_factor[block]);

    return sigma * cell_viscosity[q] * cell_solution_gradient[index(q, block)];
  }

  /*
   * Newton_Implicit cell flux (tested with grad v), with eta = sum_j d nu_t / d Q_j dQ_j:
   *
   *   sigma_i (nu_t grad dQ_i + eta grad Q_i)
   *
   * For k and omega: eta = (1/omega) dk - (k/omega^2) domega (with the realizability limiter and
   * the chain rule for ln k / ln omega included in d nu_t / d Q_j).
   */
  vector
  get_diffusion_cell_flux_newton_implicit(dealii::ArrayView<scalar const> const & values,
                                          dealii::ArrayView<vector const> const & gradients,
                                          unsigned int const                      q,
                                          unsigned int const                      block) const
  {
    scalar const sigma = dealii::make_vectorized_array<Number>(data.eddy_viscosity_factor[block]);

    scalar eta = dealii::make_vectorized_array<Number>(0.0);
    for(unsigned int j = 0; j < n_variables; ++j)
      eta += cell_viscosity_derivative[index(q, j)] * values[j];

    return sigma * (cell_viscosity[q] * gradients[block] +
                    eta * cell_solution_gradient[index(q, block)]);
  }

  /*
   * SIPG face flux of the turbulent diffusion, as in DiffusiveOperator::do_face_integral():
   *
   *   value       : 0.5 (mu- grad Q- . n + mu+ grad Q+ . n) - tau mu_penalty (Q- - Q+)
   *                 (submitted as -value on e-, +value on e+)
   *   gradient_m  : -0.5 mu- (Q- - Q+)  (submitted as normal derivative on e-)
   *   gradient_p  : -0.5 mu+ (Q- - Q+)  (submitted as normal derivative on e+)
   *
   * with mu = sigma_i nu_t and mu_penalty = max(mu-, mu+) (interior face) or mu- (boundary face).
   */
  struct FaceFlux
  {
    scalar value;
    scalar gradient_m;
    scalar gradient_p;
  };

  FaceFlux
  get_diffusion_face_flux(unsigned int const q, unsigned int const block) const
  {
    scalar const sigma = dealii::make_vectorized_array<Number>(data.eddy_viscosity_factor[block]);

    scalar const viscosity_m = sigma * face_viscosity[0][q];
    scalar const viscosity_p = sigma * face_viscosity[1][q];
    scalar const viscosity_penalty =
      boundary_face ? viscosity_m : std::max(viscosity_m, viscosity_p);

    scalar const jump = face_solution[0][index(q, block)] - face_solution[1][index(q, block)];

    FaceFlux flux;
    flux.value = 0.5 * (viscosity_m * face_normal_gradient[0][index(q, block)] +
                        viscosity_p * face_normal_gradient[1][index(q, block)]) -
                 tau * viscosity_penalty * jump;
    flux.gradient_m = -0.5 * viscosity_m * jump;
    flux.gradient_p = -0.5 * viscosity_p * jump;

    return flux;
  }

  /*
   * Newton_Implicit face flux: derivative of get_diffusion_face_flux() applied to dQ.
   *
   *   values_m[j], values_p[j]                     : dQ_j on e-, e+
   *   normal_gradients_m[j], normal_gradients_p[j] : grad dQ_j . n on e-, e+
   *
   * On boundary faces only the e- arrays are used; the exterior increment follows from the
   * homogeneous boundary condition (Dirichlet: dQ+ = -dQ-, grad dQ+ . n = grad dQ- . n; Neumann:
   * dQ+ = dQ-, grad dQ+ . n = -grad dQ- . n), and eta+ = eta- (exterior nu_t = interior nu_t).
   * The penalty viscosity max(mu-, mu+) is differentiated on the larger side (lane by lane).
   */
  FaceFlux
  get_diffusion_face_flux_newton_implicit(
    dealii::ArrayView<scalar const> const & values_m,
    dealii::ArrayView<scalar const> const & values_p,
    dealii::ArrayView<scalar const> const & normal_gradients_m,
    dealii::ArrayView<scalar const> const & normal_gradients_p,
    unsigned int const                      q,
    unsigned int const                      block) const
  {
    scalar const zero  = dealii::make_vectorized_array<Number>(0.0);
    scalar const one   = dealii::make_vectorized_array<Number>(1.0);
    scalar const sigma = dealii::make_vectorized_array<Number>(data.eddy_viscosity_factor[block]);

    // eta = sum_j d nu_t / d Q_j dQ_j on both sides
    scalar eta_m = zero;
    for(unsigned int j = 0; j < n_variables; ++j)
      eta_m += face_viscosity_derivative[0][index(q, j)] * values_m[j];

    scalar eta_p = eta_m;
    if(not boundary_face)
    {
      eta_p = zero;
      for(unsigned int j = 0; j < n_variables; ++j)
        eta_p += face_viscosity_derivative[1][index(q, j)] * values_p[j];
    }

    // increment of variable "block" on the exterior side
    scalar increment_p, increment_normal_gradient_p;
    if(boundary_face)
    {
      // lane by lane: Dirichlet (mask 1): dQ+ = -dQ-, grad dQ+ . n = grad dQ- . n;
      // zero flux (mask 0): dQ+ = dQ-, grad dQ+ . n = -grad dQ- . n
      scalar const mask           = face_dirichlet_mask[block];
      increment_p                 = (one - 2.0 * mask) * values_m[block];
      increment_normal_gradient_p = (2.0 * mask - one) * normal_gradients_m[block];
    }
    else
    {
      increment_p                 = values_p[block];
      increment_normal_gradient_p = normal_gradients_p[block];
    }

    scalar const viscosity_m = sigma * face_viscosity[0][q];
    scalar const viscosity_p = sigma * face_viscosity[1][q];

    scalar const viscosity_increment_m = sigma * eta_m;
    scalar const viscosity_increment_p = sigma * eta_p;

    // penalty viscosity and its derivative
    scalar viscosity_penalty, viscosity_penalty_increment;
    if(boundary_face)
    {
      viscosity_penalty           = viscosity_m;
      viscosity_penalty_increment = viscosity_increment_m;
    }
    else
    {
      scalar const side_m_larger =
        dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(viscosity_m,
                                                                             viscosity_p,
                                                                             one,
                                                                             zero);
      viscosity_penalty = std::max(viscosity_m, viscosity_p);
      viscosity_penalty_increment =
        side_m_larger * viscosity_increment_m + (one - side_m_larger) * viscosity_increment_p;
    }

    scalar const jump = face_solution[0][index(q, block)] - face_solution[1][index(q, block)];
    scalar const increment_jump = values_m[block] - increment_p;

    scalar const normal_gradient_m = face_normal_gradient[0][index(q, block)];
    scalar const normal_gradient_p = face_normal_gradient[1][index(q, block)];

    FaceFlux flux;
    flux.value = 0.5 * (viscosity_m * normal_gradients_m[block] +
                        viscosity_increment_m * normal_gradient_m +
                        viscosity_p * increment_normal_gradient_p +
                        viscosity_increment_p * normal_gradient_p) -
                 tau * (viscosity_penalty * increment_jump + viscosity_penalty_increment * jump);
    flux.gradient_m = -0.5 * (viscosity_m * increment_jump + viscosity_increment_m * jump);
    flux.gradient_p = -0.5 * (viscosity_p * increment_jump + viscosity_increment_p * jump);

    return flux;
  }

  bool
  is_boundary_face() const
  {
    return boundary_face;
  }

  std::shared_ptr<TurbulenceModel<dim, Number>> turbulence_model_ptr;

private:
  // position of transport variable i at quadrature point q in the cell_* arrays
  unsigned int
  index(unsigned int const q, unsigned int const i) const
  {
    return q * n_variables + i;
  }

  // position of the Jacobian entry d (term of variable i) / d Q_j at quadrature point q
  unsigned int
  index(unsigned int const q, unsigned int const i, unsigned int const j) const
  {
    return (q * n_variables + i) * n_variables + j;
  }

  // copy the data of the current cell / face (current) into slot "batch" of storage, and back
  template<typename T>
  static void
  copy_to_storage(dealii::AlignedVector<T> const & current,
                  dealii::AlignedVector<T> &       storage,
                  unsigned int const               batch)
  {
    std::copy(current.begin(), current.end(), storage.begin() + batch * current.size());
  }

  template<typename T>
  static void
  copy_from_storage(dealii::AlignedVector<T> const & storage,
                    dealii::AlignedVector<T> &       current,
                    unsigned int const               batch)
  {
    std::copy(storage.begin() + batch * current.size(),
              storage.begin() + (batch + 1) * current.size(),
              current.begin());
  }

  // face data of both sides and tau of face batch "face" from the storage (see store_face())
  void
  load_face(unsigned int const face) const
  {
    for(unsigned int side = 0; side < 2; ++side)
    {
      copy_from_storage(stored_face_solution[side], face_solution[side], face);
      copy_from_storage(stored_face_normal_gradient[side], face_normal_gradient[side], face);
      copy_from_storage(stored_face_viscosity[side], face_viscosity[side], face);
      copy_from_storage(stored_face_viscosity_derivative[side],
                        face_viscosity_derivative[side],
                        face);
    }

    tau = stored_tau[face];
  }

  // face_dirichlet[i]: true if the boundary condition of variable i on boundary_id is Dirichlet,
  // false if it is (zero flux) Neumann or CellDirichlet (value imposed in the cell, face is zero flux)
  void
  set_face_dirichlet(dealii::types::boundary_id const boundary_id) const
  {
    for(unsigned int i = 0; i < n_variables; ++i)
    {
      BoundaryType const boundary_type = data.bc[i]->get_boundary_type(boundary_id);

      AssertThrow(boundary_type == BoundaryType::Dirichlet or
                  boundary_type == BoundaryType::Neumann or
                  boundary_type == BoundaryType::CellDirichlet,
                  dealii::ExcMessage(
                  "Turbulent diffusion: only Dirichlet, (zero flux) Neumann and CellDirichlet boundaries are implemented."));

      face_dirichlet[i] = (boundary_type == BoundaryType::Dirichlet);
      face_dirichlet_mask[i] = dealii::make_vectorized_array<Number>(face_dirichlet[i] ? 1.0 : 0.0);
    }
  }

  /*
     * Diffusion viscosity nu_D and d nu_D / d Q_j at face quadrature point q of side "side"
     * (Wilcox 1988: nu_D = nu_t), from the transported values in face_solution[side] and the velocity
     * gradient of that side. The face terms need no gradients of the transport variables.
     */
  void
  compute_face_viscosity(unsigned int const side,
                         unsigned int const q,
                         tensor const &     velocity_gradient) const
  {
    rans_model->set_physical_variables(
      dealii::ArrayView<scalar const>(face_solution[side].data() + index(q, 0), n_variables),
      dealii::ArrayView<vector const>(),
      point_input);
    point_input.velocity_gradient   = velocity_gradient;
    point_input.compute_derivatives = true;

    rans_model->evaluate_viscosities(point_input, point_eddy_viscosity, point_diffusion_viscosity);
    rans_model->convert_to_transported_form(point_input, point_diffusion_viscosity);

    face_viscosity[side][q] = point_diffusion_viscosity.value;
    for(unsigned int j = 0; j < n_variables; ++j)
      face_viscosity_derivative[side][index(q, j)] = point_diffusion_viscosity.derivative[j];
  }

  // wall model (enriched cells): nu_t -> min(nu_t, nu_t^psi); no dependence on Q where bounded
  void
  apply_wall_law_viscosity_bound(unsigned int const q, scalar const & nu_t_wall_law_in) const
  {
    scalar const zero = dealii::make_vectorized_array<Number>(0.0);
    scalar const one  = dealii::make_vectorized_array<Number>(1.0);

    // nu_t^psi is a polynomial field (eddy viscosity DoFs) that can undershoot below 0 near the
    // wall: the bound is >= 0 (no negative turbulent viscosity)
    scalar const nu_t_wall_law = std::max(nu_t_wall_law_in, zero);

#ifdef EXADG_DEBUG
    debug_viscosity_unbounded[q] = cell_viscosity[q];
    debug_nu_t_wall_law[q]       = nu_t_wall_law_in;
#endif

    scalar const free = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      cell_viscosity[q], nu_t_wall_law, one, zero);

    cell_viscosity[q] = std::min(cell_viscosity[q], nu_t_wall_law);
    for(unsigned int j = 0; j < n_variables; ++j)
      cell_viscosity_derivative[index(q, j)] *= free;
  }

  // wall model (enriched cells): k production nu_t G -> min(nu_t G, nu_t^psi G)
  void
  apply_wall_law_production_bound(unsigned int const q,
                                  scalar const &     nu_t_wall_law_in,
                                  tensor const &     velocity_gradient,
                                  scalar const &     tke) const
  {
    scalar const zero = dealii::make_vectorized_array<Number>(0.0);
    scalar const one  = dealii::make_vectorized_array<Number>(1.0);

    // nu_t^psi >= 0 (polynomial field, see apply_wall_law_viscosity_bound): bound >= 0, so no
    // negative k production
    scalar const nu_t_wall_law = std::max(nu_t_wall_law_in, zero);

    scalar const G =
      scalar_product(velocity_gradient + transpose(velocity_gradient), velocity_gradient);
    scalar const bound = data.k_log ? nu_t_wall_law * G / tke : nu_t_wall_law * G;

    scalar const free = dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
      cell_production[index(q, 0)], bound, one, zero);

    cell_production[index(q, 0)] = std::min(cell_production[index(q, 0)], bound);
    for(unsigned int j = 0; j < n_variables; ++j)
      cell_production_jacobian[index(q, 0, j)] *= free;

    if(data.k_log)
      cell_production_jacobian[index(q, 0, 0)] -= (one - free) * bound;
  }

  void
  apply_tke_production_from_wall_law(unsigned int const q,
                                     scalar const &     tke_production_wall_law,
                                     scalar const &     indicator,
                                     scalar const &     tke) const
  {
    scalar const one  = dealii::make_vectorized_array<Number>(1.0);

    scalar const wall_cell = indicator;

    scalar const production = data.k_log ? tke_production_wall_law / tke : tke_production_wall_law;

    cell_production[index(q, 0)] =
      wall_cell * production + (one - wall_cell) * cell_production[index(q, 0)];

    for(unsigned int j = 0; j < n_variables; ++j)
      cell_production_jacobian[index(q, 0, j)] *= (one - wall_cell);

    if(data.k_log)
      cell_production_jacobian[index(q, 0, 0)] -= wall_cell * production;
  }

  void
  compute_square_gradient(unsigned int const q) const
  {
    scalar const zero        = dealii::make_vectorized_array<Number>(0.0);
    scalar const two         = dealii::make_vectorized_array<Number>(2.0);
    scalar const diffusivity =
      dealii::make_vectorized_array<Number>(data.diffusivity);

    scalar const viscosity = cell_viscosity[q];

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      // reset row i
      cell_square_gradient[index(q, i)]                   = zero;
      cell_square_gradient_gradient_jacobian[index(q, i)] = vector();
      for(unsigned int j = 0; j < n_variables; ++j)
        cell_square_gradient_jacobian[index(q, i, j)] = zero;

      bool const logarithmic = data.turbulence_model_data.positivity_preserving_limiter[i] == PositivityPreservingLimiter::LogarithmicTransportVariable;

      if(not logarithmic)
        continue;

      scalar const eddy_viscosity_factor =
        dealii::make_vectorized_array<Number>(data.eddy_viscosity_factor[i]);

      // nu + sigma_i nu_t, same diffusion coefficient as in DiffusiveKernel
      scalar const diffusion_coefficient = diffusivity + eddy_viscosity_factor * viscosity;

      vector const & gradient         = cell_solution_gradient[index(q, i)];
      scalar const   gradient_squared = gradient * gradient;

      cell_square_gradient[index(q, i)] = diffusion_coefficient * gradient_squared;

      if(data.square_gradient_implicit)
      {
        // d SG_i / d Q_j = sigma_i |grad Q_i|^2 d nu_t / d Q_j
        for(unsigned int j = 0; j < n_variables; ++j)
        {
          cell_square_gradient_jacobian[index(q, i, j)] = eddy_viscosity_factor * gradient_squared * cell_viscosity_derivative[index(q, j)];
        }

        // d SG_i / d grad Q_i = 2 (nu + sigma_i nu_t) grad Q_i
        cell_square_gradient_gradient_jacobian[index(q, i)] = two * diffusion_coefficient * gradient;
      }
    } 
  } 

  dealii::MatrixFree<dim, Number> const * matrix_free = nullptr;

  mutable TurbulenceSourceKernelData<dim> data;

  // time at which the boundary conditions are evaluated
  mutable double evaluation_time = 0.0;

  unsigned int n_variables     = 0;
  // mutable: reinit_points() sets it for a copy of the kernel evaluated at other points
  mutable unsigned int n_q_points_cell = 0;
  // mutable: reinit_face_points() sets it for a copy of the kernel evaluated at other points
  mutable unsigned int n_q_points_face = 0;

  mutable lazy_ptr<VectorType> velocity;
  mutable lazy_ptr<BlockVectorType> solution;

  // the RANS model (owned by the EddyViscosityEvaluator of turbulence_model_ptr, alive as long as
  // turbulence_model_ptr)
  RANSModelBase<dim, Number> const * rans_model = nullptr;

  // work structs of the model at one quadrature point (physical variables), sized in reinit()
  mutable RANSPointInput<dim, Number> point_input;
  mutable RANSViscosity<Number>       point_eddy_viscosity;
  mutable RANSViscosity<Number>       point_diffusion_viscosity;
  mutable RANSSourceTerm<dim, Number> point_production;
  mutable RANSSourceTerm<dim, Number> point_dissipation;
  mutable RANSSourceTerm<dim, Number> point_cross_diffusion;
  mutable RANSSourceTerm<dim, Number> point_square_gradient;

  std::shared_ptr<IntegratorCellVelocity>      integrator_velocity;
  std::vector<std::shared_ptr<IntegratorCell>> integrator_solution;

  // wall model: upper bound nu_t^psi of the eddy viscosity (nullptr: no bound)
  VectorType const *              eddy_viscosity_wall_law = nullptr;
  std::shared_ptr<IntegratorCell> integrator_wall_law;

  VectorType const *              tke_production_from_wall_law = nullptr;
  std::shared_ptr<IntegratorCell> integrator_tke_production_from_wall_law;
  VectorType const *              wall_cell_indicator = nullptr; 
  std::shared_ptr<IntegratorCell> integrator_wall_cell_indicator;

  // wall model: exact velocity gradient of u_h = U_bar + psi U_tilde in the enriched cells
  // (component d u_i / d x_j in [i * dim + j], DG fields on the eddy viscosity DoFHandler, exact in
  // the Gauss points) and indicator (1: enriched cell, use it; 0: use grad of the velocity field).
  // nullptr: the gradient of the transferred velocity field is used everywhere.
  std::vector<VectorType> const *              velocity_gradient_enriched  = nullptr;
  VectorType const *                           velocity_gradient_indicator = nullptr;
  std::vector<std::shared_ptr<IntegratorCell>> integrator_gradient_enriched;
  std::shared_ptr<IntegratorCell>              integrator_gradient_indicator;

  // face integrators, side 0: interior side, side 1: exterior side
  std::array<std::shared_ptr<IntegratorFaceVelocity>, 2>      face_integrator_velocity;
  std::array<std::vector<std::shared_ptr<IntegratorFace>>, 2> face_integrator_solution;

  // interior penalty (as in DiffusiveKernel)
  unsigned int                  degree       = 1;
  ElementType                   element_type = ElementType::Hypercube;
  dealii::AlignedVector<scalar> array_penalty_parameter;
  mutable scalar                tau           = dealii::make_vectorized_array<Number>(0.0);
  mutable bool                  boundary_face = false;

  // cell data at Q* at the quadrature points of the current cell
  mutable dealii::AlignedVector<scalar> cell_solution;
  mutable dealii::AlignedVector<vector> cell_solution_gradient;
  mutable dealii::AlignedVector<scalar> cell_viscosity;
#ifdef EXADG_DEBUG
  // diagnostics: nu_D before the wall-law bound and the bound nu_t^psi (unclamped), per point
  mutable dealii::AlignedVector<scalar> debug_viscosity_unbounded;
  mutable dealii::AlignedVector<scalar> debug_nu_t_wall_law;
#endif
  mutable dealii::AlignedVector<scalar> cell_viscosity_derivative;
  mutable dealii::AlignedVector<scalar> cell_production;
  mutable dealii::AlignedVector<scalar> cell_dissipation;
  mutable dealii::AlignedVector<scalar> cell_square_gradient;
  mutable dealii::AlignedVector<scalar> cell_production_jacobian;
  mutable dealii::AlignedVector<scalar> cell_dissipation_jacobian;
  mutable dealii::AlignedVector<scalar> cell_square_gradient_jacobian;
  mutable dealii::AlignedVector<vector> cell_square_gradient_gradient_jacobian;
  mutable dealii::AlignedVector<scalar> cell_cross_diffusion;
  mutable dealii::AlignedVector<scalar> cell_cross_diffusion_jacobian;
  mutable dealii::AlignedVector<vector> cell_cross_diffusion_gradient_jacobian;

  // face data at the solution set by set_solution_ptr(), at the quadrature points of the current
  // face, [side] (side 0: e-, side 1: e+ or the exterior boundary values)
  mutable std::array<dealii::AlignedVector<scalar>, 2> face_solution;
  mutable std::array<dealii::AlignedVector<scalar>, 2> face_normal_gradient;
  mutable std::array<dealii::AlignedVector<scalar>, 2> face_viscosity;
  mutable std::array<dealii::AlignedVector<scalar>, 2> face_viscosity_derivative;

  // boundary face: true if the boundary condition of variable i is Dirichlet, false if Neumann
  mutable std::vector<bool> face_dirichlet;

  // the same per SIMD lane (1: Dirichlet, 0: zero flux), used by the Newton_Implicit face flux;
  // set_face_dirichlet() or set_face_dirichlet_mask() (lane-wise wall condition of enriched
  // variables)
  mutable dealii::AlignedVector<scalar> face_dirichlet_mask;

  // linearization point of the Newton_Implicit terms for all cell batches and all face batches
  // (inner faces followed by boundary faces, as numbered by matrix_free); the data of batch b are
  // stored at b * (size of the corresponding cell_* / face_* array), see store_cell(), store_face()
  mutable dealii::AlignedVector<vector> stored_cell_solution_gradient;
  mutable dealii::AlignedVector<scalar> stored_cell_viscosity;
  mutable dealii::AlignedVector<scalar> stored_cell_viscosity_derivative;
  mutable dealii::AlignedVector<scalar> stored_cell_production_jacobian;
  mutable dealii::AlignedVector<scalar> stored_cell_dissipation_jacobian;
  mutable dealii::AlignedVector<scalar> stored_cell_square_gradient_jacobian;
  mutable dealii::AlignedVector<vector> stored_cell_square_gradient_gradient_jacobian;
  mutable dealii::AlignedVector<scalar> stored_cell_cross_diffusion_jacobian;
  mutable dealii::AlignedVector<vector> stored_cell_cross_diffusion_gradient_jacobian;


  mutable std::array<dealii::AlignedVector<scalar>, 2> stored_face_solution;
  mutable std::array<dealii::AlignedVector<scalar>, 2> stored_face_normal_gradient;
  mutable std::array<dealii::AlignedVector<scalar>, 2> stored_face_viscosity;
  mutable std::array<dealii::AlignedVector<scalar>, 2> stored_face_viscosity_derivative;
  mutable dealii::AlignedVector<scalar>                stored_tau;
};

} // namespace Operators


template<int dim>
struct TurbulenceSourceOperatorData
{ 
  TurbulenceSourceOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;
};

template<int dim, typename Number>
class TurbulenceSourceOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef TurbulenceSourceOperator<dim, Number> This;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFace;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<1, dim, scalar>  vector;

  // which terms the right-hand side loops evaluate
  enum class Terms
  {
    explicit_terms,       // terms treated Explicit
    newton_explicit_terms // Newton_Explicit part of the terms treated Implicit
  };

public:
  /*
   * Constructor.
   */
  TurbulenceSourceOperator();

  /*
   * Initialization.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const &                         matrix_free,
             TurbulenceSourceOperatorData<dim> const &                       data,
             std::shared_ptr<Operators::TurbulenceSourceKernel<dim, Number>> kernel_in);

  /*
   * dst += right-hand side of the terms treated Explicit (P, D, SG and the turbulent diffusion),
   * evaluated at the previous time step solution_n. "time" is the time of the Dirichlet values
   * of the explicit turbulent diffusion.
   *
   * Call this before evaluate_add_newton_explicit(), because both set the solution of the kernel.
   */
  void
  evaluate_add_explicit(BlockVectorType &       dst,
                        BlockVectorType const & solution_n,
                        double const            time) const;

  /*
   * dst += right-hand side of the Newton_Explicit part of the terms treated Implicit, evaluated at
   * solution (Q^n or the Newton iterate). "time" is the time of the Dirichlet values (t^{n+1}).
   *
   * solution becomes the linearization point of apply_add_newton_implicit().
   */
  void
  evaluate_add_newton_explicit(BlockVectorType &       dst,
                               BlockVectorType const & solution,
                               double const            time) const;

  /*
   * dst += Newton_Implicit part of the terms treated Implicit (left-hand side) applied to the
   * increment src, linearized at the solution of the last evaluate_add_newton_explicit() call.
   * Homogeneous boundary conditions (the increment has zero Dirichlet values).
   */
  void
  apply_add_newton_implicit(BlockVectorType & dst, BlockVectorType const & src) const;

#ifdef EXADG_DEBUG
  /*
   * Diagnostics: term balance of k and the enriched variable in the enriched wall cells along the
   * column of quadrature points closest to each x station; recorded by the next
   * evaluate_add_enriched() and printed on rank 0 (production, dissipation, square gradient,
   * source = P - D + SG; the turbulent and molecular diffusion balance -source in a steady state)
   */
  void
  request_omega_balance(std::vector<double> const & x_stations, unsigned int const time_step) const
  {
    balance_x_stations = x_stations;
    balance_time_step  = time_step;
    balance_requested  = not x_stations.empty();
    balance_records.clear();
  }

  /*
   * Diagnostics: Newton_Implicit part of one group of terms only (the others switched off
   * temporarily), part 0: production, 1: dissipation, 2: square gradient, 3: cross diffusion,
   * 4: turbulent diffusion (with the molecular diffusion of the enriched wall layer)
   */
  void
  apply_add_newton_implicit_part(BlockVectorType &       dst,
                                 BlockVectorType const & src,
                                 unsigned int const      part) const
  {
    Operators::TurbulenceSourceKernelData<dim> const data_saved = kernel->get_data();
    Operators::TurbulenceSourceKernelData<dim>       data_part  = data_saved;
    data_part.production_implicit      = data_saved.production_implicit and part == 0;
    data_part.dissipation_implicit     = data_saved.dissipation_implicit and part == 1;
    data_part.square_gradient_implicit = data_saved.square_gradient_implicit and part == 2;
    data_part.cross_diffusion_implicit = data_saved.cross_diffusion_implicit and part == 3;
    data_part.viscosity_implicit       = data_saved.viscosity_implicit and part == 4;
    kernel->set_data(data_part);
    apply_add_newton_implicit(dst, src);
    kernel->set_data(data_saved);
  }
#endif

  /*
   * Enrichment of RANS variables (nullptr: none): the source terms of the enriched wall cells are
   * evaluated on FunctionEnrichmentRANS with W_h = W_bar + psi W_tilde (evaluate_add_enriched)
   */
  void
  set_function_enrichment_rans(
    std::shared_ptr<FunctionEnrichmentRANS<dim, Number>> function_enrichment_rans_in) const;

  /*
     * dst += diagonal block "block" of the Newton_Implicit cell terms applied to src, on one cell
     * batch only (element-local, no neighbors): d(-P + D - SG - CD)/dQ_block dQ_block and the cell
     * part sigma (nu_t grad dQ_block + eta grad Q_block) of the turbulent diffusion, linearized at the last evaluate_add_newton_explicit() call. dst, src: dof values of variable "block" on cell
     * batch "cell". Same signs as apply_add_newton_implicit(). Used by the matrix-free block Jacobi
     * preconditioner (no matrices are stored).
     */
  void
  apply_add_block_diagonal_elementwise(unsigned int const block,
                                       unsigned int const cell,
                                       scalar * const     dst,
                                         scalar const * const src) const;

  /*
   * true if at least one term is treated Implicit
   */
  bool
  has_implicit_terms() const;

  void
  set_velocity_ptr(VectorType const & velocity_in) const;

  void
  set_time_step_size(double const dt,
                     double const time,
                     double const end_time,
                     double const time_step_number) const;

private:
  /*
   * true if the turbulent diffusion belongs to the terms evaluated by the right-hand side loops
   */
  bool
  diffusion_in_terms() const;

  // right-hand side loops (Explicit or Newton_Explicit, selected by "terms")
  void
  cell_loop_terms(dealii::MatrixFree<dim, Number> const & matrix_free,
                  BlockVectorType &                       dst,
                  BlockVectorType const &                 src,
                  Range const &                           cell_range) const;

  void
  face_loop_terms(dealii::MatrixFree<dim, Number> const & matrix_free,
                  BlockVectorType &                       dst,
                  BlockVectorType const &                 src,
                  Range const &                           face_range) const;

  void
  boundary_face_loop_terms(dealii::MatrixFree<dim, Number> const & matrix_free,
                           BlockVectorType &                       dst,
                           BlockVectorType const &                 src,
                           Range const &                           face_range) const;

  // left-hand side loops (Newton_Implicit)
  void
  cell_loop_newton_implicit(dealii::MatrixFree<dim, Number> const & matrix_free,
                            BlockVectorType &                       dst,
                            BlockVectorType const &                 src,
                            Range const &                           cell_range) const;

  void
  face_loop_newton_implicit(dealii::MatrixFree<dim, Number> const & matrix_free,
                            BlockVectorType &                       dst,
                            BlockVectorType const &                 src,
                            Range const &                           face_range) const;

  void
  boundary_face_loop_newton_implicit(dealii::MatrixFree<dim, Number> const & matrix_free,
                                     BlockVectorType &                       dst,
                                     BlockVectorType const &                 src,
                                     Range const &                           face_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  TurbulenceSourceOperatorData<dim> data;

  std::shared_ptr<Operators::TurbulenceSourceKernel<dim, Number>> kernel;

  // enrichment of RANS variables (nullptr: none)
  mutable std::shared_ptr<FunctionEnrichmentRANS<dim, Number>> function_enrichment_rans;

  // linearization point of the enriched wall cells and their faces (Newton_Explicit terms of
  // evaluate_add_enriched()), batches of FunctionEnrichmentRANS, points of the enrichment quadrature
  mutable typename Operators::TurbulenceSourceKernel<dim, Number>::LinearizationStorage
                linearization_enriched;
  mutable bool linearization_enriched_stored = false;

#ifdef EXADG_DEBUG
  // term balance in the enriched wall cells (request_omega_balance()): request, stations, records
  // (station, x, y, k, W, P_k, D_k, P_W, D_W, SG_W, nu_D) of this process
  mutable bool                balance_requested = false;
  mutable std::vector<double> balance_x_stations;
  mutable unsigned int        balance_time_step = 0;
  mutable std::vector<double> balance_records;

  // record the points of cell batch cell (cell_loop_terms_enriched(), after evaluate_point())
  template<typename Integrator>
  void
  record_omega_balance(Integrator const & integrator_points,
                       unsigned int const cell,
                       unsigned int const n_q_points,
                       unsigned int const n_active_lanes,
                       unsigned int const block_omega) const;

  // print the records of all processes (rank 0), then clear the request
  void
  print_omega_balance(MPI_Comm const comm) const;

  static unsigned int const n_balance_entries = 13;
#endif

  // Newton_Implicit part of the enriched wall cells and their faces (apply_add_newton_implicit)
  void
  apply_add_newton_implicit_enriched(BlockVectorType & dst, BlockVectorType const & src) const;

  void
  cell_loop_newton_implicit_enriched(dealii::MatrixFree<dim, Number> const & matrix_free,
                                     std::vector<VectorType> &               dst,
                                     BlockVectorType const &                 src,
                                     Range const &                           cell_range) const;

  void
  face_loop_newton_implicit_enriched(dealii::MatrixFree<dim, Number> const & matrix_free,
                                     std::vector<VectorType> &               dst,
                                     BlockVectorType const &                 src,
                                     Range const &                           face_range) const;

  void
  boundary_face_loop_newton_implicit_enriched(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    std::vector<VectorType> &               dst,
    BlockVectorType const &                 src,
    Range const &                           face_range) const;

  void
  integrate_face_newton_implicit_enriched(dealii::MatrixFree<dim, Number> const & matrix_free,
                                          std::vector<VectorType> &               dst,
                                          BlockVectorType const &                 src,
                                          Range const &                           face_range,
                                          bool const                              boundary) const;

  // interior penalty parameter per cell batch of FunctionEnrichmentRANS (kernel's values)
  mutable dealii::AlignedVector<scalar> penalty_parameter_enriched;

  void
  update_penalty_parameter_enriched() const;

  // face loops of evaluate_add_enriched() (turbulent diffusion on the faces of enriched cells)
  void
  face_loop_terms_enriched(dealii::MatrixFree<dim, Number> const & matrix_free,
                           std::vector<VectorType> &               dst,
                           BlockVectorType const &                 src,
                           Range const &                           face_range) const;

  void
  boundary_face_loop_terms_enriched(dealii::MatrixFree<dim, Number> const & matrix_free,
                                    std::vector<VectorType> &               dst,
                                    BlockVectorType const &                 src,
                                    Range const &                           face_range) const;

  void
  integrate_face_terms_enriched(dealii::MatrixFree<dim, Number> const & matrix_free,
                                std::vector<VectorType> &               dst,
                                BlockVectorType const &                 src,
                                Range const &                           face_range,
                                bool const                              boundary) const;

  // lane mask of a face batch of the kernel's MatrixFree: 0 on faces of enriched cells
  scalar
  face_mask_not_enriched(dealii::MatrixFree<dim, Number> const & matrix_free,
                         unsigned int const                      face,
                         bool const                              boundary) const;

  // source terms of the enriched wall cells (Explicit / Newton_Explicit part as set by "terms")
  void
  evaluate_add_enriched(BlockVectorType & dst) const;

  // cell loop of evaluate_add_enriched(), MatrixFree of FunctionEnrichmentRANS, enriched cells;
  // dst[b]: v_bar rows of RANS variable b (shadow layout), dst[n_var + i]: psi v_tilde rows of
  // enriched variable i; src: bar parts (shadow), tilde parts, wall-law fields (shadow)
  void
  cell_loop_terms_enriched(dealii::MatrixFree<dim, Number> const & matrix_free,
                           std::vector<VectorType> &               dst,
                           BlockVectorType const &                 src,
                           Range const &                           cell_range) const;

  // terms evaluated by the right-hand side loops, set by evaluate_add_explicit() and
  // evaluate_add_newton_explicit()
  mutable Terms terms = Terms::explicit_terms;

  mutable bool linearization_stored = false;
};

} // namespace RANS
namespace ConvDiff
{
using RANS::TurbulenceSourceOperator;
using RANS::TurbulenceSourceOperatorData;
namespace Operators
{
using RANS::Operators::TurbulenceSourceKernel;
using RANS::Operators::TurbulenceSourceKernelData;
} // namespace Operators
} // namespace ConvDiff
} // namespace ExaDG

#endif
