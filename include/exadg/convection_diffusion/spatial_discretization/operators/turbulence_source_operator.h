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

#ifndef INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_TURBULENCE_SOURCE_OPERATOR
#define INCLUDE_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_TURBULENCE_SOURCE_OPERATOR

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/array_view.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/integrator_flags.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/utilities/lazy_ptr.h>
#include <exadg/grid/grid_data.h>
#include <exadg/operators/interior_penalty_parameter.h>

#include <exadg/convection_diffusion/spatial_discretization/turbulence_model.h>
#include <exadg/convection_diffusion/user_interface/enum_types.h>
#include <exadg/convection_diffusion/spatial_discretization/operators/weak_boundary_conditions.h>
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>

namespace ExaDG
{
namespace ConvDiff
{
namespace Operators
{
template<int dim>
struct TurbulenceSourceKernelData
{
  TurbulenceSourceKernelData()
    : dof_index(0),
    dof_index_velocity(0),
    diffusivity(0.0),
    IP_factor(1.0),
    time_step_size(0.0),
    time_step_number(0.0),
    current_time(0.0),
    end_time(0.0),
    production_implicit(false),
    dissipation_implicit(false),
    square_gradient_implicit(false),
    viscosity_implicit(false),
    k_log(false),
    omega_log(false)
  {
  }

  unsigned int dof_index;
  unsigned int dof_index_velocity;

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

    n_q_points_cell = integrator_velocity->n_q_points;

    cell_solution.resize(n_q_points_cell * n_variables);
    cell_solution_gradient.resize(n_q_points_cell * n_variables);
    cell_viscosity.resize(n_q_points_cell);
    cell_viscosity_derivative.resize(n_q_points_cell * n_variables);
    cell_production.resize(n_q_points_cell * n_variables);
    cell_dissipation.resize(n_q_points_cell * n_variables);
    cell_square_gradient.resize(n_q_points_cell * n_variables);
    cell_production_jacobian.resize(n_q_points_cell * n_variables * n_variables);
    cell_dissipation_jacobian.resize(n_q_points_cell * n_variables * n_variables);
    cell_square_gradient_jacobian.resize(n_q_points_cell * n_variables * n_variables);
    cell_square_gradient_gradient_jacobian.resize(n_q_points_cell * n_variables);

    solution_values.resize(n_variables);
    solution_derivative.resize(n_variables);
    viscosity_derivative.resize(n_variables);

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

    n_q_points_face = face_integrator_velocity[0]->n_q_points;

    for(unsigned int side = 0; side < 2; ++side)
    {
      face_solution[side].resize(n_q_points_face * n_variables);
      face_normal_gradient[side].resize(n_q_points_face * n_variables);
      face_viscosity[side].resize(n_q_points_face);
      face_viscosity_derivative[side].resize(n_q_points_face * n_variables);
    }

    face_dirichlet.resize(n_variables);

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

  unsigned int
  get_n_variables() const
  {
    return n_variables;
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
    AssertThrow(solution_in.n_blocks() == n_variables,
                dealii::ExcMessage("Number of solution blocks does not match the turbulence model."));

    solution.own() = solution_in;
    solution->update_ghost_values();
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

    for(unsigned int q = 0; q < n_q_points_cell; ++q)
    {
      for(unsigned int i = 0; i < n_variables; ++i)
      {
        cell_solution[index(q, i)]          = integrator_solution[i]->get_value(q);
        cell_solution_gradient[index(q, i)] = integrator_solution[i]->get_gradient(q);
      }

      tensor const velocity_gradient = integrator_velocity->get_gradient(q);

      // pointwise eddy viscosity nu_t(Q) and d nu_t / d Q_j, used by all terms (explicit and
      // implicit treatment)
      turbulence_model_ptr->get_physical_variables(
        dealii::ArrayView<scalar const>(cell_solution.data() + index(q, 0), n_variables),
        dealii::make_array_view(solution_values),
        dealii::make_array_view(solution_derivative));

      turbulence_model_ptr->evaluate_eddy_viscosity(dealii::make_array_view(solution_values),
                                                    velocity_gradient,
                                                    cell_viscosity[q],
                                                    dealii::make_array_view(viscosity_derivative));

      // chain rule: d nu_t / d Q_j = d nu_t / d solution_j * d solution_j / d Q_j
      for(unsigned int j = 0; j < n_variables; ++j)
        cell_viscosity_derivative[index(q, j)] = viscosity_derivative[j] * solution_derivative[j];

      compute_production(q, velocity_gradient);
      compute_dissipation(q, velocity_gradient);
      compute_square_gradient(q);
    }
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
        {
          face_solution[side][index(q, i)] = face_integrator_solution[side][i]->get_value(q);
          face_normal_gradient[side][index(q, i)] =
            face_integrator_solution[side][i]->get_normal_derivative(q);
        }

        compute_face_viscosity(side, q);
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

    for(unsigned int i = 0; i < n_variables; ++i)
    {
      BoundaryType const boundary_type = data.bc[i]->get_boundary_type(boundary_id);

      AssertThrow(boundary_type == BoundaryType::Dirichlet or
                    boundary_type == BoundaryType::Neumann,
                  dealii::ExcMessage(
                    "Turbulent diffusion: only Dirichlet and (zero flux) Neumann boundaries are "
                    "implemented."));

      face_dirichlet[i] = (boundary_type == BoundaryType::Dirichlet);
    }

    for(unsigned int q = 0; q < n_q_points_face; ++q)
    {
      for(unsigned int i = 0; i < n_variables; ++i)
      {
        scalar const value_m           = face_integrator_solution[0][i]->get_value(q);
        scalar const normal_gradient_m = face_integrator_solution[0][i]->get_normal_derivative(q);

        face_solution[0][index(q, i)]        = value_m;
        face_normal_gradient[0][index(q, i)] = normal_gradient_m;

        if(face_dirichlet[i])
        {
          auto const g = FunctionEvaluator<0, dim, Number>::value(
            *(data.bc[i]->dirichlet_bc.find(boundary_id)->second),
            face_integrator_solution[0][i]->quadrature_point(q),
            evaluation_time);

          face_solution[1][index(q, i)]        = -value_m + g * 2.0;
          face_normal_gradient[1][index(q, i)] = normal_gradient_m;
        }
        else
        {
          face_solution[1][index(q, i)]        = value_m;
          face_normal_gradient[1][index(q, i)] = -normal_gradient_m;
        }
      }

      compute_face_viscosity(0, q);

      face_viscosity[1][q] = face_viscosity[0][q];
      for(unsigned int j = 0; j < n_variables; ++j)
        face_viscosity_derivative[1][index(q, j)] = face_viscosity_derivative[0][index(q, j)];
    }

    tau = face_integrator_solution[0][0]->read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<dim, Number>(degree, element_type, data.IP_factor);
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

    return value;
  }

  /*
   * Newton_Implicit part of the source terms P, D, SG treated Implicit, at quadrature point q, row
   * "block" (equation of transport variable "block"), all columns j:
   *
   *   values[j]    : dQ_j at q
   *   gradients[j] : grad dQ_j at q
   *
   * Left-hand side signs: -P', +D', -SG'.
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
      if(face_dirichlet[block])
      {
        increment_p                 = -values_m[block];
        increment_normal_gradient_p = normal_gradients_m[block];
      }
      else
      {
        increment_p                 = values_m[block];
        increment_normal_gradient_p = -normal_gradients_m[block];
      }
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

  /*
   * Pointwise nu_t and d nu_t / d Q_j at face quadrature point q of side "side", from the
   * transported values in face_solution[side] and the velocity gradient of that side.
   */
  void
  compute_face_viscosity(unsigned int const side, unsigned int const q) const
  {
    tensor const velocity_gradient = face_integrator_velocity[side]->get_gradient(q);

    turbulence_model_ptr->get_physical_variables(
      dealii::ArrayView<scalar const>(face_solution[side].data() + index(q, 0), n_variables),
      dealii::make_array_view(solution_values),
      dealii::make_array_view(solution_derivative));

    turbulence_model_ptr->evaluate_eddy_viscosity(dealii::make_array_view(solution_values),
                                                  velocity_gradient,
                                                  face_viscosity[side][q],
                                                  dealii::make_array_view(viscosity_derivative));

    // chain rule: d nu_t / d Q_j = d nu_t / d solution_j * d solution_j / d Q_j
    for(unsigned int j = 0; j < n_variables; ++j)
      face_viscosity_derivative[side][index(q, j)] =
        viscosity_derivative[j] * solution_derivative[j];
  }

  /*
     * k-omega quantities at one quadrature point (Bassi et al. 2014, eqs. (5)-(8)), used by the
     * k-omega production and dissipation
     */
  struct KOmegaValues
  {
    // physical values k and omega (exp of the transported value if ln is transported)
    scalar tke;
    scalar omega;

    // realizability limiter: max(k, 0), max(omega, omega_lower_bound)
    scalar tke_limited;
    scalar omega_limited;

    scalar inverse_tke;
    scalar inverse_omega_limited;

    // derivative of the limiters: 1 where the limiter is inactive, 0 where it is active
    scalar tke_not_limited;
    scalar omega_not_limited;

    // (grad u + grad u^T) : grad u
    scalar velocity_gradient_product;
  };

  KOmegaValues
  k_omega_1988_values(unsigned int const q, tensor const & velocity_gradient) const
  {
    scalar const zero = dealii::make_vectorized_array<Number>(0.0);
    scalar const one  = dealii::make_vectorized_array<Number>(1.0);

    KOmegaValues values;

    values.tke = data.k_log ? std::exp(cell_solution[index(q, 0)]) : cell_solution[index(q, 0)];
    values.omega =
      data.omega_log ? std::exp(cell_solution[index(q, 1)]) : cell_solution[index(q, 1)];

    values.velocity_gradient_product =
      scalar_product(velocity_gradient + transpose(velocity_gradient), velocity_gradient);

    scalar const omega_lower_bound =
      turbulence_model_ptr->compute_realizability_alpha(velocity_gradient);

    // safe values: max(k, 0), max(omega, omega_lower_bound)
    values.tke_limited = std::max(values.tke, zero);

    values.tke_not_limited =
      dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(values.tke,
                                                                           zero,
                                                                           one,
                                                                           zero);

    // safe inverse of k (as safe_tke in the previous implementation)
    values.inverse_tke = one / std::max(values.tke, dealii::make_vectorized_array<Number>(1.0e-6));

    if(data.omega_log)
    {
      // limiter in log space: omega_r = exp(max(ln omega, ln omega_lower_bound)), no division
      scalar const log_omega_lower_bound = std::log(omega_lower_bound);
      scalar const log_omega_limited =
        std::max(cell_solution[index(q, 1)], log_omega_lower_bound);

      values.omega_limited         = std::exp(log_omega_limited);
      values.inverse_omega_limited = std::exp(-log_omega_limited);
      values.omega_not_limited =
        dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(
          cell_solution[index(q, 1)], log_omega_lower_bound, one, zero);
    }
    else
  {
      values.omega_limited         = std::max(values.omega, omega_lower_bound);
      values.inverse_omega_limited = one / values.omega_limited;
      values.omega_not_limited =
        dealii::compare_and_apply_mask<dealii::SIMDComparison::greater_than>(values.omega,
                                                                             omega_lower_bound,
                                                                             one,
                                                                             zero);
    }

    return values;
  }

  void
  compute_production(unsigned int const q,
                     tensor const &     velocity_gradient) const
  {
    switch(data.turbulence_model_data.turbulence_model)
    {
      case TurbulenceEddyViscosityModel::StandardKOmega1988:
        k_omega_1988_production(q, velocity_gradient);
        break;
      default:
        AssertThrow(false,
                    dealii::ExcMessage("Production not implemented for this turbulence model."));
        break;
    }
  }

  void
  k_omega_1988_production(unsigned int const q,
                          tensor const &     velocity_gradient) const
  {
    scalar const zero  = dealii::make_vectorized_array<Number>(0.0);
    scalar const alpha =
      dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[0]);

    KOmegaValues const values = k_omega_1988_values(q, velocity_gradient);
    scalar const       G      = values.velocity_gradient_product;

    // 1 / omega_r and its derivative w.r.t. the transported omega:
    // d omega_r / d Q_omega = omega_not_limited (omega) or omega_not_limited * omega_r (ln omega)
    scalar const inverse_omega = values.inverse_omega_limited;
    scalar const omega_limited_derivative =
      data.omega_log ? values.omega_not_limited * values.omega_limited : values.omega_not_limited;
    scalar const inverse_omega_derivative =
      -omega_limited_derivative * inverse_omega * inverse_omega;

    // row of k
    if(data.k_log)
    {
      // equation of ln k: nu_t G / k = G / omega_r (k cancels, no 1/k)
      cell_production[index(q, 0)]             = G * inverse_omega;
      cell_production_jacobian[index(q, 0, 0)] = zero;
      cell_production_jacobian[index(q, 0, 1)] = G * inverse_omega_derivative;
    }
    else {
      // equation of k: nu_t G = G max(k, 0) / omega_r
      cell_production[index(q, 0)]             = G * values.tke_limited * inverse_omega;
      cell_production_jacobian[index(q, 0, 0)] = G * values.tke_not_limited * inverse_omega;
      cell_production_jacobian[index(q, 0, 1)] = G * values.tke_limited * inverse_omega_derivative;
    }

    // row of omega (does not depend on k)
    cell_production_jacobian[index(q, 1, 0)] = zero;

    if(data.omega_log)
    {
      // equation of ln omega: alpha G / omega_r
      cell_production[index(q, 1)]             = alpha * G * inverse_omega;
      cell_production_jacobian[index(q, 1, 1)] = alpha * G * inverse_omega_derivative;
    }
    else {
      // equation of omega: alpha G omega / omega_r
      cell_production[index(q, 1)] = alpha * G * values.omega * inverse_omega;
      cell_production_jacobian[index(q, 1, 1)] =
        alpha * G * (inverse_omega + values.omega * inverse_omega_derivative);
    }

    // explicit treatment: no Jacobian
    if(not data.production_implicit)
    {
      cell_production_jacobian[index(q, 0, 0)] = zero;
      cell_production_jacobian[index(q, 0, 1)] = zero;
      cell_production_jacobian[index(q, 1, 1)] = zero;
    }
  }

  /*
     * Dissipation of all transport variables at quadrature point q: value in 
  cell_dissipation and
     * Jacobian w.r.t. the transported values in cell_dissipation_jacobian.
     */
  void
  compute_dissipation(unsigned int const q, tensor const & velocity_gradient)
  const
  {
    switch(data.turbulence_model_data.turbulence_model)
    {
      case TurbulenceEddyViscosityModel::StandardKOmega1988:
        k_omega_1988_dissipation(q, velocity_gradient);
        break;
      default:
        AssertThrow(false,
                    dealii::ExcMessage("Dissipation not implemented for this turbulence model."));
        break;
    }
  }

  void
  k_omega_1988_dissipation(unsigned int const q, tensor const & 
                           velocity_gradient) const
  {
    scalar const zero = dealii::make_vectorized_array<Number>(0.0);
    scalar const beta =
      dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[1]);
    scalar const beta_star =
      dealii::make_vectorized_array<Number>(turbulence_model_ptr->model_coefficients[2]);

    KOmegaValues const values = k_omega_1988_values(q, velocity_gradient);

    scalar const omega_limited_derivative =
      data.omega_log ? values.omega_not_limited * values.omega_limited :
      values.omega_not_limited;

    // row of k
    if(data.k_log)
    {
      // equation of ln k: beta* omega_r
      cell_dissipation[index(q, 0)]             = beta_star *
        values.omega_limited;
      cell_dissipation_jacobian[index(q, 0, 0)] = zero;
      cell_dissipation_jacobian[index(q, 0, 1)] = beta_star *
        omega_limited_derivative;
    }
    else {
      // equation of k: beta* max(k, 0) omega_r
      cell_dissipation[index(q, 0)] = beta_star * values.tke_limited *
        values.omega_limited;
      cell_dissipation_jacobian[index(q, 0, 0)] =
        beta_star * values.tke_not_limited * values.omega_limited;
      cell_dissipation_jacobian[index(q, 0, 1)] =
        beta_star * values.tke_limited * omega_limited_derivative;
    }
    // row of omega (does not depend on k)
    cell_dissipation_jacobian[index(q, 1, 0)] = zero;

    if(data.omega_log)
    {
      // equation of ln omega: beta omega_r
      cell_dissipation[index(q, 1)]             = beta * values.omega_limited;
      cell_dissipation_jacobian[index(q, 1, 1)] = beta *
        omega_limited_derivative;
    }
    else
  {
      // equation of omega: beta omega omega_r
      cell_dissipation[index(q, 1)] = beta * values.omega * values.omega_limited;
      cell_dissipation_jacobian[index(q, 1, 1)] =
        beta * (values.omega_limited + values.omega * omega_limited_derivative);
    }

    // explicit treatment: no Jacobian
    if(not data.dissipation_implicit)
    {
      cell_dissipation_jacobian[index(q, 0, 0)] = zero;
      cell_dissipation_jacobian[index(q, 0, 1)] = zero;
      cell_dissipation_jacobian[index(q, 1, 1)] = zero;
    }
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
  unsigned int n_q_points_cell = 0;
  unsigned int n_q_points_face = 0;

  mutable lazy_ptr<VectorType> velocity;
  mutable lazy_ptr<BlockVectorType> solution;

  std::shared_ptr<IntegratorCellVelocity>      integrator_velocity;
  std::vector<std::shared_ptr<IntegratorCell>> integrator_solution;

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
  mutable dealii::AlignedVector<scalar> cell_viscosity_derivative;
  mutable dealii::AlignedVector<scalar> cell_production;
  mutable dealii::AlignedVector<scalar> cell_dissipation;
  mutable dealii::AlignedVector<scalar> cell_square_gradient;
  mutable dealii::AlignedVector<scalar> cell_production_jacobian;
  mutable dealii::AlignedVector<scalar> cell_dissipation_jacobian;
  mutable dealii::AlignedVector<scalar> cell_square_gradient_jacobian;
  mutable dealii::AlignedVector<vector> cell_square_gradient_gradient_jacobian;

  mutable dealii::AlignedVector<scalar> solution_values;
  mutable dealii::AlignedVector<scalar> solution_derivative;
  mutable dealii::AlignedVector<scalar> viscosity_derivative;

  // face data at the solution set by set_solution_ptr(), at the quadrature points of the current
  // face, [side] (side 0: e-, side 1: e+ or the exterior boundary values)
  mutable std::array<dealii::AlignedVector<scalar>, 2> face_solution;
  mutable std::array<dealii::AlignedVector<scalar>, 2> face_normal_gradient;
  mutable std::array<dealii::AlignedVector<scalar>, 2> face_viscosity;
  mutable std::array<dealii::AlignedVector<scalar>, 2> face_viscosity_derivative;

  // boundary face: true if the boundary condition of variable i is Dirichlet, false if Neumann
  mutable std::vector<bool> face_dirichlet;
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

  // terms evaluated by the right-hand side loops, set by evaluate_add_explicit() and
  // evaluate_add_newton_explicit()
  mutable Terms terms = Terms::explicit_terms;
};

} // namespace ConvDiff

} // namespace ExaDG

#endif
