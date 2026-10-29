/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#include <exadg/convection_diffusion/spatial_discretization/turbulence_model.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
TurbulenceModel<dim, Number>::TurbulenceModel()
  : dealii::Subscriptor(),
  quad_index(0),
  n_variables(0),
  compute_derivative(false),
  dof_index(0),
  matrix_free(nullptr)
{
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                         TurbulenceModelData const & turbulence_model_data_in,
                                         unsigned int const          dof_index_in,
                                         unsigned int const          dof_index_viscosity_in,
                                         unsigned int const          dof_index_velocity_in,
                                         unsigned int const          quad_index_in,
                                         unsigned int const          quad_index_eddy_velocity_in,
                                         bool const                  compute_derivative_in)
{
  matrix_free = &matrix_free_in;

  dof_index                = dof_index_in;
  dof_index_viscosity      = dof_index_viscosity_in;
  dof_index_velocity       = dof_index_velocity_in;
  quad_index               = quad_index_in;
  quad_index_eddy_velocity = quad_index_eddy_velocity_in;

  turbulence_model_data = turbulence_model_data_in;

  model_coefficients = turbulence_model_data_in.turbulence_data_base->get_all_coefficients();

  evaluator.initialize(turbulence_model_data);
  n_variables = evaluator.get_n_variables();

  matrix_free->initialize_dof_vector(eddy_viscosity, dof_index_viscosity_in);

  // d nu_t / d (transported value), one block per transport variable
  compute_derivative = compute_derivative_in;
  if(compute_derivative)
  {
    eddy_viscosity_derivative.reinit(n_variables);
    for(unsigned int i = 0; i < n_variables; ++i)
      matrix_free->initialize_dof_vector(eddy_viscosity_derivative.block(i), dof_index_viscosity_in);
    eddy_viscosity_derivative.collect_sizes();
  }

  matrix_free->initialize_dof_vector(velocity.own(), dof_index_velocity_in);
  velocity.own() = 0.0;

  InverseMassOperatorData inv_mass_data;
  inv_mass_data.dof_index  = dof_index_viscosity_in;
  inv_mass_data.quad_index = quad_index_in;
  inverse_mass_viscosity.initialize(*matrix_free, inv_mass_data);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::set_viscosity(BlockVectorType const & solution)
{
  AssertThrow(solution.n_blocks() == n_variables,
              dealii::ExcMessage("Number of solution blocks does not match the turbulence model."));

  VectorType rhs_viscosity;
  matrix_free->initialize_dof_vector(rhs_viscosity, dof_index_viscosity);
  rhs_viscosity = 0.0;

  if(compute_derivative)
    eddy_viscosity_derivative.zero_out_ghost_values();

  matrix_free->cell_loop(&This::cell_loop,
                         this,
                         rhs_viscosity,
                         solution);

  eddy_viscosity = rhs_viscosity;
  // inverse_mass_viscosity.apply(eddy_viscosity, rhs_viscosity);

  for(unsigned int i = 0; i < eddy_viscosity.locally_owned_size(); ++i)
  {
    if(eddy_viscosity.local_element(i) < Number(0.0))
    {
      eddy_viscosity.local_element(i) = Number(0.0);

      // a clipped eddy viscosity does not depend on the transport variables
      if(compute_derivative)
        for(unsigned int j = 0; j < n_variables; ++j)
          eddy_viscosity_derivative.block(j).local_element(i) = Number(0.0);
    }
  }
  eddy_viscosity.update_ghost_values();

  if(compute_derivative)
    eddy_viscosity_derivative.update_ghost_values();
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::set_velocity_ptr(VectorType const & velocity_in)
{
  velocity.own() = velocity_in;
  velocity->update_ghost_values();
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType & dst,
  BlockVectorType const & src,
  Range const &      cell_range)
{
  std::vector<std::shared_ptr<IntegratorCell>> integrator_solution(n_variables);
  for(auto & integrator : integrator_solution)
  {
    integrator = std::make_shared<IntegratorCell>(matrix_free, dof_index, quad_index);
  }

  IntegratorCell integrator_viscosity(matrix_free, dof_index_viscosity, quad_index);

  std::vector<std::shared_ptr<IntegratorCell>> integrator_derivative;
  if(compute_derivative)
  {
    integrator_derivative.resize(n_variables);
    for(auto & integrator : integrator_derivative)
    {
      integrator = std::make_shared<IntegratorCell>(matrix_free, dof_index_viscosity, quad_index);
    }
  }

  CellIntegrator<dim, dim, Number> integrator_velocity(matrix_free,
                                                       dof_index_velocity,
                                                       quad_index_eddy_velocity);
  dealii::AlignedVector<scalar> transported_values(n_variables);
  dealii::AlignedVector<scalar> solution(n_variables);
  dealii::AlignedVector<scalar> solution_derivative(compute_derivative ? n_variables : 0);
  dealii::AlignedVector<scalar> viscosity_derivative(compute_derivative ? n_variables : 0);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator_solution[i]->reinit(cell);
      integrator_solution[i]->read_dof_values(src.block(i));
    }

    integrator_viscosity.reinit(cell);

    for(auto & integrator : integrator_derivative)
    {
      integrator->reinit(cell);
    }

    integrator_velocity.reinit(cell);
    integrator_velocity.gather_evaluate(*velocity, dealii::EvaluationFlags::gradients);

    for(unsigned int dof = 0; dof < integrator_viscosity.dofs_per_cell; ++dof)
    {
      for(unsigned int i = 0; i < n_variables; ++i)
        transported_values[i] = integrator_solution[i]->get_dof_value(dof);

      get_physical_variables(dealii::make_array_view(transported_values),
                             dealii::make_array_view(solution),
                             dealii::make_array_view(solution_derivative));

      auto velocity_gradient = integrator_velocity.get_gradient(dof);

      scalar viscosity;

      evaluate_eddy_viscosity(dealii::make_array_view(solution),
                              velocity_gradient,
                              viscosity,
                              dealii::make_array_view(viscosity_derivative));

#ifdef EXADG_DEBUG
      dealii::Point<dim, scalar> const q_point = integrator_velocity.quadrature_point(dof);
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      {
        Number const val = viscosity[v];
        if(!(val >= Number(0.0)) || !dealii::numbers::is_finite(val))
        {
          dealii::Point<dim> p;
          for(unsigned int d = 0; d < dim; ++d)
            p[d] = q_point[d][v];

          std::cout << "[rank " << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << "] "
            << "bad eddy_viscosity = " << val << " at cell " << cell << ", dof " << dof
            << ", lane " << v << ", coords = " << p << ", transported values = (";
          for(unsigned int i = 0; i < n_variables; ++i)
            std::cout << (i > 0 ? ", " : "") << transported_values[i][v];
          std::cout << ")" << std::endl;
        }
      }
#endif

      integrator_viscosity.submit_dof_value(viscosity, dof);

      for(unsigned int i = 0; i < integrator_derivative.size(); ++i)
        integrator_derivative[i]->submit_dof_value(viscosity_derivative[i] *
                                                   solution_derivative[i],
                                                   dof);
    }
    integrator_viscosity.set_dof_values(dst);
    for(unsigned int i = 0; i < integrator_derivative.size(); ++i)
      integrator_derivative[i]->set_dof_values(eddy_viscosity_derivative.block(i));
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::evaluate_eddy_viscosity(
  dealii::ArrayView<scalar const> const & solution,
  dealii::Tensor<2, dim, scalar> const &  velocity_gradient,
  scalar &                                viscosity,
  dealii::ArrayView<scalar> const &       viscosity_derivative) const
{
  evaluator.evaluate_eddy_viscosity(solution, velocity_gradient, viscosity, 
                                    viscosity_derivative);
}

template<int dim, typename Number>
EddyViscosityEvaluator<dim, Number> const &
TurbulenceModel<dim, Number>::get_eddy_viscosity_evaluator() const
{
  return evaluator;
}

template<int dim, typename Number>
typename TurbulenceModel<dim, Number>::scalar
TurbulenceModel<dim, Number>::compute_realizability_alpha(
  dealii::Tensor<2, dim, scalar> const & grad_u) const
{
  return evaluator.compute_realizability_alpha(grad_u);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::get_eddy_viscosity(VectorType & dst) const
{
  dst.equ(1.0, eddy_viscosity);
}

template<int dim, typename Number>
typename TurbulenceModel<dim, Number>::VectorType const &
TurbulenceModel<dim, Number>::get_eddy_viscosity_ref() const
{
  return eddy_viscosity;
}

template<int dim, typename Number>
typename TurbulenceModel<dim, Number>::BlockVectorType const &
TurbulenceModel<dim, Number>::get_eddy_viscosity_derivative_ref() const
{
  AssertThrow(compute_derivative,
              dealii::ExcMessage("Eddy viscosity derivative is not computed, initialize the turbulence model with compute_derivative = true."));

  return eddy_viscosity_derivative;
}

template<int dim, typename Number>
unsigned int
TurbulenceModel<dim, Number>::get_n_variables() const
{
  return n_variables;
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::get_physical_variables(
  dealii::ArrayView<scalar const> const & transported_values,
  dealii::ArrayView<scalar> const &       solution,
  dealii::ArrayView<scalar> const &       solution_derivative) const
{
    evaluator.get_physical_variables(transported_values, solution, solution_derivative);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::get_turbulent_kinetic_energy(VectorType & dst,
                                                           BlockVectorType const & solution) const
{
  matrix_free->cell_loop(&This::cell_loop_extract_tke,
                         this,
                         dst,
                         solution);
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::cell_loop_extract_tke(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType & dst,
  BlockVectorType const & src,
  Range const &      cell_range) const
{
  IntegratorCell integrator(matrix_free, dof_index, quad_index);
  IntegratorCell integrator_tke(matrix_free, dof_index_viscosity, quad_index);

  // block 0 is the turbulent kinetic energy k (or ln k) in all two-equation models
  if(turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKEpsilon or
     turbulence_model_data.turbulence_model == TurbulenceEddyViscosityModel::StandardKOmega1988)
  {
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src.block(0));

      integrator_tke.reinit(cell);
      for(unsigned int dof = 0; dof < integrator_tke.dofs_per_cell; ++dof)
      {
        scalar tke = integrator.get_dof_value(dof);

        if(turbulence_model_data.positivity_preserving_limiter[0] == PositivityPreservingLimiter::LogarithmicTransportVariable)
        {
          tke = std::exp(tke);
        }

        integrator_tke.submit_dof_value(tke, dof);
      }
      integrator_tke.set_dof_values(dst);
    }
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("Turbulent kinetic energy: not implemented for this model."));
  }
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::enforce_positivity(BlockVectorType & solution) const
{
  BlockVectorType tmp;
  tmp.reinit(solution);

  matrix_free->cell_loop(&This::cell_loop_enforce_positivity,
                         this,
                         tmp,
                         solution);
  solution = tmp;
}

template<int dim, typename Number>
void
TurbulenceModel<dim, Number>::cell_loop_enforce_positivity(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  BlockVectorType & dst,
  BlockVectorType const & src,
  Range const & cell_range) const
{
  IntegratorCell integrator(matrix_free, dof_index, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    for(unsigned int i = 0; i < n_variables; ++i)
    {
      integrator.read_dof_values(src.block(i));

      if(turbulence_model_data.positivity_preserving_limiter[i] ==
        PositivityPreservingLimiter::Clipper)
      {
        for(unsigned int dof = 0; dof < integrator.dofs_per_cell; ++dof)
          integrator.submit_dof_value(std::max(integrator.get_dof_value(dof),
                                               dealii::make_vectorized_array<Number>(0.0)),
                                      dof);
      }

      integrator.set_dof_values(dst.block(i));
    }
  }
}

template class TurbulenceModel<2, float>;
template class TurbulenceModel<2, double>;
template class TurbulenceModel<3, float>;
template class TurbulenceModel<3, double>;
} // namespace RANS
} // namespace ExaDG
