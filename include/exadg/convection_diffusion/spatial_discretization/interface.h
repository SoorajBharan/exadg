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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_INTERFACE_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_INTERFACE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>

// ExaDG
#include <exadg/time_integration/interpolate.h>

namespace ExaDG
{
namespace ConvDiff
{
namespace Interface
{
template<typename Number>
class Operator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  Operator()
  {
  }

  virtual ~Operator()
  {
  }

  // explicit time integration: evaluate operator
  virtual void
  evaluate_explicit_time_int(BlockVectorType &       dst,
                             BlockVectorType const & src,
                             double const       evaluation_time,
                             VectorType const * velocity = nullptr) const = 0;

  // implicit time integration: calculate right-hand side of linear system of equations
  virtual void
  rhs(BlockVectorType &       dst,
      BlockVectorType const & src,
      double const       evaluation_time = 0.0,
      VectorType const * velocity        = nullptr) const = 0;

  // implicit time integration: solve linear system of equations
  virtual unsigned int
  solve(BlockVectorType &       sol,
        BlockVectorType const & rhs,
        bool const         update_preconditioner,
        double const       scaling_factor  = -1.0,
        double const       evaluation_time = -1.0,
        VectorType const * velocity        = nullptr) = 0;

  // time integration: initialize dof vector
  virtual void
  initialize_dof_vector(BlockVectorType & src) const = 0;

  virtual void
  initialize_dof_vector_velocity(VectorType & src) const = 0;

  virtual void
  project_velocity(VectorType & velocity, double const time) const = 0;

  // time integration: prescribe initial conditions
  virtual void
  prescribe_initial_conditions(BlockVectorType & src, double const evaluation_time) const = 0;

  // time step calculation: max efficiency
  virtual double
  calculate_time_step_max_efficiency(unsigned int const order_time_integrator) const = 0;

  // time step calculation: global CFL condition
  virtual double
  calculate_time_step_cfl_global(double const time) const = 0;

  // time step calculation: local CFL condition
  virtual double
  calculate_time_step_cfl_analytical_velocity(double const time) const = 0;

  virtual double
  calculate_time_step_cfl_numerical_velocity(VectorType const & velocity) const = 0;

  // needed for time step calculation
  virtual double
  calculate_time_step_diffusion() const = 0;
};
} // namespace Interface

template<typename Number>
class OperatorExplRK
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  OperatorExplRK(std::shared_ptr<ConvDiff::Interface::Operator<Number>> operator_in,
                 bool const                                             numerical_velocity_field_in)
    : pde_operator(operator_in), numerical_velocity_field(numerical_velocity_field_in)
  {
    if(numerical_velocity_field)
      initialize_dof_vector_velocity(velocity_interpolated);
  }

  void
  set_velocities_and_times(std::vector<VectorType const *> const & velocities_in,
                           std::vector<double> const &             times_in)
  {
    velocities = velocities_in;
    times      = times_in;
  }

  void
  evaluate(BlockVectorType & dst,
           BlockVectorType const & src,
           double const evaluation_time) const
  {
    if(numerical_velocity_field)
    {
      interpolate(velocity_interpolated, evaluation_time, velocities, times);

      pde_operator->evaluate_explicit_time_int(dst, src, evaluation_time, &velocity_interpolated);
    }
    else
    {
      pde_operator->evaluate_explicit_time_int(dst, src, evaluation_time);
    }
  }

  void
  initialize_dof_vector(BlockVectorType & src) const
  {
    pde_operator->initialize_dof_vector(src);
  }

  void
  initialize_dof_vector_velocity(VectorType & src) const
  {
    pde_operator->initialize_dof_vector_velocity(src);
  }

private:
  std::shared_ptr<ConvDiff::Interface::Operator<Number>> pde_operator;

  bool                            numerical_velocity_field;
  std::vector<VectorType const *> velocities;
  std::vector<double>             times;
  VectorType mutable velocity_interpolated;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_INTERFACE_H_ */
