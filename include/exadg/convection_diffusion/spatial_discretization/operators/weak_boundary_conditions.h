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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_

#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/operator_type.h>

namespace ExaDG
{
namespace ConvDiff
{
template <int dim, int n_components, typename Number>
struct BCEvaluator
{
using value_type = typename FaceIntegrator<dim, n_components, Number>::value_type;

  static value_type
  evaluate(dealii::Function<dim> const & function,
           dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
           double const time)
  {
    const_cast<dealii::Function<dim>&>(function).set_time(time);

    value_type result;
    for(unsigned int c = 0; c < n_components; ++c)
    {
      for(unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
      {
        dealii::Point<dim> p;
        for(unsigned int d = 0; d < dim; ++d)
          p[d] = q_points[d][v];

        result[c][v] = function.value(p, c);
      }
    }
    return result;
  }
};

// Fast specialization for single-component scalar systems
template <int dim, typename Number>
struct BCEvaluator<dim, 1, Number>
{
  using value_type = typename FaceIntegrator<dim, 1, Number>::value_type;

  static value_type
  evaluate(dealii::Function<dim> const & function,
           dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_points,
           double const time)
  {
    return FunctionEvaluator<0, dim, Number>::value(const_cast<dealii::Function<dim>&>(function), q_points, time);
  }
};
/*
 *  The following two functions calculate the interior_value/exterior_value
 *  depending on the operator type, the type of the boundary face
 *  and the given boundary conditions.
 *
 *                            +----------------------+--------------------+
 *                            | Dirichlet boundaries | Neumann boundaries |
 *  +-------------------------+----------------------+--------------------+
 *  | full operator           | phi⁺ = -phi⁻ + 2g    | phi⁺ = phi⁻        |
 *  +-------------------------+----------------------+--------------------+
 *  | homogeneous operator    | phi⁺ = -phi⁻         | phi⁺ = phi⁻        |
 *  +-------------------------+----------------------+--------------------+
 *  | inhomogeneous operator  | phi⁻ = 0, phi⁺ = 2g  | phi⁻ = 0, phi⁺ = 0 |
 *  +-------------------------+----------------------+--------------------+
 */
template<int dim, int n_components, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  typename FaceIntegrator<dim, n_components, Number>::value_type
  calculate_interior_value(unsigned int const                     q,
                           FaceIntegrator<dim, n_components, Number> const & integrator,
                           OperatorType const &                   operator_type)
{
  using value_type = typename FaceIntegrator<dim, n_components, Number>::value_type;
  value_type value_m{};

  if(operator_type == OperatorType::full or operator_type == OperatorType::homogeneous)
  {
    value_m = integrator.get_value(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    // do nothing (value_m already initialized with 0.0)
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));
  }

  return value_m;
}

template<int dim, int n_components, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  typename FaceIntegrator<dim, n_components, Number>::value_type
  calculate_exterior_value(typename FaceIntegrator<dim, n_components, Number>::value_type const &        value_m,
                           unsigned int const                             q,
                           FaceIntegrator<dim, n_components, Number> const &         integrator,
                           OperatorType const &                           operator_type,
                           BoundaryType const &                           boundary_type,
                           dealii::types::boundary_id const               boundary_id,
                           std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor,
                           double const &                                 time)
{
  using value_type = typename FaceIntegrator<dim, n_components, Number>::value_type;
  value_type value_p{};

  if(boundary_type == BoundaryType::Dirichlet)
  {
    if(operator_type == OperatorType::full or operator_type == OperatorType::inhomogeneous)
    {
      auto bc       = boundary_descriptor->dirichlet_bc.find(boundary_id)->second;
      auto q_points = integrator.quadrature_point(q);

      // auto g = FunctionEvaluator<(n_components > 1 ? 1 : 0), dim, Number>::value(*bc, q_points, time);
      auto g = BCEvaluator<dim, n_components, Number>::evaluate(*bc, q_points, time);

      value_p = -value_m + g * 2.0;
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      value_p = -value_m;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else if(boundary_type == BoundaryType::Neumann)
  {
    value_p = value_m;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return value_p;
}

// clang-format off
  /*
   *  The following two functions calculate the interior/exterior gradient
   *  in normal direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *                            +-----------------------------------------------+------------------------------------------------------+
   *                            | Dirichlet boundaries                          | Neumann boundaries                                   |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | full operator           | grad(phi⁺)*n = grad(phi⁻)*n                   | grad(phi⁺)*n = -grad(phi⁻)*n + 2h                    |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | homogeneous operator    | grad(phi⁺)*n = grad(phi⁻)*n                   | grad(phi⁺)*n = -grad(phi⁻)*n                         |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | inhomogeneous operator  | grad(phi⁺)*n = grad(phi⁻)*n, grad(phi⁻)*n = 0 | grad(phi⁺)*n = -grad(phi⁻)*n + 2h, grad(phi⁻)*n  = 0 |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *
   *                            +-----------------------------------------------+------------------------------------------------------+
   *                            | Dirichlet boundaries                          | Neumann boundaries                                   |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | full operator           | {{grad(phi)}}*n = grad(phi⁻)*n                | {{grad(phi)}}*n = h                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | homogeneous operator    | {{grad(phi)}}*n = grad(phi⁻)*n                | {{grad(phi)}}*n = 0                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   *  | inhomogeneous operator  | {{grad(phi)}}*n = 0                           | {{grad(phi)}}*n = h                                  |
   *  +-------------------------+-----------------------------------------------+------------------------------------------------------+
   */
// clang-format on
template<int dim, int n_components, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  typename FaceIntegrator<dim, n_components, Number>::value_type
  calculate_interior_normal_gradient(unsigned int const                     q,
                                     FaceIntegrator<dim, n_components, Number> const & integrator,
                                     OperatorType const &                   operator_type)
{
  using value_type = typename FaceIntegrator<dim, n_components, Number>::value_type;
  value_type value_p{};

  value_type normal_gradient_m = value_p;

  if(operator_type == OperatorType::full or operator_type == OperatorType::homogeneous)
  {
    normal_gradient_m = integrator.get_normal_derivative(q);
  }
  else if(operator_type == OperatorType::inhomogeneous)
  {
    // do nothing (normal_gradient_m already initialized with 0.0)
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));
  }

  return normal_gradient_m;
}

template<int dim, int n_components, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  typename FaceIntegrator<dim, n_components, Number>::value_type
  calculate_exterior_normal_gradient(
    typename FaceIntegrator<dim, n_components, Number>::value_type const &        normal_gradient_m,
    unsigned int const                             q,
    FaceIntegrator<dim, n_components, Number> const &         integrator,
    OperatorType const &                           operator_type,
    BoundaryType const &                           boundary_type,
    dealii::types::boundary_id const               boundary_id,
    std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor,
    double const &                                 time)
{
  using value_type = typename FaceIntegrator<dim, n_components, Number>::value_type;
  value_type normal_gradient_p{};

  if(boundary_type == BoundaryType::Dirichlet)
  {
    normal_gradient_p = normal_gradient_m;
  }
  else if(boundary_type == BoundaryType::Neumann)
  {
    if(operator_type == OperatorType::full or operator_type == OperatorType::inhomogeneous)
    {
      auto bc       = boundary_descriptor->neumann_bc.find(boundary_id)->second;
      auto q_points = integrator.quadrature_point(q);

      // auto h = FunctionEvaluator<(n_components > 1 ? 1 : 0), dim, Number>::value(*bc, q_points, time);
      auto h = BCEvaluator<dim, n_components, Number>::evaluate(*bc, q_points, time);

      normal_gradient_p = -normal_gradient_m + h * 2.0;
    }
    else if(operator_type == OperatorType::homogeneous)
    {
      normal_gradient_p = -normal_gradient_m;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Specified OperatorType is not implemented!"));
    }
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return normal_gradient_p;
}

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_OPERATORS_WEAK_BOUNDARY_CONDITIONS_H_ \
        */
