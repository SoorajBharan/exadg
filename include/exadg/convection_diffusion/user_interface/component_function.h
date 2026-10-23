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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_COMPONENT_FUNCTION_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_COMPONENT_FUNCTION_H_

// C/C++
#include <memory>

// deal.II
#include <deal.II/base/function.h>

// ExaDG
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>

namespace ExaDG
{
namespace ConvDiff
{

template<int dim>
class ComponentOfFunction : public dealii::Function<dim>
{
public:
  ComponentOfFunction(std::shared_ptr<dealii::Function<dim>> function_in,
                      unsigned int const                     component_in)
    : dealii::Function<dim>(1, function_in->get_time()),
    function(function_in),
    component(component_in)
  {
    AssertThrow(component < function->n_components,
                dealii::ExcMessage("ComponentOfFunction: component " +
                                   std::to_string(component) + " requested from a function with " + std::to_string(function->n_components) + " components."));
  }

  void
  set_time(double const new_time) override
  {
    dealii::Function<dim>::set_time(new_time);
    function->set_time(new_time);
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/ = 0) const
  override
  {
    return function->value(p, component);
  }

  dealii::Tensor<1, dim>
  gradient(dealii::Point<dim> const & p, unsigned int const /*component*/ = 0)
  const override
  {
    return function->gradient(p, component);
  }

private:
  std::shared_ptr<dealii::Function<dim>> function;

  unsigned int const component;
};

template<int dim>
std::shared_ptr<dealii::Function<dim>>
get_component_function(std::shared_ptr<dealii::Function<dim>> const & function,
                       unsigned int const                             component)
{
  if(not function)
    return function;

  if(function->n_components == 1)
    return function;

  return std::make_shared<ComponentOfFunction<dim>>(function, component);
}


template<int dim>
std::shared_ptr<BoundaryDescriptor<dim>>
get_boundary_descriptor_of_component(BoundaryDescriptor<dim> const &boundary_descriptor,
                                     unsigned int const component)
{
  std::shared_ptr<BoundaryDescriptor<dim>> bc_component = std::make_shared<BoundaryDescriptor<dim>>();

  for(auto const & [boundary_id, function] : boundary_descriptor.dirichlet_bc)
  bc_component->dirichlet_bc.insert({boundary_id, get_component_function(function, component)});

  for(auto const & [boundary_id, function] : boundary_descriptor.neumann_bc)
  bc_component->neumann_bc.insert({boundary_id, get_component_function(function, component)});


  for(auto const & [boundary_id, mixed_data] : boundary_descriptor.mixed_bc)
  {
    AssertThrow(component < mixed_data.component_bc_type.size(),
                dealii::ExcMessage("Mixed boundary condition: no boundary type specified for component " + std::to_string(component) + "."));

    BoundaryType const type = mixed_data.component_bc_type[component];

    if(type == BoundaryType::Dirichlet)
      bc_component->dirichlet_bc.insert({boundary_id, get_component_function(mixed_data.function, component)});
    else if(type == BoundaryType::Neumann)
      bc_component->neumann_bc.insert({boundary_id, get_component_function(mixed_data.function, component)});
    else
      AssertThrow(false,
                  dealii::ExcMessage("Mixed boundary condition: only Dirichlet or Neumann allowed per component."));
  }

  return bc_component;
}

} // namespace ConvDiff
} // namespace ExaDG


 #endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_USER_INTERFACE_COMPONENT_FUNCTION_H_*/

