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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_TURBULENCE_MODEL_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/convection_diffusion/user_interface/turbulence_model_data.h>



namespace ExaDG
{
namespace ConvDiff
{
/*
 *  Base class for variable viscosity models.
 */
template<int dim, int n_components, typename Number>
class TurbulenceModel : public dealii::Subscriptor
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, n_components, Number> IntegratorCell;
  typedef FaceIntegrator<dim, n_components, Number> IntegratorFace;

  typedef CellIntegrator<dim, 1, Number> IntegratorCellScalar;

  typedef dealii::VectorizedArray<Number> scalar;
public:
  /*
   * Constructor.
   */
  TurbulenceModel();

  /*
   * Destructor.
   */
  virtual ~TurbulenceModel(){};

  /*
   * Initialization function of base class.
   */
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             TurbulenceModelData const &             turbulence_model_data_in,
             unsigned int const                      dof_index_in,
             unsigned int const                      dof_index_viscosity_in,
             unsigned int const                      quad_index_in);

  /**
   * Function for *setting* the eddy viscosity
   */
  void
  set_viscosity(VectorType const & solution);

  void
  get_eddy_viscosity(VectorType & dst) const;

  dealii::LinearAlgebra::distributed::Vector<Number> const &
  get_eddy_viscosity_ref() const;

  void
  get_turbulent_kinetic_energy(VectorType & dst, VectorType const & solution) const;

  double       diffusivity;
  unsigned int quad_index;

  std::vector<double> model_coefficients;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & data,
            VectorType &,
            VectorType const & src,
            Range const &      cell_range);

  template<typename DataType>
  void
  evaluate_eddy_viscosity(DataType const & solution_values,
                          scalar & viscosity) const;

  void
  standard_k_epsilon_model(dealii::Tensor<1, n_components, scalar> const & solution_values,
                           scalar & viscosity) const;

  void
  standard_k_omega_1988_model(dealii::Tensor<1, n_components, scalar> const & solution_values,
                              scalar & viscosity) const;

  void
  cell_loop_extract_tke(dealii::MatrixFree<dim, Number> const & data,
                        VectorType &,
                        VectorType const & src,
                        Range const &      cell_range) const;

  TurbulenceModelData                                   turbulence_model_data;

  VectorType eddy_viscosity;
  VectorType effective_viscosity;

protected:
  unsigned int dof_index;
  unsigned int dof_index_viscosity;

  dealii::MatrixFree<dim, Number> const * matrix_free;
};

} // namespace RANS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_RANS_EQUATIONS_SPATIAL_DISCRETIZATION_VISCOSITY_MODEL_BASE_H_ \
        */
