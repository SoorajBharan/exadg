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
#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/array_view.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/convection_diffusion/user_interface/turbulence_model_data.h>
#include <exadg/utilities/lazy_ptr.h>

#include <exadg/convection_diffusion/spatial_discretization/eddy_viscosity_evaluator.h>

#include <exadg/operators/inverse_mass_operator.h>


namespace ExaDG
{
namespace ConvDiff
{
/*
 *  Base class for variable viscosity models.
 */
template<int dim, typename Number>
class TurbulenceModel : public dealii::Subscriptor
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  typedef TurbulenceModel<dim, Number> This;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, 1, Number> IntegratorCell;
  typedef FaceIntegrator<dim, 1, Number> IntegratorFace;

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

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             TurbulenceModelData const &             turbulence_model_data_in,
             unsigned int const                      dof_index_in,
             unsigned int const                      dof_index_viscosity_in,
             unsigned int const                      dof_index_velocity_in,
             unsigned int const                      quad_index_in,
             unsigned int const                      quad_index_eddy_velocity_in,
             bool const                              compute_derivative_in = false);

  void
  set_viscosity(BlockVectorType const & solution);

  void
  get_eddy_viscosity(VectorType & dst) const;

  VectorType const &
  get_eddy_viscosity_ref() const;

  BlockVectorType const &
  get_eddy_viscosity_derivative_ref() const;

  unsigned int
  get_n_variables() const;

  void
  get_physical_variables(dealii::ArrayView<scalar const> const & transported_values,
                         dealii::ArrayView<scalar> const &       solution,
                         dealii::ArrayView<scalar> const &       solution_derivative) const;

  scalar
  compute_realizability_alpha(dealii::Tensor<2, dim, scalar> const & grad_u) const;

  void
  get_turbulent_kinetic_energy(VectorType & dst, BlockVectorType const & solution) const;

  void
  set_velocity_ptr(VectorType const & velocity_in);

  void
  enforce_positivity(BlockVectorType & solution) const;

  double       diffusivity;
  unsigned int quad_index;
  unsigned int quad_index_eddy_velocity;

  std::vector<double> model_coefficients;

  // pointwise eddy viscosity and its derivative w.r.t. the transport variables (also used by the
  // TurbulenceSourceKernel at the quadrature points)
  void
  evaluate_eddy_viscosity(dealii::ArrayView<scalar const> const & solution,
                          dealii::Tensor<2, dim, scalar> const &  velocity_gradient,
                          scalar &                                viscosity,
                          dealii::ArrayView<scalar> const &       viscosity_derivative) const;

  EddyViscosityEvaluator<dim, Number> const &
  get_eddy_viscosity_evaluator() const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
            VectorType &                            dst,
            BlockVectorType const &                 src,
            Range const &                           cell_range);

  void
  cell_loop_extract_tke(dealii::MatrixFree<dim, Number> const & matrix_free,
                        VectorType &                            dst,
                        BlockVectorType const &                 src,
                        Range const &                           cell_range) const;

  void
  cell_loop_enforce_positivity(dealii::MatrixFree<dim, Number> const & matrix_free,
                               BlockVectorType &                       dst,
                               BlockVectorType const &                 src,
                               Range const &                           cell_range) const;

  TurbulenceModelData                                  turbulence_model_data;

  EddyViscosityEvaluator<dim, Number> evaluator;

  unsigned int n_variables;

  VectorType eddy_viscosity;
  BlockVectorType eddy_viscosity_derivative;
  bool compute_derivative;
  VectorType effective_viscosity;
  InverseMassOperator<dim, 1, Number> inverse_mass_viscosity;

  mutable lazy_ptr<VectorType> velocity;

protected:
  unsigned int dof_index;
  unsigned int dof_index_viscosity;
  unsigned int dof_index_velocity;

  dealii::MatrixFree<dim, Number> const * matrix_free;
};

} // namespace RANS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_RANS_EQUATIONS_SPATIAL_DISCRETIZATION_VISCOSITY_MODEL_BASE_H_ \
        */
