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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_RANS_MODEL_BASE_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_RANS_MODEL_BASE_H_

// C++
#include <vector>

// deal.II
#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

// ExaDG
#include <exadg/convection_diffusion/user_interface/turbulence_model_data.h>

namespace ExaDG
{
namespace ConvDiff
{
/*
   * Input of a RANS model at one quadrature point, in PHYSICAL variables (k, omega), never in the
   * transported ones (ln omega). The kernel fills it; the model only reads it.
   */
template<int dim, typename Number>
struct RANSPointInput
{
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<1, dim, scalar>  vector;
  typedef dealii::Tensor<2, dim, scalar>  tensor;

  void
  reinit(unsigned int const n_variables_in)
  {
    n_variables = n_variables_in;
    values.resize(n_variables);
    gradients.resize(n_variables);
  }

  unsigned int n_variables = 0;

  // phi_j: physical value of variable j (k, omega)
  dealii::AlignedVector<scalar> values;

  // grad phi_j: physical gradient of variable j (grad omega, not grad ln omega)
  dealii::AlignedVector<vector> gradients;

  // velocity gradient, [i][j] = d u_i / d x_j
  tensor velocity_gradient;

  // false: values only (Explicit treatment), the model may skip all Jacobians
  bool compute_derivatives = true;
};

/*
   * A viscosity at one quadrature point (eddy viscosity nu_t or diffusion viscosity nu_D) and its
   * derivative with respect to the physical variables.
   */
template<typename Number>
struct RANSViscosity
{
  typedef dealii::VectorizedArray<Number> scalar;

  void
  reinit(unsigned int const n_variables)
  {
    derivative.resize(n_variables);
  }

  // nu_t or nu_D
  scalar value;

  // d nu / d phi_j
  dealii::AlignedVector<scalar> derivative;
};

/*
   * One group of source terms (production, dissipation or gradient source) at one quadrature point,
   * for all equations i, with the Jacobians needed by the Newton linearization.
   */
template<int dim, typename Number>
struct RANSSourceTerm
{
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<1, dim, scalar>  vector;

  void
  reinit(unsigned int const n_variables_in)
  {
    n_variables = n_variables_in;
    value.resize(n_variables);
    jacobian.resize(n_variables * n_variables);
    gradient_jacobian.resize(n_variables * n_variables);
  }

  // position of the entry (equation i, variable j) in jacobian and gradient_jacobian
  unsigned int
  index(unsigned int const i, unsigned int const j) const
  {
    return i * n_variables + j;
  }

  // models fill only their non-zero entries, so everything is reset before each evaluation
  void
  set_zero()
  {
    for(unsigned int i = 0; i < value.size(); ++i)
      value[i] = scalar(0.0);
    for(unsigned int i = 0; i < jacobian.size(); ++i)
    {
      jacobian[i]          = scalar(0.0);
      gradient_jacobian[i] = vector();
    }
  }

  unsigned int n_variables = 0;

  // S_i: source in the equation of variable i (physical form)
  dealii::AlignedVector<scalar> value;

  // d S_i / d phi_j, stored at index(i, j)
  dealii::AlignedVector<scalar> jacobian;

  // d S_i / d grad phi_j, stored at index(i, j) (cross-diffusion, square gradient)
  dealii::AlignedVector<vector> gradient_jacobian;
};

/*
   * Base class of the RANS turbulence models. A model works at ONE quadrature point and in PHYSICAL
   * variables (k, omega). The conversion from and to the transported variables (ln k, ln omega) is
   * done here, once for all models.
   */
template<int dim, typename Number>
class RANSModelBase
{
public:
  typedef dealii::VectorizedArray<Number> scalar;
  typedef dealii::Tensor<1, dim, scalar>  vector;

  RANSModelBase(TurbulenceModelData const & turbulence_model_data_in)
    : turbulence_model_data(turbulence_model_data_in),
    coefficients(get_coefficients(turbulence_model_data_in)),
    is_logarithmic(turbulence_model_data_in.positivity_preserving_limiter.size(), false)
  {
    for(unsigned int i = 0; i < is_logarithmic.size(); ++i)
      is_logarithmic[i] = (turbulence_model_data.positivity_preserving_limiter[i] ==
        PositivityPreservingLimiter::LogarithmicTransportVariable);
  }

  virtual ~RANSModelBase() = default;

  // number of transport variables (k-omega: 2)
  virtual unsigned int
  get_n_variables() const = 0;

  // sigma_i: the turbulent diffusion in the equation of variable i is div(sigma_i nu_D grad phi_i)
  virtual double
  get_diffusion_factor(unsigned int const i) const = 0;

  // nu_t (momentum equation) and nu_D (turbulent diffusion), with d nu / d phi_j
  virtual void
  evaluate_viscosities(RANSPointInput<dim, Number> const & input,
                       RANSViscosity<Number> &             eddy_viscosity,
                       RANSViscosity<Number> &             diffusion_viscosity) const = 0;

  // source terms in physical form; eddy_viscosity is the result of evaluate_viscosities() at the
  // same point (models that need nu_t in a source do not compute it twice)
  virtual void
  evaluate_sources(RANSPointInput<dim, Number> const & input,
                   RANSViscosity<Number> const &       eddy_viscosity,
                   RANSSourceTerm<dim, Number> &       production,
                   RANSSourceTerm<dim, Number> &       dissipation,
                   RANSSourceTerm<dim, Number> &       gradient_source) const = 0;

  /*
     * Transported -> physical: phi_j = Q_j (linear) or exp(Q_j) (logarithmic), and
     * grad phi_j = grad Q_j or phi_j grad Q_j. An empty transported_gradients (faces) skips the
     * gradients.
     */
  void
  set_physical_variables(dealii::ArrayView<scalar const> const & transported_values,
                         dealii::ArrayView<vector const> const & transported_gradients,
                         RANSPointInput<dim, Number> &           input) const
  {
    AssertDimension(transported_values.size(), is_logarithmic.size());

    for(unsigned int j = 0; j < transported_values.size(); ++j)
    {
      input.values[j] =
        is_logarithmic[j] ? std::exp(transported_values[j]) : transported_values[j];

      if(transported_gradients.size() > 0)
        input.gradients[j] = is_logarithmic[j] ? input.values[j] * transported_gradients[j] :
          transported_gradients[j];
    }
  }

  /*
     * Physical -> transported form of a source group, in place. With s_i = 1 / phi_i (ln phi_i
     * transported, equation divided by phi_i) or 1, and c_j = d phi_j / d Q_j = phi_j or 1:
     *
     *   T_i           = s_i S_i
     *   d T_i / d Q_j = s_i (d S_i / d phi_j c_j + [ln phi_j] d S_i / d grad phi_j . grad phi_j)
     *                   - [ln phi_i, i = j] T_i
     *   d T_i / d grad Q_j = s_i c_j d S_i / d grad phi_j
     */
  void
  convert_to_transported_form(RANSPointInput<dim, Number> const & input,
                              RANSSourceTerm<dim, Number> &       term) const
  {
    unsigned int const n = input.n_variables;
    AssertDimension(term.n_variables, n);
    AssertDimension(is_logarithmic.size(), n);

    scalar const one = dealii::make_vectorized_array<Number>(1.0);

    for(unsigned int i = 0; i < n; ++i)
    {
      scalar const s_i = is_logarithmic[i] ? one / input.values[i] : one;

      term.value[i] = s_i * term.value[i];

      if(not input.compute_derivatives)
        continue;

      for(unsigned int j = 0; j < n; ++j)
      {
        scalar const       c_j = is_logarithmic[j] ? input.values[j] : one;
        unsigned int const ij  = term.index(i, j);

        // value Jacobian, including the dependence through grad phi_j = phi_j grad Q_j
        scalar jacobian = term.jacobian[ij] * c_j;
        if(is_logarithmic[j])
          jacobian += term.gradient_jacobian[ij] * input.gradients[j];

        term.jacobian[ij]          = s_i * jacobian;
        term.gradient_jacobian[ij] = s_i * c_j * term.gradient_jacobian[ij];
      }

      // derivative of the factor s_i = exp(-Q_i): (d s_i / d Q_i) S_i = - T_i
      if(is_logarithmic[i])
        term.jacobian[term.index(i, i)] -= term.value[i];
    }
  }

  // physical -> transported derivative of a viscosity: d nu / d Q_j = d nu / d phi_j c_j
  void
  convert_to_transported_form(RANSPointInput<dim, Number> const & input,
                              RANSViscosity<Number> &             viscosity) const
  {
    if(not input.compute_derivatives)
      return;

    for(unsigned int j = 0; j < input.n_variables; ++j)
      if(is_logarithmic[j])
        viscosity.derivative[j] *= input.values[j];
  }

protected:
  // switches of the model (realizability limiter, stress limiter, ...)
  TurbulenceModelData const turbulence_model_data;

  // closure coefficients in the order of TurbulenceDataBase::get_all_coefficients()
  std::vector<double> const coefficients;

private:
  static std::vector<double>
  get_coefficients(TurbulenceModelData const & data)
  {
    AssertThrow(data.turbulence_data_base != nullptr,
                dealii::ExcMessage("Turbulence data base not initialized."));
    return data.turbulence_data_base->get_all_coefficients();
  }

  // true: ln phi_i is transported (PositivityPreservingLimiter::LogarithmicTransportVariable)
  std::vector<bool> is_logarithmic;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_RANS_MODEL_BASE_H_ */
