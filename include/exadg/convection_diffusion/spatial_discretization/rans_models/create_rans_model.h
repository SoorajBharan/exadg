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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_CREATE_RANS_MODEL_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_CREATE_RANS_MODEL_H_

// C++
#include <memory>

// ExaDG
#include <exadg/convection_diffusion/spatial_discretization/rans_models/k_omega_1988.h>
#include <exadg/convection_diffusion/spatial_discretization/rans_models/k_omega_2006.h>
#include <exadg/convection_diffusion/spatial_discretization/rans_models/rans_model_base.h>

namespace ExaDG
{
namespace ConvDiff
{
/*
   * Creates the RANS model selected in TurbulenceModelData. The model object is shared (read only) by
   * all users of the eddy viscosity and the source terms (TurbulenceModel, TurbulenceSourceKernel).
   */
template<int dim, typename Number>
std::shared_ptr<RANSModelBase<dim, Number> const>
create_rans_model(TurbulenceModelData const & turbulence_model_data)
{
  AssertThrow(turbulence_model_data.is_active,
              dealii::ExcMessage("create_rans_model(): the turbulence model is not active."));

  std::shared_ptr<RANSModelBase<dim, Number> const> model;

  switch(turbulence_model_data.turbulence_model)
  {
    case TurbulenceEddyViscosityModel::StandardKOmega1988:
      model = std::make_shared<KOmega1988Model<dim, Number> const>(turbulence_model_data);
      break;
    case TurbulenceEddyViscosityModel::StandardKOmega2006:
      model = std::make_shared<KOmega2006Model<dim, Number> const>(turbulence_model_data);
      break;
    case TurbulenceEddyViscosityModel::StandardKEpsilon:
      AssertThrow(false, dealii::ExcMessage("RANS model k-epsilon: not implemented yet."));
      break;
    case TurbulenceEddyViscosityModel::Undefined:
    default:
      AssertThrow(false, dealii::ExcMessage("create_rans_model(): invalid turbulence model."));
      break;
  }

  // one positivity preserving limiter (linear / logarithmic) per transport variable
  AssertThrow(turbulence_model_data.positivity_preserving_limiter.size() ==
              model->get_n_variables(),
              dealii::ExcMessage(
              "One positivity preserving limiter per transport variable has to be specified."));

  return model;
}

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_SPATIAL_DISCRETIZATION_RANS_MODELS_CREATE_RANS_MODEL_H_ */
