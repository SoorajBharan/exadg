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
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_ENRICHED_INVERSE_MASS_PRECONDITIONER_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_ENRICHED_INVERSE_MASS_PRECONDITIONER_H_

#include <deal.II/lac/la_parallel_block_vector.h>

#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * Preconditioner for operators of the form K = s*M + (...) acting on the wall-enriched
 * velocity (block(0) = U_bar, block(1) = U_tilde):
 *
 *   P^{-1} = (1/s) * M^{-1}
 *
 * M^{-1} is the block inverse mass operator of SpatialOperatorBase:
 *   - standard DG inverse mass on non-enriched cells
 *   - exact 2x2 inverse via Schur complement on enriched cells
 *     (requires WallDGCoupler::precompute_schur_matrices() for the current psi)
 *
 * Symmetric positive definite -> usable with CG.
 */
template<int dim, typename Number>
class EnrichedInverseMassPreconditioner
{
public:
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  EnrichedInverseMassPreconditioner() : spatial_operator(nullptr), inv_scaling(1.0)
  {
  }

  void
  initialize(SpatialOperatorBase<dim, Number> const & spatial_operator_in)
  {
    spatial_operator = &spatial_operator_in;
  }

  // s = scaling factor of the mass term (gamma0/dt); call before every solve
  void
  set_scaling_factor_mass(double const s)
  {
    AssertThrow(s > 0.0, dealii::ExcMessage("Scaling factor of the mass term must be positive."));
    inv_scaling = 1.0 / s;
  }

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    AssertThrow(spatial_operator != nullptr,
                dealii::ExcMessage("EnrichedInverseMassPreconditioner not initialized."));

    spatial_operator->apply_inverse_mass_operator(dst, src);
    dst *= inv_scaling;
  }

private:
  SpatialOperatorBase<dim, Number> const * spatial_operator;
  double                                   inv_scaling;
};

} // namespace IncNS
} // namespace ExaDG

#endif
