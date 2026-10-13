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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_ENRICHMENT_CONSTRAINED_OPERATOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_ENRICHMENT_CONSTRAINED_OPERATOR_H_

#include <deal.II/lac/la_parallel_block_vector.h>

#include <exadg/operators/wall_modelling/coupler.h>

namespace ExaDG
{
namespace IncNS
{
/*
 * Galerkin restriction of a block operator K on (U_bar, U_tilde) to the constrained space
 *
 *   K_c = E K E + (I - E),     E = [ E_bar  0 ]
 *                                  [ 0      P ]
 *
 *   P     : projector onto the well-posed enrichment modes (Step 5b,
 *           WallDGCoupler::apply_enrichment_projector)
 *   E_bar : diagonal, 0 on U_bar DoFs fixed by the strong no-slip condition (Step 7c,
 *           Krank et al. 2018, App. C), 1 otherwise; E_bar = I if free_bar == nullptr
 *
 * Solve K_c X = E B with X_0 <- E X_0. K_c is SPD if K is (E = E^T = E^2), the solution satisfies
 * (I - E) X = 0 and E K E X = E B (Galerkin on range(E)).
 */
template<int dim, typename Number, typename Operator>
class EnrichmentConstrainedOperator
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number>      VectorType;
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  // free_bar: diagonal of E_bar in the global DG layout of U_bar (nullptr: no fixed DoFs)
  EnrichmentConstrainedOperator(Operator const &                   op_in,
                                WallDGCoupler<dim, Number> const & coupler_in,
                                VectorType const *                 free_bar_in = nullptr)
    : op(op_in), coupler(coupler_in), free_bar(free_bar_in)
  {
  }

  // x <- E x
  void
  constrain(BlockVectorType & x) const
  {
    if(free_bar != nullptr)
      x.block(0).scale(*free_bar);
    coupler.apply_enrichment_projector(x.block(1), x.block(1), false);
  }

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    if(tmp.n_blocks() != src.n_blocks() or tmp.size() != src.size())
      tmp.reinit(src);

    // E X
    tmp = src;
    constrain(tmp);

    // E K E X
    op.vmult(dst, tmp);
    constrain(dst);

    // + (I - E) X = X - E X
    dst += src;
    dst -= tmp;
  }

private:
  Operator const &                   op;
  WallDGCoupler<dim, Number> const & coupler;
  VectorType const *                 free_bar;

  mutable BlockVectorType tmp;
};

} // namespace IncNS
} // namespace ExaDG

#endif
