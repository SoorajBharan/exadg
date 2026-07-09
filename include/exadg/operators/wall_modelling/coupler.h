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

#ifndef INCLUDE_OPERATORS_WALL_MODELLING_COUPLER_H_
#define INCLUDE_OPERATORS_WALL_MODELLING_COUPLER_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <exadg/operators/wall_modelling/function_enrichment.h>

namespace ExaDG
{
template<int dim, typename Number>
class WallDGCoupler
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;
  using scalar     = dealii::VectorizedArray<Number>;

  WallDGCoupler();

  void initialize(
    dealii::MatrixFree<dim, Number> const & matrix_free_global_in,
    unsigned int                            dof_index_global_u_in,
    std::shared_ptr<FunctionEnrichment<dim, Number>> function_enrichment_in);

  void precompute_schur_matrices();

  void 
  apply_schur_complement(
    VectorType & dst_global,
    VectorType const & src_global,
    VectorType & dst_enrichment,
    VectorType const & src_enrichment) const;

  void
  apply_schur_inverse_mass(VectorType & dst_bar,
                           VectorType const & src_bar,
                           VectorType & dst_tilde,
                           VectorType const & src_tilde) const;

  std::vector<std::vector<dealii::VectorizedArray<Number>>>
  invert_matrix_simd(std::vector<std::vector<dealii::VectorizedArray<Number>>> M, unsigned int n);

  void compute_enrichment_velocity_from_residual(VectorType const & dg_residual,
                                   double scaling_factor);

  void compute_enrichment_velocity(VectorType const & inter_velocity,
                                   double scaling_factor);

  void
  local_compute_enrichment_velocity(
    dealii::MatrixFree<dim, Number> const & mf_en,
    VectorType & dst,
    VectorType const & src,
    std::pair<unsigned int, unsigned int> const & range) const;

  void
  update_wall_enrichment_vectors(VectorType const & global_velocity) const;

  /*
   * Getters
   * */
  dealii::LinearAlgebra::distributed::Vector<Number> const &
  get_shadow_velocity() const;

  dealii::LinearAlgebra::distributed::Vector<Number> const &
  get_enrichment_velocity() const;

  struct SchurData {
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_bar_tilde; 
    std::vector<std::vector<dealii::VectorizedArray<Number>>> schur_inverse;
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_bar_bar_inverse;
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_bar_bar;

    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_tilde_tilde;
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_tilde_tilde_inverse;

    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_tilde_tilde_safe;
    std::vector<std::vector<dealii::VectorizedArray<Number>>> M_tilde_tilde_inverse_safe;
  };

  std::vector<SchurData> cell_schur_data;

  std::shared_ptr<FunctionEnrichment<dim, Number>> function_enrichment;

private:
  dealii::MatrixFree<dim, Number> const * matrix_free_global;
  unsigned int dof_index_global_u;

  unsigned int active_fe_index;

  Number current_scaling_factor;
};

} // namespace ExaDG

#endif
