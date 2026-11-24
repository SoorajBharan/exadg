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
 *
 * Source: https://github.com/MeltPoolDG/MeltPoolDG
 * Author: Peter Munch, Magdalena Schreter, TUM, December 2020
 */

#ifndef INCLUDE_EXADG_OPERATORS_ADAPTIVE_MESH_REFINEMENT_H_
#define INCLUDE_EXADG_OPERATORS_ADAPTIVE_MESH_REFINEMENT_H_

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools_common.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

#include <limits>
#include <string>
#include <vector>
#include <algorithm>
#include <optional>

namespace ExaDG
{
/**
   * Rule for marking cells from the error indicator (deal.II GridRefinement):
   *   FixedNumber:   the given fractions refer to the number of cells
   *                  (refine_and_coarsen_fixed_number)
   *   FixedFraction: the given fractions refer to the total error, i.e. the sum of the squared
   *                  indicators (refine_and_coarsen_fixed_fraction)
   */
enum class MarkingStrategy
{
  FixedNumber,
  FixedFraction
};

/**
   * Combination of the normalized indicators of several fields (velocity, transport variables) into
   * one indicator per cell:
   *   Maximum:     eta_K = max_f ( w_f * eta_K^f / max_K eta_K^f )
   *   WeightedSum: eta_K = sum_f ( w_f * eta_K^f / max_K eta_K^f )
   */
enum class IndicatorCombination
{
  Maximum,
  WeightedSum
};

struct AdaptiveMeshRefinementData
{
  AdaptiveMeshRefinementData()
    : trigger_every_n_time_steps(1),
      maximum_refinement_level(10),
      minimum_refinement_level(0),
      preserve_boundary_cells(false),
      fraction_of_cells_to_be_refined(0.0),
      fraction_of_cells_to_be_coarsened(0.0),
      marking_strategy(MarkingStrategy::FixedNumber),
      indicator_combination(IndicatorCombination::Maximum),
      indicator_weights()
  {
  }

  void
  print(dealii::ConditionalOStream const & pcout) const
  {
    print_parameter(pcout, "Enable adaptivity", true);
    print_parameter(pcout, "Triggered every n time steps", trigger_every_n_time_steps);
    print_parameter(pcout, "Maximum refinement level", maximum_refinement_level);
    print_parameter(pcout, "Minimum refinement level", minimum_refinement_level);
    print_parameter(pcout, "Preserve boundary cells", preserve_boundary_cells);
    print_parameter(pcout, "Fraction of cells to be refined", fraction_of_cells_to_be_refined);
    print_parameter(pcout, "Fraction of cells to be coarsened", fraction_of_cells_to_be_coarsened);
    print_parameter(pcout, "Marking strategy", marking_strategy);
    print_parameter(pcout, "Indicator combination", indicator_combination);
    for(unsigned int i = 0; i < indicator_weights.size(); ++i)
      print_parameter(pcout, "Indicator weight of field " + std::to_string(i), indicator_weights[i]);
  }

  unsigned int trigger_every_n_time_steps;

  int  maximum_refinement_level;
  int  minimum_refinement_level;
  bool preserve_boundary_cells;

  double fraction_of_cells_to_be_refined;
  double fraction_of_cells_to_be_coarsened;

  MarkingStrategy marking_strategy;
  IndicatorCombination indicator_combination;
  std::vector<double> indicator_weights;
};

/**
 * Check if adaptive mesh refinement should be triggered depending on the time_step_number.
 */
inline bool
trigger_coarsening_and_refinement_now(unsigned int const trigger_every_n_time_steps,
                                      unsigned int const time_step_number)
{
  return ((time_step_number == 0) or not(time_step_number % trigger_every_n_time_steps));
}

template<int dim>
std::vector<int>
compute_maximum_level_around_frozen_cells(dealii::Triangulation<dim> const & tria,
                                          std::vector<bool> const &          frozen_cells,
                                          int const                     maximum_refinement_level)
{
  int const no_level_limit = std::numeric_limits<int>::max();

  std::vector<int> maximum_level(tria.n_active_cells(), no_level_limit);

  // frozen cells keep their level
  for(auto const & cell : tria.active_cell_iterators())
  {
    if(not cell->is_artificial() and frozen_cells[cell->active_cell_index()])
    {
      maximum_level[cell->active_cell_index()] = cell->level();
    }
  }

  // ring n around the frozen cells gets level l + n; a cell in ring n >= maximum_refinement_level
  // cannot be refined too far anyway, so maximum_refinement_level rings are sufficient
  for(int ring = 1; ring <= maximum_refinement_level; ++ring)
  {
    // smallest maximum level of all cells at each vertex
    std::vector<int> maximum_level_at_vertex(tria.n_vertices(), no_level_limit);

    for(auto const & cell : tria.active_cell_iterators())
    {
      if(not cell->is_artificial())
      {
        for(auto const v : cell->vertex_indices())
        {
          unsigned int const vertex       = cell->vertex_index(v);
          maximum_level_at_vertex[vertex] = std::min(maximum_level_at_vertex[vertex],
                                                     maximum_level[cell->active_cell_index()]);
        }
      }
    }

    // locally owned cells: at most one level more than any cell sharing a vertex
    for(auto const & cell : tria.active_cell_iterators())
    {
      if(cell->is_locally_owned())
      {
        for(auto const v : cell->vertex_indices())
        {
          int const level_at_vertex = maximum_level_at_vertex[cell->vertex_index(v)];

          if(level_at_vertex != no_level_limit)
          {
            maximum_level[cell->active_cell_index()] =
              std::min(maximum_level[cell->active_cell_index()], level_at_vertex + 1);
          }
        }
      }
    }

    // ghost cells: the value computed by the rank that owns the cell (next ring needs it)
    if(dynamic_cast<dealii::parallel::TriangulationBase<dim> const *>(&tria) != nullptr)
    {
      dealii::GridTools::exchange_cell_data_to_ghosts<int, dealii::Triangulation<dim>>(
        tria,
        [&](typename dealii::Triangulation<dim>::active_cell_iterator const & cell)
        -> std::optional<int> { return maximum_level[cell->active_cell_index()]; },
        [&](typename dealii::Triangulation<dim>::active_cell_iterator const & cell,
            int const &                                                      level) {
          maximum_level[cell->active_cell_index()] = level;
        });
    }
  }

  return maximum_level;
}

/**
 * Limit the coarsening and refinement by adapting the flags set in the triangulation.
 * This considers the maximum and minimum refinement levels provided and might remove
 * refinement flags set on the boundary if requested.
 */
template<int dim>
void
limit_coarsening_and_refinement(dealii::Triangulation<dim> &       tria,
                                AdaptiveMeshRefinementData const & amr_data,
                                std::vector<bool> const &          frozen_cells = {})
{
  AssertThrow(frozen_cells.empty() or frozen_cells.size() == tria.n_active_cells(),
              dealii::ExcMessage("frozen_cells needs one entry per active cell."));

  // maximum level after refinement around the frozen cells (2:1 rule); empty: no frozen cells
  std::vector<int> const maximum_level = frozen_cells.empty() ? std::vector<int>() : compute_maximum_level_around_frozen_cells(tria, frozen_cells, amr_data.maximum_refinement_level);

  for(auto & cell : tria.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      // Clear refinement flags on maximum refinement level.
      if(cell->level() == amr_data.maximum_refinement_level)
      {
        cell->clear_refine_flag();
      }

      // Clear coarsening flags on minimum refinement level.
      if(cell->level() <= amr_data.minimum_refinement_level)
      {
        cell->clear_coarsen_flag();
      }

      // Do not coarsen/refine cells along boundary if requested
      if(amr_data.preserve_boundary_cells and cell->at_boundary())
      {
        cell->clear_refine_flag();
        cell->clear_coarsen_flag();
      }

      // Do not coarsen/refine frozen cells (wall enrichment layer)
      if(not frozen_cells.empty() and frozen_cells[cell->active_cell_index()])
      {
        cell->clear_refine_flag();
        cell->clear_coarsen_flag();
      }

      // Do not refine cells around the frozen cells beyond the maximum level of the 2:1 rule
      if(not maximum_level.empty() and cell->level() + 1 > maximum_level[cell->active_cell_index()])
      {
        cell->clear_refine_flag();
      }
    }
  }
}

/**
 * Call after prepare_coarsening_and_refinement(): stops if a frozen cell (wall enrichment layer)
 * is flagged, i.e. the 2:1 rule of the triangulation has set a refinement flag on it again.
 * frozen_cells as in limit_coarsening_and_refinement(); empty: nothing to check.
 */
template<int dim>
void
check_frozen_cells_not_flagged(dealii::Triangulation<dim> const & tria,
                               std::vector<bool> const &          frozen_cells)
{
  bool frozen_cells_flagged = false;

  if(not frozen_cells.empty())
  {
    for(auto const & cell : tria.active_cell_iterators())
    {
      if(cell->is_locally_owned() and frozen_cells[cell->active_cell_index()] and (cell->refine_flag_set() or cell->coarsen_flag_set()))
      {
        frozen_cells_flagged = true;
      }
    }
  }

  frozen_cells_flagged = dealii::Utilities::MPI::logical_or(frozen_cells_flagged, tria.get_communicator());

  AssertThrow(not frozen_cells_flagged,
              dealii::ExcMessage("Adaptive mesh refinement: a cell of the wall enrihched layer is flagged for refinement or coarsening after prepare_coarsening_and_refinement()"));
}

/**
 * Loop over cells and check for refinement/coarsening flags set.
 */
template<int dim>
bool
any_cells_flagged_for_coarsening_or_refinement(dealii::Triangulation<dim> const & tria)
{
  bool any_flag_set = false;

  for(auto & cell : tria.active_cell_iterators())
  {
    if(cell->is_locally_owned())
    {
      if(cell->refine_flag_set() or cell->coarsen_flag_set())
      {
        any_flag_set = true;
        break;
      }
    }
  }

  any_flag_set = dealii::Utilities::MPI::logical_or(any_flag_set, tria.get_communicator());

  return any_flag_set;
}

template<int dim, typename Number, typename VectorType>
dealii::Vector<float>
compute_kelly_indicator(
  dealii::DoFHandler<dim> const & dof_handler,
  dealii::AffineConstraints<Number> const & constraints,
  dealii::Mapping<dim> const & mapping,
  VectorType const & solution,
  unsigned int const n_face_quadrature_points)
{
  VectorType locally_relevant_solution;
  locally_relevant_solution.reinit(dof_handler.locally_owned_dofs(),
                                   dealii::DoFTools::extract_locally_relevant_dofs(dof_handler),
                                   dof_handler.get_communicator());
  locally_relevant_solution.copy_locally_owned_data_from(solution);
  constraints.distribute(locally_relevant_solution);
  locally_relevant_solution.update_ghost_values();

  dealii::QGauss<dim - 1> face_quadrature(n_face_quadrature_points);

  dealii::Vector<float> indicator(dof_handler.get_triangulation().n_active_cells());

  dealii::KellyErrorEstimator<dim>::estimate(mapping,
                                             dof_handler,
                                             face_quadrature,
                                             {}, // no Neumann boundary contributions
                                             locally_relevant_solution,
                                             indicator);

  return indicator;
}

/**
   * Divide an indicator by its global maximum (over the locally owned cells of all MPI ranks), so
   * that its values lie in [0, 1]. An indicator that is zero everywhere stays zero.
   */
inline void
normalize_indicator(dealii::Vector<float> & indicator, MPI_Comm const & mpi_comm)
{
  float const local_maximum  = indicator.linfty_norm();
  float const global_maximum = dealii::Utilities::MPI::max(local_maximum, mpi_comm);

  if(global_maximum > 0.0f)
    indicator /= global_maximum;
}

/**
 * Combine the indicators of several fields (velocity, transport variables) into one indicator per
 * cell, see IndicatorCombination: each field is normalized by its global maximum, multiplied by
 * its weight, and then the maximum or the sum over the fields is taken per cell.
 *
 * All indicators have to belong to the same triangulation (one entry per active cell, see
 * compute_kelly_indicator()). weights: one entry per field, or empty for weight 1 for all fields.
 */
inline dealii::Vector<float>
combine_indicators(std::vector<dealii::Vector<float>> indicators,
                   std::vector<double> const &        weights,
                   IndicatorCombination const         combination,
                   MPI_Comm const &                   mpi_comm)
{
  AssertThrow(not indicators.empty(),
              dealii::ExcMessage("combine_indicators(): no indicator given."));
  AssertThrow(weights.empty() or weights.size() == indicators.size(),
              dealii::ExcMessage("combine_indicators(): the number of weights has to be equal to "
                                 "the number of indicators (or zero for weight 1)."));
  if(indicators.size() == 1)
  {
    double const weight = weights.empty() ? 1.0 : weights[0];
    AssertThrow(weight >= 0.0,
                dealii::ExcMessage("combine_indicators(): the weights must not be negative."));

    if(weight > 0.0)
      return indicators[0];
    else
      return dealii::Vector<float>(indicators[0].size());
  }

  dealii::Vector<float> combined(indicators[0].size());

  for(unsigned int field = 0; field < indicators.size(); ++field)
  {
    AssertThrow(indicators[field].size() == combined.size(),
                dealii::ExcMessage("combine_indicators(): the indicators belong to different "
                                   "triangulations."));

    double const weight = weights.empty() ? 1.0 : weights[field];
    AssertThrow(weight >= 0.0,
                dealii::ExcMessage("combine_indicators(): the weights must not be negative."));

    normalize_indicator(indicators[field], mpi_comm);

    for(unsigned int cell = 0; cell < combined.size(); ++cell)
    {
      float const weighted_indicator = static_cast<float>(weight) * indicators[field][cell];

      if(combination == IndicatorCombination::Maximum)
        combined[cell] = std::max(combined[cell], weighted_indicator);
      else if(combination == IndicatorCombination::WeightedSum)
        combined[cell] += weighted_indicator;
      else
        AssertThrow(false, dealii::ExcMessage("Unknown IndicatorCombination."));
    }
  }

  return combined;
}

/**
   * Flag cells for refinement and coarsening from an indicator (one entry per active cell, see
   * compute_kelly_indicator() and combine_indicators()), with the marking rule of amr_data:
   *
   *   FixedNumber:   refine the fraction_of_cells_to_be_refined of the cells with the largest
   *                  indicator, coarsen the fraction_of_cells_to_be_coarsened of the cells with the
   *                  smallest indicator
   *   FixedFraction: refine the cells with the largest indicator that together make up the
   *                  fraction_of_cells_to_be_refined of the total error sum_K eta_K^2, coarsen the
   *                  cells with the smallest indicator that make up the
   *                  fraction_of_cells_to_be_coarsened of the total error
   *
   * The limits (minimum/maximum level, boundary cells) are applied afterwards by
   * limit_coarsening_and_refinement().
   */
template<int dim>
void
mark_cells(dealii::Triangulation<dim> &       tria,
           dealii::Vector<float> const &      indicator,
           AdaptiveMeshRefinementData const & amr_data)
{
  if(amr_data.marking_strategy == MarkingStrategy::FixedNumber)
  {
    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      tria,
      indicator,
      amr_data.fraction_of_cells_to_be_refined,
      amr_data.fraction_of_cells_to_be_coarsened);
  }
  else if(amr_data.marking_strategy == MarkingStrategy::FixedFraction)
  {
    // L2 norm: the fractions refer to sum_K eta_K^2, the Kelly estimate of the squared error
    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      tria,
      indicator,
      amr_data.fraction_of_cells_to_be_refined,
      amr_data.fraction_of_cells_to_be_coarsened,
      dealii::VectorTools::L2_norm);
  }
  else
{
    AssertThrow(false, dealii::ExcMessage("Unknown MarkingStrategy."));
  }
}

template<int dim, typename Number, typename VectorType>
void
mark_cells_kelly_error_estimator(dealii::Triangulation<dim> &              tria,
                                 dealii::DoFHandler<dim> const &           dof_handler,
                                 dealii::AffineConstraints<Number> const & constraints,
                                 dealii::Mapping<dim> const &              mapping,
                                 VectorType const &                        solution,
                                 unsigned int const                        n_face_quadrature_points,
                                 AdaptiveMeshRefinementData const &        amr_data)
{
  dealii::Vector<float> const indicator = compute_kelly_indicator(dof_handler,
                                                                  constraints,
                                                                  mapping,
                                                                  solution,
                                                                  n_face_quadrature_points);
  mark_cells(tria, indicator, amr_data);
}


} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_ADAPTIVE_MESH_REFINEMENT_H_ */
