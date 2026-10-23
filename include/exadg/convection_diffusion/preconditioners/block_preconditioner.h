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

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_PRECONDITIONERS_BLOCK_PRECONDITIONER_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_PRECONDITIONERS_BLOCK_PRECONDITIONER_H_

// C/C++
#include <memory>
#include <string>
#include <vector>

// deal.II
#include <deal.II/lac/la_parallel_block_vector.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>
#include <exadg/utilities/timer_tree.h>


namespace ExaDG
{
namespace ConvDiff
{

template<typename Number>
class BlockPreconditioner
{
public:
  typedef dealii::LinearAlgebra::distributed::BlockVector<Number>
  BlockVectorType;

  void
  initialize(std::vector<std::shared_ptr<PreconditionerBase<Number>>> const & preconditioner_in)
  {
    preconditioner = preconditioner_in;
  }

  bool
  needs_update() const
  {
    for(auto const & preconditioner_block : preconditioner)
    if(preconditioner_block and preconditioner_block->needs_update())
      return true;

    return false;
  }

  void
  update()
  {
    for(auto const & preconditioner_block : preconditioner)
    if(preconditioner_block)
      preconditioner_block->update();
  }

  void
  vmult(BlockVectorType & dst, BlockVectorType const & src) const
  {
    AssertDimension(src.n_blocks(), preconditioner.size());

    for(unsigned int b = 0; b < preconditioner.size(); ++b)
      preconditioner[b]->vmult(dst.block(b), src.block(b));
  }

  std::shared_ptr<TimerTree>
  get_timings() const
  {
    std::shared_ptr<TimerTree> timings = std::make_shared<TimerTree>();

    for(unsigned int b = 0; b < preconditioner.size(); ++b)
      if(preconditioner[b])
        timings->insert({"Block " + std::to_string(b)},
                        preconditioner[b]->get_timings());

    return timings;
  }

private:
  std::vector<std::shared_ptr<PreconditionerBase<Number>>> preconditioner;
};

  } // namespace ConvDiff
  } // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_PRECONDITIONERS_BLOCK_PRECONDITIONER_H_ */
