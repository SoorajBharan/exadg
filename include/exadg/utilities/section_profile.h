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

#ifndef INCLUDE_EXADG_UTILITIES_SECTION_PROFILE_H_
#define INCLUDE_EXADG_UTILITIES_SECTION_PROFILE_H_

#ifdef EXADG_PROFILE

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

// C++
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ExaDG
{

class SectionProfile
{
public:
  using Clock = std::chrono::steady_clock;

  static void
  lap(char const * name, Clock::time_point & t)
  {
    auto const t_now = Clock::now();
    instance().add(name, std::chrono::duration<double>(t_now - t).count());
    t = t_now;
  }

  static void
  print_and_reset(dealii::ConditionalOStream const & pcout,
                  MPI_Comm const &                   comm,
                  unsigned int const                 n_steps)
  {
    auto & entries = instance().entries;
    if(n_steps == 0 or entries.empty())
      return;

    // same order on all ranks
    std::sort(entries.begin(), entries.end(), [](Entry const & a, Entry const &
                                                 b) {
              return a.name < b.name;
              });

    std::ostringstream header;
    header << "  " << std::left << std::setw(46) << "section [ms per step]" << std::right << std::setw(8) << "calls" << std::setw(10) << "max" << std::setw(10) << "ms/call";
    pcout << std::endl << "Sections over the last " << n_steps << " time steps (max over ranks):" << std::endl << header.str() << std::endl;

    double const n = double(n_steps);
    for(auto const & e : entries)
    {
      double const t_max = dealii::Utilities::MPI::max(e.time / n, comm);
      double const calls = double(e.calls) / n;

      std::ostringstream line;
      line << "  " << std::left << std::setw(46) << e.name << std::right << std::fixed << std::setprecision(1) << std::setw(8) << calls << std::setprecision(2) << std::setw(10) << 1e3 * t_max << std::setprecision(3) << std::setw(10) << (calls > 0.0 ? 1e3 * t_max / calls : 0.0);
      pcout << line.str() << std::endl;
    }
    entries.clear();
  }

private:
  struct Entry
  {
    std::string   name;
    double        time;
    unsigned long calls;
  };

  // inline (in-class): one instance for the whole library
  static SectionProfile &
  instance()
  {
    static SectionProfile profile;
    return profile;
  }

  void
  add(char const * name, double const t)
  {
    for(auto & e : entries)
    if(e.name == name)
    {
      e.time += t;
      ++e.calls;
      return;
    }
    entries.push_back(Entry{name, t, 1});
  }

  std::vector<Entry> entries;
};

/*
   * Per-step cost profile of a time integrator. Every part starts with a barrier, 
  so
   * max - min over the ranks is the load imbalance of that part.
   */
struct StepProfile
{
  void
  start(dealii::Timer & timer, MPI_Comm const & comm) const
  {
    MPI_Barrier(comm);
    timer.restart();
  }

  void
  add(std::string const & name, dealii::Timer const & timer)
  {
    double const t = timer.wall_time();
    for(auto & entry : times)
    if(entry.first == name)
    {
      entry.second += t;
      return;
    }
    times.emplace_back(name, t);
  }

  void
  print_and_reset(dealii::ConditionalOStream const & pcout, MPI_Comm const & 
                  comm)
  {
    if(n_steps == 0)
      return;

    double t_full = 0.0, t_parts = 0.0;
    for(auto const & entry : times)
    (entry.first == "full step" ? t_full : t_parts) += entry.second;

    double const n     = double(n_steps);
    double const t_max = dealii::Utilities::MPI::max(t_full / n, comm);

    auto const row = [&](std::string const & name, double const t_local) {
      auto const         r = dealii::Utilities::MPI::min_max_avg(t_local / n,
                                                                 comm);
      std::ostringstream line;
      line << "  " << std::left << std::setw(32) << name << std::right << std::fixed << std::setprecision(2) << std::setw(10) << 1e3 * r.min << std::setw(10) << 1e3 * r.avg << std::setw(10) << 1e3 * r.max << std::setprecision(1) << std::setw(8) << 100.0 * r.max / t_max << std::setw(6) << r.max_index;
      pcout << line.str() << std::endl;
    };

    std::ostringstream header;
    header << "  " << std::left << std::setw(32) << "part [ms per step]" << std::right << std::setw(10) << "min" << std::setw(10) << "avg" << std::setw(10) << "max" << std::setw(8) << "% max" << std::setw(6) << "rank";
    pcout << std::endl << "Profile over the last " << n_steps << " time steps:" << std::endl << header.str() << std::endl;

    for(auto const & entry : times)
    if(entry.first != "full step")
      row(entry.first, entry.second);
    row("rest (postpro, CFL, I/O, wait)", t_full - t_parts);
    row("full step", t_full);

    SectionProfile::print_and_reset(pcout, comm, n_steps);

    times.clear();
    n_steps = 0;
  } 

  std::vector<std::pair<std::string, double>> times;
  dealii::Timer                                cycle_timer;
  unsigned int                                 n_steps       = 0;
  bool                                         cycle_running = false;
};

// constructed on first use (after MPI_Init); inline: one instance for the whole library
inline StepProfile &
step_profile()
{
  static StepProfile profile;
  return profile;
}

} // namespace ExaDG

#endif

#endif /* INCLUDE_EXADG_UTILITIES_SECTION_PROFILE_H_ */
