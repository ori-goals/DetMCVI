// Provides an interface for a POMDP implementation

/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>

#include "StateVector.h"

namespace DetMCVI {

class SimInterface {
 private:
  /* data */
 public:
  SimInterface() {};
  virtual ~SimInterface() {};

  // ------- obligatory functions ----------
  virtual std::tuple<State, int64_t, double, bool> Step(
      const State& sI,
      int64_t aI) const = 0;  // sI_next, oI, Reward, Done
  virtual State SampleStartState() = 0;
  virtual int64_t GetSizeOfObs() const = 0;
  virtual int64_t GetSizeOfA() const = 0;
  virtual double GetDiscount() const = 0;
  virtual int64_t GetNbAgent() const = 0;
  virtual bool isTerminal(const State& sI) const = 0;

  virtual double applyActionToState(const DetMCVI::State& state, int64_t action,
                                    DetMCVI::State& sNext) const = 0;

  // --------------------------------------------------------

  virtual std::optional<double> GetHeuristicUpper(
      const std::vector<std::pair<DetMCVI::State, double>>& /*belief*/,
      int64_t /*max_depth*/) const {
    return std::nullopt;
  }
  virtual std::optional<double> GetHeuristicLower(
      const std::vector<std::pair<DetMCVI::State, double>>& /*belief*/,
      int64_t /*max_depth*/) const {
    return std::nullopt;
  }

  virtual StateMap<double> TrueInitBelief() const { return StateMap<double>(); }
};

}  // namespace DetMCVI
