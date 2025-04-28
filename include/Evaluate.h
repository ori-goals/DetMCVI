// Provides functions for evaluating policies and storing/printing the results

/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <iostream>
#include <sstream>

#include "BeliefDistribution.h"
#include "BeliefTree.h"
#include "SimInterface.h"

namespace DetMCVI {

class Welford {
 private:
  size_t n;
  double mean;
  double M2;
  double max_val;
  double min_val;

 public:
  Welford()
      : n(0),
        mean(0.0),
        M2(0.0),
        max_val(-std::numeric_limits<double>::infinity()),
        min_val(std::numeric_limits<double>::infinity()) {}

  void update(double x) {
    n++;
    double delta = x - mean;
    mean += delta / n;
    double delta2 = x - mean;
    M2 += delta * delta2;

    // Update max and min values
    if (x > max_val) {
      max_val = x;
    }
    if (x < min_val) {
      min_val = x;
    }
  }

  double getCount() const { return n; }

  double getMean() const { return mean; }

  double getVariance() const {
    if (n < 2)
      return 0.0;
    else
      return M2 / (n - 1);
  }

  double getMax() const { return max_val; }

  double getMin() const { return min_val; }
};

typedef struct {
  Welford complete;
  Welford off_policy;
  Welford max_depth;
  Welford no_solution_on_policy;
  Welford no_solution_off_policy;
} EvaluationStats;

typedef enum TrialTermination {
  INVALID = 0,
  GOAL_REACHED,
  NO_OBSERVATION,
  MAX_DEPTH,
} TrialTermination;

using TrialResult = std::tuple<double, double, TrialTermination, bool,
                               double>;  // return, oracle, exit reason, has
                                         // solution, reward_to_go

void StatsCSVHeader(std::ostream& os);

int64_t EvaluationWithSimulationFSC(
    int64_t max_steps, int64_t num_sims, int64_t init_belief_samples,
    std::optional<StateValueFunction> valFunc, SimInterface* pomdp,
    std::mt19937_64& rng, const AlphaVectorFSC& fsc, const OptimalPath& solver,
    const std::string& alg_name, double timestamp, int64_t set_number,
    int64_t seed, int64_t policy_nodes, std::ostream& os);

int64_t EvaluationWithGreedyTreePolicy(
    std::shared_ptr<BeliefTreeNode> root, int64_t max_steps, int64_t num_sims,
    int64_t init_belief_samples, SimInterface* pomdp, std::mt19937_64& rng,
    const OptimalPath& solver, std::optional<StateValueFunction> valFunc,
    const std::string& alg_name, double timestamp, int64_t set_number,
    int64_t seed, int64_t policy_nodes, std::ostream& os);

void RunCommonEvaluation(
    std::vector<State> eval_states, SimInterface* pomdp, int64_t max_depth,
    const OptimalPath& solver, std::optional<StateValueFunction> valFunc,
    const AlphaVectorFSC& mcvi_fsc,
    const std::shared_ptr<BeliefTreeNode> aostar_root,
    const std::shared_ptr<BeliefTreeNode> anytime_aostar_root,
    const std::shared_ptr<BeliefTreeNode> qmdp_root,
    const AlphaVectorFSC& sarsop_fsc, std::ostream& os);

std::string generateSARSOPFilename(const std::string& problem, int N, int i);

}  // namespace DetMCVI
