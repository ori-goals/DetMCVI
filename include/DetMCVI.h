// Implements the DetMCVI algorithm

/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <atomic>
#include <chrono>
#include <iostream>

#include "AlphaVectorFSC.h"
#include "BeliefDistribution.h"
#include "BeliefTree.h"
#include "Bound.h"
#include "Evaluate.h"
#include "SimInterface.h"

namespace DetMCVI {

class MCVIPlanner {
 private:
  SimInterface* _pomdp;
  AlphaVectorFSC _fsc;
  BeliefStates _b0;
  const OptimalPath& _heuristic;
  std::mt19937_64& _rng;

 public:
  MCVIPlanner(SimInterface* pomdp, const AlphaVectorFSC& init_fsc,
              const BeliefStates& init_belief, const OptimalPath& heuristic,
              std::mt19937_64& rng)
      : _pomdp(pomdp),
        _fsc(init_fsc),
        _b0(init_belief),
        _heuristic(heuristic),
        _rng(rng) {}

  /// @brief Run the MCVI planner
  /// @param max_depth_sim Maximum depth to simulate
  /// @param epsilon Threshold for difference between upper and lower bounds
  /// @param max_nb_iter Maximum number of tree traversals
  /// @return The FSC for the pomdp
  std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>> Plan(
      int64_t max_depth_sim, double epsilon, int64_t max_nb_iter,
      int64_t max_computation_ms, int64_t eval_depth, double eval_epsilon,
      std::atomic<bool>& exit_flag);

  // fsc, root node, converged, timed out, reached max iter
  std::tuple<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>, double, bool,
             bool, bool>
  PlanIncrement(std::shared_ptr<BeliefTreeNode> Tr_root, double R_lower,
                int64_t iter, int64_t ms_remaining, int64_t max_depth_sim,
                double epsilon, int64_t max_nb_iter, int64_t eval_depth,
                double eval_epsilon, std::atomic<bool>& exit_flag);

  double MCVIIteration(std::shared_ptr<BeliefTreeNode> Tr_root, double R_lower,
                       int64_t ms_remaining, int64_t max_depth_sim,
                       int64_t eval_depth, double eval_epsilon,
                       std::atomic<bool>& exit_flag);

  // run evaluation after each iteration
  std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>> PlanAndEvaluate(
      int64_t max_depth_sim, double epsilon, int64_t max_nb_iter,
      int64_t max_computation_ms, int64_t eval_depth, double eval_epsilon,
      int64_t max_eval_steps, int64_t n_eval_trials, int64_t nb_particles_b0,
      int64_t eval_interval_ms, int64_t completion_threshold,
      int64_t completion_reps, std::optional<StateValueFunction> valFunc,
      std::atomic<bool>& exit_flag, int64_t set_number, int64_t seed,
      std::ostream& os);

  /// @brief Simulate an FSC execution from the initial belief
  void SimulationWithFSC(int64_t steps) const;

 private:
  int64_t GetFirstAction(std::shared_ptr<BeliefTreeNode> Tr_node,
                         double R_lower, int64_t max_depth_sim,
                         int64_t eval_depth, double eval_epsilon);

  /// @brief Perform a monte-carlo backup on the given belief node
  void BackUp(std::shared_ptr<BeliefTreeNode> Tr_node, double R_lower,
              int64_t max_depth_sim, int64_t eval_depth);

  /// @brief Find a node matching the given node and edges, or insert it if it
  /// does not exist
  int64_t FindOrInsertNode(const int64_t node_act,
                           const std::unordered_map<int64_t, int64_t>& edges);

  /// @brief Insert the given node into the fsc
  int64_t InsertNode(const int64_t node_act,
                     const std::unordered_map<int64_t, int64_t>& edges);

  void SampleBeliefs(
      std::vector<std::shared_ptr<BeliefTreeNode>>& traversal_list,
      int64_t eval_depth, double eval_epsilon, double R_lower,
      int64_t max_depth_sim);
};

}  // namespace DetMCVI
