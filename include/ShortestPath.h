// Implements a shortest path algorithm and a search algorithm

/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

#include "Cache.h"
#include "StateVector.h"

namespace DetMCVI {

/**
 * @brief Implements the Shortest Path Faster Algorithm (SPFA)
 *
 * Inherit from this class to calculate the shortest path from source to a
 * terminal node, according to edges given by `getEdges` and termination
 * according to `isTerminal`.
 */
class ShortestPathFasterAlgorithm {
 public:
  ShortestPathFasterAlgorithm() = default;

  /**
   * @brief Get the edges and weights out of `node`.
   *
   * Returns a list of destination nodes, associated edge rewards and edge
   * number.
   *
   * The edge number is only used to label which edge is taken when
   * reconstructing the path, and can be set arbitrarily.
   */
  virtual std::vector<std::tuple<State, double, int64_t>> getEdges(
      const State& node) const = 0;

  /// @brief Determine if a node is terminal
  virtual bool isTerminal(const State& node) const = 0;

  /**
   * @brief Calculate the maximal reward path between source and the terminal
   * states.
   *
   * Returns a path (not including the source node), a list of actions taken,
   * and the total reward
   */
  std::tuple<std::vector<State>, std::vector<int64_t>, double> FindShortestPath(
      const State& start_state, int64_t max_depth) const;
};

class MaximiseReward {
 private:
  double discount_factor;
  mutable LRUCache<
      State, std::unordered_map<int64_t, std::tuple<int64_t, State, double>>,
      StateHash, StateEqual>
      cache;

  std::pair<double, std::vector<std::tuple<int64_t, State, double>>>
  GetCachedOrSearch(const State& state, int64_t depth_to_go) const;

  std::pair<double, std::vector<std::tuple<int64_t, State, double>>> Search(
      const State& state, int64_t depth_to_go) const;

 public:
  MaximiseReward(double discount_factor, size_t cache_capacity = 25000)
      : discount_factor(discount_factor), cache(cache_capacity) {}

  MaximiseReward(const MaximiseReward&) = delete;
  MaximiseReward& operator=(const MaximiseReward&) = delete;

  // Available successors <action, state, reward> tuples
  virtual std::vector<std::tuple<int64_t, State, double, bool>> getSuccessors(
      const State& state) const = 0;

  // Return the maximum reward that can be obtained starting in `state`
  // up to `max_depth`, alongside the path of <action, next state,
  // immediate_reward> pairs
  std::pair<double, std::vector<std::tuple<int64_t, State, double>>>
  getMaxReward(const State& state, int64_t max_depth) const;
};

}  // namespace DetMCVI
