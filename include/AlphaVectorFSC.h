// Define a Finite State Controller

/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <queue>
#include <string>

#include "AlphaVectorNode.h"
#include "Bound.h"
#include "SimInterface.h"

namespace DetMCVI {

class AlphaVectorFSC {
 private:
  std::vector<std::unordered_map<int64_t, int64_t>> _edges;
  std::deque<std::shared_ptr<AlphaVectorNode>> _nodes;
  int64_t _start_node_index;

 public:
  /// @brief Construct an FSC with a maximum number of nodes
  /// @param max_node_size The maximum number of nodes in the FSC (unused)
  AlphaVectorFSC(int64_t max_node_size)
      : _edges(max_node_size, std::unordered_map<int64_t, int64_t>()),
        _nodes(),
        _start_node_index(-1) {}

  /// @brief Return a reference to node number nI
  const std::shared_ptr<AlphaVectorNode> GetNode(int64_t nI) const {
    return _nodes.at(nI);
  }
  std::shared_ptr<AlphaVectorNode> GetNodeModifiable(int64_t nI) {
    return _nodes[nI];
  }

  /// @brief Return the number of nodes in the FSC
  int64_t NumNodes() const { return _nodes.size(); }

  /// @brief Add a node to the FSC
  int64_t AddNode(const int64_t action);

  /// @brief Return the edges associated with node nI
  const std::unordered_map<int64_t, int64_t>& GetEdges(int64_t nI) const;

  /// @brief Return the node index assosciated with node nI and
  /// observation o. Returns -1 if it does not exist.
  int64_t GetEdgeValue(int64_t nI, int64_t o) const;

  /// @brief Set the node index assosciated with node nI and
  /// observation o to nI_new.
  void UpdateEdge(int64_t nI, int64_t o, int64_t nI_new);
  void UpdateEdge(int64_t nI,
                  const std::unordered_map<int64_t, int64_t>& edges);

  int64_t GetStartNodeIndex() const { return _start_node_index; }
  void SetStartNodeIndex(int64_t idx) { _start_node_index = idx; }

  /// @brief Generate a graphviz representation of the FSC
  /// @param ofs Output stream to write the graphviz representation
  /// @param actions Optional mapping of action indices to action names
  /// @param observations Optional mapping of observation indices to observation
  /// names
  void GenerateGraphviz(
      std::ostream& ofs, const std::vector<std::string>& actions = {},
      const std::vector<std::string>& observations = {}) const;

  /// @brief Simulate a trajectory from the start node
  /// @param nI The index of the start node
  /// @param state The initial state
  /// @param max_depth The maximum depth of the simulation
  /// @param R_lower The lower bound for the reward
  /// @param pomdp The POMDP simulator
  /// @return The total reward obtained during the simulation
  double SimulateTrajectory(int64_t nI, const State& state, int64_t max_depth,
                            double R_lower, SimInterface* pomdp);

  /// @brief Get the alpha value for a specific state and node, or initialise it
  /// if it does not exist
  /// @param state The state for which to get the alpha value
  /// @param nI The index of the node
  /// @param R_lower The lower bound for the reward
  /// @param max_depth_sim The maximum depth for the simulation
  /// @param pomdp The POMDP simulator
  /// @return The alpha value for the specified state and node
  double GetNodeAlpha(const State& state, int64_t nI, double R_lower,
                      int64_t max_depth_sim, SimInterface* pomdp);

  /// @brief Calculate the lower value bound for a belief by executing the FSC
  /// from that belief
  /// @param b0 The initial belief
  /// @param belief_depth The depth of the belief
  /// @param R_lower_lim The lower limit for the reward
  /// @param max_depth The maximum depth for the simulation
  /// @param pomdp The POMDP simulator
  double LowerBoundFromFSC(const BeliefStates& b0, int64_t belief_depth,
                           double R_lower_lim, int64_t max_depth,
                           SimInterface* pomdp);
};

/// @brief Read an FSC from a DOT file
AlphaVectorFSC ParseDotFile(const std::string& filename);

}  // namespace DetMCVI
