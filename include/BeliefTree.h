/* Implements a belief tree class
The belief tree consists of BeliefTreeNodes, which represent the
beliefs in the tree. Each belief node has a set of action edges, which
lead to action nodes. Each action node has a set of observation edges,
which lead to observation nodes.
Each observation node is the parent of exactly one belief node.
*/

/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <memory>
#include <unordered_map>

#include "AlphaVectorFSC.h"
#include "BeliefDistribution.h"
#include "Bound.h"

namespace DetMCVI {

class BeliefTreeNode;

class ObservationNode {
 private:
  double _observation;
  double _weight;
  double _sum_reward;
  double _upper_bound;
  double _lower_bound;
  int64_t _best_policy_node;
  double _best_policy_val;
  std::shared_ptr<BeliefTreeNode> _next_belief;

 public:
  ObservationNode(int64_t observation, double weight, double sum_reward,
                  std::shared_ptr<BeliefTreeNode> next_belief,
                  double next_upper, double next_lower)
      : _observation(observation),
        _weight(weight),
        _sum_reward(sum_reward),
        _upper_bound(sum_reward + next_upper),
        _lower_bound(sum_reward + next_lower),
        _best_policy_node(-1),
        _best_policy_val(-std::numeric_limits<double>::infinity()),
        _next_belief(next_belief) {}

  int64_t GetObs() const { return _observation; }

  int64_t GetBestPolicyNode() const { return _best_policy_node; }

  /// @brief Perform a backup of the observation node using both the child
  /// belief and the fsc
  void BackUp(AlphaVectorFSC& fsc, double R_lower, int64_t max_depth_sim,
              SimInterface* pomdp);

  /// @brief Perform a backup of the observation node using only the bounds of
  /// its child belief
  void BackUpFromNextBelief();

  /// @brief Perform a backup of the observation node using only the given fsc
  void BackUpFromPolicyGraph(AlphaVectorFSC& fsc, double R_lower,
                             int64_t max_depth_sim, SimInterface* pomdp);

  double GetWeight() const { return _weight; }
  double GetUpper() const { return _upper_bound; }
  double GetLower() const { return _lower_bound; }

  std::shared_ptr<BeliefTreeNode> GetBelief() const { return _next_belief; }
};

class ActionNode {
 private:
  int64_t _action;              // Action of this node
  double _avgUpper, _avgLower;  // Bounds based on observation children
  mutable bool _closed;
  std::vector<ObservationNode> _observation_edges;

 public:
  ActionNode(int64_t action, const BeliefStates& belief, int64_t belief_depth,
             const BoundFunction& upper_bound_func, int64_t eval_depth,
             const BoundFunction& lower_bound_func, SimInterface* pomdp);

  int64_t GetAction() const { return _action; }

  double GetAvgUpper() const { return _avgUpper; }
  double GetAvgLower() const { return _avgLower; }

  std::shared_ptr<BeliefTreeNode> GetChild(int64_t observation) const;
  const std::vector<ObservationNode>& GetChildren() const {
    return _observation_edges;
  }

  /// @brief Choose the next observation based on the largest excess uncertainty
  /// as per Smith, T. and Simmons, R. (2004) ‘Heuristic Search Value Iteration
  /// for POMDPs’.
  std::pair<std::shared_ptr<BeliefTreeNode>, double> ChooseObservation(
      double epsilon, double gamma, int64_t depth) const;

  /// @brief Perform a backup of the action node. Calls BackUp on each child
  /// observation node
  void BackUp(AlphaVectorFSC& fsc, double R_lower, int64_t max_depth_sim,
              SimInterface* pomdp);

  /// @brief Perform a backup of the action node using only the bounds of its
  /// children
  void BackUpNoFSC();

  void Close() { _closed = true; }
  bool IsClosed() const { return _closed; }

 private:
  /// @brief Generate a set of next beliefs mapped by observation,
  /// obtained by taking `action` in belief.
  void BeliefUpdate(const BeliefStates& belief, int64_t belief_depth,
                    const BoundFunction& upper_bound_func, int64_t eval_depth,
                    const BoundFunction& lower_bound_func, SimInterface* pomdp);

  void CalculateBounds();
};

static int64_t belief_tree_count = 0;

class BeliefTreeNode {
 private:
  BeliefStates _belief;
  std::vector<ActionNode> _action_edges;
  int64_t _belief_depth;

  int64_t _bestActUBound, _bestActLBound;
  int64_t _best_policy_node;

  double _upper_bound;
  double _lower_bound;

  mutable bool _closed;

  int64_t _index;

 public:
  BeliefTreeNode(const BeliefStates& belief, double belief_depth,
                 double upper_bound, double lower_bound)
      : _belief(belief),
        _belief_depth(belief_depth),
        _bestActUBound(-1),
        _bestActLBound(-1),
        _best_policy_node(-1),
        _upper_bound(upper_bound),
        _lower_bound(lower_bound),
        _closed(false),
        _index(belief_tree_count++) {}

  /// @brief Add a child action node to this belief node.
  void AddChild(int64_t action, const BoundFunction& upper_bound_func,
                int64_t eval_depth, const BoundFunction& lower_bound_func,
                SimInterface* pomdp);
  /// @brief Get or add a child action node to this belief node.
  const ActionNode& GetOrAddChildren(int64_t action,
                                     const BoundFunction& upper_bound_func,
                                     int64_t eval_depth,
                                     const BoundFunction& lower_bound_func,
                                     SimInterface* pomdp);

  const BeliefStates& GetBelief() const { return _belief; }

  void SetBestPolicyNode(int64_t idx) { _best_policy_node = idx; }
  int64_t GetBestPolicyNode() const { return _best_policy_node; }

  void UpdateBestAction();
  int64_t GetBestActLBound() const { return _bestActLBound; }
  int64_t GetBestActUBound() const { return _bestActUBound; }

  double GetUpper() const { return _upper_bound; }
  double GetLower() const { return _lower_bound; }
  int64_t GetDepth() const { return _belief_depth; }

  /// @brief Get the child belief node corresponding to the given action and
  /// observation
  std::shared_ptr<BeliefTreeNode> GetChild(int64_t action,
                                           int64_t observation) const;
  /// @brief Get the child obsevation nodes corresponding to the given action
  const std::vector<ObservationNode>& GetChildren(int64_t action) const;

  /// @brief Choose the next observation based on the largest excess uncertainty
  std::pair<std::shared_ptr<BeliefTreeNode>, double> ChooseObservation(
      double epsilon, double gamma);

  /// @brief Perform a backup of the belief node. Calls BackUp on each child
  /// action node
  void BackUpActions(AlphaVectorFSC& fsc, double R_lower, int64_t max_depth_sim,
                     SimInterface* pomdp);

  void BackUpBestActionUpperNoFSC();
  void BackUpBestActionLowerNoFSC();

  int64_t GetId() const { return _index; }

  void DrawBeliefTree(std::ostream& ofs) const;

  int64_t DrawPolicyTree(std::ostream& ofs) const;

  void Close() { _closed = true; }
  bool IsClosed() const { return _closed; }

 private:
  void GenerateGraphviz(std::ostream& out) const;

  void DrawPolicyBranch(std::ostream& ofs, int64_t& i) const;
};

/// @brief Create a new belief tree node with the given belief, depth, and
/// bounds
std::shared_ptr<BeliefTreeNode> CreateBeliefTreeNode(const BeliefStates& belief,
                                                     int64_t belief_depth,
                                                     double upper_bound,
                                                     double lower_bound);

}  // namespace DetMCVI
