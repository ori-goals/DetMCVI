/* Implement the Anytime AO* algorithm for POMDPs as per Bonet, B. and Geffner,
 * H. (2021) ‘Action Selection for MDPs: Anytime AO* Versus UCT’, Proceedings of
 * the AAAI Conference on Artificial Intelligence, 26(1), pp. 1749–1755.
 * Available at: https://doi.org/10.1609/aaai.v26i1.8369.
 */

/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include "AOStar.h"
#include "BeliefTree.h"
#include "DetMCVI.h"

namespace DetMCVI {

BeliefTreeNodeHistory ChooseNodeWithProbability(
    std::vector<BeliefTreeNodeHistory>& fringe,
    const std::vector<std::shared_ptr<BeliefTreeNode>>& graph,
    std::mt19937_64& rng, double p, AOSearchType method) {
  if (fringe.empty()) return {nullptr, {}};
  static std::uniform_real_distribution<double> distribution(0.0, 1.0);

  // Create a vector containing nodes that exist in both fringe and graph if
  // in_graph is true, or not in graph if in_graph is false
  std::vector<BeliefTreeNodeHistory> in_graph_nodes;
  std::vector<BeliefTreeNodeHistory> not_in_graph_nodes;
  for (const auto& node_ptr : fringe) {
    if (std::find(graph.begin(), graph.end(), node_ptr.first) != graph.end()) {
      in_graph_nodes.push_back(node_ptr);
    } else {
      not_in_graph_nodes.push_back(node_ptr);
    }
  }

  const bool in_graph = distribution(rng) >= p;
  std::vector<BeliefTreeNodeHistory>& common_nodes = not_in_graph_nodes;
  if ((in_graph && !in_graph_nodes.empty()) || not_in_graph_nodes.empty())
    common_nodes = in_graph_nodes;
  if (common_nodes.empty()) return {nullptr, {}};

  switch (method) {
    case AOSearchType::random: {
      std::uniform_int_distribution<size_t> distribution(
          0, common_nodes.size() - 1);
      return common_nodes.at(distribution(rng));
    }
    case AOSearchType::depth_first:
      return *std::max_element(common_nodes.begin(), common_nodes.end(),
                               CmpNodeDepth);
    case AOSearchType::breadth_first:
      return *std::min_element(common_nodes.begin(), common_nodes.end(),
                               CmpNodeDepth);
    default:
      return common_nodes.at(0);
  }
  return common_nodes.at(0);
}

void AnytimeAOStarIter(std::vector<std::shared_ptr<BeliefTreeNode>>& graph,
                       std::vector<BeliefTreeNodeHistory>& fringe,
                       const OptimalPath& /*heuristic*/, int64_t eval_depth,
                       std::mt19937_64& rng, double p, SimInterface* pomdp,
                       AOSearchType method) {
  const auto [belief_node, history] =
      ChooseNodeWithProbability(fringe, graph, rng, p, method);

  // remove node from fringe set
  auto it = std::find(fringe.begin(), fringe.end(),
                      std::make_pair(belief_node, history));
  if (it != fringe.end())
    fringe.erase(it);
  else
    throw std::logic_error("Cannot find fringe node to erase");

  auto upperBoundFunc = [&rng](const BeliefStates& belief,
                               int64_t /*belief_depth*/, int64_t eval_depth,
                               SimInterface* sim) {
    return FindRLower(sim, belief, 1e-6, eval_depth, rng);
  };

  // expand node
  auto new_history = history;
  new_history.push_back(belief_node);
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    const auto actNode = belief_node->GetOrAddChildren(
        a, upperBoundFunc, eval_depth, InfBoundFunc, pomdp);
    for (const auto& obsNode : actNode.GetChildren())
      fringe.push_back({obsNode.GetBelief(), new_history});
  }

  // back up the graph
  for (auto it = new_history.rbegin(); it < new_history.rend(); ++it) {
    (*it)->BackUpBestActionUpperNoFSC();
    (*it)->UpdateBestAction();
  }

  // rebuild the the graph
  graph = {graph.at(0)};
  size_t i = 0;
  while (i < graph.size()) {
    auto node = graph.at(i);
    if (node->GetBestActUBound() != -1) {
      for (const auto& obsNode : node->GetChildren(node->GetBestActUBound()))
        graph.push_back(obsNode.GetBelief());
    }
    ++i;
  }
}

void RunAnytimeAOStar(std::shared_ptr<BeliefTreeNode> initial_belief,
                      int64_t max_iter, int64_t max_computation_ms,
                      const OptimalPath& heuristic, int64_t eval_depth,
                      std::mt19937_64& rng, double p, SimInterface* pomdp,
                      AOSearchType method = AOSearchType::random) {
  std::vector<BeliefTreeNodeHistory> fringe = {{initial_belief, {}}};
  std::vector<std::shared_ptr<BeliefTreeNode>> graph = {initial_belief};

  const auto ao_start = std::chrono::steady_clock::now();
  int64_t iter = 0;
  while (++iter <= max_iter && fringe.size() > 0) {
    AnytimeAOStarIter(graph, fringe, heuristic, eval_depth, rng, p, pomdp,
                      method);
    if (AOStarTimeExpired(ao_start, max_computation_ms)) {
      std::cout
          << "Anytime AO* planning complete, reached maximum computation time."
          << std::endl;
      return;
    }
  }
  std::cout << "Anytime AO* planning complete, reached maximum iterations."
            << std::endl;
}

void RunAnytimeAOStarAndEvaluate(
    std::shared_ptr<BeliefTreeNode> initial_belief, int64_t max_iter,
    int64_t max_computation_ms, const OptimalPath& heuristic,
    int64_t eval_depth, int64_t max_eval_steps, int64_t n_eval_trials,
    int64_t nb_particles_b0, int64_t eval_interval_ms,
    int64_t completion_threshold, int64_t completion_reps, int64_t node_limit,
    std::mt19937_64& rng, const OptimalPath& solver,
    std::optional<StateValueFunction> valFunc, double p, SimInterface* pomdp,
    int64_t set_number, int64_t seed, std::ostream& os,
    AOSearchType method = AOSearchType::random) {
  std::vector<std::pair<std::shared_ptr<BeliefTreeNode>,
                        std::vector<std::shared_ptr<BeliefTreeNode>>>>
      fringe = {{initial_belief, {}}};
  std::vector<std::shared_ptr<BeliefTreeNode>> graph = {initial_belief};

  int64_t iter = 0;
  int64_t time_sum = 0;
  int64_t last_eval = -eval_interval_ms * 1000;
  int64_t completed_times = 0;
  while (++iter <= max_iter && fringe.size() > 0) {
    const auto iter_start = std::chrono::steady_clock::now();

    AnytimeAOStarIter(graph, fringe, heuristic, eval_depth, rng, p, pomdp,
                      method);

    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - iter_start);
    time_sum += elapsed.count();

    if (time_sum - last_eval >= eval_interval_ms * 1000) {
      last_eval = time_sum;
      const int64_t completed_count =
          RunAOEvaluation(initial_belief, time_sum, max_eval_steps,
                          n_eval_trials, nb_particles_b0, rng, solver, valFunc,
                          pomdp, set_number, seed, os, true);
      completed_times =
          (completed_count >= completion_threshold) ? completed_times + 1 : 0;
      if (completed_times >= completion_reps) return;

      const std::string policy_tree_file =
          "greedy_policy_tree_" + std::to_string(time_sum) + ".dot";
      std::fstream policy_tree(policy_tree_file, std::fstream::out);
      const int64_t node_count = initial_belief->DrawPolicyTree(policy_tree);
      policy_tree.close();
      std::remove(policy_tree_file.c_str());
      if (node_count >= node_limit) {
        std::cout << "AO* planning complete, reached node limit." << std::endl;
        return;
      }
    }
    if (time_sum >= max_computation_ms * 1000) {
      std::cout << "AO* planning complete, reached computation time."
                << std::endl;
      RunAOEvaluation(initial_belief, time_sum, max_eval_steps, n_eval_trials,
                      nb_particles_b0, rng, solver, valFunc, pomdp, set_number,
                      seed, os, true);
      return;
    }
  }
  std::cout << "AO* planning complete, reached maximum iterations."
            << std::endl;
  RunAOEvaluation(initial_belief, time_sum, max_eval_steps, n_eval_trials,
                  nb_particles_b0, rng, solver, valFunc, pomdp, set_number,
                  seed, os, true);
  return;
}

}  // namespace DetMCVI
