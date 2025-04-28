#include "BeliefTree.h"

#include <limits>

namespace DetMCVI {

void ObservationNode::BackUp(AlphaVectorFSC& fsc, double R_lower,
                             int64_t max_depth_sim, SimInterface* pomdp) {
  BackUpFromNextBelief();

  BackUpFromPolicyGraph(fsc, R_lower, max_depth_sim, pomdp);
}

void ObservationNode::BackUpFromNextBelief() {
  double nextLower = _next_belief->GetLower() + _sum_reward;
  double nextUpper = _next_belief->GetUpper() + _sum_reward;

  if (nextLower > _lower_bound) {
    _lower_bound = nextLower;
    _best_policy_val = nextLower;
    _best_policy_node = _next_belief->GetBestPolicyNode();
  }

  if (nextUpper < _upper_bound) _upper_bound = nextUpper;
}

ActionNode::ActionNode(int64_t action, const BeliefStates& belief,
                       int64_t belief_depth,
                       const BoundFunction& upper_bound_func,
                       int64_t eval_depth,
                       const BoundFunction& lower_bound_func,
                       SimInterface* pomdp)
    : _action(action), _closed(false) {
  BeliefUpdate(belief, belief_depth, upper_bound_func, eval_depth,
               lower_bound_func, pomdp);
  CalculateBounds();
}

void ActionNode::BeliefUpdate(const BeliefStates& belief, int64_t belief_depth,
                              const BoundFunction& upper_bound_func,
                              int64_t eval_depth,
                              const BoundFunction& lower_bound_func,
                              SimInterface* pomdp) {
  std::unordered_map<int64_t, BeliefDistribution> next_beliefs;
  std::unordered_map<int64_t, double> reward_map;

  std::vector<std::tuple<State, double, double, int64_t>> next_states(
      belief.size(), {{}, 0.0, -std::numeric_limits<double>::infinity(), -1});

  for (size_t i = 0; i < belief.size(); ++i) {
    if (pomdp->isTerminal(belief[i].first)) continue;
    auto [sNext, obs, reward, done] = pomdp->Step(belief[i].first, GetAction());
    next_states[i] = {sNext, belief[i].second, reward, obs};
  }

  for (const auto& [sNext, prob, reward, obs] : next_states) {
    if (sNext.empty() || prob == 0) continue;
    reward_map[obs] += reward * prob;
    next_beliefs[obs][sNext] += prob;
  }

  for (auto& [o, belief] : next_beliefs) {
    double w = 0.0;
    for (const auto& [s, p] : belief) w += p;
    // Renormalize next probabilities
    for (auto& [s, p] : belief) p /= w;
    const auto b = beliefDistributionToStates(belief);

    const auto belief_upper = CalculateUpperBound(
        b, belief_depth + 1, eval_depth, upper_bound_func, pomdp);
    const auto belief_lower = CalculateLowerBound(
        b, belief_depth + 1, eval_depth, lower_bound_func, pomdp);
    const auto belief_node =
        CreateBeliefTreeNode(b, belief_depth + 1, belief_upper, belief_lower);
    const double discounted_reward =
        std::pow(pomdp->GetDiscount(), belief_depth) * reward_map[o] / w;
    _observation_edges.emplace_back(o, w, discounted_reward, belief_node,
                                    belief_node->GetUpper(),
                                    belief_node->GetLower());
    if (eval_depth <= 1 || IsClosed()) belief_node->Close();
  }
}

void ActionNode::CalculateBounds() {
  double upper = 0;
  double lower = 0;

  for (size_t i = 0; i < _observation_edges.size(); ++i) {
    upper +=
        _observation_edges[i].GetUpper() * _observation_edges[i].GetWeight();
    lower +=
        _observation_edges[i].GetLower() * _observation_edges[i].GetWeight();
  }
  _avgUpper = upper;
  _avgLower = lower;
}

std::shared_ptr<BeliefTreeNode> ActionNode::GetChild(
    int64_t observation) const {
  auto it = std::find_if(_observation_edges.begin(), _observation_edges.end(),
                         [&observation](const ObservationNode& o) {
                           return o.GetObs() == observation;
                         });
  if (it == _observation_edges.end()) return nullptr;
  return it->GetBelief();
}

static bool CmpObsPair(const std::pair<int64_t, double>& a,
                       const std::pair<int64_t, double>& b) {
  return a.second < b.second;
}

std::pair<std::shared_ptr<BeliefTreeNode>, double>
ActionNode::ChooseObservation(double epsilon, double gamma,
                              int64_t depth) const {
  std::vector<std::pair<int64_t, double>> obs_diff(
      _observation_edges.size() + 1,
      {-1, -std::numeric_limits<double>::infinity()});  // obs, weighted_diff

  for (size_t i = 0; i < _observation_edges.size(); ++i) {
    if (_observation_edges[i].GetBelief()->IsClosed()) continue;
    const double diff =
        (_observation_edges[i].GetUpper() - _observation_edges[i].GetLower()) -
        epsilon * std::pow(gamma, -depth);
    const double weighted_diff = diff * _observation_edges[i].GetWeight();
    obs_diff[i] = {_observation_edges[i].GetObs(), weighted_diff};
  }

  const auto [best_obs, excess_uncertainty] =
      *std::max_element(obs_diff.begin(), obs_diff.end(), CmpObsPair);
  if (best_obs == -1) throw std::logic_error("Failed to find best observation");
  return {GetChild(best_obs), excess_uncertainty};
}

void ActionNode::BackUp(AlphaVectorFSC& fsc, double R_lower,
                        int64_t max_depth_sim, SimInterface* pomdp) {
  bool all_closed = true;
  for (size_t i = 0; i < _observation_edges.size(); ++i) {
    _observation_edges[i].BackUp(fsc, R_lower, max_depth_sim, pomdp);
    if (!_observation_edges[i].GetBelief()->IsClosed()) all_closed = false;
  }
  CalculateBounds();

  if (all_closed) Close();
}

void ActionNode::BackUpNoFSC() {
  for (size_t i = 0; i < _observation_edges.size(); ++i)
    _observation_edges[i].BackUpFromNextBelief();

  CalculateBounds();
}

void BeliefTreeNode::AddChild(int64_t action,
                              const BoundFunction& upper_bound_func,
                              int64_t eval_depth,
                              const BoundFunction& lower_bound_func,
                              SimInterface* pomdp) {
  _action_edges.emplace_back(action, GetBelief(), _belief_depth,
                             upper_bound_func, eval_depth, lower_bound_func,
                             pomdp);
  if (eval_depth <= 0 || IsClosed()) _action_edges.back().Close();
}

void BeliefTreeNode::UpdateBestAction() {
  if (_action_edges.size() < 1) return;
  bool all_closed = true;

  // find best bounds at the belief
  _lower_bound = -std::numeric_limits<double>::infinity();
  _upper_bound = -std::numeric_limits<double>::infinity();
  _bestActLBound = -1;
  _bestActUBound = -1;

  for (const auto& actNode : _action_edges) {
    if (_lower_bound <= actNode.GetAvgLower()) {
      _lower_bound = actNode.GetAvgLower();
      _bestActLBound = actNode.GetAction();
    }
    if (_upper_bound <= actNode.GetAvgUpper()) {
      _upper_bound = actNode.GetAvgUpper();
      _bestActUBound = actNode.GetAction();
    }
    if (!actNode.IsClosed()) all_closed = false;
  }

  if (all_closed) Close();
}

std::shared_ptr<BeliefTreeNode> BeliefTreeNode::GetChild(
    int64_t action, int64_t observation) const {
  auto it = std::find_if(
      _action_edges.begin(), _action_edges.end(),
      [&action](const ActionNode& a) { return a.GetAction() == action; });
  if (it == _action_edges.cend()) return nullptr;
  return it->GetChild(observation);
}

const std::vector<ObservationNode>& BeliefTreeNode::GetChildren(
    int64_t action) const {
  auto it = std::find_if(
      _action_edges.begin(), _action_edges.end(),
      [&action](const ActionNode& a) { return a.GetAction() == action; });
  if (it == _action_edges.cend())
    throw std::logic_error("No observation nodes for action " +
                           std::to_string(action));
  return it->GetChildren();
}

std::pair<std::shared_ptr<BeliefTreeNode>, double>
BeliefTreeNode::ChooseObservation(double epsilon, double gamma) {
  int64_t action = _bestActUBound;
  auto it = std::find_if(
      _action_edges.begin(), _action_edges.end(),
      [&action](const ActionNode& a) { return a.GetAction() == action; });
  if (it == _action_edges.cend()) UpdateBestAction();
  it = std::find_if(
      _action_edges.begin(), _action_edges.end(),
      [&action](const ActionNode& a) { return a.GetAction() == action; });
  if (it == _action_edges.cend())
    throw std::logic_error("Could not find best action");

  // If action is closed, pick another one
  if (it->IsClosed()) {
    std::cout << "Action " << action << " is closed" << std::endl;
    for (const auto& actNode : _action_edges) {
      if (!actNode.IsClosed()) {
        it = _action_edges.begin() + action;
        break;
      }
    }
  }

  return it->ChooseObservation(epsilon, gamma, _belief_depth + 1);
}

const ActionNode& BeliefTreeNode::GetOrAddChildren(
    int64_t action, const BoundFunction& upper_bound_func, int64_t eval_depth,
    const BoundFunction& lower_bound_func, SimInterface* pomdp) {
  const auto it = std::find_if(
      _action_edges.begin(), _action_edges.end(),
      [&action](const ActionNode& a) { return a.GetAction() == action; });
  if (it != _action_edges.cend()) return *it;
  AddChild(action, upper_bound_func, eval_depth, lower_bound_func, pomdp);
  UpdateBestAction();
  return _action_edges.at(action);
}

void BeliefTreeNode::BackUpBestActionUpperNoFSC() {
  _action_edges.at(_bestActUBound).BackUpNoFSC();
  UpdateBestAction();
}

void BeliefTreeNode::BackUpBestActionLowerNoFSC() {
  _action_edges.at(_bestActLBound).BackUpNoFSC();
  UpdateBestAction();
}

void BeliefTreeNode::BackUpActions(AlphaVectorFSC& fsc, double R_lower,
                                   int64_t max_depth_sim, SimInterface* pomdp) {
  for (size_t i = 0; i < _action_edges.size(); ++i)
    _action_edges[i].BackUp(fsc, R_lower, max_depth_sim, pomdp);
  UpdateBestAction();
}

void ObservationNode::BackUpFromPolicyGraph(AlphaVectorFSC& fsc, double R_lower,
                                            int64_t max_depth_sim,
                                            SimInterface* pomdp) {
  for (int64_t nI = 0; nI < fsc.NumNodes(); ++nI) {
    double node_policy_value_sum = 0.0;

    const auto& b = _next_belief->GetBelief();

    for (size_t i = 0; i < b.size(); ++i) {
      const double V_nI_sNext =
          fsc.GetNodeAlpha(b[i].first, nI, R_lower, max_depth_sim, pomdp);
      node_policy_value_sum += V_nI_sNext * b[i].second;
    }
    node_policy_value_sum *=
        std::pow(pomdp->GetDiscount(), _next_belief->GetDepth() - 1);
    node_policy_value_sum += _sum_reward;

    if (node_policy_value_sum > _best_policy_val) {
      _best_policy_val = node_policy_value_sum;
      _best_policy_node = nI;
      _lower_bound = node_policy_value_sum;
    }
  }
}

std::shared_ptr<BeliefTreeNode> CreateBeliefTreeNode(const BeliefStates& belief,
                                                     int64_t belief_depth,
                                                     double upper_bound,
                                                     double lower_bound) {
  const auto node = std::make_shared<BeliefTreeNode>(belief, belief_depth,
                                                     upper_bound, lower_bound);
  return node;
}

void BeliefTreeNode::GenerateGraphviz(std::ostream& out) const {
  out << "tr" << GetId() << " [label=<<B>" << _belief << "</B><BR/>"
      << "BestPolicyNode: " << GetBestPolicyNode() << "<BR/>"
      << "BestActLBound: " << GetBestActLBound() << "<BR/>"
      << "BestActUBound: " << GetBestActUBound() << "<BR/>"
      << "UpperBound: " << GetUpper() << "<BR/>"
      << "LowerBound: " << GetLower() << ">];" << std::endl;

  for (const auto& actNode : _action_edges) {
    out << "tr" << GetId() << "_" << actNode.GetAction()
        << " [shape=point, style=filled, fillcolor=black];" << std::endl;
    out << "tr" << GetId() << " -> "
        << "tr" << GetId() << "_" << actNode.GetAction()
        << " [label=<a: " << actNode.GetAction() << ">];" << std::endl;
    for (const auto& obsChild : actNode.GetChildren()) {
      out << "tr" << GetId() << "_" << actNode.GetAction() << " -> "
          << "tr" << obsChild.GetBelief()->GetId()
          << " [label=<o: " << obsChild.GetObs() << ">];" << std::endl;
      obsChild.GetBelief()->GenerateGraphviz(out);
    }
  }
}

void BeliefTreeNode::DrawBeliefTree(std::ostream& ofs) const {
  ofs << "digraph BeliefTree {" << std::endl;
  GenerateGraphviz(ofs);
  ofs << "}" << std::endl;
}

void BeliefTreeNode::DrawPolicyBranch(std::ostream& out, int64_t& i) const {
  out << "po" << GetId() << " [label=<<B>" << _belief << "</B><BR/>"
      << "BestAction: " << GetBestActUBound() << "<BR/>"
      << "UpperBound: " << GetUpper() << ">];" << std::endl;
  ++i;

  const auto act = GetBestActUBound();
  const auto it = std::find_if(
      _action_edges.begin(), _action_edges.end(),
      [&act](const ActionNode& a) { return a.GetAction() == act; });
  if (it == _action_edges.end()) return;
  out << "po" << GetId() << "_" << act
      << " [shape=point, style=filled, fillcolor=black];" << std::endl;
  out << "po" << GetId() << " -> "
      << "po" << GetId() << "_" << act << " [label=<a: " << act << ">];"
      << std::endl;
  for (const auto& obsChild : it->GetChildren()) {
    out << "po" << GetId() << "_" << act << " -> "
        << "po" << obsChild.GetBelief()->GetId()
        << " [label=<o: " << obsChild.GetObs() << ">];" << std::endl;
    obsChild.GetBelief()->DrawPolicyBranch(out, i);
  }
}

int64_t BeliefTreeNode::DrawPolicyTree(std::ostream& ofs) const {
  int64_t i = 0;
  ofs << "digraph GreedyPolicyTree {" << std::endl;
  DrawPolicyBranch(ofs, i);
  ofs << "}" << std::endl;
  return i;
}

}  // namespace DetMCVI
