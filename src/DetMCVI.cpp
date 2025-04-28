
#include "DetMCVI.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace DetMCVI {

int64_t MCVIPlanner::InsertNode(
    const int64_t node_act, const std::unordered_map<int64_t, int64_t>& edges) {
  const int64_t nI = _fsc.AddNode(node_act);
  _fsc.UpdateEdge(nI, edges);
  return nI;
}

int64_t MCVIPlanner::FindOrInsertNode(
    const int64_t node_act, const std::unordered_map<int64_t, int64_t>& edges) {
  for (int64_t nI = 0; nI < _fsc.NumNodes(); ++nI) {
    // First check the best action
    if (_fsc.GetNode(nI)->GetBestAction() != node_act) continue;
    const auto& check_edges = _fsc.GetEdges(nI);
    if (check_edges == edges) return nI;
  }
  return InsertNode(node_act, edges);
}

void MCVIPlanner::BackUp(std::shared_ptr<BeliefTreeNode> Tr_node,
                         double R_lower, int64_t max_depth_sim,
                         int64_t eval_depth) {
  // Initialise node with all action children if not already done
  auto lowerBoundFunc = [this, R_lower](const BeliefStates& belief,
                                        int64_t belief_depth,
                                        int64_t eval_depth, SimInterface* sim) {
    return this->_fsc.LowerBoundFromFSC(belief, belief_depth, R_lower,
                                        eval_depth, sim);
  };
  auto upperBoundFunc = [this](const BeliefStates& belief, int64_t belief_depth,
                               int64_t eval_depth, SimInterface* sim) {
    return UpperBoundEvaluation(belief, this->_heuristic, sim->GetDiscount(),
                                belief_depth, belief_depth + eval_depth);
  };

  for (int64_t action = 0; action < _pomdp->GetSizeOfA(); ++action)
    Tr_node->GetOrAddChildren(action, upperBoundFunc, eval_depth,
                              lowerBoundFunc, _pomdp);

  Tr_node->BackUpActions(_fsc, R_lower, max_depth_sim, _pomdp);
  Tr_node->UpdateBestAction();

  const int64_t best_act = Tr_node->GetBestActLBound();
  std::unordered_map<int64_t, int64_t> node_edges;
  for (const auto& next_belief : Tr_node->GetChildren(best_act)) {
    if (next_belief.GetBestPolicyNode() >= 0)
      node_edges[next_belief.GetObs()] = next_belief.GetBestPolicyNode();
  }

  if (node_edges.empty()) return;  // Terminal belief

  const int64_t nI = FindOrInsertNode(best_act, node_edges);
  Tr_node->SetBestPolicyNode(nI);
}

static double s_time_diff(const std::chrono::steady_clock::time_point& begin,
                          const std::chrono::steady_clock::time_point& end) {
  return (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
              .count()) /
         1e6;
}

void MCVIPlanner::SampleBeliefs(
    std::vector<std::shared_ptr<BeliefTreeNode>>& traversal_list,
    int64_t eval_depth, double eval_epsilon, double R_lower,
    int64_t max_depth_sim) {
  const auto node = traversal_list.back();
  if (node == nullptr) throw std::logic_error("Invalid node");
  // Initialise node with all action children if not already done
  auto lowerBoundFunc = [this, R_lower](const BeliefStates& belief,
                                        int64_t belief_depth,
                                        int64_t eval_depth, SimInterface* sim) {
    return this->_fsc.LowerBoundFromFSC(belief, belief_depth, R_lower,
                                        eval_depth, sim);
  };
  auto upperBoundFunc = [this](const BeliefStates& belief, int64_t belief_depth,
                               int64_t eval_depth, SimInterface* sim) {
    return UpperBoundEvaluation(belief, this->_heuristic, sim->GetDiscount(),
                                belief_depth, belief_depth + eval_depth);
  };

  for (int64_t action = 0; action < _pomdp->GetSizeOfA(); ++action)
    node->GetOrAddChildren(action, upperBoundFunc, eval_depth, lowerBoundFunc,
                           _pomdp);
  node->BackUpActions(_fsc, R_lower, max_depth_sim, _pomdp);
  node->UpdateBestAction();
  if (node->IsClosed()) return;

  try {
    const auto [next_node, excess_uncertainty] =
        node->ChooseObservation(eval_epsilon, _pomdp->GetDiscount());
    if (excess_uncertainty < 0) return;
    traversal_list.push_back(next_node);
  } catch (std::logic_error& e) {
    if (std::string(e.what()) == "Failed to find best observation") return;
    throw(e);
  }
}

static bool MCVITimeExpired(const std::chrono::steady_clock::time_point& begin,
                            int64_t max_computation_ms) {
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
  if (elapsed.count() >= max_computation_ms * 1000) {
    std::cout << "detMCVI planning complete, reached maximum computation time."
              << std::endl;
    return true;
  }
  return false;
}

double MCVIPlanner::MCVIIteration(std::shared_ptr<BeliefTreeNode> Tr_root,
                                  double R_lower, int64_t ms_remaining,
                                  int64_t max_depth_sim, int64_t eval_depth,
                                  double eval_epsilon,
                                  std::atomic<bool>& exit_flag) {
  std::cout << "Belief Expand Process" << std::flush;
  std::vector<std::shared_ptr<BeliefTreeNode>> traversal_list = {Tr_root};

  auto begin = std::chrono::steady_clock::now();
  for (int64_t depth = 0; depth < max_depth_sim; ++depth) {
    SampleBeliefs(traversal_list, eval_depth - depth, eval_epsilon, R_lower,
                  max_depth_sim - depth);
    if (exit_flag.load()) break;
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(now - begin);
    if (elapsed.count() >= ms_remaining * 1000 / 2) break;
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << " (" << s_time_diff(begin, end) << " seconds)" << std::endl;

  std::cout << "Backup Process" << std::flush;
  begin = std::chrono::steady_clock::now();
  while (!traversal_list.empty()) {
    auto tr_node = traversal_list.back();
    BackUp(tr_node, R_lower, max_depth_sim, eval_depth);
    traversal_list.pop_back();
    if (exit_flag.load()) break;
  }
  end = std::chrono::steady_clock::now();
  std::cout << " (" << s_time_diff(begin, end) << " seconds)" << std::endl;

  if (Tr_root->GetBestPolicyNode() < 0)
    throw std::runtime_error("No best policy node found");

  _fsc.SetStartNodeIndex(Tr_root->GetBestPolicyNode());

  std::cout << "Tr_root upper bound: " << Tr_root->GetUpper() << std::endl;
  std::cout << "Tr_root lower bound: " << Tr_root->GetLower() << std::endl;
  const double new_precision = Tr_root->GetUpper() - Tr_root->GetLower();
  std::cout << "Precision: " << new_precision << std::endl;
  return new_precision;
}

int64_t MCVIPlanner::GetFirstAction(std::shared_ptr<BeliefTreeNode> Tr_node,
                                    double R_lower, int64_t max_depth_sim,
                                    int64_t eval_depth, double eval_epsilon) {
  auto lowerBoundFunc = [eval_epsilon, this](
                            const BeliefStates& belief, int64_t belief_depth,
                            int64_t eval_depth, SimInterface* sim) {
    return FindRLower(sim, belief, belief_depth - eval_epsilon, eval_depth,
                      this->_rng);
  };
  auto upperBoundFunc = [this](const BeliefStates& belief, int64_t belief_depth,
                               int64_t eval_depth, SimInterface* sim) {
    return UpperBoundEvaluation(belief, this->_heuristic, sim->GetDiscount(),
                                belief_depth, belief_depth + eval_depth);
  };

  for (int64_t action = 0; action < _pomdp->GetSizeOfA(); ++action)
    Tr_node->GetOrAddChildren(action, upperBoundFunc, eval_depth,
                              lowerBoundFunc, _pomdp);

  Tr_node->BackUpActions(_fsc, R_lower, max_depth_sim, _pomdp);
  Tr_node->UpdateBestAction();
  return Tr_node->GetBestActUBound();
}

// fsc, root node, precision, converged, timed out, reached max iter
std::tuple<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>, double, bool, bool,
           bool>
MCVIPlanner::PlanIncrement(std::shared_ptr<BeliefTreeNode> Tr_root,
                           double R_lower, int64_t iter, int64_t ms_remaining,
                           int64_t max_depth_sim, double epsilon,
                           int64_t max_nb_iter, int64_t eval_depth,
                           double eval_epsilon, std::atomic<bool>& exit_flag) {
  if (iter == 0) {
    const auto action = GetFirstAction(Tr_root, R_lower, max_depth_sim,
                                       eval_depth, eval_epsilon);
    _fsc.AddNode(action);
    _fsc.SetStartNodeIndex(0);
    Tr_root->SetBestPolicyNode(0);
  }
  const double precision_before = Tr_root->GetUpper() - Tr_root->GetLower();
  if (ms_remaining <= 0)
    return {_fsc, Tr_root, precision_before, false, true, false};
  if (iter >= max_nb_iter) {
    return {_fsc, Tr_root, precision_before, false, false, true};
  }
  const auto iter_start = std::chrono::steady_clock::now();
  std::cout << "--- Iter " << iter << " ---" << std::endl;
  const double precision =
      MCVIIteration(Tr_root, R_lower, ms_remaining, max_depth_sim, eval_depth,
                    eval_epsilon, exit_flag);
  return {_fsc,
          Tr_root,
          precision,
          std::abs(precision) < epsilon,
          MCVITimeExpired(iter_start, ms_remaining),
          iter >= max_nb_iter - 1};
}

std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>> MCVIPlanner::Plan(
    int64_t max_depth_sim, double epsilon, int64_t max_nb_iter,
    int64_t max_computation_ms, int64_t eval_depth, double eval_epsilon,
    std::atomic<bool>& exit_flag) {
  // Calculate the lower bound
  const double R_lower =
      FindRLower(_pomdp, _b0, eval_epsilon, eval_depth, _rng);
  auto upperBoundFunc = [this](const BeliefStates& belief, int64_t belief_depth,
                               int64_t eval_depth, SimInterface* sim) {
    return UpperBoundEvaluation(belief, this->_heuristic, sim->GetDiscount(),
                                belief_depth, belief_depth + eval_depth);
  };
  const auto init_upper =
      CalculateUpperBound(_b0, 0, eval_depth, upperBoundFunc, _pomdp);
  std::shared_ptr<BeliefTreeNode> Tr_root =
      CreateBeliefTreeNode(_b0, 0, init_upper, R_lower);

  const auto iter_start = std::chrono::steady_clock::now();
  int64_t i = 0;

  while (!exit_flag.load()) {
    std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    const auto ms_remaining =
        max_computation_ms -
        std::chrono::duration_cast<std::chrono::microseconds>(now - iter_start)
                .count() /
            1000;
    std::cerr << " ms remaining " << ms_remaining << " max time "
              << max_computation_ms << std::endl;
    const auto [fsc, root, precision, converged, timed_out, max_iter] =
        PlanIncrement(Tr_root, R_lower, i, ms_remaining, max_depth_sim, epsilon,
                      max_nb_iter, eval_depth, eval_epsilon, exit_flag);
    _fsc = fsc;
    Tr_root = root;
    if (converged || timed_out || max_iter) break;
    ++i;
  }
  return {_fsc, Tr_root};
}

std::pair<AlphaVectorFSC, std::shared_ptr<BeliefTreeNode>>
MCVIPlanner::PlanAndEvaluate(int64_t max_depth_sim, double epsilon,
                             int64_t max_nb_iter, int64_t max_computation_ms,
                             int64_t eval_depth, double eval_epsilon,
                             int64_t max_eval_steps, int64_t n_eval_trials,
                             int64_t nb_particles_b0, int64_t eval_interval_ms,
                             int64_t completion_threshold,
                             int64_t completion_reps,
                             std::optional<StateValueFunction> valFunc,
                             std::atomic<bool>& exit_flag, int64_t set_number,
                             int64_t seed, std::ostream& os) {
  OptimalPath solver(_pomdp);
  // Calculate the lower bound
  const double R_lower =
      FindRLower(_pomdp, _b0, eval_epsilon, eval_depth, _rng);
  auto upperBoundFunc = [this](const BeliefStates& belief, int64_t belief_depth,
                               int64_t eval_depth, SimInterface* sim) {
    return UpperBoundEvaluation(belief, this->_heuristic, sim->GetDiscount(),
                                belief_depth, belief_depth + eval_depth);
  };
  const auto init_upper =
      CalculateUpperBound(_b0, 0, eval_depth, upperBoundFunc, _pomdp);
  std::shared_ptr<BeliefTreeNode> Tr_root =
      CreateBeliefTreeNode(_b0, 0, init_upper, R_lower);

  int64_t i = 0;
  int64_t time_sum = 0;
  int64_t last_eval = -eval_interval_ms * 1000;
  int64_t completed_times = 0;

  while (!exit_flag.load()) {
    const std::chrono::steady_clock::time_point iter_start =
        std::chrono::steady_clock::now();
    const auto ms_remaining = max_computation_ms - time_sum / 1e3;

    const auto [fsc, root, precision, converged, timed_out, max_iter] =
        PlanIncrement(Tr_root, R_lower, i, ms_remaining, max_depth_sim, epsilon,
                      max_nb_iter, eval_depth, eval_epsilon, exit_flag);

    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - iter_start);
    time_sum += elapsed.count();

    _fsc = fsc;
    Tr_root = root;
    ++i;

    if (time_sum - last_eval >= eval_interval_ms * 1000) {
      last_eval = time_sum;

      const int64_t completed_count = EvaluationWithSimulationFSC(
          max_eval_steps, n_eval_trials, nb_particles_b0, valFunc, _pomdp, _rng,
          _fsc, solver, "detMCVI", time_sum / 1e6, set_number, seed,
          _fsc.NumNodes(), os);

      completed_times =
          (completed_count >= completion_threshold) ? completed_times + 1 : 0;
      if (completed_times >= completion_reps) {
        std::cout
            << "detMCVI planning complete, reached number of completion reps."
            << std::endl;
        break;
      }
    }
    if (converged || timed_out || max_iter) {
      std::cout << "detMCVI planning complete. Converged: " << converged
                << " Timed out: " << timed_out
                << " Maxed iterations: " << max_iter << std::endl;
      break;
    }
    if (time_sum >= max_computation_ms * 1000) break;
  }

  EvaluationWithSimulationFSC(max_eval_steps, n_eval_trials, nb_particles_b0,
                              valFunc, _pomdp, _rng, _fsc, _heuristic,
                              "detMCVI", time_sum / 1e6, set_number, seed,
                              _fsc.NumNodes(), os);
  return {_fsc, Tr_root};
}

bool CmpActionPair(const std::pair<int64_t, double>& a,
                   const std::pair<int64_t, double>& b) {
  return a.second < b.second;
}

void MCVIPlanner::SimulationWithFSC(int64_t steps) const {
  const double gamma = _pomdp->GetDiscount();
  State state = SampleOneState(beliefStatesToDistribution(_b0), _rng);
  double sum_r = 0.0;
  int64_t nI = _fsc.GetStartNodeIndex();
  for (int64_t i = 0; i < steps; ++i) {
    if (nI == -1) {
      std::cout << "Reached end of policy." << std::endl;
      break;
    }
    const int64_t action = _fsc.GetNode(nI)->GetBestAction();
    std::cout << "---------" << std::endl;
    std::cout << "step: " << i << std::endl;
    std::cout << "state: <";
    for (const auto& state_elem : state) std::cout << state_elem << ", ";
    std::cout << ">" << std::endl;
    std::cout << "perform action: " << action << std::endl;
    const auto [sNext, obs, reward, done] = _pomdp->Step(state, action);

    std::cout << "receive obs: " << obs << std::endl;
    if (nI != -1) {
      std::cout << "nI: " << nI << std::endl;
      std::cout << "nI value: " << _fsc.GetNode(nI)->V_node() << std::endl;
    }
    std::cout << "reward: " << reward << std::endl;

    sum_r += std::pow(gamma, i) * reward;
    if (nI != -1) nI = _fsc.GetEdgeValue(nI, obs);

    if (done) {
      std::cout << "Reached terminal state." << std::endl;
      break;
    }
    state = sNext;
  }
  std::cout << "sum reward: " << sum_r << std::endl;
}

}  // namespace DetMCVI
