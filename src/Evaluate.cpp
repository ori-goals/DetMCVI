#include "Evaluate.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace DetMCVI {

static std::string print_inf(double num) {
  if (std::isinf(num)) {
    if (num > 0)
      return "inf";
    else
      return "-inf";
  } else {
    std::ostringstream stream;
    stream << std::fixed << num;
    return stream.str();
  }
}

void StatsCSVHeader(std::ostream& os) {
  os << "algorithm,timestamp,set number,seed,policy nodes,";
  for (const std::string termination : {"completed", "exited", "truncated"}) {
    os << termination << " count,";
    for (const std::string metric : {"reward", "oracle", "regret", "ratio"}) {
      for (const std::string stat : {"mean", "var", "max", "min"}) {
        os << termination << " " << metric << " " << stat;
        if (termination == "exited" && metric == "reward" && stat == "min") {
          for (const std::string stat2 : {"mean", "var", "max", "min"})
            os << "," << termination << " " << metric << " to go " << stat2;
        }
        if (!(termination == "truncated" && metric == "ratio" && stat == "min"))
          os << ",";
      }
    }
  }
  os << std::endl;
}

static void StatsToCSV(const std::string& algorithm, double timestamp,
                       int64_t set_number, int64_t seed, int64_t policy_nodes,
                       const EvaluationStats& reward,
                       const EvaluationStats& oracle,
                       const EvaluationStats& regret,
                       const EvaluationStats& ratio, const Welford& path_to_go,
                       std::ostream& os) {
  os << algorithm << "," << timestamp << "," << set_number << "," << seed << ","
     << policy_nodes << ",";

  os << print_inf(reward.complete.getCount()) << ",";
  for (const auto& metric : {reward, oracle, regret, ratio}) {
    const auto a = metric.complete;
    os << print_inf(a.getMean()) << "," << print_inf(a.getVariance()) << ","
       << print_inf(a.getMax()) << "," << print_inf(a.getMin()) << ",";
  }
  os << print_inf(reward.off_policy.getCount()) << ",";
  for (const auto metric : {&reward, &oracle, &regret, &ratio}) {
    const auto a = metric->off_policy;
    os << print_inf(a.getMean()) << "," << print_inf(a.getVariance()) << ","
       << print_inf(a.getMax()) << "," << print_inf(a.getMin()) << ",";
    if (metric == &reward) {
      os << print_inf(path_to_go.getMean()) << ","
         << print_inf(path_to_go.getVariance()) << ","
         << print_inf(path_to_go.getMax()) << ","
         << print_inf(path_to_go.getMin()) << ",";
    }
  }
  os << print_inf(reward.max_depth.getCount()) << ",";
  for (const auto metric : {&reward, &oracle, &regret, &ratio}) {
    const auto a = metric->max_depth;
    os << print_inf(a.getMean()) << "," << print_inf(a.getVariance()) << ","
       << print_inf(a.getMax()) << "," << print_inf(a.getMin());
    if (metric != &ratio) os << ",";
  }
  os << std::endl;
}

static int64_t ProcessStats(const std::vector<TrialResult>& results,
                            const std::string& alg_name, double timestamp,
                            int64_t set_number, int64_t seed,
                            int64_t policy_nodes, std::ostream& os) {
  EvaluationStats reward_stats;  // reward
  EvaluationStats oracle_stats;  // best-case
  EvaluationStats regret_stats;  // regret
  EvaluationStats ratio_stats;   // competitive ratio
  Welford path_to_go;  // remaining reward to reach a goal state (according to
                       // oracle)

  for (const auto& [ret, oracle_ret, termination, has_solution, reward_to_go] :
       results) {
    if (termination == TrialTermination::GOAL_REACHED) {
      reward_stats.complete.update(ret);
      oracle_stats.complete.update(oracle_ret);
      regret_stats.complete.update(oracle_ret - ret);
      ratio_stats.complete.update((ret) / oracle_ret);
    }
    if (termination == TrialTermination::NO_OBSERVATION && has_solution) {
      reward_stats.off_policy.update(ret);
      oracle_stats.off_policy.update(oracle_ret);
      regret_stats.off_policy.update(oracle_ret - ret);
      ratio_stats.off_policy.update((ret) / oracle_ret);
      path_to_go.update(reward_to_go);
    }
    if (termination == TrialTermination::MAX_DEPTH && has_solution) {
      reward_stats.max_depth.update(ret);
      oracle_stats.max_depth.update(oracle_ret);
      regret_stats.max_depth.update(oracle_ret - ret);
      ratio_stats.max_depth.update((ret) / oracle_ret);
    }
    if (!has_solution && termination == TrialTermination::NO_OBSERVATION) {
      reward_stats.no_solution_off_policy.update(ret);
      oracle_stats.no_solution_off_policy.update(oracle_ret);
      regret_stats.no_solution_off_policy.update(oracle_ret - ret);
      ratio_stats.no_solution_off_policy.update((ret) / oracle_ret);
    }
    if (!has_solution && (termination == TrialTermination::GOAL_REACHED ||
                          termination == TrialTermination::MAX_DEPTH)) {
      reward_stats.no_solution_on_policy.update(ret);
      oracle_stats.no_solution_on_policy.update(oracle_ret);
      regret_stats.no_solution_on_policy.update(oracle_ret - ret);
      ratio_stats.no_solution_on_policy.update((ret) / oracle_ret);
    }
  }

  StatsToCSV(alg_name, timestamp, set_number, seed, policy_nodes, reward_stats,
             oracle_stats, regret_stats, ratio_stats, path_to_go, os);
  return reward_stats.complete.getCount();
}

// Return the shortest path reward and whether any terminal state is reachable
// from this state (assumed true, user must implement StateValueFunction to
// access this functionality)
static std::pair<double, bool> OracleReward(const State& state,
                                            const OptimalPath& solver,
                                            int64_t max_depth) {
  const auto [sum_reward, path] = solver.getMaxReward(state, max_depth);
  const bool can_reach_terminal = true;
  return {sum_reward, can_reach_terminal};
}

static TrialResult runFSC(const AlphaVectorFSC& fsc, State state,
                          int64_t max_steps, SimInterface* pomdp,
                          const OptimalPath& solver,
                          std::optional<StateValueFunction> valFunc) {
  const double gamma = pomdp->GetDiscount();
  const auto [optimal, has_soln] = (valFunc.has_value())
                                       ? valFunc.value()(state, max_steps)
                                       : OracleReward(state, solver, max_steps);

  double sum_r = 0.0;
  int64_t nI = fsc.GetStartNodeIndex();
  for (int64_t i = 0; i < max_steps; ++i) {
    if (nI == -1) {
      const auto [remaining_path, has_soln_rem] =
          (valFunc.has_value()) ? valFunc.value()(state, max_steps - i)
                                : OracleReward(state, solver, max_steps - i);

      return std::make_tuple(sum_r, optimal, TrialTermination::NO_OBSERVATION,
                             has_soln, remaining_path);
    }

    const int64_t action = fsc.GetNode(nI)->GetBestAction();
    const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
    sum_r += std::pow(gamma, i) * reward;

    if (done)
      return std::make_tuple(sum_r, optimal, TrialTermination::GOAL_REACHED,
                             has_soln, 0.0);

    nI = fsc.GetEdgeValue(nI, obs);

    state = sNext;
  }
  return std::make_tuple(sum_r, optimal, TrialTermination::MAX_DEPTH, has_soln,
                         0.0);
}

int64_t EvaluationWithSimulationFSC(
    int64_t max_steps, int64_t num_sims, int64_t init_belief_samples,
    std::optional<StateValueFunction> valFunc, SimInterface* pomdp,
    std::mt19937_64& rng, const AlphaVectorFSC& fsc, const OptimalPath& solver,
    const std::string& alg_name, double timestamp, int64_t set_number,
    int64_t seed, int64_t policy_nodes, std::ostream& os) {
  std::vector<TrialResult> results(
      num_sims, std::make_tuple(0.0, 0.0, TrialTermination::INVALID, false,
                                0.0));  // return, oracle, type, has_soln
  const BeliefDistribution init_belief =
      SampleInitialBelief(init_belief_samples, pomdp);

  for (int64_t sim = 0; sim < num_sims; ++sim) {
    State state = SampleOneState(init_belief, rng);
    results[sim] = runFSC(fsc, state, max_steps, pomdp, solver, valFunc);
  }

  return ProcessStats(results, alg_name, timestamp, set_number, seed,
                      policy_nodes, os);
}

static TrialResult runTree(const std::shared_ptr<BeliefTreeNode> root,
                           State state, int64_t max_steps, SimInterface* pomdp,
                           const OptimalPath& solver,
                           std::optional<StateValueFunction> valFunc) {
  if (!root)
    return std::make_tuple(0.0, 0.0, TrialTermination::INVALID, false, 0.0);
  const double gamma = pomdp->GetDiscount();
  const auto [optimal, has_soln] = (valFunc.has_value())
                                       ? valFunc.value()(state, max_steps)
                                       : OracleReward(state, solver, max_steps);
  double sum_r = 0.0;
  auto node = root;
  for (int64_t i = 0; i < max_steps; ++i) {
    if (node && node->GetBestActUBound() == -1) node = nullptr;
    if (!node) {  // off policy
      const auto [remaining_path, has_soln_rem] =
          (valFunc.has_value()) ? valFunc.value()(state, max_steps - i)
                                : OracleReward(state, solver, max_steps - i);

      return std::make_tuple(sum_r, optimal, TrialTermination::NO_OBSERVATION,
                             has_soln, remaining_path);
    }
    const int64_t action = node->GetBestActUBound();
    const auto [sNext, obs, reward, done] = pomdp->Step(state, action);
    sum_r += std::pow(gamma, i) * reward;

    if (done)
      return std::make_tuple(sum_r, optimal, TrialTermination::GOAL_REACHED,
                             has_soln, 0.0);

    node = node->GetChild(action, obs);
    state = sNext;
  }
  return std::make_tuple(sum_r, optimal, TrialTermination::MAX_DEPTH, has_soln,
                         0.0);
}

int64_t EvaluationWithGreedyTreePolicy(
    std::shared_ptr<BeliefTreeNode> root, int64_t max_steps, int64_t num_sims,
    int64_t init_belief_samples, SimInterface* pomdp, std::mt19937_64& rng,
    const OptimalPath& solver, std::optional<StateValueFunction> valFunc,
    const std::string& alg_name, double timestamp, int64_t set_number,
    int64_t seed, int64_t policy_nodes, std::ostream& os) {
  std::vector<TrialResult> results(
      num_sims, std::make_tuple(0.0, 0.0, TrialTermination::INVALID, false,
                                0.0));  // return, oracle, type, has_soln
  const BeliefDistribution init_belief =
      SampleInitialBelief(init_belief_samples, pomdp);

  for (int64_t sim = 0; sim < num_sims; ++sim) {
    State state = SampleOneState(init_belief, rng);
    results[sim] = runTree(root, state, max_steps, pomdp, solver, valFunc);
  }

  return ProcessStats(results, alg_name, timestamp, set_number, seed,
                      policy_nodes, os);
}

void RunCommonEvaluation(
    std::vector<State> eval_states, SimInterface* pomdp, int64_t max_depth,
    const OptimalPath& solver, std::optional<StateValueFunction> valFunc,
    const AlphaVectorFSC& mcvi_fsc,
    const std::shared_ptr<BeliefTreeNode> aostar_root,
    const std::shared_ptr<BeliefTreeNode> anytime_aostar_root,
    const std::shared_ptr<BeliefTreeNode> qmdp_root,
    const AlphaVectorFSC& sarsop_fsc, std::ostream& os) {
  std::map<std::string, std::vector<TrialResult>> results;
  for (const auto& state : eval_states) {
    results["DetMCVI"].push_back(
        runFSC(mcvi_fsc, state, max_depth, pomdp, solver, valFunc));
    results["AO*"].push_back(
        runTree(aostar_root, state, max_depth, pomdp, solver, valFunc));
    results["Anytime AO*"].push_back(
        runTree(anytime_aostar_root, state, max_depth, pomdp, solver, valFunc));
    results["QMDP"].push_back(
        runTree(qmdp_root, state, max_depth, pomdp, solver, valFunc));
    results["SARSOP"].push_back(
        runFSC(sarsop_fsc, state, max_depth, pomdp, solver, valFunc));
  }

  const int64_t threshold = 7 * eval_states.size() / 10;
  const std::vector<std::string> alg_names = {"DetMCVI", "AO*", "Anytime AO*",
                                              "QMDP", "SARSOP"};
  for (const auto& name : alg_names) {
    const int64_t completed = std::count_if(
        results[name].begin(), results[name].end(),
        [](const TrialResult& result) {
          return std::get<2>(result) == TrialTermination::GOAL_REACHED;
        });
    std::cout << name << " completed: " << completed << std::endl;
    if (completed < threshold) results.erase(name);
  }

  std::map<std::string, std::vector<TrialResult>> good_results;
  for (size_t i = 0; i < eval_states.size(); ++i) {
    bool good = true;
    for (const auto& [name, res] : results) {
      if (std::get<2>(res[i]) != TrialTermination::GOAL_REACHED) {
        good = false;
        break;
      }
    }
    if (good) {
      for (const auto& [name, res] : results) {
        good_results[name].push_back(res[i]);
      }
    }
  }

  for (const auto& [name, res] : good_results) {
    ProcessStats(res, name, 0, 0, 0, 0, os);
  }
}

std::string generateSARSOPFilename(const std::string& problem, int N, int i) {
  std::ostringstream oss;
  oss << "experiments/" << problem << "/evaluation/sarsop/" << N << "x" << N
      << "/" << problem << "_policy_" << N << "_" << i << ".dot";
  return oss.str();
}

}  // namespace DetMCVI
