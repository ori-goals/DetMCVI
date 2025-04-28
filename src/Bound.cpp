#include "Bound.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace DetMCVI {

double UpperBoundEvaluation(const BeliefStates& belief,
                            const OptimalPath& solver, double gamma,
                            int64_t belief_depth, int64_t max_depth) {
  double V_upper_bound = 0.0;

  for (size_t i = 0; i < belief.size(); ++i) {
    const auto [reward, path] = solver.getMaxReward(belief[i].first, max_depth);
    V_upper_bound += std::pow(gamma, belief_depth) * reward * belief[i].second;
  }

  return V_upper_bound;
}

double RunSim(SimInterface* pomdp, State state, int64_t max_depth,
              double epsilon, std::mt19937_64& rng) {
  int64_t step = 0;
  double val = 0;
  std::uniform_int_distribution<int> dist(0, pomdp->GetSizeOfA() - 1);
  while ((step < max_depth) &&
         (std::pow(pomdp->GetDiscount(), step) >= epsilon)) {
    const auto [sNext, obs, reward, done] = pomdp->Step(state, dist(rng));
    val += std::pow(pomdp->GetDiscount(), step) * reward;
    if (done) break;
    state = sNext;
    ++step;
  }
  return val;
}

double FindRLower(SimInterface* pomdp, const BeliefStates& b0, double epsilon,
                  int64_t max_depth, std::mt19937_64& rng) {
  double belief_val = 0;

  for (size_t i = 0; i < b0.size(); ++i) {
    belief_val +=
        RunSim(pomdp, b0[i].first, max_depth, epsilon, rng) * b0[i].second;
  }
  return belief_val;
}

std::vector<std::tuple<int64_t, State, double, bool>>
OptimalPath::getSuccessors(const State& state) const {
  const bool isTerminal = pomdp->isTerminal(state);
  if (isTerminal) return {{-1, state, 0.0, true}};
  std::vector<std::tuple<int64_t, State, double, bool>> successors;
  for (int64_t a = 0; a < pomdp->GetSizeOfA(); ++a) {
    State sNext;
    const auto& reward = pomdp->applyActionToState(state, a, sNext);
    successors.push_back({a, sNext, reward, isTerminal});
  }
  return successors;
}

double CalculateUpperBound(const BeliefStates& belief, int64_t belief_depth,
                           int64_t eval_depth, const BoundFunction& func,
                           SimInterface* sim) {
  const auto H_Uval = sim->GetHeuristicUpper(belief, eval_depth - belief_depth);
  if (H_Uval.has_value())
    return std::pow(sim->GetDiscount(), belief_depth) * H_Uval.value();
  return func(belief, belief_depth, eval_depth, sim);
}

double CalculateLowerBound(const BeliefStates& belief, int64_t belief_depth,
                           int64_t eval_depth, const BoundFunction& func,
                           SimInterface* sim) {
  const auto H_Lval = sim->GetHeuristicLower(belief, eval_depth);
  if (H_Lval.has_value())
    return std::pow(sim->GetDiscount(), belief_depth) * H_Lval.value();
  return func(belief, belief_depth, eval_depth, sim);
}

}  // namespace DetMCVI
