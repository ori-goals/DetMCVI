#include "BeliefDistribution.h"

namespace DetMCVI {

BeliefStates beliefDistributionToStates(const BeliefDistribution& b) {
  return std::vector<std::pair<State, double>>(b.begin(), b.end());
}

BeliefDistribution beliefStatesToDistribution(const BeliefStates& b) {
  BeliefDistribution d;
  for (const auto& [s, p] : b) d[s] += p;
  return d;
}

State SampleOneState(const BeliefDistribution& belief, std::mt19937_64& rng) {
  return SamplePMF<State>(belief, rng);
}

std::ostream& operator<<(std::ostream& os, const BeliefDistribution& bd) {
  if (bd.size() > 5) {
    os << "{ States: " << bd.size() << "}";
  } else {
    os << "{ ";
    for (const auto& pair : bd) {
      os << "[";
      const auto& v = pair.first;
      for (const auto& state_elem : v) {
        os << state_elem << ", ";
      }
      os << "]: " << pair.second << ", ";
    }
    os << "}";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const BeliefStates& b) {
  os << beliefStatesToDistribution(b);
  return os;
}

BeliefDistribution SampleInitialBelief(size_t N, SimInterface* pomdp) {
  std::vector<State> samples(N, State({}));
  for (size_t i = 0; i < N; ++i) {
    samples[i] = pomdp->SampleStartState();
  }
  StateMap<int64_t> state_counts;
  for (size_t i = 0; i < N; ++i) state_counts[samples[i]] += 1;
  auto init_belief = BeliefDistribution();
  for (const auto& [state, count] : state_counts)
    init_belief[state] = (double)count / N;
  return init_belief;
}

BeliefDistribution DownsampleBelief(const BeliefDistribution& belief,
                                    int64_t max_belief_samples,
                                    std::mt19937_64& rng) {
  const auto shuffled_init = weightedShuffle(belief, rng, max_belief_samples);
  double prob_sum = 0.0;
  for (const auto& [state, prob] : shuffled_init) prob_sum += prob;
  auto b = BeliefDistribution();
  for (const auto& [state, prob] : shuffled_init) b[state] = prob / prob_sum;
  return b;
}
}  // namespace DetMCVI
