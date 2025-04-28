#include "AlphaVectorNode.h"

#include <algorithm>
#include <limits>

namespace DetMCVI {

static bool CmpPair(
    const std::pair<State, std::pair<double, std::list<State>::iterator>>& p1,
    const std::pair<State, std::pair<double, std::list<State>::iterator>>& p2) {
  return p1.second.first < p2.second.first;
}

AlphaVectorNode::AlphaVectorNode(int64_t init_best_action)
    : _best_action(init_best_action), _alpha(2000000) {}

std::optional<double> AlphaVectorNode::GetAlpha(const State& state) const {
  const auto it = _alpha.find(state);
  const std::optional<double> alpha =
      (it == _alpha.cend()) ? std::nullopt
                            : std::make_optional(it->second.first);
  return alpha;
}

double AlphaVectorNode::V_node() const {
  auto v = std::max_element(_alpha.cbegin(), _alpha.cend(), CmpPair);
  const double V = (v == _alpha.cend()) ? 0 : v->second.first;
  return V;
}

void AlphaVectorNode::SetAlpha(const State& state, double value) {
  _alpha[state] = value;
}

}  // namespace DetMCVI
