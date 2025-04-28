#include "ShortestPath.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <stdexcept>

namespace DetMCVI {

std::tuple<std::vector<State>, std::vector<int64_t>, double>
ShortestPathFasterAlgorithm::FindShortestPath(const State& start_state,
                                              int64_t max_depth) const {
  std::deque<State> queue = {start_state};
  StateMap<bool> in_queue = {{start_state, true}};
  StateMap<int64_t> depth = {{start_state, 0}};
  StateMap<double> returns = {{start_state, 0.0}};
  StateMap<std::pair<State, int64_t>> predecessors = {
      {start_state, {State(), -1}}};
  std::pair<State, double> best_terminal = {
      {}, -std::numeric_limits<double>::infinity()};

  State current_state;

  while (!queue.empty()) {
    current_state = queue.front();
    queue.pop_front();
    in_queue[current_state] = false;

    if (isTerminal(current_state)) {
      if (returns[current_state] > best_terminal.second) {
        best_terminal = {current_state, returns[current_state]};
      }
      continue;
    }

    for (const auto& [next_state, reward, action] : getEdges(current_state)) {
      if (next_state != current_state)
        depth[next_state] =
            std::max(depth[next_state], depth[current_state] + 1);
      if (depth[next_state] > max_depth) {  // max depth reached
        continue;
      }
      double new_return = returns[current_state] + reward;

      if (returns.find(next_state) == returns.end() ||
          new_return > returns[next_state]) {
        returns[next_state] = new_return;
        predecessors[next_state] = std::make_pair(current_state, action);

        if (!in_queue[next_state]) {
          queue.push_back(next_state);
          in_queue[next_state] = true;
        }
      }
    }
  }

  std::vector<State> path;
  std::vector<int64_t> actions;
  if (best_terminal.first.empty()) {
    return {path, actions, -std::numeric_limits<double>::infinity()};
  }
  State state = best_terminal.first;
  double total_reward = best_terminal.second;
  int64_t d = depth[state];
  while (d >= 0) {
    auto [prev_state, action] = predecessors[state];
    if (prev_state == State()) break;
    path.push_back(state);
    actions.push_back(action);
    state = prev_state;
    d--;
  }
  std::reverse(path.begin(), path.end());
  std::reverse(actions.begin(), actions.end());

  return {path, actions, total_reward};
}

std::pair<double, std::vector<std::tuple<int64_t, State, double>>>
MaximiseReward::getMaxReward(const State& init_state, int64_t max_depth) const {
  return GetCachedOrSearch(init_state, max_depth);
}

static bool CmpPathValue(
    const std::pair<double, std::vector<std::tuple<int64_t, State, double>>>&
        p1,
    const std::pair<double, std::vector<std::tuple<int64_t, State, double>>>&
        p2) {
  return p1.first < p2.first;
}

std::pair<double, std::vector<std::tuple<int64_t, State, double>>>
MaximiseReward::GetCachedOrSearch(const State& state,
                                  int64_t depth_to_go) const {
  if (depth_to_go <= 0) return {0, {}};

  // Read from the cache
  const auto state_cache = cache.find(state);
  if (state_cache == cache.end()) {
    return Search(state, depth_to_go);
  }
  const auto pos = state_cache->second.first.find(depth_to_go);
  if (pos == state_cache->second.first.end()) {
    return Search(state, depth_to_go);
  }

  // follow the path to reconstruct
  const auto& [action, successor, immediate_rw] = pos->second;
  const auto& [next_reward, next_path] =
      GetCachedOrSearch(successor, depth_to_go - 1);
  const double total_rw = immediate_rw + discount_factor * next_reward;
  auto new_path = next_path;
  new_path.insert(new_path.begin(),
                  std::make_tuple(action, successor, immediate_rw));

  return {total_rw, new_path};
}

std::pair<double, std::vector<std::tuple<int64_t, State, double>>>
MaximiseReward::Search(const State& state, int64_t depth_to_go) const {
  if (depth_to_go <= 0) return {0, {}};

  std::vector<
      std::pair<double, std::vector<std::tuple<int64_t, State, double>>>>
      paths;
  for (const auto& [action, successor, immediate_rw, state_terminal] :
       getSuccessors(state)) {
    if (state_terminal) return {0, {}};

    const auto& [next_reward, next_path] =
        GetCachedOrSearch(successor, depth_to_go - 1);
    const double total_rw = immediate_rw + discount_factor * next_reward;
    auto new_path = next_path;
    new_path.insert(new_path.begin(),
                    std::make_tuple(action, successor, immediate_rw));
    paths.push_back({total_rw, new_path});
  }

  const auto best_path =
      std::max_element(paths.begin(), paths.end(), CmpPathValue);
  if (best_path == paths.end())
    throw std::runtime_error("Failed to find best path for remaining depth " +
                             std::to_string(depth_to_go));

  cache[state][depth_to_go] = best_path->second.at(0);
  return *best_path;
}

}  // namespace DetMCVI
