#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BeliefDistribution.h"
#include "Cache.h"
#include "ShortestPath.h"
#include "SimInterface.h"

struct pairhash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};

static bool CmpPair(const std::pair<std::pair<int64_t, int64_t>, double>& p1,
                    const std::pair<std::pair<int64_t, int64_t>, double>& p2) {
  return p1.second < p2.second;
}

class GraphPath : public DetMCVI::ShortestPathFasterAlgorithm {
 private:
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
      _edges;  // bidirectional, smallest node is first, double is weight
  int64_t _goal;

 public:
  GraphPath(
      std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> edges,
      int64_t goal)
      : _edges(edges), _goal(goal) {}

  std::vector<std::tuple<DetMCVI::State, double, int64_t>> getEdges(
      const DetMCVI::State& node) const override {
    std::vector<std::tuple<DetMCVI::State, double, int64_t>> out;
    for (const auto& e : _edges) {
      if (e.first.first == node.at(0))
        out.push_back({{e.first.second}, -e.second, e.first.second});
      else if (e.first.second == node.at(0))
        out.push_back({{e.first.first}, -e.second, e.first.first});
    }
    return out;
  }

  bool isTerminal(const DetMCVI::State& state) const override {
    return state.at(0) == _goal;
  }
};

class CTP : public DetMCVI::SimInterface,
            public DetMCVI::ShortestPathFasterAlgorithm {
 protected:
  std::mt19937_64& rng;
  std::vector<int64_t> nodes;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
      edges;  // bidirectional, smallest node is first, double is weight
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
      stoch_edges;  // probability of being blocked
  int64_t origin;
  int64_t goal;
  std::map<std::string, size_t> state_factor_sizes;
  int64_t max_obs_width;
  std::vector<std::string> actions;
  std::vector<std::string> observations;
  double _idle_reward;
  double _bad_action_reward;
  double _complete_reward = 0;
  mutable DetMCVI::LRUCache<DetMCVI::State, bool, DetMCVI::StateHash,
                            DetMCVI::StateEqual>
      goal_reachable;
  mutable DetMCVI::LRUCache<DetMCVI::State, double, DetMCVI::StateHash,
                            DetMCVI::StateEqual>
      state_value;

 public:
  CTP(std::mt19937_64& rng, const std::vector<int64_t>& nodes,
      const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>&
          edges,
      const std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>&
          stoch_edges,
      int64_t origin, int64_t goal)
      : rng(rng),
        nodes(nodes),
        edges(edges),
        stoch_edges(stoch_edges),
        origin(origin),
        goal(goal),
        state_factor_sizes(initStateSpace()),
        max_obs_width(initObsWidth()),
        actions(initActions()),
        observations(initObs()),
        _idle_reward(initIdleReward()),
        _bad_action_reward(initBadReward()),
        goal_reachable(250000),
        state_value(250000) {}

  int64_t GetSizeOfObs() const override { return nodes.size() * max_obs_width; }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }
  int64_t getGoal() const { return goal; }
  bool isTerminal(const DetMCVI::State& sI) const override {
    return sI.at(sfIdx("loc")) == goal;
  }

  std::optional<double> GetHeuristicUpper(const DetMCVI::BeliefStates& belief,
                                          int64_t max_depth) const override {
    double val = 0;
    for (size_t i = 0; i < belief.size(); ++i) {
      val +=
          get_state_value(belief[i].first, max_depth).first * belief[i].second;
    }
    return val;
  }

  std::tuple<DetMCVI::State, int64_t, double, bool> Step(
      const DetMCVI::State& sI, int64_t aI) const override {
    DetMCVI::State sNext;
    const double reward = applyActionToState(sI, aI, sNext);
    const int64_t oI = observeState(sNext);
    const bool finished = checkFinished(sI, aI, sNext);
    // sI_next, oI, Reward, Done
    return std::tuple<DetMCVI::State, int64_t, double, bool>(sNext, oI, reward,
                                                             finished);
  }

  DetMCVI::State SampleStartState() override {
    std::uniform_real_distribution<> unif(0, 1);
    std::map<std::string, int64_t> state;
    // agent starts at special initial state (for init observation)
    state["loc"] = (int64_t)nodes.size();
    // stochastic edge status
    for (const auto& [edge, p] : stoch_edges)
      state[edge2str(edge)] = (unif(rng)) < p ? 0 : 1;
    return names2state(state);
  }

  void visualiseGraph(std::ostream& os) {
    if (!os) throw std::logic_error("Unable to write graph to file");

    os << "graph G {" << std::endl;

    for (const auto& i : nodes) {
      os << "  " << i << " [label=\"" << i << "\"";
      if (i == origin) os << ", fillcolor=\"#ff7f0e\", style=filled";
      if (i == goal) os << ", fillcolor=\"#2ca02c\", style=filled";
      os << "];" << std::endl;
    }

    for (const auto& [edge, weight] : edges) {
      auto stochEdge = stoch_edges.find(edge);
      if (stochEdge != stoch_edges.end()) {
        os << "  " << edge.first << " -- " << edge.second << " [label=\""
           << stochEdge->second << " : " << weight << "\", style=dashed];"
           << std::endl;
      } else {
        os << "  " << edge.first << " -- " << edge.second << " [label=\""
           << weight << "\"];" << std::endl;
      }
    }

    os << "}" << std::endl;
  }

  std::pair<double, bool> get_state_value(const DetMCVI::State& state,
                                          int64_t max_depth) const {
    const auto f = state_value.find(state);
    if (f != state_value.cend()) {
      const auto val = f->second.first;
      return {val, true};
    }
    const auto b = bestPath(state, max_depth);
    state_value.put(state, b.first);
    return b;
  }

  double applyActionToState(const DetMCVI::State& state, int64_t action,
                            DetMCVI::State& sNext) const override {
    sNext = state;
    if (isTerminal(state)) return 0;
    const int64_t loc_idx = sfIdx("loc");
    const int64_t loc = state.at(loc_idx);
    if (loc == (int64_t)nodes.size()) {  // special initial state
      sNext[loc_idx] = origin;
      return 0;
    }
    if (loc == goal) return _complete_reward;  // goal is absorbing

    if (actions.at(action) == "decide_goal_unreachable") {
      if (!goalUnreachable(state)) return _bad_action_reward;
      sNext[loc_idx] = goal;
      return _idle_reward;
    }

    const int64_t dest_loc = nodes.at(action);
    if (loc == dest_loc) return _idle_reward;  // idling

    // invalid move
    if (!nodesAdjacent(loc, dest_loc, state)) return _bad_action_reward;

    // moving
    sNext[loc_idx] = dest_loc;
    return dest_loc < loc ? -edges.at({dest_loc, loc})
                          : -edges.at({loc, dest_loc});
  }

 protected:
  std::string edge2str(std::pair<int64_t, int64_t> e) const {
    return "e" + std::to_string(e.first) + "_" + std::to_string(e.second);
  }

  DetMCVI::State names2state(
      const std::map<std::string, int64_t>& names) const {
    assert(names.size() == state_factor_sizes.size());
    std::vector<int64_t> state;
    for (const auto& [name, state_elem] : names) {
      const auto sf_sz = state_factor_sizes.find(name);
      assert(sf_sz != state_factor_sizes.cend());
      //   assert(sf_sz->second > state_elem);
      state.push_back(state_elem);
    }
    return state;
  }

  int64_t sfIdx(const std::string& state_factor) const {
    const auto sf_sz = state_factor_sizes.find(state_factor);
    assert(sf_sz != state_factor_sizes.cend());
    return (int64_t)std::distance(state_factor_sizes.cbegin(), sf_sz);
  }

  // Return all stochastic edges adjacent to `node`
  std::vector<std::pair<int64_t, int64_t>> AdjacentStochEdges(
      int64_t node) const {
    std::vector<std::pair<int64_t, int64_t>> edges;
    for (const auto& [edge, p] : stoch_edges) {
      if (edge.first == node || edge.second == node) {
        edges.push_back(edge);
      }
    }

    auto compareEdges = [node](const std::pair<int64_t, int64_t>& edge1,
                               const std::pair<int64_t, int64_t>& edge2) {
      int64_t other1 = (edge1.first == node) ? edge1.second : edge1.first;
      int64_t other2 = (edge2.first == node) ? edge2.second : edge2.first;
      return other1 < other2;
    };

    std::sort(edges.begin(), edges.end(), compareEdges);

    return edges;
  }

  bool goalUnreachable(const DetMCVI::State& state) const {
    // check if goal is reachable from the origin (cached)
    DetMCVI::State origin_state = state;
    origin_state[sfIdx("loc")] = origin;
    const auto ret = goal_reachable.find(origin_state);
    if (ret != goal_reachable.end()) {
      return !ret->second.first;
    }

    // find shortest path to goal, return true if none exists
    std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>
        state_edges;
    for (const auto& e : edges) {
      if (!stoch_edges.contains(e.first) ||
          state.at(sfIdx(edge2str(e.first))) == 1)
        state_edges.insert(e);
    }
    auto gp = GraphPath(state_edges, goal);
    const auto [path, actions, total_reward] =
        gp.FindShortestPath({origin}, state_edges.size() + 1);
    const bool reaches_goal = (path.size() > 0) && (path.back()[0] == goal);
    goal_reachable[origin_state] = reaches_goal;

    return !reaches_goal;
  }

 private:
  std::vector<std::string> initActions() const {
    std::vector<std::string> acts;
    for (const auto& n : nodes) acts.push_back(std::to_string(n));
    acts.push_back("decide_goal_unreachable");
    return acts;
  }

  std::map<std::string, size_t> initStateSpace() const {
    std::map<std::string, size_t> state_factors;
    // agent location
    state_factors["loc"] =
        nodes.size() + 1;  // special state for init observation
    // stochastic edge status
    for (const auto& [edge, _] : stoch_edges)
      state_factors[edge2str(edge)] = 2;  // 0 = blocked, 1 = unblocked

    double p = 1.0;
    for (const auto& [sf, sz] : state_factors) p *= sz;
    std::cout << "State space size: " << p << std::endl;

    return state_factors;
  }

  int64_t initObsWidth() const {
    size_t max_stoch_edges_at_node = 0;
    for (const auto& node : nodes)
      max_stoch_edges_at_node =
          std::max(max_stoch_edges_at_node, AdjacentStochEdges(node).size());

    return std::pow(2, max_stoch_edges_at_node);
  }

  std::vector<std::string> initObs() const {
    std::vector<std::string> obs;  // Observation space can be very large!
    return obs;
  }

  bool nodesAdjacent(int64_t a, int64_t b, const DetMCVI::State& state) const {
    if (a == (int64_t)nodes.size() || b == (int64_t)nodes.size()) return false;
    if (a == b) return true;
    const auto edge = a < b ? std::pair(a, b) : std::pair(b, a);
    if (edges.find(edge) == edges.end()) return false;  // edge does not exist

    // check if edge is stochastic
    const auto stoch_ptr = stoch_edges.find(edge);
    if (stoch_ptr == stoch_edges.end()) return true;  // deterministic edge

    // check if edge is unblocked
    return state.at(sfIdx(edge2str(edge))) == 1;  // traversable
  }

  int64_t observeState(const DetMCVI::State& state) const {
    int64_t observation = 0;

    int64_t loc = state.at(sfIdx("loc"));
    if (loc == (int64_t)nodes.size())
      loc = origin;  // observe initial state as if at origin
    const int64_t loc_idx = std::distance(
        nodes.begin(), std::find(nodes.begin(), nodes.end(), loc));

    // stochastic edge status
    int64_t n = 0;
    for (const auto& edge : AdjacentStochEdges(loc)) {
      if (state.at(sfIdx(edge2str(edge)))) observation |= ((int64_t)1 << n);
      ++n;
    }

    observation += loc_idx * max_obs_width;

    return observation;
  }

  bool checkFinished(const DetMCVI::State& sI, int64_t aI,
                     const DetMCVI::State& sNext) const {
    const int64_t loc_idx = sfIdx("loc");
    if (sI.at(loc_idx) == (int64_t)nodes.size()) return false;
    if (actions.at(aI) == "decide_goal_unreachable" && goalUnreachable(sI))
      return true;
    return sNext.at(loc_idx) == goal;
  }

  double initIdleReward() const {
    const double max_edge =
        std::max_element(edges.begin(), edges.end(), CmpPair)->second;
    return -5 * max_edge;
  }

  double initBadReward() const {
    const double max_edge =
        std::max_element(edges.begin(), edges.end(), CmpPair)->second;
    return -50 * max_edge;
  }

  std::vector<int64_t> sfToIndices(
      const std::vector<std::string>& sf_keys) const {
    std::vector<int64_t> indices;
    for (const auto& key : sf_keys) indices.push_back(sfIdx(key));
    return indices;
  }

  double sumStateValues(const DetMCVI::State& state,
                        const std::vector<int64_t>& sf_indices) const {
    return std::accumulate(
        sf_indices.begin(), sf_indices.end(), 0.0,
        [&state](double sum, int64_t index) { return sum + state[index]; });
  }

  DetMCVI::State findMaxSumElement(const DetMCVI::StateMap<double>& belief,
                                   const std::vector<int64_t>& indices) const {
    DetMCVI::State maxElement;
    double maxSum = -std::numeric_limits<double>::infinity();

    for (const auto& [state, value] : belief) {
      double currentSum = sumStateValues(state, indices);
      if (currentSum > maxSum) {
        maxSum = currentSum;
        maxElement = state;
      }
    }

    return maxElement;
  }

  DetMCVI::State findMinSumElement(const DetMCVI::StateMap<double>& belief,
                                   const std::vector<int64_t>& indices) const {
    DetMCVI::State minElement;
    double minSum = std::numeric_limits<double>::infinity();

    for (const auto& [state, value] : belief) {
      double currentSum = sumStateValues(state, indices);
      if (currentSum < minSum) {
        minSum = currentSum;
        minElement = state;
      }
    }

    return minElement;
  }

  // find an upper bound for the value of a belief
  double heuristicUpper(const DetMCVI::StateMap<double>& belief,
                        int64_t max_depth) const {
    std::vector<std::string> sf_keys;
    for (const auto& [e, p] : stoch_edges) sf_keys.push_back(edge2str(e));
    const auto sf_indices = sfToIndices(sf_keys);
    // get state with most traversable edges
    DetMCVI::State best_case_state = findMaxSumElement(belief, sf_indices);
    if (best_case_state.at(sfIdx("loc")) == (int64_t)nodes.size())
      best_case_state[sfIdx("loc")] = origin;

    const auto it = state_value.find(best_case_state);
    if (it != state_value.end()) {
      return it->second.first;
    }

    const auto [val, _] = get_state_value(best_case_state, max_depth);
    state_value[best_case_state] = val;
    return val;
  }

  double heuristicLower(const DetMCVI::StateMap<double>& belief,
                        int64_t max_depth) const {
    std::vector<std::string> sf_keys;
    for (const auto& [e, p] : stoch_edges) sf_keys.push_back(edge2str(e));
    const auto sf_indices = sfToIndices(sf_keys);
    // get state with most traversable edges
    DetMCVI::State worst_case_state = findMinSumElement(belief, sf_indices);
    if (worst_case_state.at(sfIdx("loc")) == (int64_t)nodes.size())
      worst_case_state[sfIdx("loc")] = origin;

    const auto it = state_value.find(worst_case_state);
    if (it != state_value.end()) {
      return it->second.first;
    }

    const auto [val, _] = get_state_value(worst_case_state, max_depth);
    state_value[worst_case_state] = val;
    return val;
  }

 public:
  std::vector<std::tuple<DetMCVI::State, double, int64_t>> getEdges(
      const DetMCVI::State& state) const {
    if (isTerminal(state)) return {};
    std::vector<std::tuple<DetMCVI::State, double, int64_t>> successors;
    for (int64_t a = 0; a < GetSizeOfA(); ++a) {
      DetMCVI::State sNext;
      const auto& reward = applyActionToState(state, a, sNext);
      successors.push_back({sNext, reward, a});
    }
    return successors;
  }

 private:
  std::pair<double, bool> bestPath(const DetMCVI::State& state,
                                   int64_t max_depth) const {
    const auto [path, actions, total_reward] =
        FindShortestPath(state, max_depth);

    return {total_reward,
            total_reward != -std::numeric_limits<double>::infinity()};
  }

  std::vector<DetMCVI::State> enumerateStates(
      size_t max_size = std::numeric_limits<int64_t>::max()) const {
    std::vector<DetMCVI::State> enum_states;
    std::vector<size_t> sizes;
    for (const auto& pair : state_factor_sizes) sizes.push_back(pair.second);

    // Initialize a state vector with the first state (all zeros)
    DetMCVI::State current_state(state_factor_sizes.size(), 0);
    enum_states.push_back(current_state);

    // Generate all combinations of state factors
    while (true) {
      // Find the rightmost factor that can be incremented
      size_t factor_index = state_factor_sizes.size();
      while (factor_index > 0) {
        factor_index--;
        if (current_state[factor_index] + 1 < (int64_t)sizes[factor_index]) {
          current_state[factor_index]++;
          break;
        } else {
          // Reset this factor and carry over to the next factor
          current_state[factor_index] = 0;
        }
      }

      // If we completed a full cycle (all factors are zero again), we are done
      if (factor_index == 0 && current_state[0] == 0) {
        break;
      }
      // warn if overflow
      if (enum_states.size() >= max_size)
        throw std::runtime_error("Maximum size exceeded.");
      enum_states.push_back(current_state);
    }
    return enum_states;
  }

  double init_prob(const DetMCVI::State& s) const {
    double prob = 1.0;
    for (const auto& [edge, p] : stoch_edges)
      if (s[sfIdx(edge2str(edge))])
        prob *= (1 - p);
      else
        prob *= p;
    return prob;
  }

 public:
  DetMCVI::BeliefDistribution TrueInitBelief() const override {
    std::vector<DetMCVI::State> state_enum = enumerateStates();
    DetMCVI::StateMap<double> b;
    for (const auto& s : state_enum) {
      if (s[sfIdx("loc")] == (int64_t)nodes.size()) b[s] = init_prob(s);
    }
    std::cerr << "Initial belief size " << b.size() << std::endl;
    return DetMCVI::BeliefDistribution(b);
  }

 public:
  void toSARSOP(std::ostream& os) {
    std::vector<DetMCVI::State> state_enum = enumerateStates();
    const size_t num_states = state_enum.size();
    os << "discount: " << std::exp(std::log(0.01) / (20.0 * nodes.size()))
       << std::endl;
    os << "values: reward" << std::endl;
    os << "states: " << num_states << std::endl;
    os << "actions: " << GetSizeOfA() << std::endl;
    os << "observations: " << GetSizeOfObs() << std::endl << std::endl;

    // Initial belief
    os << "start: " << std::endl;
    for (const auto& s : state_enum) {
      if (s[sfIdx("loc")] != (int64_t)nodes.size())
        os << "0 ";
      else
        os << std::fixed << init_prob(s) << " ";
    }
    os << std::endl << std::endl;

    // Transition probabilities  T : <action> : <start-state> : <end-state> %f
    // Observation probabilities O : <action> : <end-state> : <observation> %f
    // Reward     R: <action> : <start-state> : <end-state> : <observation> %f
    for (size_t sI = 0; sI < state_enum.size(); ++sI) {
      for (int64_t a = 0; a < GetSizeOfA(); ++a) {
        DetMCVI::State sNext;
        double reward = applyActionToState(state_enum[sI], a, sNext);
        if (isTerminal(state_enum[sI])) {
          reward = 0;
          sNext = state_enum[sI];
        }
        const int64_t obs = observeState(state_enum[sI]);
        const size_t eI = std::distance(
            state_enum.begin(),
            std::find(state_enum.begin(), state_enum.end(), sNext));
        os << "T : " << a << " : " << sI << " : " << eI << " 1.0" << std::endl;
        os << "O : " << a << " : " << sI << " : " << obs << " 1.0" << std::endl;
        os << "R : " << a << " : " << sI << " : " << eI << " : * " << reward
           << std::endl;
      }
    }
  }
};

void ctpGraphFromFile(
    const std::string& filename, std::vector<int64_t>& CTPNodes,
    std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>& CTPEdges,
    std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash>&
        CTPStochEdges,
    int64_t& CTPOrigin, int64_t& CTPGoal) {
  std::ifstream file(filename);
  std::string line = "";
  std::string key = "";

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    iss >> key;

    if (key == "CTPNodes:") {
      int64_t node;
      while (iss >> node) {
        CTPNodes.push_back(node);
      }
    } else if (key == "CTPEdges:") {
      while (true) {
        const std::streampos oldpos = file.tellg();
        if (!std::getline(file, line) || line.empty()) break;
        if (line.find(':') != std::string::npos) {
          file.seekg(oldpos);
          break;
        }
        std::istringstream edgeStream(line);
        int64_t from, to;
        double weight;
        edgeStream >> from >> to >> weight;
        CTPEdges[{from, to}] = weight;
      }
    } else if (key == "CTPStochEdges:") {
      while (true) {
        const std::streampos oldpos = file.tellg();
        if (!std::getline(file, line) || line.empty()) break;
        if (line.find(':') != std::string::npos) {
          file.seekg(oldpos);
          break;
        }
        std::istringstream edgeStream(line);
        int64_t from, to;
        double weight;
        edgeStream >> from >> to >> weight;
        CTPStochEdges[{from, to}] = weight;
      }
    } else if (key == "CTPOrigin:") {
      iss >> CTPOrigin;
    } else if (key == "CTPGoal:") {
      iss >> CTPGoal;
    }
  }
}
