#pragma once

#include <cassert>
#include <random>

#include "BeliefDistribution.h"
#include "Sample.h"
#include "ShortestPath.h"
#include "SimInterface.h"

class Wumpus : public DetMCVI::SimInterface,
               public DetMCVI::ShortestPathFasterAlgorithm {
 private:
  int64_t grid_size;
  std::vector<std::string> actions = {
      "forward", "turn_90_deg_left", "turn_90_deg_right", "grab", "shoot",
      "climb"};
  std::vector<std::string> observations;

  double pit_probability;

  size_t heuristic_samples;

  std::mt19937_64& rng;
  std::map<std::string, size_t> state_factor_sizes;

  double _failed_reward;
  double _success_reward = -1;
  double _action_reward = -10;
  double _shoot_reward = -10;
  double _grab_gold_reward = -10;
  double _bad_action_reward = -1000;

  mutable DetMCVI::LRUCache<DetMCVI::State, std::pair<double, bool>,
                            DetMCVI::StateHash, DetMCVI::StateEqual>
      state_value;

 public:
  Wumpus(int32_t grid_size, size_t heuristic_samples, std::mt19937_64& rng)
      : grid_size(grid_size),
        observations(initObs()),
        pit_probability(0.2),
        heuristic_samples(heuristic_samples),
        rng(rng),
        state_factor_sizes(initStateSpace()),
        _failed_reward(-100 * (grid_size - 1) - 30),
        state_value(25000) {}

  int64_t GetSizeOfObs() const override { return observations.size(); }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }
  bool isTerminal(const DetMCVI::State& sI) const override {
    return sI[sfIdx("player_state")] == 1 || sI[sfIdx("player_state")] == 3;
  }

  void drawWumpusWorld(const DetMCVI::State& state) const {
    std::vector<std::vector<std::string>> grid(
        grid_size, std::vector<std::string>(grid_size, "         "));

    // Player position and direction
    int64_t player_pos = state[sfIdx("player_pos")];
    int64_t player_x = player_pos / grid_size;
    int64_t player_y = player_pos % grid_size;
    char player_dir;
    switch (state[sfIdx("player_dir")]) {
      case 1:
        player_dir = '>';
        break;
      case 2:
        player_dir = 'V';
        break;
      case 3:
        player_dir = '<';
        break;
      default:
        player_dir = 'A';
    }
    grid[player_x][player_y][7] = player_dir;

    // Player carries gold
    if (state[sfIdx("player_gold")] == 1) {
      grid[player_x][player_y][6] = 'g';
    }

    // Player carries arrow
    if (state[sfIdx("player_arrow")] == 1) {
      grid[player_x][player_y][8] = 'q';
    }

    // Wumpus position
    int64_t wumpus_x = state[sfIdx("wumpus_x")];
    int64_t wumpus_y = state[sfIdx("wumpus_y")];
    if (wumpus_x != grid_size && wumpus_y != grid_size) {
      grid[wumpus_x][wumpus_y][1] = 'W';
    }

    // Gold position
    int64_t gold_x = state[sfIdx("gold_x")];
    int64_t gold_y = state[sfIdx("gold_y")];
    if (gold_x != grid_size && gold_y != grid_size) {
      grid[gold_x][gold_y][0] = 'G';
    }

    // Pits
    for (int64_t x = 0; x < grid_size; ++x) {
      for (int64_t y = 0; y < grid_size; ++y) {
        if (state[sfIdx(coord2str(x, y) + "_pit")] == 1) {
          grid[x][y][4] = 'P';
        }
      }
    }

    // Draw the grid
    for (int64_t y = grid_size - 1; y >= 0; --y) {
      for (int64_t x = 0; x < grid_size; ++x) {
        std::cout << "+---";
      }
      std::cout << "+" << std::endl;
      for (int64_t x = 0; x < grid_size; ++x) {
        std::cout << "|" << grid[x][y][0] << grid[x][y][1] << grid[x][y][2];
      }
      std::cout << "|" << std::endl;
      for (int64_t x = 0; x < grid_size; ++x) {
        std::cout << "|" << grid[x][y][3] << grid[x][y][4] << grid[x][y][5];
      }
      std::cout << "|" << std::endl;
      for (int64_t x = 0; x < grid_size; ++x) {
        std::cout << "|" << grid[x][y][6] << grid[x][y][7] << grid[x][y][8];
      }
      std::cout << "|" << std::endl;
    }
    for (int64_t x = 0; x < grid_size; ++x) {
      std::cout << "+---";
    }
    std::cout << "+" << std::endl;
  }

  std::optional<double> GetHeuristicUpper(const DetMCVI::BeliefStates& belief,
                                          int64_t max_depth) const override {
    return heuristicUpper(belief, max_depth);
  }

  std::pair<double, bool> get_state_value(const DetMCVI::State& state,
                                          int64_t max_depth) const {
    const auto f = state_value.find(state);
    if (f != state_value.cend()) {
      const auto val = f->second.first;
      return val;
    }
    const auto b = bestPath(state, max_depth);
    state_value.put(state, b);
    return b;
  }

  std::tuple<DetMCVI::State, int64_t, double, bool> Step(
      const DetMCVI::State& sI, int64_t aI) const override {
    DetMCVI::State sNext;
    const double reward = applyActionToState(sI, aI, sNext);
    const int64_t oI = observeState(sI, sNext, aI);
    const bool finished = isTerminal(sNext);
    // sI_next, oI, Reward, Done
    return std::tuple<DetMCVI::State, int64_t, double, bool>(sNext, oI, reward,
                                                             finished);
  }

  DetMCVI::State SampleStartState() override {
    static std::uniform_real_distribution<> pit_dist(0, 1);
    static std::uniform_int_distribution<int64_t> grid_dist(
        1, grid_size * grid_size - 1);

    std::map<std::string, int64_t> state_factors;

    // the entrance/exit are fixed at (0,0) but agent starts in special init
    // state for init observation
    state_factors["player_pos"] = 0;
    state_factors["player_dir"] = 1;  // Facing East
    state_factors["player_state"] = 2;
    state_factors["player_gold"] = 0;
    state_factors["player_arrow"] = 1;

    const int64_t wumpus_loc = grid_dist(rng);
    const int64_t gold_loc = grid_dist(rng);
    state_factors["wumpus_x"] = wumpus_loc / grid_size;
    state_factors["wumpus_y"] = wumpus_loc % grid_size;
    state_factors["gold_x"] = gold_loc / grid_size;
    state_factors["gold_y"] = gold_loc % grid_size;

    for (int x = 0; x < grid_size; ++x) {
      for (int y = 0; y < grid_size; ++y) {
        state_factors[coord2str(x, y) + "_pit"] =
            pit_dist(rng) < pit_probability;
      }
    }
    state_factors[coord2str(0, 0) + "_pit"] = 0;

    const auto st = names2state(state_factors);
    return st;
  }

  double applyActionToState(const DetMCVI::State& sI, int64_t aI,
                            DetMCVI::State& sNext) const {
    sNext = sI;
    if (isTerminal(sI)) return 0;
    if (sI[sfIdx("player_state")] == 2) {  // initial state
      sNext[sfIdx("player_state")] = 0;
      return 0;
    }

    const int64_t loc = sI[sfIdx("player_pos")];
    const int64_t dir = sI[sfIdx("player_dir")];
    const int64_t loc_x = loc / grid_size;
    const int64_t loc_y = loc % grid_size;
    const int64_t x_inc = (dir == 1) ? 1 : ((dir == 3) ? -1 : 0);
    const int64_t y_inc = (dir == 0) ? 1 : ((dir == 2) ? -1 : 0);

    if (actions[aI] == "forward") {
      // calc next coords
      const int64_t x = (loc_x + x_inc >= 0 && loc_x + x_inc < grid_size)
                            ? loc_x + x_inc
                            : loc_x;
      const int64_t y = (loc_y + y_inc >= 0 && loc_y + y_inc < grid_size)
                            ? loc_y + y_inc
                            : loc_y;
      const int64_t next_loc = x * grid_size + y;
      // move
      sNext[sfIdx("player_pos")] = next_loc;
      // check for wumpus/pit
      if ((sI[sfIdx("wumpus_x")] == x && sI[sfIdx("wumpus_y")] == y) ||
          coordHasItem(sI, x, y, "pit")) {
        sNext[sfIdx("player_state")] = 3;  // dead
        return _failed_reward;
      }
      return _action_reward;

    } else if (actions[aI] == "turn_90_deg_left") {  // turn left
      sNext[sfIdx("player_dir")] = (sI[sfIdx("player_dir")] + 3) % 4;
      return _action_reward;

    } else if (actions[aI] == "turn_90_deg_right") {  // turn right
      sNext[sfIdx("player_dir")] = (sI[sfIdx("player_dir")] + 1) % 4;
      return _action_reward;

    } else if (actions[aI] == "grab" &&
               (sI[sfIdx("gold_x")] == loc_x &&
                sI[sfIdx("gold_y")] == loc_y)) {  // grab gold
      // remove gold from world
      sNext[sfIdx("gold_x")] = grid_size;
      sNext[sfIdx("gold_y")] = grid_size;
      // put gold in inventory
      sNext[sfIdx("player_gold")] = 1;
      return _grab_gold_reward;

    } else if (actions[aI] == "shoot" &&
               sI[sfIdx("player_arrow")] == 1) {  // shoot arrow
      // remove an arrow from inventory
      sNext[sfIdx("player_arrow")] = 0;
      // kill wumpus in that direction
      int64_t x = loc_x + x_inc;
      int64_t y = loc_y + y_inc;
      int64_t x_end = (x_inc != 0) ? (x_inc > 0 ? grid_size : -1) : loc_x;
      int64_t y_end = (y_inc != 0) ? (y_inc > 0 ? grid_size : -1) : loc_y;

      while ((x_inc != 0 && x != x_end) || (y_inc != 0 && y != y_end)) {
        if (sI[sfIdx("wumpus_x")] == x && sI[sfIdx("wumpus_y")] == y) {
          sNext[sfIdx("wumpus_x")] = grid_size;
          sNext[sfIdx("wumpus_y")] = grid_size;
          break;
        }
        x += x_inc;
        y += y_inc;
      }

      return _shoot_reward;

    } else if (actions[aI] == "climb" && loc == 0) {
      if (sI[sfIdx("player_gold")] == 1) {
        sNext[sfIdx("player_state")] = 1;  // success
        return _success_reward;
      }
      return _bad_action_reward;
    }

    return _bad_action_reward;
  }

  int64_t sfIdx(const std::string& state_factor) const {
    const auto sf_sz = state_factor_sizes.find(state_factor);
    if (sf_sz == state_factor_sizes.cend())
      throw std::logic_error("Could not find state factor " + state_factor);
    return (int64_t)std::distance(state_factor_sizes.cbegin(), sf_sz);
  }

 private:
  std::string coord2str(int64_t i, int64_t j) const {
    return std::to_string(i) + "_" + std::to_string(j);
  }

  bool coordHasItem(const DetMCVI::State& state, int64_t x, int64_t y,
                    const std::string& item) const {
    return state[sfIdx(coord2str(x, y) + "_" + item)];
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

  static inline bool isBitSet(int64_t num, size_t bit) {
    return 1 == ((num >> bit) & 1);
  }

  std::vector<std::string> initObs() const {
    std::vector<std::string> obs_components = {"stench", "breeze", "glitter",
                                               "bump", "scream"};

    std::vector<std::string> obs;
    for (int64_t b = 0; b < (1 << obs_components.size()); ++b) {
      std::string o = "*";
      for (size_t i = 0; i < obs_components.size(); ++i)
        if (isBitSet(b, i)) o += obs_components[i];
      obs.push_back(o);
    }
    return obs;
  }

  std::map<std::string, size_t> initStateSpace() const {
    std::map<std::string, size_t> state_factors;

    // For each square the following conditions can be met:
    // contains wumpus
    // contains pit
    // contains gold
    // Other variables are:
    // player position
    // game state

    for (int x = 0; x < grid_size; ++x) {
      for (int y = 0; y < grid_size; ++y) {
        state_factors[coord2str(x, y) + "_pit"] = 2;  // 0=no pit, 1=pit
      }
    }
    state_factors["player_pos"] =
        grid_size * grid_size;  // grid coord (NxM grid in the form of M*x+y)
    state_factors["player_dir"] = 4;  // 0=N, 1=E, 2=S, 3=W
    state_factors["player_gold"] = 2;
    state_factors["player_arrow"] = 2;
    state_factors["player_state"] = 4;  // 2=init, 0=playing, 1=success, 3=dead
    state_factors["wumpus_x"] = grid_size + 1;
    state_factors["wumpus_y"] = grid_size + 1;
    state_factors["gold_x"] = grid_size + 1;
    state_factors["gold_y"] = grid_size + 1;

    size_t p = 1;
    for (const auto& [sf, sz] : state_factors) p *= sz;
    std::cout << "State space size: " << p << std::endl;

    return state_factors;
  }

  int64_t observeState(const DetMCVI::State& sPrev, const DetMCVI::State& sI,
                       int64_t aI) const {
    std::string obs = "*";
    if (sI[sfIdx("player_state")] == 1 ||
        sI[sfIdx("player_state")] == 3) {  // terminal state
      return (int64_t)std::distance(
          observations.cbegin(),
          std::find(observations.cbegin(), observations.cend(), obs));
    }

    const int64_t loc = sI[sfIdx("player_pos")];
    const int64_t loc_x = loc / grid_size;
    const int64_t loc_y = loc % grid_size;

    for (int i = 0; i < 4; ++i) {  // adjacent wumpus
      const int64_t x = (i == 1) ? loc_x + 1 : ((i == 3) ? loc_x - 1 : loc_x);
      const int64_t y = (i == 0) ? loc_y + 1 : ((i == 2) ? loc_y - 1 : loc_y);
      if (x >= grid_size || x < 0 || y >= grid_size || y < 0) continue;
      if ((sI[sfIdx("wumpus_x")] == x && sI[sfIdx("wumpus_y")] == y)) {
        obs += "stench";
        break;
      }
    }

    for (int i = 0; i < 4; ++i) {  // adjacent pit
      const int64_t x = (i == 1) ? loc_x + 1 : ((i == 3) ? loc_x - 1 : loc_x);
      const int64_t y = (i == 0) ? loc_y + 1 : ((i == 2) ? loc_y - 1 : loc_y);
      if (x >= grid_size || x < 0 || y >= grid_size || y < 0) continue;
      if (coordHasItem(sI, x, y, "pit")) {
        obs += "breeze";
        break;
      }
    }

    if ((sI[sfIdx("gold_x")] == loc_x && sI[sfIdx("gold_y")] == loc_y))
      obs += "glitter";

    if (actions[aI] == "forward") {  // bump into wall
      const int64_t prev_loc = sPrev[sfIdx("player_pos")];
      const int64_t prev_loc_x = prev_loc / grid_size;
      const int64_t prev_loc_y = prev_loc % grid_size;
      const int64_t dir = sPrev[sfIdx("player_dir")];
      const int64_t x_inc = (dir == 1) ? 1 : ((dir == 3) ? -1 : 0);
      const int64_t y_inc = (dir == 0) ? 1 : ((dir == 2) ? -1 : 0);

      if (prev_loc_x + x_inc < 0 || prev_loc_x + x_inc >= grid_size ||
          prev_loc_y + y_inc < 0 || prev_loc_y + y_inc >= grid_size)
        obs += "bump";
    } else if (actions[aI] == "shoot" && sPrev[sfIdx("player_arrow")] == 1 &&
               sPrev[sfIdx("wumpus_x")] != grid_size &&
               sPrev[sfIdx("wumpus_y")] != grid_size &&
               sI[sfIdx("wumpus_x")] == grid_size &&
               sI[sfIdx("wumpus_y")] == grid_size) {  // hear wumpus scream
      obs += "scream";
    }

    return (int64_t)std::distance(
        observations.cbegin(),
        std::find(observations.cbegin(), observations.cend(), obs));
  }

  // find an upper bound for the value of a belief
  double heuristicUpper(const DetMCVI::BeliefStates& belief,
                        int64_t max_depth) const {
    double val = 0;
    if (belief.size() <= heuristic_samples) {
      // #pragma omp parallel for reduction(+ : val)
      for (size_t i = 0; i < belief.size(); ++i) {
        val += get_state_value(belief[i].first, max_depth).first *
               belief[i].second;
      }
    } else {
      const auto b = DetMCVI::beliefStatesToDistribution(belief);
      for (size_t i = 0; i < heuristic_samples; ++i) {
        const auto state = DetMCVI::SampleOneState(b, rng);
        val += get_state_value(state, max_depth).first;
      }
      val /= heuristic_samples;
    }

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

 public:
  void toSARSOP(std::ostream& os, int64_t init_belief_sz) {
    std::vector<DetMCVI::State> state_enum = enumerateStates();
    const size_t num_states = state_enum.size();
    os << "discount: " << GetDiscount() << std::endl;
    os << "values: reward" << std::endl;
    os << "states: " << num_states << std::endl;
    os << "actions: " << GetSizeOfA() << std::endl;
    os << "observations: " << GetSizeOfObs() << std::endl << std::endl;

    // Initial belief
    os << "start: " << std::endl;
    auto init_belief = SampleInitialBelief(init_belief_sz, this);
    for (const auto& s : state_enum) os << std::fixed << init_belief[s] << " ";
    os << std::endl << std::endl;

    // Transition probabilities  T : <action> : <start-state> : <end-state> %f
    // Observation probabilities O : <action> : <end-state> : <observation> %f
    // Reward     R: <action> : <start-state> : <end-state> : <observation> %f
    for (size_t sI = 0; sI < state_enum.size(); ++sI) {
      for (int64_t a = 0; a < GetSizeOfA(); ++a) {
        DetMCVI::State sNext;
        const double reward = applyActionToState(state_enum[sI], a, sNext);
        const int64_t obs = observeState(state_enum[sI], state_enum[sI], a);
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

void ReadWumpusParams(const std::string& filename, int64_t& grid_size) {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Unable to open file: " + filename);
  if (!(file >> grid_size))
    throw std::runtime_error("Error reading the grid_size from file");
  file.close();
}
