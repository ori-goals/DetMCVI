#pragma once

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "BeliefDistribution.h"
#include "Cache.h"
#include "ShortestPath.h"
#include "SimInterface.h"

class Maze : public DetMCVI::SimInterface,
             public DetMCVI::ShortestPathFasterAlgorithm {
 private:
  std::vector<std::string> _maze;
  std::vector<std::string> actions = {"forward", "turn_right", "turn_around",
                                      "turn_left"};
  std::vector<std::string> observations;  // observe the surrounding walls

  std::mt19937_64& rng;

  double _success_reward = -1;
  double _move_reward = -1;
  double _bad_action_reward = -10;

  int64_t state_space_sz;
  mutable DetMCVI::LRUCache<DetMCVI::State, double, DetMCVI::StateHash,
                            DetMCVI::StateEqual>
      state_value_cache;

 public:
  Maze(std::vector<std::string> maze, std::mt19937_64& rng)
      : _maze(maze),
        observations(initObs()),
        rng(rng),
        state_space_sz(4 * (countBlankSpaces(_maze)) + 1),
        state_value_cache(250000) {}

  int64_t GetSizeOfObs() const override { return observations.size(); }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  const std::vector<std::string>& getActions() const { return actions; }
  const std::vector<std::string>& getObs() const { return observations; }
  bool isTerminal(const DetMCVI::State& sI) const override {
    return sI[0] == 0 && sI[2] == 0;
  }

  void drawState(const DetMCVI::State& sI) const {
    const auto m = indexToPlayerLocation(_maze, sI[0], sI[1]);
    return printMaze(m);
  }

  std::optional<double> GetHeuristicUpper(const DetMCVI::BeliefStates& belief,
                                          int64_t max_depth) const override {
    return heuristicUpper(belief, max_depth);
  }

  std::pair<double, bool> get_state_value(const DetMCVI::State& state,
                                          int64_t max_depth) const {
    const auto f = state_value_cache.find(state);
    if (f != state_value_cache.cend()) return {f->second.first, true};
    const auto b = bestPath(state, max_depth);
    if (b.second) state_value_cache.put(state, b.first);
    return b;
  }

  std::tuple<DetMCVI::State, int64_t, double, bool> Step(
      const DetMCVI::State& sI, int64_t aI) const override {
    DetMCVI::State sNext;
    // drawState(sI);
    const double reward = applyActionToState(sI, aI, sNext);
    const int64_t oI = observeState(sNext);
    const bool finished = isTerminal(sNext);
    // std::cout << "Action: " << actions[aI]
    //   << " observation: " << observations[oI] << std::endl;
    // std::cout << "Reward: " << reward << std::endl;
    // drawState(sNext);
    // std::cout << std::endl << std::endl;
    // sI_next, oI, Reward, Done
    return std::tuple<DetMCVI::State, int64_t, double, bool>(sNext, oI, reward,
                                                             finished);
  }

  DetMCVI::State SampleStartState() override {
    // Start in any available position other than the goal
    static std::uniform_int_distribution<int64_t> ss_dist(
        1, (state_space_sz - 1) / 4);
    static std::uniform_int_distribution<int64_t> o_dist(0, 3);
    return {ss_dist(rng), o_dist(rng), 1};
  }

  double applyActionToState(const DetMCVI::State& sI, int64_t aI,
                            DetMCVI::State& sNext) const {
    sNext = sI;
    if (isTerminal(sI)) return 0;
    if (sI[2] != 0) {  // init state
      sNext[2] = 0;
      return 0;
    }
    const auto curr_maze = indexToPlayerLocation(_maze, sI[0], sI[1]);
    const auto& [x, y, o] = findPlayerLocation(curr_maze);
    if (x == -1 || y == -1)
      throw std::logic_error("Cannot find player location for state " +
                             std::to_string(sI[0]));

    auto next_maze = curr_maze;
    const std::string action = actions[aI];

    if (action == "forward") {
      char above = (x > 0) ? curr_maze[x - 1][y] : '#';
      char right =
          (y < (int64_t)_maze[0].size() - 1) ? curr_maze[x][y + 1] : '#';
      char below = (x < (int64_t)_maze.size() - 1) ? curr_maze[x + 1][y] : '#';
      char left = (y > 0) ? curr_maze[x][y - 1] : '#';

      std::vector<char> walls_seen = {above, right, below, left};
      std::vector<std::pair<int64_t, int64_t>> next_xy = {
          {x - 1, y}, {x, y + 1}, {x + 1, y}, {x, y - 1}};
      // walls_seen: in front, right, behind, left
      if (walls_seen[o] == ' ' || walls_seen[o] == 'G') {
        next_maze[next_xy[o].first][next_xy[o].second] = "^>V<"[o];
        next_maze[x][y] = ' ';
      } else {
        return _bad_action_reward;
      }
    } else if (action == "turn_right") {
      const char new_orientation = ">V<^"[o];
      next_maze[x][y] = new_orientation;
    } else if (action == "turn_left") {
      const char new_orientation = "<^>V"[o];
      next_maze[x][y] = new_orientation;
    } else if (action == "turn_around") {
      const char new_orientation = "V<^>"[o];
      next_maze[x][y] = new_orientation;
    }

    // Player location is same as goal
    if (findGoalLocation(next_maze).first == -1) {
      sNext = {0, 0, 0};
      return _success_reward;
    }

    const auto& [index, orientation] = playerLocationToIndex(next_maze);
    sNext = {index, orientation, 0};
    return _move_reward;
  }

 private:
  void printMaze(const std::vector<std::string>& maze) const {
    for (const std::string& row : maze) {
      std::cout << row << std::endl;
    }
  }

  std::tuple<int64_t, int64_t, uint8_t> findPlayerLocation(
      const std::vector<std::string>& maze) const {
    for (int64_t r = 0; r < (int64_t)maze.size(); ++r) {
      size_t col = maze[r].find_first_of("^<>V");
      if (col != std::string::npos) {
        uint8_t orientation = 0;
        switch (maze[r][col]) {
          case '^':
            orientation = 0;
            break;
          case '>':
            orientation = 1;
            break;
          case 'V':
            orientation = 2;
            break;
          case '<':
            orientation = 3;
            break;
        }
        return {r, static_cast<int64_t>(col), orientation};
      }
    }
    return {-1, -1, 0};  // not found
  }

  std::pair<int64_t, int64_t> findGoalLocation(
      const std::vector<std::string>& maze) const {
    for (int64_t r = 0; r < (int64_t)maze.size(); ++r) {
      size_t col = maze[r].find('G');
      if (col != std::string::npos) {
        return {r, static_cast<int64_t>(col)};
      }
    }
    return {-1, -1};  // not found
  }

  std::pair<int64_t, uint8_t> playerLocationToIndex(
      const std::vector<std::string>& maze) const {
    if (std::get<0>(findPlayerLocation(maze)) == -1) return {-1, 0};
    if (findGoalLocation(maze).first == -1) return {0, 0};

    int64_t index = 1;  // Start at 1 since G is index 0

    for (int64_t r = 0; r < (int64_t)maze.size(); ++r) {
      for (int64_t c = 0; c < (int64_t)maze[r].size(); ++c) {
        if (maze[r][c] == ' ' || maze[r][c] == '^' || maze[r][c] == '>' ||
            maze[r][c] == 'V' || maze[r][c] == '<') {
          if (maze[r][c] == '^' || maze[r][c] == '>' || maze[r][c] == 'V' ||
              maze[r][c] == '<') {
            uint8_t orientation = 0;
            switch (maze[r][c]) {
              case '^':
                orientation = 0;
                break;
              case '>':
                orientation = 1;
                break;
              case 'V':
                orientation = 2;
                break;
              case '<':
                orientation = 3;
                break;
            }
            return {index, orientation};
          }
          ++index;
        }
      }
    }

    return {-1, 0};  // No player location is found
  }

  std::vector<std::string> indexToPlayerLocation(
      const std::vector<std::string>& maze, int64_t index,
      uint8_t orientation) const {
    std::vector<std::string> blankMaze = maze;
    const char player = "^>V<"[orientation];

    if (index == 0) {  // player is at goal location
      const auto goal_loc = findGoalLocation(maze);
      if (goal_loc.first == -1 || goal_loc.second == -1)
        return blankMaze;  // bad
      blankMaze[goal_loc.first][goal_loc.second] = player;
      return blankMaze;
    }

    int64_t currentIndex = 1;
    for (int64_t r = 0; r < (int64_t)blankMaze.size(); ++r) {
      for (int64_t c = 0; c < (int64_t)blankMaze[r].size(); ++c) {
        if (blankMaze[r][c] == ' ') {
          if (index == currentIndex) {
            blankMaze[r][c] = player;
            return blankMaze;
          }
          ++currentIndex;
        }
      }
    }

    return blankMaze;
  }

  static inline bool isBitSet(int64_t num, size_t bit) {
    return 1 == ((num >> bit) & 1);
  }

  std::vector<std::string> initObs() const {
    std::vector<std::string> obs_components = {"in_front", "right", "behind",
                                               "left"};

    std::vector<std::string> obs;
    for (int64_t b = 0; b < (1 << obs_components.size()); ++b) {
      std::string o = "*";
      for (size_t i = 0; i < obs_components.size(); ++i)
        if (isBitSet(b, i)) o += obs_components[i];
      obs.push_back(o);
    }
    return obs;
  }

  int64_t observeState(const DetMCVI::State& sI) const {
    std::string obs = "*";
    const auto curr_maze = indexToPlayerLocation(_maze, sI[0], sI[1]);
    const auto& [x, y, o] = findPlayerLocation(curr_maze);
    if (x == -1 || y == -1)
      throw std::logic_error("Cannot find player location for state " +
                             std::to_string(sI[0]));

    char above = (x > 0) ? curr_maze[x - 1][y] : '#';
    char right = (y < (int64_t)_maze[0].size() - 1) ? curr_maze[x][y + 1] : '#';
    char below = (x < (int64_t)_maze.size() - 1) ? curr_maze[x + 1][y] : '#';
    char left = (y > 0) ? curr_maze[x][y - 1] : '#';

    std::vector<char> walls_seen = {above, right, below, left};
    std::rotate(walls_seen.begin(), walls_seen.begin() + o,
                walls_seen.end());  // rotate according to orientation

    if (walls_seen[0] != ' ' && walls_seen[0] != 'G') obs += "in_front";
    if (walls_seen[1] != ' ' && walls_seen[1] != 'G') obs += "right";
    if (walls_seen[2] != ' ' && walls_seen[2] != 'G') obs += "behind";
    if (walls_seen[3] != ' ' && walls_seen[3] != 'G') obs += "left";

    return (int64_t)std::distance(
        observations.cbegin(),
        std::find(observations.cbegin(), observations.cend(), obs));
  }

  double heuristicUpper(const DetMCVI::BeliefStates& belief,
                        int64_t max_depth) const {
    double val = 0;
    // #pragma omp parallel for reduction(+ : val)
    for (size_t i = 0; i < belief.size(); ++i) {
      val +=
          get_state_value(belief[i].first, max_depth).first * belief[i].second;
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

  int64_t countBlankSpaces(const std::vector<std::string>& maze) {
    int64_t count = 0;
    for (const auto& row : maze) {
      for (char ch : row) {
        if (ch == ' ') ++count;
      }
    }
    return count;
  }

 public:
  void toSARSOP(std::ostream& os) {
    std::vector<DetMCVI::State> state_enum;
    state_enum.push_back({0, 0, 0});  // goal state
    const int num_blank_spaces = (state_space_sz - 1) / 4;
    for (int64_t s = 1; s < num_blank_spaces + 1; ++s)
      for (int64_t o = 0; o < 4; ++o) state_enum.push_back({s, o, 0});
    for (int64_t s = 1; s < num_blank_spaces + 1; ++s)
      for (int64_t o = 0; o < 4; ++o) state_enum.push_back({s, o, 1});
    const size_t num_states = state_enum.size();
    const double n = 4 * std::sqrt(((state_space_sz - 1) / 4 + 2) / 2);
    os << "discount: " << std::exp(std::log(0.01) / (2.0 * n * n + n))
       << std::endl;
    os << "values: reward" << std::endl;
    os << "states: " << num_states << std::endl;
    os << "actions: ";
    for (const auto& act : actions) os << act << " ";
    os << std::endl;
    os << "observations: " << GetSizeOfObs() << std::endl << std::endl;

    // Initial belief
    os << "start include: ";
    for (int64_t s = state_space_sz; s < 2 * state_space_sz - 1; ++s)
      os << s << " ";
    os << std::endl;

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

void ReadMazeParams(const std::string& filename,
                    std::vector<std::string>& maze) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    maze.push_back(line);
  }

  file.close();
}
