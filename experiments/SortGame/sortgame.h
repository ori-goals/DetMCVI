#pragma once

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

#include "SimInterface.h"

class SortGame : public DetMCVI::SimInterface {
 private:
  int item_count;
  std::vector<std::pair<int, int>> actions;
  DetMCVI::State goal_state;
  std::vector<int> observations;

  std::mt19937_64& rng;

  double _success_reward = -10;
  double _guess_reward = -10;

 public:
  SortGame(int item_count, std::mt19937_64& rng)
      : item_count(item_count),
        actions(initActions()),
        goal_state(initGoalState()),
        observations(initObs()),
        rng(rng) {}

  int64_t GetSizeOfObs() const override { return observations.size(); }
  int64_t GetSizeOfA() const override { return actions.size(); }
  double GetDiscount() const override { return 1.0; }
  int64_t GetNbAgent() const override { return 1; }
  bool isTerminal(const DetMCVI::State& sI) const override {
    return sI == goal_state;
  }

  std::tuple<DetMCVI::State, int64_t, double, bool> Step(
      const DetMCVI::State& sI, int64_t aI) const override {
    DetMCVI::State sNext;
    const double reward = applyActionToState(sI, aI, sNext);
    const int64_t oI = observeState(sNext);
    const bool finished = isTerminal(sNext);
    // sI_next, oI, Reward, Done
    return std::tuple<DetMCVI::State, int64_t, double, bool>(sNext, oI, reward,
                                                             finished);
  }

  DetMCVI::State initGoalState() {
    std::vector<int64_t> goal_state;
    for (int i = 0; i < item_count; ++i) goal_state.push_back(i);
    return goal_state;
  }

  DetMCVI::State SampleStartState() override {
    for (int i = 0; i < 100; ++i) {
      std::vector<int64_t> items(item_count);
      std::iota(items.begin(), items.end(), 0);
      std::shuffle(items.begin(), items.end(), rng);
      if (items != goal_state) return items;
    }
    throw std::runtime_error("Could not generate a start state");
  }

  double applyActionToState(const DetMCVI::State& sI, int64_t aI,
                            DetMCVI::State& sNext) const {
    sNext = sI;
    if (isTerminal(sI)) return 0;

    const auto& [idx1, idx2] = actions.at(aI);
    std::swap(sNext.at(idx1), sNext.at(idx2));
    return sNext == goal_state ? _success_reward : _guess_reward;
  }

 private:
  int SpearmanFootruleDistance(const DetMCVI::State& sI) const {
    int distance = 0;
    for (int i = 0; i < item_count; ++i) {
      distance += std::abs(sI[i] - goal_state[i]);
    }
    return distance;
  }

  std::vector<std::pair<int, int>> initActions() const {
    std::vector<std::pair<int, int>> actions;
    for (int i = 0; i < item_count; ++i) {
      for (int j = i + 1; j < item_count; ++j) {
        actions.push_back({i, j});
      }
    }
    return actions;
  }

  std::vector<int> initObs() const {
    auto reversed_goal_state = goal_state;
    std::reverse(reversed_goal_state.begin(), reversed_goal_state.end());
    const auto max_distance = SpearmanFootruleDistance(reversed_goal_state);
    std::vector<int> observations;
    for (int i = 0; i <= max_distance; ++i) observations.push_back(i);
    return observations;
  }

  int64_t observeState(const DetMCVI::State& sI) const {
    return SpearmanFootruleDistance(sI);
  }
};

void ReadSortGameParams(const std::string& filename, int& item_count) {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Unable to open file: " + filename);
  if (!(file >> item_count))
    throw std::runtime_error("Error reading the item_count from file");
  file.close();
}
