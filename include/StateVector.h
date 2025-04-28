// A class for representing a factored state representation

/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace DetMCVI {

using State = std::vector<int64_t>;

struct StateHash {
  std::size_t operator()(const State& vec) const {
    std::size_t hash = 0;
    std::hash<int64_t> hasher;
    for (int64_t i : vec)
      hash ^= hasher(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct StateEqual {
  bool operator()(const State& lhs, const State& rhs) const {
    return lhs == rhs;
  }
};

template <typename T>
using StateMap = std::unordered_map<State, T, StateHash, StateEqual>;

}  // namespace DetMCVI
