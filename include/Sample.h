// Provides functions for probability mass functions

/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 *
 */

#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace DetMCVI {

/// @brief Sum the probabilities of a PMF
template <typename T, typename Hash = std::hash<T>,
          typename Equal = std::equal_to<T>>
double SumPMF(const std::unordered_map<T, double, Hash, Equal>& pmf) {
  double sum_p = 0.0;
  for (const auto& [s, p] : pmf) sum_p += p;
  return sum_p;
}

/// @brief Sample from a PMF
template <typename T, typename Hash = std::hash<T>,
          typename Equal = std::equal_to<T>>
T SamplePMF(const std::unordered_map<T, double, Hash, Equal>& pmf,
            std::mt19937_64& rng) {
  const double max_p = SumPMF<T>(pmf);
  std::uniform_real_distribution<> dist(0, max_p);
  const double u = dist(rng);
  double sum_p = 0.0;
  for (const auto& [s, p] : pmf) {
    sum_p += p;
    if (sum_p > u) return s;
  }
  throw std::runtime_error("Unable to sample from pmf");
}

/// @brief Perform a weighted shuffle of a PMF
/// @details This function takes a probability mass function (PMF) and
/// generates a weighted shuffle of its elements. The weights are
/// determined by the inverse of the PMF values, and the elements are
/// sorted based on these weights. The function returns a vector of
/// pairs, where each pair contains an element from the PMF and its
/// corresponding weight.
/// @param pmf The probability mass function to shuffle.
/// @param rng A random number generator.
/// @param sample_cap The maximum number of elements to include in the
/// shuffle.
/// @return The shuffled elements and their weights.
template <typename T, typename Hash = std::hash<T>,
          typename Equal = std::equal_to<T>>
std::vector<std::pair<T, double>> weightedShuffle(
    const std::unordered_map<T, double, Hash, Equal>& pmf, std::mt19937_64& rng,
    size_t sample_cap) {
  auto exp_dist = std::exponential_distribution<double>();

  std::vector<std::pair<double, std::pair<T, double>>> index_pairs;
  index_pairs.reserve(pmf.size());
  for (const auto& elem : pmf) {
    const double p = elem.second;
    index_pairs.emplace_back(exp_dist(rng) / p, elem);
  }
  std::sort(index_pairs.begin(), index_pairs.end());

  std::vector<std::pair<T, double>> indices;
  indices.reserve(std::min(pmf.size(), sample_cap));
  for (const auto& [w, pair] : index_pairs) {
    if (indices.capacity() == indices.size()) break;
    indices.emplace_back(pair);
  }
  return indices;
}

}  // namespace DetMCVI
