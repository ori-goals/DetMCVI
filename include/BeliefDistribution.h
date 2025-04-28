// Provides two classes for representing a belief distribution:
//  * BeliefDistribution: a map from states to probabilities
//  * BeliefStates: a vector of pairs of states and probabilities

/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once
#include <Sample.h>

#include <iostream>

#include "SimInterface.h"
#include "StateVector.h"

namespace DetMCVI {

using BeliefDistribution = StateMap<double>;
using BeliefStates = std::vector<std::pair<State, double>>;

/// @brief Convert a belief distribution to a vector of belief states
BeliefStates beliefDistributionToStates(const BeliefDistribution& b);

/// @brief Convert a vector of belief states to a belief distribution
BeliefDistribution beliefStatesToDistribution(const BeliefStates& b);

/// @brief Sample a state from a belief distribution
State SampleOneState(const BeliefDistribution& belief, std::mt19937_64& rng);

std::ostream& operator<<(std::ostream& os, const BeliefDistribution& bd);
std::ostream& operator<<(std::ostream& os, const BeliefStates& b);

/// @brief Sample an initial belief distribution from a POMDP
/// @param N The number of samples to draw
/// @param pomdp The POMDP simulator
/// @return A belief distribution
BeliefDistribution SampleInitialBelief(size_t N, SimInterface* pomdp);

/// @brief Downsample a belief distribution to a maximum number of states
/// @param belief The original belief distribution
/// @param max_belief_samples The maximum number of states in the downsampled
/// distribution
/// @param rng A random number generator
/// @return A downsampled belief distribution
BeliefDistribution DownsampleBelief(const BeliefDistribution& belief,
                                    int64_t max_belief_samples,
                                    std::mt19937_64& rng);
}  // namespace DetMCVI
