#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>

#include "AnytimeAOStar.h"
#include "CTP.h"
#include "DetMCVI.h"
#include "Params.h"
#include "QMDPTree.h"

using namespace DetMCVI;

std::atomic<bool> exit_flag = false;

AlphaVectorFSC runMCVIIncrements(
    CTP* pomdp, const BeliefStates& init_belief, std::mt19937_64& rng,
    int64_t max_sim_depth, int64_t max_node_size, int64_t eval_depth,
    int64_t eval_epsilon, double converge_thresh, int64_t max_time_ms,
    int64_t max_eval_steps, int64_t n_eval_trials, int64_t nb_particles_b0,
    int64_t eval_interval_ms, int64_t completion_threshold,
    int64_t completion_reps, OptimalPath& solver, int64_t set_number,
    int64_t seed, std::ostream& os) {
  // Initialise the FSC
  const auto init_fsc = AlphaVectorFSC(max_node_size);

  // Run MCVI
  std::cout << "Running DetMCVI" << std::endl;
  auto planner = MCVIPlanner(pomdp, init_fsc, init_belief, solver, rng);
  const auto [fsc, root] = planner.PlanAndEvaluate(
      max_sim_depth, converge_thresh, std::numeric_limits<int64_t>::max(),
      max_time_ms, eval_depth, eval_epsilon, max_eval_steps, n_eval_trials,
      nb_particles_b0, eval_interval_ms, completion_threshold, completion_reps,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      exit_flag, set_number, seed, os);
  return fsc;
}

std::shared_ptr<BeliefTreeNode> runAOStarIncrements(
    CTP* pomdp, const BeliefStates& init_belief, std::mt19937_64& rng,
    int64_t eval_depth, int64_t max_time_ms, int64_t max_eval_steps,
    int64_t n_eval_trials, int64_t nb_particles_b0, int64_t eval_interval_ms,
    int64_t completion_threshold, int64_t completion_reps, int64_t node_limit,
    OptimalPath& solver, int64_t set_number, int64_t seed, std::ostream& os) {
  // Initialise heuristic
  OptimalPath heuristic(pomdp);

  // Create root belief node
  auto upperBoundFunc = [&heuristic](const BeliefStates& belief,
                                     int64_t belief_depth, int64_t eval_depth,
                                     SimInterface* sim) {
    return UpperBoundEvaluation(belief, heuristic, sim->GetDiscount(),
                                belief_depth, belief_depth + eval_depth);
  };
  const double init_upper =
      CalculateUpperBound(init_belief, 0, eval_depth, upperBoundFunc, pomdp);
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, init_upper, -std::numeric_limits<double>::infinity());

  // Run AO*
  std::cout << "Running AO* on belief tree" << std::endl;
  RunAOStarAndEvaluate(
      root, std::numeric_limits<int64_t>::max(), max_time_ms, heuristic,
      eval_depth, max_eval_steps, n_eval_trials, nb_particles_b0,
      eval_interval_ms, completion_threshold, completion_reps, node_limit, rng,
      solver,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      pomdp, set_number, seed, os);
  return root;
}

std::shared_ptr<BeliefTreeNode> runAnytimeAOStarIncrements(
    CTP* pomdp, const BeliefStates& init_belief, std::mt19937_64& rng,
    int64_t eval_depth, int64_t max_time_ms, int64_t max_eval_steps,
    int64_t n_eval_trials, int64_t nb_particles_b0, int64_t eval_interval_ms,
    int64_t completion_threshold, int64_t completion_reps, int64_t node_limit,
    OptimalPath& solver, int64_t set_number, int64_t seed, std::ostream& os) {
  // Initialise heuristic
  OptimalPath heuristic(pomdp);

  // Create root belief node
  auto upperBoundFunc = [&heuristic](const BeliefStates& belief,
                                     int64_t belief_depth, int64_t eval_depth,
                                     SimInterface* sim) {
    return UpperBoundEvaluation(belief, heuristic, sim->GetDiscount(),
                                belief_depth, belief_depth + eval_depth);
  };
  const double init_upper =
      CalculateUpperBound(init_belief, 0, eval_depth, upperBoundFunc, pomdp);
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, init_upper, -std::numeric_limits<double>::infinity());

  // Run AO*
  std::cout << "Running AO* on belief tree" << std::endl;
  RunAnytimeAOStarAndEvaluate(
      root, std::numeric_limits<int64_t>::max(), max_time_ms, heuristic,
      eval_depth, max_eval_steps, n_eval_trials, nb_particles_b0,
      eval_interval_ms, completion_threshold, completion_reps, node_limit, rng,
      solver,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      0.5, pomdp, set_number, seed, os);
  return root;
}

std::shared_ptr<BeliefTreeNode> runQMDPIncrements(
    CTP* pomdp, const BeliefStates& init_belief, std::mt19937_64& rng,
    int64_t eval_depth, int64_t max_time_ms, int64_t max_eval_steps,
    int64_t n_eval_trials, int64_t nb_particles_b0, int64_t eval_interval_ms,
    int64_t completion_threshold, int64_t completion_reps, int64_t node_limit,
    OptimalPath& solver, int64_t set_number, int64_t seed, std::ostream& os) {
  // Initialise heuristic
  OptimalPath heuristic(pomdp);

  // Create root belief node
  auto upperBoundFunc = [&heuristic](const BeliefStates& belief,
                                     int64_t belief_depth, int64_t eval_depth,
                                     SimInterface* sim) {
    return UpperBoundEvaluation(belief, heuristic, sim->GetDiscount(),
                                belief_depth, belief_depth + eval_depth);
  };
  const double init_upper =
      CalculateUpperBound(init_belief, 0, eval_depth, upperBoundFunc, pomdp);
  std::shared_ptr<BeliefTreeNode> root = CreateBeliefTreeNode(
      init_belief, 0, init_upper, -std::numeric_limits<double>::infinity());

  // Run QMDP
  std::cout << "Running QMDP on belief tree" << std::endl;
  RunQMDPAndEvaluate(
      root, std::numeric_limits<int64_t>::max(), max_time_ms, heuristic,
      eval_depth, max_eval_steps, n_eval_trials, nb_particles_b0,
      eval_interval_ms, completion_threshold, completion_reps, node_limit, rng,
      solver,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      pomdp, set_number, seed, os);
  return root;
}

int main(int argc, char* argv[]) {
  const EvalParams params = parseArgs(argc, argv);
  std::mt19937_64 rng(params.seed);

  std::ofstream evaluation_file(params.evaluation_file);
  if (!evaluation_file.is_open()) {
    std::cerr << "Failed to open evaluation file" << std::endl;
    return 1;
  }
  StatsCSVHeader(evaluation_file);

  // Initialise the POMDP
  std::cout << "Initialising CTP" << std::endl;
  std::vector<int64_t> nodes;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> edges;
  std::unordered_map<std::pair<int64_t, int64_t>, double, pairhash> stoch_edges;
  int64_t origin;
  int64_t goal;
  ctpGraphFromFile(params.datafile, nodes, edges, stoch_edges, origin, goal);
  auto pomdp = CTP(rng, nodes, edges, stoch_edges, origin, goal);
  OptimalPath solver(&pomdp);

  std::cout << "Observation space size: " << pomdp.GetSizeOfObs() << std::endl;

  // Sample the initial belief
  std::cout << "Sampling initial belief" << std::endl;
  auto init_belief = SampleInitialBelief(params.nb_particles_b0, &pomdp);
  if (params.max_belief_samples < (int64_t)init_belief.size()) {
    std::cout << "Initial belief size: " << init_belief.size() << std::endl;
    std::cout << "Downsampling belief" << std::endl;
    init_belief = DownsampleBelief(init_belief, params.max_belief_samples, rng);
  }
  std::cout << "Initial belief size: " << init_belief.size() << std::endl;
  const auto init_belief_states = beliefDistributionToStates(init_belief);

  //   AlphaVectorFSC mcvi_fsc(params.max_node_size);
  //   std::shared_ptr<BeliefTreeNode> aostar_root = nullptr;
  //   std::shared_ptr<BeliefTreeNode> anytime_aostar_root = nullptr;
  //   std::shared_ptr<BeliefTreeNode> qmdp_root = nullptr;
  //   AlphaVectorFSC sarsop_fsc(params.max_node_size);

  {  // Run MCVI
    auto mcvi_ctp = CTP(rng, nodes, edges, stoch_edges, origin, goal);
    runMCVIIncrements(&mcvi_ctp, init_belief_states, rng, params.max_sim_depth,
                      params.max_node_size, params.max_sim_depth,
                      params.eval_epsilon, params.converge_thresh,
                      params.max_time_ms, params.max_sim_depth,
                      params.n_eval_trials, 10 * params.nb_particles_b0,
                      params.eval_interval_ms, params.completion_threshold,
                      params.completion_reps, solver, params.set_number,
                      params.seed, evaluation_file);
  }
  {  // Compare to AO*
    auto aostar_ctp = CTP(rng, nodes, edges, stoch_edges, origin, goal);
    runAOStarIncrements(&aostar_ctp, init_belief_states, rng,
                        params.max_sim_depth, params.max_time_ms,
                        params.max_sim_depth, params.n_eval_trials,
                        10 * params.nb_particles_b0, params.eval_interval_ms,
                        params.completion_threshold, params.completion_reps,
                        params.max_node_size, solver, params.set_number,
                        params.seed, evaluation_file);
  }
  {  // Compare to AO*
    auto aostar_ctp = CTP(rng, nodes, edges, stoch_edges, origin, goal);
    runAnytimeAOStarIncrements(
        &aostar_ctp, init_belief_states, rng, params.max_sim_depth,
        params.max_time_ms, params.max_sim_depth, params.n_eval_trials,
        10 * params.nb_particles_b0, params.eval_interval_ms,
        params.completion_threshold, params.completion_reps,
        params.max_node_size, solver, params.set_number, params.seed,
        evaluation_file);
  }
  {  // Compare to QMDP
    auto qmdp_ctp = CTP(rng, nodes, edges, stoch_edges, origin, goal);
    runQMDPIncrements(&qmdp_ctp, init_belief_states, rng, params.max_sim_depth,
                      params.max_time_ms, params.max_sim_depth,
                      params.n_eval_trials, 10 * params.nb_particles_b0,
                      params.eval_interval_ms, params.completion_threshold,
                      params.completion_reps, params.max_node_size, solver,
                      params.set_number, params.seed, evaluation_file);
  }
  //   {  // Compare to SARSOP
  //     const std::string sarsop_policy =
  //         generateSARSOPFilename("CTP", nodes.size(), params.set_number);

  //     if (std::ifstream(sarsop_policy)) {
  //       sarsop_fsc = ParseDotFile(sarsop_policy);

  //       EvaluationWithSimulationFSC(
  //           params.max_sim_depth, params.n_eval_trials,
  //           params.nb_particles_b0,
  //           [&pomdp](const State& state, int64_t value) {
  //             return pomdp.get_state_value(state, value);
  //           },
  //           &pomdp, rng, sarsop_fsc, solver, "SARSOP", 0, params.set_number,
  //           params.seed, sarsop_fsc.NumNodes(), evaluation_file);
  //     }
  //   }

  //   std::vector<State> eval_states;
  //   for (int i = 0; i < params.n_eval_trials; ++i) {
  //     eval_states.push_back(pomdp.SampleStartState());
  //   }
  //   RunCommonEvaluation(
  //       eval_states, &pomdp, params.max_sim_depth, solver,
  //       [&pomdp](const State& state, int64_t value) {
  //         return pomdp.get_state_value(state, value);
  //       },
  //       mcvi_fsc, aostar_root, anytime_aostar_root, qmdp_root, sarsop_fsc,
  //       evaluation_file);

  evaluation_file.close();
  return 0;
}
