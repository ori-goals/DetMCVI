#include "CTP.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>

#include "AOStar.h"
#include "DetMCVI.h"
#include "Params.h"
#include "QMDPTree.h"

using namespace DetMCVI;

std::atomic<bool> exit_flag = false;

static double s_time_diff(const std::chrono::steady_clock::time_point& begin,
                          const std::chrono::steady_clock::time_point& end) {
  return (std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
              .count()) /
         1000.0;
}

void runMCVI(CTP* pomdp, const BeliefStates& init_belief, std::mt19937_64& rng,
             int64_t max_sim_depth, int64_t max_node_size, int64_t eval_depth,
             int64_t eval_epsilon, double converge_thresh, int64_t max_iter,
             int64_t max_computation_ms, int64_t max_eval_steps,
             int64_t n_eval_trials, int64_t nb_particles_b0,
             OptimalPath& solver, int64_t set_number, int64_t seed,
             std::ostream& os) {
  // Initialise heuristic
  OptimalPath heuristic(pomdp);

  // Initialise the FSC
  std::cout << "Initialising FSC" << std::endl;
  const auto init_fsc = AlphaVectorFSC(max_node_size);

  // Run MCVI
  std::cout << "Running DetMCVI" << std::endl;
  const std::chrono::steady_clock::time_point mcvi_begin =
      std::chrono::steady_clock::now();
  auto planner = MCVIPlanner(pomdp, init_fsc, init_belief, heuristic, rng);
  const auto [fsc, root] =
      planner.Plan(max_sim_depth, converge_thresh, max_iter, max_computation_ms,
                   eval_depth, eval_epsilon, exit_flag);
  const std::chrono::steady_clock::time_point mcvi_end =
      std::chrono::steady_clock::now();
  std::cout << "detMCVI complete (" << s_time_diff(mcvi_begin, mcvi_end)
            << " seconds)" << std::endl;

  // Draw FSC plot
  std::fstream fsc_graph("fsc.dot", std::fstream::out);
  fsc.GenerateGraphviz(fsc_graph, pomdp->getActions(), pomdp->getObs());
  fsc_graph.close();

  // Draw the internal belief tree
  std::fstream belief_tree("belief_tree.dot", std::fstream::out);
  root->DrawBeliefTree(belief_tree);
  belief_tree.close();

  // Simulate the resultant FSC
  std::cout << "Simulation with up to " << max_eval_steps
            << " steps:" << std::endl;
  planner.SimulationWithFSC(max_eval_steps);
  std::cout << std::endl;

  // Evaluate the FSC policy
  EvaluationWithSimulationFSC(
      max_eval_steps, n_eval_trials, nb_particles_b0,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      pomdp, rng, fsc, solver, "detMCVI", s_time_diff(mcvi_begin, mcvi_end),
      set_number, seed, fsc.NumNodes(), os);
  std::cout << std::endl;
}

void runAOStar(CTP* pomdp, const BeliefStates& init_belief,
               std::mt19937_64& rng, int64_t eval_depth, int64_t max_iter,
               int64_t max_computation_ms, int64_t max_eval_steps,
               int64_t n_eval_trials, int64_t nb_particles_b0,
               OptimalPath& solver, int64_t set_number, int64_t seed,
               std::ostream& os) {
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
  const std::chrono::steady_clock::time_point ao_begin =
      std::chrono::steady_clock::now();
  RunAOStar(root, max_iter, max_computation_ms, heuristic, eval_depth, rng,
            pomdp);
  const std::chrono::steady_clock::time_point ao_end =
      std::chrono::steady_clock::now();
  std::cout << "AO* complete (" << s_time_diff(ao_begin, ao_end) << " seconds)"
            << std::endl;

  // Draw policy tree
  std::fstream policy_tree("greedy_policy_tree.dot", std::fstream::out);
  root->DrawPolicyTree(policy_tree);
  policy_tree.close();

  // Evaluate policy
  int64_t time_sum =
      std::chrono::duration_cast<std::chrono::microseconds>(ao_end - ao_begin)
          .count();
  RunAOEvaluation(
      root, time_sum, max_eval_steps, n_eval_trials, nb_particles_b0, rng,
      solver,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      pomdp, set_number, seed, os);
}

void runQMDP(CTP* pomdp, const BeliefStates& init_belief, std::mt19937_64& rng,
             int64_t eval_depth, int64_t max_time_ms, int64_t max_eval_steps,
             int64_t n_eval_trials, int64_t nb_particles_b0,
             OptimalPath& solver, int64_t set_number, int64_t seed,
             std::ostream& os) {
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
  const int64_t t = RunQMDP(root, max_time_ms, heuristic, eval_depth, pomdp);

  // Draw policy tree
  std::fstream policy_tree("qmdp_policy_tree.dot", std::fstream::out);
  root->DrawPolicyTree(policy_tree);
  policy_tree.close();

  RunQMDPEvaluation(
      root, t, max_eval_steps, n_eval_trials, nb_particles_b0, rng, solver,
      [&pomdp](const State& state, int64_t value) {
        return pomdp->get_state_value(state, value);
      },
      pomdp, set_number, seed, os);
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

  std::ofstream graph_file("ctp_graph.dot");
  if (!graph_file.is_open()) {
    std::cerr << "Failed to open graph file" << std::endl;
    return 1;
  }
  pomdp.visualiseGraph(graph_file);
  graph_file.close();

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

  {  // Run MCVI
    auto mcvi = CTP(rng, nodes, edges, stoch_edges, origin, goal);
    runMCVI(&mcvi, init_belief_states, rng, params.max_sim_depth,
            params.max_node_size, params.max_sim_depth, params.eval_epsilon,
            params.converge_thresh, params.max_iterations, params.max_time_ms,
            params.max_sim_depth, params.n_eval_trials, params.nb_particles_b0,
            solver, params.set_number, params.seed, evaluation_file);
  }
  {  // Compare to AO*
    auto aostar = CTP(rng, nodes, edges, stoch_edges, origin, goal);
    runAOStar(&aostar, init_belief_states, rng, params.max_sim_depth,
              params.max_iterations, params.max_time_ms, params.max_sim_depth,
              params.n_eval_trials, params.nb_particles_b0, solver,
              params.set_number, params.seed, evaluation_file);
  }
  {  // QMDP
    auto qmdp = CTP(rng, nodes, edges, stoch_edges, origin, goal);
    runQMDP(&qmdp, init_belief_states, rng, params.max_sim_depth,
            params.max_time_ms, params.max_sim_depth, params.n_eval_trials,
            10 * params.nb_particles_b0, solver, params.set_number, params.seed,
            evaluation_file);
  }
  return 0;
}
