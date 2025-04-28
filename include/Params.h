// Provides the interface for the algorithm parameters

/* This file has been written and/or modified by the following people:
 *
 * Alex Schutz
 *
 */

#pragma once

#include <inttypes.h>
#include <string.h>

#include <iostream>
#include <limits>
#include <sstream>
#include <string>

struct EvalParams {
  std::string datafile;                            // problem data file
  std::string evaluation_file = "eval_stats.csv";  // output file
  int64_t nb_particles_b0 = 100000;                // num init belief samples
  int64_t max_belief_samples = 10000;              // downsampled belief size
  int64_t max_node_size = 3125000;                 // num FSC nodes
  int64_t max_sim_depth = 100;                     // trajectory depth
  double eval_epsilon = 0.005;     // trajectory cumulative discount limit
  double converge_thresh = 0.005;  // upper and lower bound diff
  int64_t max_iterations =
      std::numeric_limits<int64_t>::max();  // MCVI iterations
  int64_t max_time_ms = 60000;              // MCVI computation time
  int64_t n_eval_trials = 10000;            // num trials for evaluation
  int64_t eval_interval_ms =
      100;  // how long to compute before re-evaluation (timeseries only)
  int64_t completion_threshold =
      9900;  // number of completed runs to consider planning finished
  int completion_reps = 3;  // number of consecutive times to achieve completion
                            // threshold before exiting
  int64_t seed = 42;        // random seed
  int64_t set_number = 0;
};

EvalParams parseArgs(int argc, char** argv) {
  EvalParams params;
  std::stringstream ss;
  ss << "Usage: " << argv[0] << " <datafile> <output_file> [options]\n"
     << "Options:\n"
     << "  --nb_particles_b0 <int64_t>       Number of initial belief "
        "samples\n"
     << "  --max_belief_samples <int64_t>    Max downsampled belief size\n"
     << "  --max_node_size <int64_t>         Max number of FSC nodes\n"
     << "  --max_sim_depth <int64_t>         Max trajectory depth\n"
     << "  --eval_epsilon <double>           Trajectory cumulative "
        "discount limit\n"
     << "  --converge_thresh <double>        Convergence threshold "
        "(upper and lower bound difference)\n"
     << "  --max_iterations <int64_t>        MCVI iterations\n"
     << "  --max_time_ms <int64_t>           MCVI computation time\n"
     << "  --n_eval_trials <int64_t>         Number of trials for evaluation\n"
     << "  --eval_interval_ms <int64_t>      How long to compute before "
        "re-evaluation (timeseries only)\n"
     << "  --completion_threshold <int64_t>  Number of completed runs to "
        "consider planning finished\n"
     << "  --completion_reps <int>           Number of consecutive times to "
        "achieve completion threshold before exiting\n"
     << "  --seed <int64_t>                  Random seed\n"
     << "  --set_number <int64_t>            Set number\n"
     << "  --help                            Show this help message\n";

  if (argc < 3) {
    std::cerr << "Error: Missing datafile or output file argument.\n";
    std::cerr << ss.str();
    std::exit(1);
  }

  params.datafile = argv[1];
  params.evaluation_file = argv[2];

  for (int i = 3; i < argc; ++i) {
    if (strcmp(argv[i], "--nb_particles_b0") == 0 && i + 1 < argc) {
      params.nb_particles_b0 = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--max_belief_samples") == 0 && i + 1 < argc) {
      params.max_belief_samples = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--max_node_size") == 0 && i + 1 < argc) {
      params.max_node_size = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--max_sim_depth") == 0 && i + 1 < argc) {
      params.max_sim_depth = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--eval_epsilon") == 0 && i + 1 < argc) {
      params.eval_epsilon = std::stod(argv[++i]);
    } else if (strcmp(argv[i], "--converge_thresh") == 0 && i + 1 < argc) {
      params.converge_thresh = std::stod(argv[++i]);
    } else if (strcmp(argv[i], "--max_iterations") == 0 && i + 1 < argc) {
      params.max_iterations = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--max_time_ms") == 0 && i + 1 < argc) {
      params.max_time_ms = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--n_eval_trials") == 0 && i + 1 < argc) {
      params.n_eval_trials = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--eval_interval_ms") == 0 && i + 1 < argc) {
      params.eval_interval_ms = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--completion_threshold") == 0 && i + 1 < argc) {
      params.completion_threshold = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--completion_reps") == 0 && i + 1 < argc) {
      params.completion_reps = std::stoi(argv[++i]);
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      params.seed = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--set_number") == 0 && i + 1 < argc) {
      params.set_number = std::stoll(argv[++i]);
    } else if (strcmp(argv[i], "--help") == 0) {
      std::cout << ss.str();
      std::exit(0);
    }
  }

  return params;
}
