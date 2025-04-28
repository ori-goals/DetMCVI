#include "AlphaVectorFSC.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace DetMCVI {

int64_t AlphaVectorFSC::GetEdgeValue(int64_t nI, int64_t observation) const {
  const std::unordered_map<int64_t, int64_t>& m = _edges[nI];
  const auto it = m.find(observation);
  if (it != m.cend()) return it->second;
  return -1;
}

const std::unordered_map<int64_t, int64_t>& AlphaVectorFSC::GetEdges(
    int64_t nI) const {
  return _edges.at(nI);
}

int64_t AlphaVectorFSC::AddNode(const int64_t action) {
  _nodes.emplace_back(std::make_shared<AlphaVectorNode>(action));
  return _nodes.size() - 1;
}

void AlphaVectorFSC::UpdateEdge(int64_t nI, int64_t o, int64_t nI_new) {
  _edges[nI][o] = nI_new;
}

void AlphaVectorFSC::UpdateEdge(
    int64_t nI, const std::unordered_map<int64_t, int64_t>& edges) {
  _edges[nI] = edges;
}

void AlphaVectorFSC::GenerateGraphviz(
    std::ostream& ofs, const std::vector<std::string>& actions,
    const std::vector<std::string>& observations) const {
  if (!ofs) throw std::logic_error("Invalid output file");

  ofs << "digraph AlphaVectorFSC {" << std::endl;

  ofs << "node [shape=circle];" << std::endl;

  // Loop through each node
  for (int64_t i = 0; i < NumNodes(); ++i) {
    const std::shared_ptr<AlphaVectorNode> node = GetNode(i);

    std::string action = actions.empty() ? std::to_string(node->GetBestAction())
                                         : actions[node->GetBestAction()];
    ofs << " n" << i << " [label=<<B>" << i << "</B><BR/>a: " << action
        << "<BR/>V: " << node->V_node() << ">";
    if (i == GetStartNodeIndex()) {  // highlight start node with double outline
      ofs << ", penwidth=3";
    }
    ofs << "];" << std::endl;

    // Loop through edges from this node
    for (const auto& edge : _edges[i]) {
      std::string observation = observations.empty()
                                    ? std::to_string(edge.first)
                                    : observations[edge.first];
      int64_t target_node = edge.second;
      ofs << "n" << i << " -> n" << target_node << " [label=<" << observation
          << ">];";
    }
    ofs << std::endl;
  }

  ofs << "}" << std::endl;
}

double AlphaVectorFSC::SimulateTrajectory(int64_t nI, const State& state,
                                          int64_t max_depth, double R_lower,
                                          SimInterface* pomdp) {
  const double gamma = pomdp->GetDiscount();
  double V_n_s = 0.0;
  int64_t nI_current = nI;
  State curr_state = state;
  for (int64_t step = 0; step < max_depth; ++step) {
    if (nI_current == -1) {
      V_n_s += std::pow(gamma, step) * R_lower;
      break;
    }

    const int64_t action = GetNode(nI_current)->GetBestAction();
    const auto [sNext, obs, reward, done] = pomdp->Step(curr_state, action);
    nI_current = GetEdgeValue(nI_current, obs);
    V_n_s += std::pow(gamma, step) * reward;
    if (done) break;
    curr_state = sNext;
  }

  return V_n_s;
}

double AlphaVectorFSC::GetNodeAlpha(const State& state, int64_t nI,
                                    double R_lower, int64_t max_depth_sim,
                                    SimInterface* pomdp) {
  const std::optional<double> val = GetNode(nI)->GetAlpha(state);
  if (val.has_value()) return val.value();
  const double V = SimulateTrajectory(nI, state, max_depth_sim, R_lower, pomdp);
  GetNodeModifiable(nI)->SetAlpha(state, V);
  return V;
}

double AlphaVectorFSC::LowerBoundFromFSC(const BeliefStates& b0,
                                         int64_t belief_depth,
                                         double R_lower_lim, int64_t max_depth,
                                         SimInterface* pomdp) {
  double belief_value = 0.0;

  // #pragma omp parallel for reduction(+ : belief_value)
  for (size_t i = 0; i < b0.size(); ++i) {
    const auto alpha = GetNodeAlpha(b0[i].first, GetStartNodeIndex(),
                                    R_lower_lim, max_depth, pomdp);
    belief_value += alpha * b0[i].second;
  }
  return std::pow(pomdp->GetDiscount(), belief_depth) * belief_value;
}

int64_t CountNodes(const std::string& filename) {
  std::ifstream file(filename);
  std::string line;
  int64_t node_count = 0;

  // Regular expression for matching node lines
  std::regex node_regex(R"((\w+)\s+\[label=\".*?A\s+\((\d+)\).*\"]\;)");

  while (std::getline(file, line)) {
    if (std::regex_search(line, node_regex)) {
      ++node_count;
    }
  }

  return node_count;
}

AlphaVectorFSC ParseDotFile(const std::string& filename) {
  std::ifstream file(filename);
  std::string line;
  std::unordered_map<std::string, int64_t> node_indices;
  AlphaVectorFSC fsc(CountNodes(filename));
  int64_t current_index = 0;
  std::string start_node;

  std::regex node_regex(R"((\w+)\s+\[label=\".*?A\s+\((\d+)\).*\"]\;)");
  std::regex start_node_regex(
      R"((\w+)\s+\[label=\".*?A\s+\((\d+)\).*\" shape=doublecircle.*\]\;)");
  std::regex edge_regex(
      R"((\w+)\s+->\s+(\w+)\s+\[label=\"o\s+\((\d+)\).*\"]\;)");

  while (std::getline(file, line)) {
    std::smatch match;

    // Check for start node (doublecircle shape)
    if (std::regex_search(line, match, start_node_regex)) {
      std::string node_name = match[1];
      int64_t action = std::stoll(match[2]);
      node_indices[node_name] = current_index;
      fsc.AddNode(action);
      start_node = node_name;
      ++current_index;
    }
    // Check for regular nodes
    else if (std::regex_search(line, match, node_regex)) {
      std::string node_name = match[1];
      int64_t action = std::stoll(match[2]);
      node_indices[node_name] = current_index;
      fsc.AddNode(action);
      ++current_index;
    }
    // Check for edges
    else if (std::regex_search(line, match, edge_regex)) {
      std::string start = match[1];
      std::string end = match[2];
      int64_t observation = std::stoll(match[3]);
      int64_t start_index = node_indices[start];
      int64_t end_index = node_indices[end];
      fsc.UpdateEdge(start_index, observation, end_index);
    }
  }

  // Set the start node index
  if (!start_node.empty()) {
    fsc.SetStartNodeIndex(node_indices[start_node]);
  }
  return fsc;
}

}  // namespace DetMCVI
