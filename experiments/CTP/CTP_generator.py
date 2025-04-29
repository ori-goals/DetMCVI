#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import sys
from typing import TextIO


def plot_nx_graph(G: nx.Graph, origin, goal):
    # Plot graph
    node_colour = []
    for node in G.nodes:
        c = "white"
        if node == goal:
            c = "#2ca02c"
        elif node == origin:
            c = "#ff7f0e"
        node_colour.append(c)
    edge_labels = []
    probs = nx.get_edge_attributes(G, "blocked_prob")
    weights = nx.get_edge_attributes(G, "weight")
    edge_labels = {
        e: (f"{w}\np: {probs[e]}" if e in probs else f"{w}") for e, w in weights.items()
    }
    edge_style = ["dashed" if edge in probs.keys() else "solid" for edge in G.edges]
    pos = nx.get_node_attributes(G, "pos")
    nx.draw(
        G,
        with_labels=True,
        node_size=500,
        node_color=node_colour,
        edgecolors="black",
        pos=pos,
        style=edge_style,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos={p: (v[0], v[1]) for p, v in pos.items()},
        edge_labels=edge_labels,
        bbox={"boxstyle": "square", "pad": 0, "color": "white"},
        rotate=False,
        font_size=8,
        verticalalignment="baseline",
        clip_on=False,
    )
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx * dx + dy * dy)


def edge_distance(e, nodes):
    return distance(nodes[e[0]], nodes[e[1]])


def generate_graph(
    n_nodes: int,
    seed: int,
    use_edge_weights=False,
    prop_stoch=0.4,
    plot=False,
    grid_size=10,
) -> tuple[nx.Graph, int, int, bool]:
    # grid_size=100
    # seed=57043235
    # Define world parameters
    xmax = grid_size  # max x-grid coordinate
    ymax = grid_size  # max y-grid coordinate

    # Fix seed for graph generation
    np.random.seed(seed)

    # Generate random points in a grid
    node_pos = np.random.choice(xmax * ymax, n_nodes, replace=False)
    grid_nodes = [(i // (ymax + 1), i % (ymax + 1)) for i in node_pos]

    # Define origin and goal nodes based on furthest apart nodes
    goal = int(
        np.argmax(
            [distance(grid_nodes[0], grid_nodes[i]) for i in range(len(grid_nodes))]
        )
    )
    origin = int(
        np.argmax(
            [distance(grid_nodes[goal], grid_nodes[i]) for i in range(len(grid_nodes))]
        )
    )

    # Apply Delaunay triangulation
    delaunay = Delaunay(grid_nodes)
    simplices = delaunay.simplices

    # Add to networkx object
    G = nx.Graph()
    for i, node in enumerate(grid_nodes):
        G.add_node(i, pos=node)
    for path in simplices:
        nx.add_path(G, path)
    weights = {}
    for e in G.edges():
        weights[e] = (
            max(
                round(edge_distance(e, grid_nodes), 2),
                distance(grid_nodes[goal], grid_nodes[origin]) / (n_nodes),
            )
            if use_edge_weights
            else 1
        )
    nx.set_edge_attributes(G, values=weights, name="weight")

    # Define stochastic edges and probabilities
    num_stoch_edges = round(prop_stoch * len(G.edges))
    stoch_edge_idx = np.random.choice(len(G.edges), num_stoch_edges, replace=False)
    # stoch_edge_idx = [6, 12, 9, 13, 7, 4, 15]
    edge_probs: dict[tuple[int, int], float] = {}
    for i, e in enumerate(G.edges):
        if i in stoch_edge_idx:
            edge_probs[e] = round(np.random.uniform(), 2)
    nx.set_edge_attributes(G, edge_probs, "blocked_prob")

    # basic solvability check
    solvable = nx.has_path(G, origin, goal)

    if plot:
        plot_nx_graph(G, origin, goal)

    return G, origin, goal, solvable


def generate_delaunay_graph_set(location_count: int, set_size: int, seed: int):
    """
    Delaunay graphs with 100 vertices whose coordinates are randomly chosen over the region [1, 100] x [1, 100] on the plane.
    Edge lengths are set to the Euclidean distance between their end vertices and the two farthest vertices of the graph are designated as the starting and termination vertices, respectively.
    Each grid edge has a 0.25 probability of being stochastic and marks of stochastic edges are sampled from the uniform distribution. (Aksakalli et al. 2016)
    """
    problem_set = []
    for n in range(set_size):
        while True:
            G, origin, goal, solvable = generate_graph(
                location_count, seed, True, 0.25, False, 100
            )
            if solvable:
                problem_set.append((G, origin, goal, seed + n))
                break
        #     else:
        #         seed += 1
        # seed += 1
    return problem_set


def ctp_to_file(G: nx.Graph, origin: int, goal: int, f: TextIO):
    print("CTPNodes: " + " ".join(map(str, G.nodes)), file=f)
    print("CTPEdges:", file=f)
    for e, w in nx.get_edge_attributes(G, "weight").items():
        print(f"{min(e)} {max(e)} {w}", file=f)
    print("CTPStochEdges:", file=f)
    for e, p in nx.get_edge_attributes(G, "blocked_prob").items():
        print(f"{min(e)} {max(e)} {p}", file=f)
    print(f"CTPOrigin: {origin}", file=f)
    print(f"CTPGoal: {goal}", file=f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a CTP problem example")
    parser.add_argument(
        "-n", "--nodes", help="Number of nodes", required=True, type=int
    )
    parser.add_argument(
        "-p",
        "--perc-stoch-edges",
        help="Percentage of stochastic edges",
        default=0.4,
        type=float,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Random seed",
        default=np.random.randint(0, 9999999),
        type=int,
    )
    args = parser.parse_args()
    seed = args.seed

    N = args.nodes
    perc_stoch = args.perc_stoch_edges
    G, origin, goal, solvable = generate_graph(
        N, seed, True, prop_stoch=perc_stoch, plot=False
    )
    print("seed:", seed, "nodes:", N, "solvable:", solvable, file=sys.stderr)
    ctp_to_file(G, origin, goal, sys.stdout)
    G, origin, goal, solvable = generate_graph(
        N, seed, True, prop_stoch=perc_stoch, plot=True
    )
