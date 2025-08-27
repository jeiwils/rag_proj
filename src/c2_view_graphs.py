#!/usr/bin/env python3
"""view_graph.py – View or save graphs produced by c_graphing.py."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def load_graph(path: Path) -> nx.Graph:
    ext = path.suffix.lower()
    if ext == ".gpickle":
        return nx.read_gpickle(path)
    if ext == ".graphml":
        return nx.read_graphml(path)
    raise ValueError(f"Unsupported graph format: {ext}")


def draw_graph(G: nx.Graph, out: Path | None, show: bool) -> None:
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=300, font_size=8)
    if out:
        plt.savefig(out, bbox_inches="tight")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View/save graph files (.gpickle or .graphml)"
    )
    parser.add_argument("graph_file", help="Path to graph file")
    parser.add_argument("--out", help="Optional path to save rendered image")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save image only; do not display window",
    )
    args = parser.parse_args()

    G = load_graph(Path(args.graph_file))
    draw_graph(G, Path(args.out) if args.out else None, not args.no_show)


if __name__ == "__main__":
    main()
