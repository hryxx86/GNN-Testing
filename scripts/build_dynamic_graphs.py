"""
Dynamic Graph Construction Module for GNN Stock Prediction.

Builds rolling-window correlation graphs from stock return data.
Designed with an extensible interface: swap in different graph construction
methods (partial correlation, transfer entropy, etc.) by subclassing GraphBuilder.

Usage:
    from build_dynamic_graphs import PearsonGraphBuilder, build_graph_snapshots

    builder = PearsonGraphBuilder()
    config = {"window_sizes": [63, 126, 252], "thresholds": [0.4, 0.5, 0.6, 0.7]}
    stats_df = build_graph_snapshots(returns, builder, config)
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
import networkx as nx
from torch_geometric.utils import degree


# ---------------------------------------------------------------------------
# Abstract interface — extend this for new graph construction methods
# ---------------------------------------------------------------------------

class GraphBuilder(ABC):
    """Abstract base class: build edge_index from a returns window."""

    @abstractmethod
    def build_edge_index(
        self, returns_window: pd.DataFrame, threshold: float
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        returns_window : DataFrame, shape (num_days, num_stocks)
            Daily returns for a rolling window.
        threshold : float
            Edge inclusion threshold (method-specific interpretation).

        Returns
        -------
        edge_index : Tensor, shape [2, num_edges]
            COO sparse format, self-loops removed.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class PearsonGraphBuilder(GraphBuilder):
    """Rolling-window Pearson correlation graph."""

    def build_edge_index(self, returns_window, threshold):
        corr = torch.tensor(returns_window.corr().values, dtype=torch.float32)
        mask = corr.abs() > threshold
        mask.fill_diagonal_(False)
        return mask.nonzero().t()

    @property
    def name(self):
        return "pearson"


# Future extensions (implement when needed):
# class PartialCorrGraphBuilder(GraphBuilder): ...
# class TransferEntropyGraphBuilder(GraphBuilder): ...


# ---------------------------------------------------------------------------
# Statistics for a single graph snapshot
# ---------------------------------------------------------------------------

def compute_graph_stats(
    edge_index: torch.Tensor,
    num_nodes: int,
    ticker_names: list,
    top_k: int = 10,
) -> dict:
    """
    Compute structural statistics for one graph snapshot.

    Returns dict with: num_edges, density, avg_degree, max_degree,
    clustering_coeff, num_components, top_k_hubs (list of (ticker, degree)).
    """
    num_edges = edge_index.shape[1]
    max_possible = num_nodes * (num_nodes - 1)
    density = num_edges / max_possible if max_possible > 0 else 0.0

    # Degree stats
    d = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
    avg_deg = d.float().mean().item()
    max_deg = d.max().item()

    # Top-k hubs
    k = min(top_k, num_nodes)
    top_vals, top_idx = torch.topk(d, k)
    top_hubs = [
        (ticker_names[idx.item()], val.item())
        for idx, val in zip(top_idx, top_vals)
    ]

    # NetworkX stats (clustering, components) — use sparse graph for speed
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges_np = edge_index.t().numpy()
    G.add_edges_from(edges_np.tolist())

    clustering = nx.average_clustering(G)
    num_components = nx.number_connected_components(G)

    return {
        "num_edges": num_edges,
        "density": round(density, 6),
        "avg_degree": round(avg_deg, 2),
        "max_degree": int(max_deg),
        "clustering_coeff": round(clustering, 4),
        "num_components": num_components,
        "top10_hubs": top_hubs,
    }


# ---------------------------------------------------------------------------
# Main entry point — build all snapshots
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "window_sizes": [63, 126, 252],
    "thresholds": [0.4, 0.5, 0.6, 0.7],
    "step_size": 21,
}


def build_graph_snapshots(
    returns: pd.DataFrame,
    builder: GraphBuilder,
    config: dict = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate rolling-window graph snapshots with statistics.

    Parameters
    ----------
    returns : DataFrame, shape (num_days, num_stocks)
        Daily returns. Index should be date-like.
    builder : GraphBuilder
        Graph construction method.
    config : dict, optional
        Keys: window_sizes, thresholds, step_size.
        Defaults to DEFAULT_CONFIG.
    verbose : bool
        Print progress updates.

    Returns
    -------
    DataFrame with one row per (window_size, threshold, time_step) snapshot.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    window_sizes = cfg["window_sizes"]
    thresholds = cfg["thresholds"]
    step_size = cfg["step_size"]

    ticker_names = returns.columns.tolist()
    num_nodes = len(ticker_names)
    dates = returns.index
    total_days = len(returns)

    records = []
    total_combos = len(window_sizes) * len(thresholds)
    combo_idx = 0

    for ws in window_sizes:
        for thr in thresholds:
            combo_idx += 1
            steps = list(range(ws, total_days, step_size))
            if verbose:
                print(
                    f"[{combo_idx}/{total_combos}] "
                    f"method={builder.name} window={ws} threshold={thr} "
                    f"steps={len(steps)}"
                )

            for t in steps:
                window_returns = returns.iloc[t - ws : t]
                edge_index = builder.build_edge_index(window_returns, thr)
                stats = compute_graph_stats(
                    edge_index, num_nodes, ticker_names
                )

                # Serialize top10 hubs as string for CSV storage
                hubs_str = "; ".join(
                    f"{tk}({deg})" for tk, deg in stats["top10_hubs"]
                )

                records.append(
                    {
                        "method": builder.name,
                        "window": ws,
                        "threshold": thr,
                        "step": t,
                        "date": str(dates[t - 1]),
                        "num_edges": stats["num_edges"],
                        "density": stats["density"],
                        "avg_degree": stats["avg_degree"],
                        "max_degree": stats["max_degree"],
                        "clustering_coeff": stats["clustering_coeff"],
                        "num_components": stats["num_components"],
                        "top10_hubs": hubs_str,
                    }
                )

    df = pd.DataFrame(records)
    if verbose:
        print(f"\nDone. {len(df)} snapshots generated.")
    return df
