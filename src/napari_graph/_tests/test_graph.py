import pytest
import numpy as np
import pandas as pd
from typing import Tuple

from napari_graph.graph import UndirectedGraph


def make_graph(size: int, sparsity: float) -> Tuple[pd.DataFrame, np.ndarray]:
    # TODO:
    #  - move to pytest fixture

    adj_matrix = np.random.uniform(size=(size, size)) < sparsity
    np.fill_diagonal(adj_matrix, 0)

    edges = np.stack(adj_matrix.nonzero()).T
    nodes_df = pd.DataFrame(np.random.randn(size, 3), columns=["z", "y", "x"])

    return nodes_df, edges


@pytest.mark.parametrize("n_prealloc_edges", [0, 2, 5])
def test_undirected_init_from_dataframe(n_prealloc_edges: int) -> None:
    nodes_df = pd.DataFrame(
        [[0, 2.5],
         [4, 2.5],
         [1, 0,],
         [2, 3.5],
         [3, 0]],
        columns=["y", "x"],
    )

    edges = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 4]]
    
    graph = UndirectedGraph(n_nodes=nodes_df.shape[0], ndim=nodes_df.shape[1], n_edges=n_prealloc_edges)

    graph.init_nodes_from_dataframe(nodes_df, ["y", "x"])
    graph.add_edges(edges)

    for node_idx in nodes_df.index:
        node_edges = graph.edges(node_idx)

        # checking if two edges per node and connecting only two nodes
        assert node_edges.shape == (2, 2)

        # checking if the given index is the source
        assert np.all(node_edges[:, 0] == node_idx) 

        # checking if the edges are corrected
        for edge in node_edges:
            assert sorted(edge) in edges


def test_benchmark_construction_speed() -> None:
    # FIXME: remove this, maybe create an airspeed velocity CI
    import time
    nodes_df, edges = make_graph(20000, 0.01)

    print('# edges', len(edges))
    start = time.time()
    graph = UndirectedGraph(n_nodes=nodes_df.shape[0], ndim=nodes_df.shape[1], n_edges=len(edges))
    graph.init_nodes_from_dataframe(nodes_df, ["z", "y", "x"])
    graph.add_edges(edges)
    print('init time', time.time() - start)
