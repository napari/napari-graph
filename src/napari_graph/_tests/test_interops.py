import itertools
from typing import Any, Callable, List

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from napari_graph import (
    BaseGraph,
    DirectedGraph,
    UndirectedGraph,
    to_napari_graph,
    to_networkx,
)


def _graph_list() -> List[BaseGraph]:

    coords = pd.DataFrame(
        [
            [0, 2.5],
            [4, 2.5],
            [1, 0],
            [2, 3.5],
            [3, 0],
        ],
        columns=["y", "x"],
    )

    edges = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])

    empty_graph = UndirectedGraph()

    graph = UndirectedGraph(edges)

    digraph = DirectedGraph(edges)

    spatial_graph = UndirectedGraph(
        edges=edges,
        coords=coords,
    )

    spatial_digraph = DirectedGraph(
        edges=edges,
        coords=coords,
    )

    only_coords = UndirectedGraph(coords=coords)

    only_coords_di = DirectedGraph(coords=coords)

    return [
        empty_graph,
        graph,
        digraph,
        spatial_graph,
        spatial_digraph,
        only_coords,
        only_coords_di,
    ]


@pytest.mark.parametrize(
    "in_graph,to_class", itertools.product(_graph_list(), [to_networkx])
)
def test_conversion(
    in_graph: BaseGraph, to_class: Callable[[BaseGraph], Any]
) -> None:

    nxgraph = to_class(in_graph)
    out_graph = to_napari_graph(nxgraph)

    assert np.array_equal(in_graph.get_nodes(), out_graph.get_nodes())

    if in_graph.is_spatial():
        assert np.array_equal(
            in_graph.get_coordinates(), out_graph.get_coordinates()
        )

    if in_graph.n_edges == 0:
        assert out_graph.n_edges == 0

    else:
        in_graph_edges = np.concatenate(in_graph.get_edges(), axis=0).sort()
        out_graph_edges = np.concatenate(out_graph.get_edges(), axis=0).sort()
        assert np.array_equal(in_graph_edges, out_graph_edges)


def test_weighted_networkx_graph() -> None:

    nxgraph = nx.DiGraph()
    nxgraph.add_edge(10, 11, weight=0.1)
    nxgraph.add_edge(11, 12, weight=0.2)
    nxgraph.add_edge(12, 10, weight=0.3)

    nxgraph_edges = np.asarray(nxgraph.edges).sort()

    graph = to_napari_graph(nxgraph)
    graph_edges = np.concatenate(graph.get_edges(), axis=0).sort()

    assert np.array_equal(nxgraph_edges, graph_edges)
