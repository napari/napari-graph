import itertools
from typing import Any, Callable, List

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from napari_graph import (  # noqa
    BaseGraph,
    DirectedGraph,
    UndirectedGraph,
    to_napari_graph,
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
    "in_graph,to_class",
    itertools.product(_graph_list(), [BaseGraph.to_networkx]),
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


def test_table_like_graphs() -> None:

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
    coords.index = np.arange(
        10, 5, -1
    )  # testing index that don't start with zero

    # testing pandas dataframe
    graph = to_napari_graph(coords)
    assert np.allclose(graph.get_coordinates(), coords)

    # testing numpy array
    graph = to_napari_graph(coords.to_numpy())
    assert np.allclose(graph.get_coordinates(), coords.to_numpy())

    # testing bad table
    not_a_table = np.ones((5, 5, 5))
    with pytest.raises(ValueError):
        graph = to_napari_graph(not_a_table)


def test_networkx_non_integer_ids():
    """Check that passing nx graph with non-integer IDs doesn't crash."""
    g = nx.hexagonal_lattice_graph(5, 5, with_positions=True)
    with pytest.warns(UserWarning, match='Node IDs must be integers.'):
        BaseGraph.from_networkx(g)


def test_networkx_basic_roundtrip():
    g = nx.hexagonal_lattice_graph(5, 5, with_positions=True)
    gint = nx.convert_node_labels_to_integers(g)
    ng = BaseGraph.from_networkx(gint)
    g2 = ng.to_networkx()
    # convert positions to tuples because nx comparison tools don't like arrays
    for node in g2.nodes():
        g2.nodes[node]['pos'] = tuple(g2.nodes[node]['pos'])
    assert nx.utils.edges_equal(gint.edges, g2.edges)
    assert set(gint.nodes) == set(g2.nodes)
    for node in gint.nodes:
        np.testing.assert_allclose(
            gint.nodes[node]['pos'], g2.nodes[node]['pos']
        )
