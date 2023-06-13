from typing import List

import numpy as np
import pandas as pd
import pytest

from napari_graph import (
    BaseGraph,
    DirectedGraph,
    UndirectedGraph,
    from_networkx,
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


@pytest.mark.parametrize("in_graph", _graph_list())
def test_networkx_conversion(in_graph: BaseGraph) -> None:

    nxgraph = to_networkx(in_graph)
    out_graph = from_networkx(nxgraph)

    assert np.array_equal(in_graph.get_nodes(), out_graph.get_nodes())

    if in_graph.is_spatial():
        assert np.array_equal(
            in_graph.get_coordinates(), out_graph.get_coordinates()
        )

    if in_graph.n_edges == 0:
        assert out_graph.n_edges == 0

    else:
        in_graph_edges = np.lexsort(in_graph.get_edges())
        out_graph_edges = np.lexsort(out_graph.get_edges())
        assert np.array_equal(in_graph_edges, out_graph_edges)
