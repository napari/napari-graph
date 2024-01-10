from itertools import product
from typing import Any, Callable, List, Optional, Protocol, Type

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike

from napari_graph import DirectedGraph, UndirectedGraph
from napari_graph.base_graph import _EDGE_EMPTY_PTR, BaseGraph
from napari_graph.undirected_graph import _LL_UN_EDGE_POS, _UN_EDGE_SIZE


@pytest.mark.parametrize("n_prealloc_edges", [0, 2, 5])
def test_undirected_edge_addition(n_prealloc_edges: int) -> None:
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

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]]

    graph = UndirectedGraph(
        coords=coords,
        n_edges=n_prealloc_edges,
    )
    graph.add_edges(edges)

    for node_idx, node_edges in zip(
        coords.index, graph.get_edges(coords.index)
    ):
        # checking if two edges per node and connecting only two nodes
        assert node_edges.shape == (2, 2)

        # checking if the given index is the source
        assert np.all(node_edges[:, 0] == node_idx)

        # checking if the edges are corrected
        for edge in node_edges:
            assert sorted(edge) in edges


@pytest.mark.parametrize("n_prealloc_edges", [0, 2, 5])
def test_directed_edge_addition(n_prealloc_edges: int) -> None:
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

    graph = DirectedGraph(
        coords=coords,
        n_edges=n_prealloc_edges,
    )
    graph.add_edges(edges)

    source_edges = np.asarray(graph.get_source_edges(coords.index))
    target_edges = np.asarray(
        graph.get_target_edges(np.roll(coords.index, -1))
    )
    assert np.all(source_edges == edges[:, np.newaxis, :])
    assert np.all(target_edges == edges[:, np.newaxis, :])


@pytest.mark.parametrize("n_prealloc_nodes", [0, 3, 6, 12])
def test_node_addition_indices_coords(n_prealloc_nodes: int) -> None:
    # test node addition with indices and coords and different pre-allocation size
    size = 6
    ndim = 3

    indices = np.random.choice(range(100), size=size, replace=False)
    coords = np.random.randn(size, ndim)

    graph = DirectedGraph(ndim=ndim, n_nodes=n_prealloc_nodes)
    for i in range(size):
        graph.add_nodes(indices=indices[i], coords=coords[i])
        assert len(graph) == i + 1

    np.testing.assert_allclose(graph._coords[: len(graph)], coords)
    np.testing.assert_array_equal(graph._buffer2world[: len(graph)], indices)
    np.testing.assert_array_equal(
        graph._map_world2buffer(indices), range(size)
    )


def test_node_addition_non_spatial() -> None:
    graph = DirectedGraph()

    size = 5
    new_indices = graph.add_nodes(count=size)

    np.testing.assert_array_equal(new_indices, np.arange(size))

    new_indices = graph.add_nodes(indices=[10, 11])
    nodes = graph.get_nodes()

    np.testing.assert_equal(new_indices, [10, 11])
    np.testing.assert_array_equal(nodes[:size], np.arange(size))
    np.testing.assert_array_equal(nodes[size:], [10, 11])

    with pytest.raises(ValueError):
        graph.add_nodes(coords=[[0, 1], [2, 3]])

    graph.add_nodes(indices=15)
    assert len(graph) == 8

    with pytest.raises(ValueError):
        graph.add_nodes(indices=15)


def test_node_addition_spatial() -> None:
    graph = DirectedGraph(ndim=2)

    with pytest.raises(ValueError):
        graph.add_nodes(indices=5, coords=[0, 0], count=1)

    new_index = graph.add_nodes(coords=[2, 2])
    assert len(graph) == 1
    assert new_index == 0
    np.testing.assert_equal(graph.get_coordinates(), [[2, 2]])

    new_index = graph.add_nodes(indices=[10, 11], coords=[[0, 0], [1, 1]])
    assert len(graph) == 3
    np.testing.assert_equal(new_index, [10, 11])

    with pytest.raises(ValueError):
        graph.add_nodes(indices=13)

    with pytest.raises(ValueError):
        graph.add_nodes(count=2)

    with pytest.raises(ValueError):
        graph.add_nodes(indices=[2, 3], coords=[[1, 2]])

    assert len(graph) == 3


@pytest.mark.parametrize(
    "graph_type,ndim",
    product([UndirectedGraph, DirectedGraph], [None, 2, 3]),
)
def test_empty_graph(graph_type: Type[BaseGraph], ndim: Optional[int]) -> None:
    _ = graph_type(ndim=ndim)


class TestGraph:
    _GRAPH_CLASS: Type[BaseGraph]
    __test__ = False  # ignored for testing
    _index_shift = 0  # shift used to test special indexing

    def setup_method(self, method: Callable) -> None:
        self.coords = pd.DataFrame(
            [
                [0, 2.5],
                [4, 2.5],
                [1, 0],
                [2, 3.5],
                [3, 0],
            ],
            index=np.arange(5) + self._index_shift,
            columns=["y", "x"],
        )

        self.edges = (
            np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=int)
            + self._index_shift
        )

        self.graph = self._GRAPH_CLASS(
            edges=self.edges,
            coords=self.coords,
        )

    @staticmethod
    def contains(edge: ArrayLike, edges: List[ArrayLike]) -> bool:
        return any(
            np.allclose(e, edge) if len(e) > 0 else False for e in edges
        )

    def teardown_method(self, method: Callable) -> None:
        del self.edges, self.graph
        if hasattr(self, "coords"):
            del self.coords

    def test_edge_buffers(self) -> None:
        # testing buffer correctness on a non-trivial case when a node was removed
        node_id = 3 + self._index_shift
        self.graph.remove_node(node_id)

        valid_edges = np.logical_not(np.any(self.edges == node_id, axis=1))

        expected_edges = self.edges[valid_edges, :]
        (expected_indices,) = np.nonzero(valid_edges)
        expected_indices *= (
            self.graph._EDGE_SIZE * self.graph._EDGE_DUPLICATION
        )

        indices, edges = self.graph.get_edges_buffers()

        assert np.allclose(expected_indices, indices)
        assert np.allclose(expected_edges, edges)

    def test_subgraph_edges(self) -> None:
        # 3 is disconnected in subgraph
        nodes_ids = np.asarray([0, 1, 3]) + self._index_shift

        subgraph = self.graph.subgraph_edges(nodes_ids)
        expected_subgraph = np.asarray([[0, 1]]) + self._index_shift

        np.testing.assert_array_equal(subgraph, expected_subgraph)

        buffer_node_ids = np.asarray([0, 1, 3])
        buffer_subgraph = self.graph.subgraph_edges(
            buffer_node_ids, is_buffer_domain=True
        )

        np.testing.assert_equal(buffer_subgraph, [[0, 1]])


class TestDirectedGraph(TestGraph):
    graph: DirectedGraph  # required by typing
    _GRAPH_CLASS = DirectedGraph
    __test__ = True

    def test_edge_removal(self) -> None:
        edge = np.asarray([0, 1]) + self._index_shift
        self.graph.remove_edges(edge)
        assert self.graph.n_edges == 4
        assert self.graph.n_empty_edges == self.graph._ALLOC_MIN - 4
        assert not self.contains(edge, self.graph.get_source_edges())
        assert not self.contains(np.flip(edge), self.graph.get_target_edges())

        edges = np.asarray([[1, 2], [2, 3]]) + self._index_shift
        self.graph.remove_edges(edges)
        assert self.graph.n_edges == 2
        assert self.graph.n_empty_edges == self.graph._ALLOC_MIN - 2
        assert not self.contains(edges[0], self.graph.get_source_edges())
        assert not self.contains(edges[1], self.graph.get_source_edges())

        assert self.graph.n_allocated_edges == self.graph._ALLOC_MIN

    def test_node_removal(self) -> None:
        nodes = np.asarray([3, 4, 1]) + self._index_shift
        original_size = len(self.graph)

        for i in range(len(nodes)):
            node = nodes[i]
            self.graph.remove_node(node)

            for edge in self.graph.get_source_edges():
                assert node not in edge

            for edge in self.graph.get_target_edges():
                assert node not in edge

            assert node not in self.graph.get_nodes()
            assert len(self.graph) == original_size - i - 1

    def test_edge_coordinates(self) -> None:
        coords = self.coords.loc[self.edges.ravel()].to_numpy()
        coords = coords.reshape(self.edges.shape + (-1,))

        source_edge_coords = np.concatenate(
            self.graph.get_source_edges(mode='coords'), axis=0
        )
        assert np.allclose(coords, source_edge_coords)

        target_edges_coords = np.concatenate(
            self.graph.get_target_edges(mode='coords'), axis=0
        )
        rolled_coords = np.roll(coords, shift=1, axis=0)
        assert np.allclose(rolled_coords, target_edges_coords)


class TestUndirectedGraph(TestGraph):
    _GRAPH_CLASS = UndirectedGraph
    __test__ = True

    def test_edge_removal(self) -> None:
        edge = np.asarray([0, 1]) + self._index_shift
        self.graph.remove_edges(edge)
        assert self.graph.n_edges == 4
        assert self.graph.n_empty_edges == self.graph._ALLOC_MIN - 4
        assert not self.contains(edge, self.graph.get_edges())
        assert not self.contains(np.flip(edge), self.graph.get_edges())

        edges = np.asarray([[1, 2], [2, 3]]) + self._index_shift
        self.graph.remove_edges(edges)
        assert self.graph.n_edges == 2
        assert self.graph.n_empty_edges == self.graph._ALLOC_MIN - 2
        assert not self.contains(edges[0], self.graph.get_edges())
        assert not self.contains(edges[1], self.graph.get_edges())

        assert self.graph.n_allocated_edges == self.graph._ALLOC_MIN

        self.assert_empty_linked_list_pairs_are_neighbors()

    def test_node_removal(self) -> None:
        nodes = np.asarray([3, 4, 1]) + self._index_shift
        original_size = len(self.graph)

        for i in range(len(nodes)):
            node = nodes[i]
            self.graph.remove_node(node)

            for edge in self.graph.get_edges():
                assert node not in edge

            assert node not in self.graph.get_nodes()
            assert len(self.graph) == original_size - i - 1

        self.assert_empty_linked_list_pairs_are_neighbors()

    def test_edge_coordinates(self) -> None:
        edge_coords = self.graph.get_edges(mode='coords')

        for node, coords in zip(self.graph.get_nodes(), edge_coords):
            for i, edge in enumerate(self.graph.get_edges(node)):
                assert np.allclose(
                    self.coords.loc[edge, ["y", "x"]].to_numpy(), coords[i]
                )

    def assert_empty_linked_list_pairs_are_neighbors(self) -> None:
        # testing if empty edges linked list pairs are neighbors
        empty_idx = self.graph._empty_edge_idx
        while empty_idx != _EDGE_EMPTY_PTR:
            next_empty_idx = self.graph._edges_buffer[
                empty_idx * _UN_EDGE_SIZE + _LL_UN_EDGE_POS
            ]
            assert empty_idx + 1 == next_empty_idx
            # skipping one
            empty_idx = self.graph._edges_buffer[
                next_empty_idx * _UN_EDGE_SIZE + _LL_UN_EDGE_POS
            ]


class NonSpatialMixin(Protocol):
    # required by typing
    _GRAPH_CLASS: Any
    graph: BaseGraph
    edges: np.ndarray
    _index_shift: int

    def setup_method(self, method: Callable) -> None:
        self.edges = (
            np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=int)
            + self._index_shift
        )

        self.graph = self._GRAPH_CLASS(edges=self.edges)

    def test_edge_coordinates(self) -> None:
        pytest.skip("Non-spatial graph has no coordinates.")


class TestNonSpatialDirectedGraph(NonSpatialMixin, TestDirectedGraph):
    pass


class TestNonSpatialUnirectedGraph(NonSpatialMixin, TestUndirectedGraph):
    pass


class TestDirectedGraphSpecialIndex(TestDirectedGraph):
    _index_shift = 10


class TestUndirectedGraphSpecialIndex(TestUndirectedGraph):
    _index_shift = 10


@pytest.mark.parametrize(
    "node_id",
    [0, 1],
)
def test_removing_last_edge_from_digraph(node_id: int) -> None:
    # regression test from bug reported on Zulip:
    # https://napari.zulipchat.com/#narrow/stream/360030-working-group-graph/topic/Node.20removal.20bug/near/411290388
    coords = np.asarray([[0, 0], [20, 20], [100, 100], [120, 120]])

    graph = DirectedGraph(edges=[[0, 1]], coords=coords)

    # removing node 3 is fine since it is not connected to the sole edge
    graph.remove_node(3)

    # removing either 0 or 1 (source or destination) caused error
    graph.remove_node(node_id)

    assert graph.n_edges == 0
    assert graph.n_nodes == 2
