
from typing import List, Tuple, Optional
from numpy.typing import ArrayLike

import numpy as np

from numba import njit, typed

from napari_graph._base_graph import BaseGraph, _iterate_edges, _EDGE_EMPTY_PTR, _remove_edge


# undirected edge constants
_UN_EDGE_SIZE = 3
_LL_UN_EDGE_POS = 2


@njit
def _add_undirected_edge(buffer: np.ndarray, node2edge: np.ndarray, empty_idx: int, src_node: int, tgt_node: int) -> int:
    """
    Adds a single edge (`src_idx`, `tgt_idx`) to `buffer`, updating the edge linked list (present in the buffer)
    and the `node2edges` mapping (head of linked list).

    NOTE: Edges are added at the beginning of the linked list so we don't have to track its
          tail and the operation can be done in O(1). This might decrease cash hits because
          they're sorted in memory in the opposite direction we iterate it.

    Returns
    -------
    int
        Next empty edge index position.
    """

    if empty_idx == _EDGE_EMPTY_PTR:
        raise ValueError("Edge buffer is full.")

    elif empty_idx < 0:
        raise ValueError("Invalid empty index.")
    
    next_edge = node2edge[src_node]
    node2edge[src_node] = empty_idx

    buffer_index = empty_idx * _UN_EDGE_SIZE
    next_empty = buffer[buffer_index + _LL_UN_EDGE_POS]

    buffer[buffer_index] = src_node
    buffer[buffer_index + 1] = tgt_node
    buffer[buffer_index + _LL_UN_EDGE_POS] = next_edge

    return next_empty


@njit
def _add_undirected_edges(
    buffer: np.ndarray,
    edges: np.ndarray,
    empty_idx: int,
    n_edges: int,
    node2edge: np.ndarray
) -> Tuple[int, int]:
    """Adds an array of edges into the `buffer`.
       Edges are duplicated so both directions are available for fast graph transversal.
    """
    size = edges.shape[0]
    for i in range(size):

        # adding (u, v)
        empty_idx = _add_undirected_edge(buffer, node2edge, empty_idx, edges[i, 0], edges[i, 1])
        # adding (v, u)
        empty_idx = _add_undirected_edge(buffer, node2edge, empty_idx, edges[i, 1], edges[i, 0])

        n_edges += 1

    return empty_idx, n_edges


@njit
def _remove_undirected_edge(
    src_node: int,
    tgt_node: int,
    empty_idx: int,
    edges_buffer: np.ndarray,
    node2edges: np.ndarray,
) -> int:
    """Removes a single edge (and its duplicated sibling edge) from the buffer.
       NOTE: Edges are removed such that empty pairs are consecutive in memory.
    """
    empty_idx = _remove_edge(
        tgt_node, src_node, empty_idx, edges_buffer, node2edges, _UN_EDGE_SIZE, _LL_UN_EDGE_POS,
    )

    empty_idx = _remove_edge(
        src_node, tgt_node, empty_idx, edges_buffer, node2edges, _UN_EDGE_SIZE, _LL_UN_EDGE_POS,
    )

    return empty_idx


@njit
def _remove_undirected_edges(
    edges: np.ndarray,
    empty_idx: int,
    edges_buffer: np.ndarray,
    node2edges: np.ndarray,
) -> int:
    """Removes an array of edges (and their duplicated siblings edges) from the buffer."""

    for i in range(edges.shape[0]):
        empty_idx = _remove_undirected_edge(
            edges[i, 0], edges[i, 1], empty_idx, edges_buffer, node2edges
        )
    return empty_idx


@njit
def _remove_incident_undirected_edges(
    node: int,
    empty_idx: int,
    n_edges: int,
    edges_buffer: np.ndarray,
    node2edges: np.ndarray,
) -> Tuple[int, int]:
    """Removes every edges that contains `node_idx`.
       NOTE: Edges are removed such that empty pairs are consecutive in memory.
    """

    # the edges are removed such that the empty edges linked list contains
    # two positions adjacent in memory so we can serialize the edges using numpy vectorization
    idx = node2edges[node]

    while idx != _EDGE_EMPTY_PTR:
        buffer_idx = idx * _UN_EDGE_SIZE
        next_idx = edges_buffer[buffer_idx + _LL_UN_EDGE_POS]
        # checking if sibling edges is before or after current node
        if (buffer_idx > 0 and
            edges_buffer[buffer_idx - _UN_EDGE_SIZE + 1] == edges_buffer[buffer_idx] and
            edges_buffer[buffer_idx - _UN_EDGE_SIZE] == edges_buffer[buffer_idx + 1]):

            src_node = edges_buffer[buffer_idx + 1]
            tgt_node = edges_buffer[buffer_idx]
        else:
            src_node = edges_buffer[buffer_idx]
            tgt_node = edges_buffer[buffer_idx + 1]

        empty_idx = _remove_edge(
            tgt_node, src_node, empty_idx, edges_buffer, node2edges, _UN_EDGE_SIZE, _LL_UN_EDGE_POS
        )

        empty_idx = _remove_edge(
            src_node, tgt_node, empty_idx, edges_buffer, node2edges, _UN_EDGE_SIZE, _LL_UN_EDGE_POS
        )

        idx = next_idx
        n_edges = n_edges - 1

    return empty_idx, n_edges


@njit
def _iterate_undirected_edges(edge_ptr_indices: np.ndarray, edges_buffer: np.ndarray) -> typed.List:
    """Helper function to inline the edges size and linked list position shift."""
    return _iterate_edges(edge_ptr_indices, edges_buffer, _UN_EDGE_SIZE, _LL_UN_EDGE_POS)


class UndirectedGraph(BaseGraph):
    """Undirected graph class.

    Parameters
    ----------
    n_nodes : int
        Number of nodes to allocate to graph.
    ndim : int
        Number of spatial dimensions of graph.
    n_edges : int
        Number of edges of the graph.
    """

    _EDGE_DUPLICATION = 2
    _EDGE_SIZE = _UN_EDGE_SIZE
    _LL_EDGE_POS = _LL_UN_EDGE_POS

    def _add_edges(self, edges: np.ndarray) -> None:
        self._empty_edge_idx, self._n_edges = _add_undirected_edges(
            self._edges_buffer,
            edges,
            self._empty_edge_idx,
            self._n_edges,
            self._node2edges,
        )

    def edges(self, nodes: Optional[ArrayLike] = None) -> List[np.ndarray]:
        """Returns the edges of the given nodes, if none is provided all edges are returned.

        Parameters
        ----------
        nodes : Optional[ArrayLike], optional
            Node indices, by default None

        Returns
        -------
        List[np.ndarray]
            List of N_i x 2 arrays, where N_i is the number of edges at the ith node.
        """
        return self._iterate_edges(
            nodes,
            node2edges=self._node2edges,
            iterate_edges_func=_iterate_undirected_edges,
        )
    
    def _remove_edges(self, edges: np.ndarray) -> None:
        self._empty_edge_idx = _remove_undirected_edges(
            edges, self._empty_edge_idx, self._edges_buffer, self._node2edges
        )

    def _remove_node_edges(self, node_buffer_index: int) -> None:
        self._empty_edge_idx, self._n_edges = _remove_incident_undirected_edges(
            node_buffer_index, self._empty_edge_idx, self._n_edges, self._edges_buffer, self._node2edges
        )
