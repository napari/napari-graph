from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit, typed
from numpy.typing import ArrayLike

from napari_graph._base_graph import (
    _EDGE_EMPTY_PTR,
    BaseGraph,
    _iterate_edges,
    _remove_edge,
)
from napari_graph.undirected_graph import _UN_EDGE_SIZE

"""
Directed edge constants for accessing the directed graph buffer data.
Each edge occupies _DI_EDGE_SIZE spaces on the graph buffer.
_LL_DI_EDGE_POS indicates the displacement between the edge initial index and
the **target** edge directed linked list position.

Example of a directed graph edge buffer:
[
    source_node_buffer_id_0,
    target_node_buffer_id_0,
    source_edge_linked_list_0,
    target_edge_linked_List_0,
    source_node_buffer_id_1,
    target_node_buffer_id_1,
    source_edge_linked_list_1,
    target_edge_linked_List_1,
    ...
]
"""
_DI_EDGE_SIZE = _UN_EDGE_SIZE + 1
_LL_DI_EDGE_POS = 2


@njit
def _add_directed_edge(
    buffer: np.ndarray,
    node2src_edges: np.ndarray,
    node2tgt_edges: np.ndarray,
    empty_idx: int,
    src_node: int,
    tgt_node: int,
) -> int:
    """Add a single directed edge to `buffer`.

    This updates the `buffer`'s source and target linked list
    and the nodes to edges mappings.

    NOTE: see `_add_undirected_edge` docs for comment about cache misses.

    Parameters
    ----------
    buffer : np.ndarray
        Edges buffer.
    node2src_edges : np.ndarray
        Mapping from node indices to source edge buffer indices -- head of edges linked list.
    node2tgt_edges : np.ndarray
        Mapping from node indices to target edge buffer indices -- head of edges linked list.
    empty_idx : int
        First index of empty edges linked list.
    src_node : int
        Source node of added edge.
    tgt_node : int
        Target node of added edge.

    Returns
    -------
    int
        New first index of empty edges linked list.
    """
    if empty_idx == _EDGE_EMPTY_PTR:
        raise ValueError("Edge buffer is full.")

    elif empty_idx < 0:
        raise ValueError("Invalid empty index.")

    next_src_edge = node2src_edges[src_node]
    next_tgt_edge = node2tgt_edges[tgt_node]
    node2src_edges[src_node] = empty_idx
    node2tgt_edges[tgt_node] = empty_idx

    buffer_index = empty_idx * _DI_EDGE_SIZE
    next_empty = buffer[buffer_index + _LL_DI_EDGE_POS]

    buffer[buffer_index] = src_node
    buffer[buffer_index + 1] = tgt_node
    buffer[buffer_index + _LL_DI_EDGE_POS] = next_src_edge
    buffer[buffer_index + _LL_DI_EDGE_POS + 1] = next_tgt_edge

    return next_empty


@njit
def _add_directed_edges(
    buffer: np.ndarray,
    edges: np.ndarray,
    empty_idx: int,
    n_edges: int,
    node2src_edges: np.ndarray,
    node2tgt_edges: np.ndarray,
) -> Tuple[int, int]:
    """Add an array of edges into the `buffer`.

    Directed edges contains two linked lists, outgoing (source) and incoming
    (target) edges.
    """
    size = edges.shape[0]
    for i in range(size):

        empty_idx = _add_directed_edge(
            buffer,
            node2src_edges,
            node2tgt_edges,
            empty_idx,
            edges[i, 0],
            edges[i, 1],
        )
        n_edges += 1

    return empty_idx, n_edges


@njit(inline='always')
def _remove_target_edge(
    src_node: int,
    tgt_node: int,
    edges_buffer: np.ndarray,
    node2tgt_edges: np.ndarray,
) -> None:
    """Remove edge from target edges linked list.
    It doesn't clean the buffer, because it'll be used later.

    Parameters
    ----------
    src_node : int
        Source node of added edge.
    tgt_node : int
        Target node of added edge.
    edges_buffer : np.ndarray
        Edges buffer.
    node2tgt_edges : np.ndarray
        Mapping from node indices to target edge buffer indices -- head of edges linked list.
    """
    idx = node2tgt_edges[tgt_node]  # different indexing from source edge
    prev_buffer_idx = _EDGE_EMPTY_PTR

    for _ in range(edges_buffer.shape[0] // _DI_EDGE_SIZE):
        if idx == _EDGE_EMPTY_PTR:
            raise ValueError(
                "Could not find target node at directed edge removal."
            )

        buffer_idx = idx * _DI_EDGE_SIZE
        next_edge_idx = edges_buffer[buffer_idx + _LL_DI_EDGE_POS + 1]

        # edge found
        if (
            edges_buffer[buffer_idx] == src_node
        ):  # different indexing from source edge
            # skipping found edge from linked list
            if prev_buffer_idx == _EDGE_EMPTY_PTR:
                node2tgt_edges[
                    tgt_node
                ] = next_edge_idx  # different indexing from source edge
            else:
                edges_buffer[
                    prev_buffer_idx + _LL_DI_EDGE_POS + 1
                ] = next_edge_idx

            edges_buffer[buffer_idx + _LL_DI_EDGE_POS + 1] = _EDGE_EMPTY_PTR
            break

        # moving to next edge
        idx = next_edge_idx
        prev_buffer_idx = buffer_idx
    else:
        raise ValueError(
            "Infinite loop detected at target edge removal, edges buffer must be corrupted."
        )


@njit
def _remove_directed_edge(
    src_node: int,
    tgt_node: int,
    empty_idx: int,
    edges_buffer: np.ndarray,
    node2src_edges: np.ndarray,
    node2tgt_edges: np.ndarray,
) -> int:
    """Remove a single directed edge from the edges buffer."""

    # must be executed before default edge removal and cleanup
    _remove_target_edge(src_node, tgt_node, edges_buffer, node2tgt_edges)

    empty_idx = _remove_edge(
        src_node,
        tgt_node,
        empty_idx,
        edges_buffer,
        node2src_edges,
        _DI_EDGE_SIZE,
        _LL_DI_EDGE_POS,
    )

    return empty_idx


@njit
def _remove_directed_edges(
    edges: np.ndarray,
    empty_idx: int,
    edges_buffer: np.ndarray,
    node2src_edges: np.ndarray,
    node2tgt_edges: np.ndarray,
) -> int:
    """Remove an array of edges from the edges buffer."""

    for i in range(edges.shape[0]):
        empty_idx = _remove_directed_edge(
            edges[i, 0],
            edges[i, 1],
            empty_idx,
            edges_buffer,
            node2src_edges,
            node2tgt_edges,
        )
    return empty_idx


@njit
def _remove_unidirectional_incident_edges(
    node: int,
    empty_idx: int,
    n_edges: int,
    edges_buffer: np.ndarray,
    node2src_edges: np.ndarray,
    node2tgt_edges: np.ndarray,
    is_target: int,
) -> Tuple[int, int]:
    """Remove directed edges from the buffer that contain the given `node`.

    The parameter `is_target` should be zero to remove edges where `node` is
    the source node, and 1 for the target node.

    Parameters
    ----------
    node : int
        Node index in the buffer domain.
    empty_idx : int
        Index first edge (head of) linked list
    n_edges : int
        Current number of total edges
    edges_buffer : np.ndarray
        Buffer containing the edges data
    node2src_edges : np.ndarray
        Mapping from node indices to source edge buffer indices -- head of edges linked list.
    node2tgt_edges : np.ndarray
        Mapping from node indices to target edge buffer indices -- head of edges linked list.
    is_target : int
        Binary integer flag indicating if it is a target or not, used to shift linked list position.

    Returns
    -------
    Tuple[int, int]
        New empty linked list head, new number of edges
    """
    if is_target:
        idx = node2tgt_edges[node]
    else:
        idx = node2src_edges[node]

    # safe guard against a corrupted buffer causing an infite loop
    for _ in range(edges_buffer.shape[0] // _DI_EDGE_SIZE):
        if idx == _EDGE_EMPTY_PTR:
            break  # no edges left at the given node

        buffer_idx = idx * _DI_EDGE_SIZE
        next_idx = edges_buffer[buffer_idx + _LL_DI_EDGE_POS + is_target]

        src_node = edges_buffer[buffer_idx]
        tgt_node = edges_buffer[buffer_idx + 1]

        # must be removed before source edge, due to information cleanup
        _remove_target_edge(src_node, tgt_node, edges_buffer, node2tgt_edges)
        empty_idx = _remove_edge(
            src_node,
            tgt_node,
            empty_idx,
            edges_buffer,
            node2src_edges,
            _DI_EDGE_SIZE,
            _LL_DI_EDGE_POS,
        )

        idx = next_idx
        n_edges = n_edges - 1
    else:
        raise ValueError(
            "Infinite loop detected at directed graph node removal, edges buffer must be corrupted."
        )

    return empty_idx, n_edges


@njit
def _iterate_directed_source_edges(
    edge_ptr_indices: np.ndarray, edges_buffer: np.ndarray
) -> typed.List:
    """Inline the edges size and linked list position shift."""
    return _iterate_edges(
        edge_ptr_indices, edges_buffer, _DI_EDGE_SIZE, _LL_DI_EDGE_POS
    )


@njit
def _iterate_directed_target_edges(
    edge_ptr_indices: np.ndarray, edges_buffer: np.ndarray
) -> typed.List:
    """Inline the edges size and linked list position shift."""
    return _iterate_edges(
        edge_ptr_indices, edges_buffer, _DI_EDGE_SIZE, _LL_DI_EDGE_POS + 1
    )


class DirectedGraph(BaseGraph):
    """Directed graph class.

    Parameters
    ----------
    n_nodes : int
        Number of nodes to allocate to graph.
    ndim : int
        Number of spatial dimensions of graph.
    n_edges : int
        Number of edges of the graph.
    """

    _EDGE_DUPLICATION = 1
    _EDGE_SIZE = _DI_EDGE_SIZE
    _LL_EDGE_POS = _LL_DI_EDGE_POS

    def __init__(self, n_nodes: int, ndim: int, n_edges: int):
        super().__init__(n_nodes, ndim, n_edges)
        self._node2tgt_edges = np.full(
            n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int
        )

    def init_nodes_from_dataframe(
        self,
        nodes_df: pd.DataFrame,
        coordinates_columns: List[str],
    ) -> None:
        """Initialize graph nodes from dataframe.

        Graph nodes will be indexed by dataframe indices.

        Parameters
        ----------
        nodes_df : pd.DataFrame
            Data frame containing node's features and coordinates.
        coordinates_columns : List[str]
            Names of coordinate columns.
        """
        super().init_nodes_from_dataframe(nodes_df, coordinates_columns)
        n_nodes = len(nodes_df)
        if len(self._node2tgt_edges) < n_nodes:
            self._node2tgt_edges = np.full(
                n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int
            )
        else:
            self._node2tgt_edges.fill(_EDGE_EMPTY_PTR)

    def _realloc_nodes_buffers(self, size: int) -> None:
        diff_size = size - self.n_allocated_nodes
        super()._realloc_nodes_buffers(size)
        self._node2tgt_edges = np.append(
            self._node2tgt_edges,
            np.full(diff_size, fill_value=self._NODE_EMPTY_PTR, dtype=int),
        )

    def _add_edges(self, edges: np.ndarray) -> None:
        self._empty_edge_idx, self._n_edges = _add_directed_edges(
            self._edges_buffer,
            edges,
            self._empty_edge_idx,
            self._n_edges,
            self._node2edges,
            self._node2tgt_edges,
        )

    def source_edges(
        self, nodes: Optional[ArrayLike] = None, mode: str = 'indices'
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Return the source edges (outgoing) of the given nodes.

        If no nodes are provided, all source edges are returned.

        Parameters
        ----------
        nodes : Optional[ArrayLike], optional
            Node indices, by default None
        mode : str
            Type of data queried from the edges. For example, `indices` or
            `coords`.

        Returns
        -------
        List[np.ndarray]
            List of (N_i) x 2 x D arrays, where N_i is the number of edges at
            the ith node.  D is the dimensionality of `coords` when
            mode == `coords` and is ignored when mode == `indices`. N_i
            dimension is ignored when N_i is 1.
        """
        return self._iterate_edges_generic(
            nodes,
            node2edges=self._node2edges,
            iterate_edges_func=_iterate_directed_source_edges,
            mode=mode,
        )

    def target_edges(
        self, nodes: Optional[ArrayLike] = None, mode: str = 'indices'
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Return the target edges (incoming) of the given nodes.

        If no nodes are provided, all target edges are returned.

        Parameters
        ----------
        nodes : Optional[ArrayLike], optional
            Node indices, by default None
        mode : str
            Type of data queried from the edges. For example, `indices` or
            `coords`.

        Returns
        -------
        List[np.ndarray]
            List of (N_i) x 2 x D arrays, where N_i is the number of edges at
            the ith node.  D is the dimensionality of `coords` when
            mode == `coords` and it's ignored when mode == `indices`. N_i
            dimension is ignored when N_i is 1.
        """
        return self._iterate_edges_generic(
            nodes,
            node2edges=self._node2tgt_edges,
            iterate_edges_func=_iterate_directed_target_edges,
            mode=mode,
        )

    def _remove_edges(self, edges: np.ndarray) -> None:
        self._empty_edge_idx = _remove_directed_edges(
            edges,
            self._empty_edge_idx,
            self._edges_buffer,
            self._node2edges,
            self._node2tgt_edges,
        )

    def _remove_node_edges(self, node_buffer_index: int) -> None:
        """Remove directed edges that contain `node` in either direction."""
        for is_target in (1, 0):
            (
                self._empty_edge_idx,
                self._n_edges,
            ) = _remove_unidirectional_incident_edges(
                node_buffer_index,
                self._empty_edge_idx,
                self._n_edges,
                self._edges_buffer,
                self._node2edges,
                self._node2tgt_edges,
                is_target,
            )
