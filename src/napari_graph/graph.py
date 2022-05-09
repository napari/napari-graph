
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from numba import njit, typed
from numba.core import types

from undirectedgraph import EDGE_SIZE


# Numba constants have to be outside classes

_EDGE_SIZE = 3
_LL_EDGE_POS = _EDGE_SIZE - 1
_EDGE_EMPTY_PTR = -1
_EDGE_BUFFER_FULL = -2


@njit
def _add_edge(buffer: np.ndarray, node2edge: np.ndarray, free_idx: int, src: int, dst: int) -> int:
    """
    TODO: doc

    NOTE:
      - edges are added at the beginning of the linked list so we don't have to track its
        tail and the operation can be done in O(1). This might decrease cash coherence because
        they're sorted in memory in the opposite direction we're iterating
    """

    if free_idx == _EDGE_EMPTY_PTR:
        return _EDGE_BUFFER_FULL  # buffer is full
    elif free_idx < 0:
        return -3  # invalid index
    
    next_edge = node2edge[src]
    node2edge[src] = free_idx

    buffer_index = free_idx * _EDGE_SIZE
    next_empty = buffer[buffer_index + _LL_EDGE_POS]

    buffer[buffer_index] = src
    buffer[buffer_index + 1] = dst
    buffer[buffer_index + _LL_EDGE_POS] = next_edge

    return next_empty


@njit
def _add_undirected_edges(buffer: np.ndarray, edges: np.ndarray, empty_idx: int, n_edges: int, node2edge: np.ndarray) -> Tuple[int, int]:
    # TODO: doc
    """
    Returns next empty index, -1 if full, -2 if there was some error
    """
    size = edges.shape[0]
    for i in range(size):

        # adding (u, v)
        if empty_idx == _EDGE_BUFFER_FULL:
            return _EDGE_BUFFER_FULL, n_edges
        empty_idx = _add_edge(buffer, node2edge, empty_idx, edges[i, 0], edges[i, 1])
        n_edges += 1

        # adding (v, u)
        if empty_idx == _EDGE_BUFFER_FULL:
            return _EDGE_BUFFER_FULL, n_edges
        empty_idx = _add_edge(buffer, node2edge, empty_idx, edges[i, 1], edges[i, 0])
        n_edges += 1

    return empty_idx, n_edges


@njit
def _add_directed_edges(buffer: np.ndarray, edges: np.ndarray, empty_idx: int, n_edges: int, node2edge: np.ndarray) -> Tuple[int, int]:
    # TODO: doc
    """
    Returns next empty index, -1 if full, -2 if there was some error
    """
    size = edges.shape[0]
    for i in range(size):

        if empty_idx == _EDGE_BUFFER_FULL:
            return _EDGE_BUFFER_FULL, n_edges
        empty_idx = _add_edge(buffer, node2edge, empty_idx, edges[i, 0], edges[i, 1])
        n_edges += 1

    return empty_idx, n_edges


@njit
def _iterate_edges(edge_ptr_idx: int, edges_buffer: np.ndarray) -> typed.List:
    """TODO: doc"""
    edges = typed.List.empty_list(types.int64)

    while edge_ptr_idx != _EDGE_EMPTY_PTR:
        buffer_idx = edge_ptr_idx * _EDGE_SIZE
        edges.append(edges_buffer[buffer_idx])      # src
        edges.append(edges_buffer[buffer_idx + 1])  # dst
        edge_ptr_idx = edges_buffer[buffer_idx + _LL_EDGE_POS]
    
    return edges


@njit
def _create_world2buffer_map(world_idx: np.ndarray) -> typed.Dict:
    """
    Fills world indices to buffer indices mapping.
    """
    world2buffer = typed.Dict.empty(types.int64, types.int64)

    for i in range(world_idx.shape[0]):
        world2buffer[world_idx[i]] = i
    
    return world2buffer


@njit
def _vmap_world2buffer(world2buffer: typed.Dict, world_idx: np.ndarray) -> typed.Dict:
    """
    Maps world indices to buffer indices.
    """
    buffer_idx = np.empty(world_idx.shape[0], dtype=int)
    for i in range(world_idx.shape[0]):
        buffer_idx[i] = world2buffer[world_idx[i]]
    return buffer_idx


class BaseGraph:
    # TODO: doc
    _EDGE_DUPLICATION = 1
    _NODE_EMPTY_PTR = -1

    def __init__(self, n_nodes: int, ndim: int, n_edges: int):
        self._active = np.ones(n_nodes, dtype=bool)
        self._coords = np.zeros((n_nodes, ndim), dtype=np.float32)
        self._feats: Dict[str, np.ndarray] = {}

        self._empty_nodes: List[int] = []
        self._node2edges = np.full(n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int)
        self._empty_edge_idx = 0 if n_edges > 0 else _EDGE_EMPTY_PTR
        self._n_edges = 0

        self._edges_buffer = np.full(n_edges * self._EDGE_DUPLICATION * _EDGE_SIZE, fill_value=-1, dtype=int)
        self._edges_buffer[_LL_EDGE_POS : -_EDGE_SIZE :_EDGE_SIZE] = np.arange(1, self._EDGE_DUPLICATION * n_edges)

        self._world2buffer = typed.Dict.empty(types.int64, types.int64)
        self._buffer2world = np.full(n_nodes, fill_value=self._NODE_EMPTY_PTR, dtype=int)
    
    def init_nodes_from_dataframe(
        self,
        nodes_df: pd.DataFrame,
        coordinates_columns: List[str],
    ) -> None:
        # TODO: doc

        if nodes_df.index.dtype != np.int64:
            raise ValueError(f"Nodes indices must be int64. Found {nodes_df.index.dtype}.")
 
        n_nodes = len(nodes_df)

        if  n_nodes > self._coords.shape[0] or len(coordinates_columns) != self._coords.shape[1]:
            self._coords = nodes_df[coordinates_columns].values.astype(np.float32, copy=True)
            self._active = np.ones(n_nodes, dtype=bool)
            self._node2edges = np.full(n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int)
            self._buffer2world = nodes_df.index.values.astype(np.uint64, copy=True)
            self._empty_nodes = []
        else:
            self._coords[:n_nodes] = nodes_df[coordinates_columns].values
            self._active[:n_nodes] = True
            self._node2edges[:n_nodes] = _EDGE_EMPTY_PTR
            self._buffer2world[:n_nodes] = nodes_df.index.values
            self._empty_nodes = list(reversed(range(n_nodes, len(self._active))))  # reversed so we add nodes to the end of it

        self._world2buffer = _create_world2buffer_map(self._buffer2world[:n_nodes])

        # NOTE:
        #  - feats and buffers arrays length may not match after this
        #  - feats should be indexed by their pandas DataFrame index (world index)
        self._feats = nodes_df.drop(coordinates_columns, axis=1)

    def add_node(self, coords: np.ndarray, features: Dict = {}) -> int:
        # TODO: doc
        pass
    
    def _realloc_edges_buffer(self, n_edges: int) -> None:
        # TODO: doc
        # augmenting size to match dummy edges
        n_edges = n_edges * self._EDGE_DUPLICATION
        old_n_allocated = self.n_allocated_edges * self._EDGE_DUPLICATION
        n_allocated = n_edges - old_n_allocated

        if n_allocated < 0:
            raise NotImplementedError("Edge buffer size decrease not implemented.")
        elif n_allocated == 0:
            raise ValueError("Tried to realloc to current buffer size.")

        old_buffer_size = len(self._edges_buffer)
        buffer_size = n_edges * _EDGE_SIZE

        new_edges_buffer = np.full(buffer_size, fill_value=-1, dtype=int)
        new_edges_buffer[:len(self._edges_buffer)] = self._edges_buffer  # filling previous buffer data
        self._edges_buffer = new_edges_buffer

        # fills empty edges ptr
        self._edges_buffer[old_buffer_size + _LL_EDGE_POS : -_EDGE_SIZE : _EDGE_SIZE] =\
             np.arange(old_n_allocated + 1, n_edges) 

        # appends existing empty edges linked list to the end of the new list
        self._edges_buffer[_LL_EDGE_POS - _EDGE_SIZE] = self._empty_edge_idx
        self._empty_edge_idx = old_n_allocated

    @property
    def n_allocated_edges(self) -> int:
        return len(self._edges_buffer) // (self._EDGE_DUPLICATION * _EDGE_SIZE)

    @property
    def n_empty_edges(self) -> int:
        return self.n_allocated_edges - self.n_edges
    
    @property
    def n_edges(self) -> int:
        return self._n_edges

    def _validate_edge_addition(self, edges: Union[np.ndarray, List[Tuple[int, int]]]) -> np.ndarray:
        edges = np.asarray(edges)

        if edges.ndim == 1:
            edges = edges[np.newaxis, ...]

        if edges.ndim != 2:
            raise ValueError(f"Edges must be 1- or 2-dimensional. Found {edges.ndim}-dimensional.")
        
        if edges.shape[1] != 2:
            raise ValueError(f"Edges must be a sequence of length 2 arrays. Found length {edges.shape[1]}")

        if self.n_empty_edges < len(edges):
            self._realloc_edges_buffer(len(edges))

        return edges
    
    def _add_edges(self, edges: np.ndarray) -> None:
        """Abstract method, different implementation for undirected and directed graph."""
        raise NotImplementedError

    def add_edges(self, edges: Union[np.ndarray, List[Tuple[int, int]]]) -> np.ndarray:
        # TODO: 
        #   - doc
        #   - edges features
        edges = self._validate_edge_addition(edges)
        self._add_edges(edges)
 
    def _map_world2buffer(self, world_idx: np.ndarray) -> np.ndarray:
        """Flattens the world indices buffer maps it to buffer coordinates and reshape back to original space."""
        shape = world_idx.shape
        buffer_idx = _vmap_world2buffer(self._world2buffer, world_idx.reshape(-1))
        return buffer_idx.reshape(shape)

    def edges(self, node_idx: int) -> List[int]:
        """
        TODO:
            - doc
            - implement numba version that iterates multiple nodes' edges in a single call.
        """
        flat_edges = _iterate_edges(
            self._node2edges[self._world2buffer[node_idx]],
            self._edges_buffer,
        )
        flat_edges = self._buffer2world[flat_edges]
        return np.asarray(flat_edges.reshape(-1, 2))


class UndirectedGraph(BaseGraph):
    # TODO: doc

    _EDGE_DUPLICATION = 2

    def _add_edges(self, edges: np.ndarray) -> None:
        self._empty_edge_idx, self._n_edges = _add_undirected_edges(
            self._edges_buffer,
            edges,
            self._empty_edge_idx,
            self._n_edges,
            self._node2edges,
        )
        

class DirectedGraph(BaseGraph):
    # TODO: doc

    def _add_edges(self, edges: np.ndarray) -> None:
        self._empty_edge_idx, self._n_edges = _add_directed_edges(
            self._edges_buffer,
            edges,
            self._empty_edge_idx,
            self._n_edges,
            self._node2edges,
        )
