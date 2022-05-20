
from typing import Dict, List, Tuple, Callable, Optional
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd

from numba import njit, typed, prange
from numba.core import types

# Numba constants have to be outside classes :(

# undirected edge constants
_UN_EDGE_SIZE = 3
_LL_UN_EDGE_POS = 2

# directed edge constants
_DI_EDGE_SIZE = _UN_EDGE_SIZE + 1
_LL_DI_EDGE_POS = 2

# generic constants
_EDGE_EMPTY_PTR = -1
_EDGE_BUFFER_FULL = -2
_EDGE_INVALID_INDEX = -3


################################
##  edge insertion functions  ##
################################

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
def _add_directed_edge(
    buffer: np.ndarray,
    node2src_edge: np.ndarray,
    node2tgt_edge: np.ndarray,
    empty_idx: int,
    src_node: int,
    tgt_node: int,
) -> int:
    """
    Adds a single directed edge to `buffer`, updating the `buffer`'s source and target linked list 
    nd the nodes to edges mappings.

    NOTE: see `_add_undirected_edge` docs for comment about cache misses.

     Returns
    -------
    int
        Next empty edge index position.
    """
    if empty_idx == _EDGE_EMPTY_PTR:
        raise ValueError("Edge buffer is full.")

    elif empty_idx < 0:
        raise ValueError("Invalid empty index.")
    
    next_src_edge = node2src_edge[src_node]
    next_tgt_edge = node2tgt_edge[tgt_node]
    node2src_edge[src_node] = empty_idx
    node2tgt_edge[tgt_node] = empty_idx

    buffer_index = empty_idx * _DI_EDGE_SIZE
    next_empty = buffer[buffer_index + _LL_DI_EDGE_POS]

    buffer[buffer_index] = src_node
    buffer[buffer_index + 1] = tgt_node
    buffer[buffer_index + _LL_UN_EDGE_POS] = next_src_edge
    buffer[buffer_index + _LL_DI_EDGE_POS + 1] = next_tgt_edge

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
def _add_directed_edges(
    buffer: np.ndarray,
    edges: np.ndarray,
    empty_idx: int,
    n_edges: int,
    node2src_edge: np.ndarray,
    node2tgt_edge: np.ndarray,
) -> Tuple[int, int]:
    """Adds an array of edges into the `buffer`.
       Directed edges contains two linked lists, outgoing (source) and incoming (target) edges.
    """
    size = edges.shape[0]
    for i in range(size):

        if empty_idx == _EDGE_BUFFER_FULL:
            return _EDGE_BUFFER_FULL, n_edges

        empty_idx = _add_directed_edge(
            buffer, node2src_edge, node2tgt_edge, empty_idx, edges[i, 0], edges[i, 1]
        )
        n_edges += 1

    return empty_idx, n_edges


############################
## edge removal functions ##
############################

@njit(inline='always')
def _remove_edge(
    src_node: int,
    tgt_node: int,
    empty_idx: int,
    edges_buffer: np.ndarray,
    node2edges: np.ndarray,
    edge_size: int,
    ll_edge_pos: int,
) -> int:
    """Generic function to remove directed or undirected nodes.
       An additional removal of the target edges linked list is necessary for directed edges.

    Parameters
    ----------
    src_node : int
        Source node buffer index.
    tgt_node : int
        Target node buffer index.
    empty_idx : int
        First index of empty edges linked list.
    edges_buffer : np.ndarray
        Buffer of edges data.
    node2edges : np.ndarray
        Head of edges linked list, a mapping from node buffer indices to edge buffer indices.
    edge_size : int
        Size of the edges on the buffer. It should be inlined when compiled.
    ll_edge_pos : int
        Position (shift) of the edge linked list on the edge buffer. It should be inlined when compiled.

    Returns
    -------
    int
        New first index of empty edges linked list.
    """

    idx = node2edges[src_node]
    prev_buffer_idx = _EDGE_EMPTY_PTR

    while idx != _EDGE_EMPTY_PTR:
        buffer_idx = idx * edge_size
        next_edge_idx = edges_buffer[buffer_idx + ll_edge_pos] 

        # edge found
        if edges_buffer[buffer_idx + 1] == tgt_node:
            # skipping found edge from linked list
            if prev_buffer_idx == _EDGE_EMPTY_PTR:
                node2edges[src_node] = next_edge_idx
            else:
                edges_buffer[prev_buffer_idx + ll_edge_pos] = next_edge_idx

            # clean up not necessary but good practice
            edges_buffer[buffer_idx:buffer_idx + edge_size] = _EDGE_EMPTY_PTR

            edges_buffer[buffer_idx + ll_edge_pos] = empty_idx
            return idx
        # moving to next edge
        idx = next_edge_idx
        prev_buffer_idx = buffer_idx
    
    raise ValueError("Found an invalid edge at edge removal")


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


@njit(inline='always')
def _remove_target_edge(
    src_node: int,
    tgt_node: int,
    edges_buffer: np.ndarray,
    node2tgt_edges: np.ndarray,
) -> None:
    """Removes edge from target edges linked list. It doesn't clean the buffer, because it'll be used later."""

    idx = node2tgt_edges[tgt_node]  # different indexing from normal edge
    prev_buffer_idx = _EDGE_EMPTY_PTR

    while idx != _EDGE_EMPTY_PTR:
        buffer_idx = idx * _DI_EDGE_SIZE
        next_edge_idx = edges_buffer[buffer_idx + _LL_DI_EDGE_POS + 1] 

        # edge found
        if edges_buffer[buffer_idx] == src_node:   # different indexing from normal edge
            # skipping found edge from linked list
            if prev_buffer_idx == _EDGE_EMPTY_PTR:
                node2tgt_edges[src_node] = next_edge_idx
            else:
                edges_buffer[prev_buffer_idx + _LL_DI_EDGE_POS + 1] = next_edge_idx
            edges_buffer[buffer_idx + _LL_DI_EDGE_POS + 1] = _EDGE_EMPTY_PTR
            return

        # moving to next edge
        idx = next_edge_idx
        prev_buffer_idx = buffer_idx
    
    raise ValueError("Found an invalid edge at edge removal")


@njit
def _remove_directed_edge(
    src_node: int,
    tgt_node: int,
    empty_idx: int,
    edges_buffer: np.ndarray,
    node2src_edges: np.ndarray,
    node2tgt_edges: np.ndarray,
) -> int:
    """Removes a single directed edge from the edges buffer."""

    # must be executed before default edge removal and cleanup
    _remove_target_edge(src_node, tgt_node, edges_buffer, node2tgt_edges)

    empty_idx = _remove_edge(
        src_node, tgt_node, empty_idx, edges_buffer, node2src_edges, _DI_EDGE_SIZE, _LL_DI_EDGE_POS,
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
    """Removes an array of edges from the edges buffer."""

    for i in range(edges.shape[0]):
        empty_idx = _remove_directed_edge(
            edges[i, 0], edges[i, 1], empty_idx, edges_buffer, node2src_edges, node2tgt_edges,
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
    """Removes directed edge from the buffer that contains the given `node` in one of the selected directions.
       The parameter `is_target` should be zero to remove collisions with source nodes and 1 for target nodes.
    """
    if is_target:
        idx = node2tgt_edges[node]
    else:
        idx = node2src_edges[node]

    while idx != _EDGE_EMPTY_PTR:
        buffer_idx = idx * _DI_EDGE_SIZE
        next_idx = edges_buffer[buffer_idx + _LL_DI_EDGE_POS + is_target]

        src_node = edges_buffer[buffer_idx]
        tgt_node = edges_buffer[buffer_idx + 1]

        _remove_target_edge(src_node, tgt_node, edges_buffer, node2tgt_edges)
        empty_idx = _remove_edge(
            src_node, tgt_node, empty_idx, edges_buffer, node2src_edges, _DI_EDGE_SIZE, _LL_DI_EDGE_POS,
        )

        idx = next_idx
        n_edges = n_edges - 1
    
    return empty_idx, n_edges


@njit
def _remove_incident_directed_edges(
    node: int,
    empty_idx: int,
    n_edges: int,
    edges_buffer: np.ndarray,
    node2src_edges: np.ndarray,
    node2tgt_edges: np.ndarray,
) -> Tuple[int, int]:
    """Remove directed edges that contains `node` in either direction."""
    # does it need to be jitted? why not

    empty_idx, n_edges = _remove_unidirectional_incident_edges(
        node, empty_idx, n_edges, edges_buffer, node2src_edges, node2tgt_edges, is_target=0,
    )

    empty_idx, n_edges = _remove_unidirectional_incident_edges(
        node, empty_idx, n_edges, edges_buffer, node2src_edges, node2tgt_edges, is_target=1,
    )

    return empty_idx, n_edges


################################
##  edge iteration functions  ##
################################


@njit(inline='always')
def _iterate_edges(
    edge_ptr_indices: np.ndarray,
    edges_buffer: np.ndarray,
    edge_size: int,
    ll_edge_pos: int,
) -> typed.List:
    """Iterate over the edges linked lists given their starting edges.
       It returns list of multiplicity 2, where each pair is an edge.

    Parameters
    ----------
    edge_ptr_indices : np.ndarray
        Array of starting indices.
    edges_buffer : np.ndarray
        Edges buffer.
    edge_size : int
        Size of the edges on the buffer. It should be inlined when compiled.
    ll_edge_pos : int
        Position (shift) of the edge linked list on the edge buffer. It should be inlined when compiled.

    Returns
    -------
    typed.List
        List of lists edges, adjacent nodes are at indices (k, k+1) such that k is even.
    """
    edges_list = typed.List()

    for idx in edge_ptr_indices:
        edges = typed.List.empty_list(types.int64)
        edges_list.append(edges)

        while idx != _EDGE_EMPTY_PTR:
            buffer_idx = idx * edge_size
            edges.append(edges_buffer[buffer_idx])      # src
            edges.append(edges_buffer[buffer_idx + 1])  # tgt
            idx = edges_buffer[buffer_idx + ll_edge_pos]
    
    return edges_list


@njit
def _iterate_undirected_edges(edge_ptr_indices: np.ndarray, edges_buffer: np.ndarray) -> typed.List:
    """Helper function to inline the edges size and linked list position shift."""
    return _iterate_edges(edge_ptr_indices, edges_buffer, _UN_EDGE_SIZE, _LL_UN_EDGE_POS)


@njit
def _iterate_directed_source_edges(edge_ptr_indices: np.ndarray, edges_buffer: np.ndarray) -> typed.List:

    """Helper function to inline the edges size and linked list position shift."""
    return _iterate_edges(edge_ptr_indices, edges_buffer, _DI_EDGE_SIZE, _LL_DI_EDGE_POS)


@njit
def _iterate_directed_target_edges(edge_ptr_indices: np.ndarray, edges_buffer: np.ndarray) -> typed.List:
    """Helper function to inline the edges size and linked list position shift."""
    return _iterate_edges(edge_ptr_indices, edges_buffer, _DI_EDGE_SIZE, _LL_DI_EDGE_POS + 1)


##############################
##  edge mapping functions  ##
##############################

@njit
def _create_world2buffer_map(world_idx: np.ndarray) -> typed.Dict:
    """Fills world indices to buffer indices mapping."""
    world2buffer = typed.Dict.empty(types.int64, types.int64)

    for i in range(world_idx.shape[0]):
        world2buffer[world_idx[i]] = i
    
    return world2buffer


@njit(parallel=True)  # TODO: benchmark if parallel is worth it
def _vmap_world2buffer(world2buffer: typed.Dict, world_idx: np.ndarray) -> typed.Dict:
    """Maps world indices to buffer indices."""
    buffer_idx = np.empty(world_idx.shape[0], dtype=types.int64)
    for i in prange(world_idx.shape[0]):
        buffer_idx[i] = world2buffer[world_idx[i]]
    return buffer_idx


class BaseGraph:
    """Abstract base graph class.

    Parameters
    ----------
    n_nodes : int
        Number of nodes to allocate to graph.
    ndim : int
        Number of spatial dimensions of graph.
    n_edges : int
        Number of edges of the graph.
    """
    _NODE_EMPTY_PTR = -1

    # abstract constants
    _EDGE_DUPLICATION: int = ...
    _EDGE_SIZE: int = ...
    _LL_EDGE_POS: int = ...

    # allocation constants
    _ALLOC_MULTIPLIER = 1.1
    _ALLOC_MIN = 25

    def __init__(self, n_nodes: int, ndim: int, n_edges: int):
        # node-wise buffers
        self._coords = np.zeros((n_nodes, ndim), dtype=np.float32)
        self._feats = pd.DataFrame()
        self._empty_nodes: List[int] = list(reversed(range(n_nodes)))
        self._node2edges = np.full(n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int)
        self._world2buffer = typed.Dict.empty(types.int64, types.int64)
        self._buffer2world = np.full(n_nodes, fill_value=self._NODE_EMPTY_PTR, dtype=int)
 
        # edge-wise buffers
        self._empty_edge_idx = 0 if n_edges > 0 else _EDGE_EMPTY_PTR
        self._n_edges = 0
        self._edges_buffer = np.full(n_edges * self._EDGE_DUPLICATION * self._EDGE_SIZE, fill_value=_EDGE_EMPTY_PTR, dtype=int)
        self._edges_buffer[self._LL_EDGE_POS : -self._EDGE_SIZE :self._EDGE_SIZE] = np.arange(1, self._EDGE_DUPLICATION * n_edges)
   
    def init_nodes_from_dataframe(
        self,
        nodes_df: pd.DataFrame,
        coordinates_columns: List[str],
    ) -> None:
        """Initializes graph nodes from data frame data. Graph nodes will be indexed by data frame indices.

        Parameters
        ----------
        nodes_df : pd.DataFrame
            Data frame containing node's features and coordinates.
        coordinates_columns : List[str]
            Names of coordinate columns.
        """
        if nodes_df.index.dtype != np.int64:
            raise ValueError(f"Nodes indices must be int64. Found {nodes_df.index.dtype}.")
 
        n_nodes = len(nodes_df)

        if  n_nodes > self._coords.shape[0] or len(coordinates_columns) != self._coords.shape[1]:
            self._coords = nodes_df[coordinates_columns].values.astype(np.float32, copy=True)
            self._node2edges = np.full(n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int)
            self._buffer2world = nodes_df.index.values.astype(np.uint64, copy=True)
            self._empty_nodes = []
        else:
            self._coords[:n_nodes] = nodes_df[coordinates_columns].values
            self._node2edges.fill(_EDGE_EMPTY_PTR)
            self._buffer2world[:n_nodes] = nodes_df.index.values
            self._empty_nodes = list(reversed(range(n_nodes, len(self._buffer2world))))  # reversed so we add nodes to the end of it

        self._world2buffer = _create_world2buffer_map(self._buffer2world[:n_nodes])

        # NOTE:
        #  - feats and buffers arrays length may not match after this
        #  - feats should be indexed by their pandas DataFrame index (world index)
        self._feats = nodes_df.drop(coordinates_columns, axis=1)

    @property
    def n_allocated_nodes(self) -> int:
        """Number of total allocated nodes."""
        return len(self._buffer2world)

    @property
    def n_empty_nodes(self) -> int:
        """Number of nodes allocated but not used."""
        return len(self._empty_nodes)
    
    @property
    def n_nodes(self) -> int:
        """Number of nodes in use."""
        return self.n_allocated_nodes - self.n_empty_nodes
    
    def nodes(self) -> np.ndarray:
        """Indices of graph nodes."""
        return self._buffer2world[self._buffer2world != self._NODE_EMPTY_PTR]
    
    def coordinates(self, node_indices: Optional[ArrayLike] = None) -> np.ndarray:
        """Coordinates of the given nodes, if none is provided it returns the coordinates of every node.
        """
        node_indices = self._validate_nodes(node_indices)
        node_indices = self._map_world2buffer(node_indices)
        return self._coords[node_indices]

    def _realloc_nodes_buffers(self, size: int) -> None:
        """Rellocs the nodes buffers and copies existing data.

        NOTE: Currently, only increasing the buffers' size is implemented.

        Parameters
        ----------
        size : int
            New buffer size.
        """
        prev_size = self.n_allocated_nodes
        size_diff = size - prev_size

        if size_diff < 0:
            raise NotImplementedError("Node buffers size decrease not implemented.")

        elif size_diff == 0:
            raise ValueError("Tried to realloc to current buffer size.")

        self._coords.resize((size, self._coords.shape[1])) # zero-filled
        self._node2edges = np.append(
            self._node2edges,
            np.full(size_diff, fill_value=_EDGE_EMPTY_PTR, dtype=int),
        )
        self._buffer2world = np.append(
            self._buffer2world,
            np.full(size_diff, fill_value=self._NODE_EMPTY_PTR, dtype=int)
        )
        self._empty_nodes = list(reversed(range(prev_size, size)))

        # FIXME: self._feats --- how should it be pre-allocated?

    def add_node(self, index: int, coords: np.ndarray, features: Dict = {}) -> None:
        """Adds node to graph.

        Parameters
        ----------
        index : int
            Node index.
        coords : np.ndarray
            Node coordinates.
        features : Dict, optional
            Node features, by default {}
        """
        
        if self.n_empty_nodes == 0:
            self._realloc_nodes_buffers(
                max(self.n_allocated_nodes * self._ALLOC_MULTIPLIER, self._ALLOC_MIN)
            )
        
        buffer_index = self._empty_nodes.pop()
        self._coords[buffer_index, :] = coords
        self._world2buffer[index] = buffer_index
        self._buffer2world[buffer_index] = index
        # FIXME: not efficient, how should we pre alloc the features?
        if len(features) > 0:
            features = pd.DataFrame([features], index=[index])
        else:
            features = pd.DataFrame(np.NaN, index=[index], columns=self._feats.keys())

        self._feats = pd.concat((self._feats, features))

    def remove_node(self, index: int) -> None:
        """Remove node of given `index`."""
        buffer_index = self._world2buffer.pop(index)
        self._remove_node_edges(buffer_index)
        self._buffer2world[buffer_index] = self._NODE_EMPTY_PTR
        self._empty_nodes.append(buffer_index)

    def _remove_node_edges(self, node_buffer_index: int) -> None:
        """Abstract method, removes node at given buffer index."""
        raise NotImplementedError

    def _realloc_edges_buffers(self, n_edges: int) -> None:
        """Reallocs the edges buffer and copies existing data.

        NOTE: Currently, only increasing the buffers' size is implemented.

        Parameters
        ----------
        n_edges : int
            New number of edges.
        """

        # augmenting size to match dummy edges
        size = n_edges * self._EDGE_DUPLICATION
        prev_size = self.n_allocated_edges * self._EDGE_DUPLICATION
        diff_size = size - prev_size

        if diff_size < 0:
            raise NotImplementedError("Edge buffer size decrease not implemented.")
        elif diff_size == 0:
            raise ValueError("Tried to realloc to current buffer size.")

        prev_buffer_size = len(self._edges_buffer)

        self._edges_buffer = np.append(
            self._edges_buffer,
            np.full(diff_size * self._EDGE_SIZE, fill_value=_EDGE_EMPTY_PTR, dtype=int)
        )

        # fills empty edges ptr
        self._edges_buffer[prev_buffer_size + self._LL_EDGE_POS : -self._EDGE_SIZE :self._EDGE_SIZE] =\
             np.arange(prev_size + 1, size) 

        # appends existing empty edges linked list to the end of the new list
        self._edges_buffer[self._LL_EDGE_POS - self._EDGE_SIZE] = self._empty_edge_idx
        self._empty_edge_idx = prev_size

    @property
    def n_allocated_edges(self) -> int:
        """Number of total allocated edges."""
        return len(self._edges_buffer) // (self._EDGE_DUPLICATION * self._EDGE_SIZE)

    @property
    def n_empty_edges(self) -> int:
        """Number of allocated edges but not used."""
        return self.n_allocated_edges - self.n_edges
    
    @property
    def n_edges(self) -> int:
        """Number of edges in use."""
        return self._n_edges

    def _validate_nodes(self, node_indices: Optional[ArrayLike] = None) -> np.ndarray:
        """Converts and validates the nodes indices."""

        # NOTE: maybe the nodes could be mappend inside this function
        if node_indices is None:
            return self.nodes()
        
        node_indices = np.atleast_1d(node_indices)

        if not np.issubdtype(node_indices.dtype, np.integer):
            raise ValueError(f"Node indices must be integer. Found {node_indices.dtype}.")
 
        if node_indices.ndim != 1:
            raise ValueError(f"Node indices must be 1-dimensional. Found {node_indices.ndim}-dimensional.")
        
        return node_indices

    def _validate_edges(self, edges: ArrayLike) -> np.ndarray:
        """Converts and validates the edges."""
        edges = np.atleast_2d(edges)

        if not np.issubdtype(edges.dtype, np.integer):
            raise ValueError(f"Edges must be integer. Found {edges.dtype}.")

        if edges.ndim != 2:
            raise ValueError(f"Edges must be 1- or 2-dimensional. Found {edges.ndim}-dimensional.")
        
        if edges.shape[1] != 2:
            raise ValueError(f"Edges must be a sequence of length 2 arrays. Found length {edges.shape[1]}")

        return edges
    
    def _add_edges(self, edges: np.ndarray) -> None:
        """Abstract method, different implementation for undirected and directed graph."""
        raise NotImplementedError

    def add_edges(self, edges: ArrayLike) -> None:
        """Add edges into the graph.

        TODO : add parameter for edges features

        Parameters
        ----------
        edges : ArrayLike
            A list of 2-dimensional tuples or an Nx2 array with a pair of nodes indices.
        """
        edges = self._validate_edges(edges)

        if self.n_empty_edges < len(edges):
            self._realloc_edges_buffers(len(edges))

        self._add_edges(edges)
    
    def _remove_edges(self, edges: np.ndarray) -> None:
        raise NotImplementedError
    
    def remove_edges(self, edges: ArrayLike) -> None:
        """Remove edges from the graph.

        Parameters
        ----------
        edges : ArrayLike
            A list of 2-dimensional tuples or an Nx2 array with a pair of nodes indices.
        """
        edges = self._validate_edges(edges)
        edges = self._map_world2buffer(edges)
        self._remove_edges(edges)

        # FIXME: this can lead to inconsistency in the count if removing edges raises an error
        self._n_edges -= len(edges)
 
    def _map_world2buffer(self, world_idx: np.ndarray) -> np.ndarray:
        """Flattens the world indices buffer maps it to buffer coordinates and reshape back to original space."""
        shape = world_idx.shape
        buffer_idx = _vmap_world2buffer(self._world2buffer, world_idx.reshape(-1))
        return buffer_idx.reshape(shape)

    def _iterate_edges(
        self,
        node_world_indices: ArrayLike,
        node2edges: np.ndarray,
        iterate_edges_func: Callable[[np.ndarray, np.ndarray], List[np.ndarray]],
    ) -> List[np.ndarray]:
        """Helper function to iterate over any kind of edges.

        Parameters
        ----------
        node_world_indices : ArrayLike
            Nodes world indices.
        node2edges : np.ndarray
            Mapping from nodes to edges (edges linked list heads).
        iterate_edges_func : Callable[[np.ndarray, np.ndarray], List[np.ndarray]]
            Function that iterates the edges from `edges_ptr_indices` and `edges_buffer`.

        Returns
        -------
        List[np.ndarray]
            List of N_i x 2 arrays, where N_i is the number of edges at the ith node.
        """
        node_world_indices = self._validate_nodes(node_world_indices)

        flat_edges = iterate_edges_func(
            node2edges[self._map_world2buffer(node_world_indices)],
            self._edges_buffer,
        )
        return [
            self._buffer2world[e].reshape(-1, 2) if len(e) > 0 else np.empty((0,2))
            for e in flat_edges
        ]


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
        self._node2tgt_edges = np.full(n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int)

    def init_nodes_from_dataframe(
        self,
        nodes_df: pd.DataFrame,
        coordinates_columns: List[str],
    ) -> None:
        """Initializes graph nodes from data frame data. Graph nodes will be indexed by data frame indices.

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
            self._node2tgt_edges = np.full(n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int)
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
 
    def source_edges(self, nodes: Optional[ArrayLike] = None) -> List[np.ndarray]:
        """Returns the source edges (outgoing) of the given nodes, if none is provided all source edges are returned.

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
            iterate_edges_func=_iterate_directed_source_edges,
        )

    def target_edges(self, nodes: Optional[ArrayLike] = None) -> List[np.ndarray]:
        """Returns the target edges (incoming) of the given nodes, if none is provided all target edges are returned.

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
            node2edges=self._node2tgt_edges,
            iterate_edges_func=_iterate_directed_target_edges,
        )

    def _remove_edges(self, edges: np.ndarray) -> None:
        self._empty_edge_idx = _remove_directed_edges(
            edges, self._empty_edge_idx, self._edges_buffer, self._node2edges, self._node2tgt_edges,
        )

    def _remove_node_edges(self, node_buffer_index: int) -> None:
        self._empty_edge_idx, self._n_edges = _remove_incident_directed_edges(
            node_buffer_index, self._empty_edge_idx, self._n_edges, self._edges_buffer,
            self._node2edges, self._node2tgt_edges
        )
