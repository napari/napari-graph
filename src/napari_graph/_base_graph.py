from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit, typed
from numba.core import types
from numpy.typing import ArrayLike

"""
_EDGE_EMPTY_PTR is used to fill the values of uninitialized/empty/removed nodes or edges
"""
_EDGE_EMPTY_PTR = -1


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

    An additional removal of the target edges linked list is necessary for
    directed edges.

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
        Mapping from node indices to edge buffer indices -- head of edges linked list.
    edge_size : int
        Size of the edges on the buffer. It should be inlined when compiled.
    ll_edge_pos : int
        Position (shift) of the edge linked list on the edge buffer. It should
        be inlined when compiled.

    Returns
    -------
    int
        New first index of empty edges linked list.
    """

    idx = node2edges[src_node]
    prev_buffer_idx = _EDGE_EMPTY_PTR

    # safe guard against a corrupted buffer causing an infite loop
    for _ in range(edges_buffer.shape[0] // edge_size):
        if idx == _EDGE_EMPTY_PTR:
            raise ValueError("Could not find/remove edge.")

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
            edges_buffer[buffer_idx : buffer_idx + edge_size] = _EDGE_EMPTY_PTR
            edges_buffer[buffer_idx + ll_edge_pos] = empty_idx

            break

        # moving to next edge
        idx = next_edge_idx
        prev_buffer_idx = buffer_idx
    else:
        raise ValueError(
            "Infinite loop detected at edge removal, edges buffer must be corrupted."
        )

    return idx


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
        Position (shift) of the edge linked list on the edge buffer. It should
        be inlined when compiled.

    Returns
    -------
    typed.List
        List of lists edges, adjacent nodes are at indices (k, k+1) such that k
        is even.
    """
    edges_list = typed.List()

    for idx in edge_ptr_indices:
        edges = typed.List.empty_list(types.int64)
        edges_list.append(edges)

        while idx != _EDGE_EMPTY_PTR:
            buffer_idx = idx * edge_size
            edges.append(edges_buffer[buffer_idx])  # src
            edges.append(edges_buffer[buffer_idx + 1])  # tgt
            idx = edges_buffer[buffer_idx + ll_edge_pos]

    return edges_list


@njit
def _create_world2buffer_map(world_idx: np.ndarray) -> typed.Dict:
    """Fills world indices to buffer indices mapping."""
    world2buffer = typed.Dict.empty(types.int64, types.int64)

    for i in range(world_idx.shape[0]):
        world2buffer[world_idx[i]] = i

    return world2buffer


@njit
def _vmap_world2buffer(
    world2buffer: typed.Dict, world_idx: np.ndarray
) -> typed.Dict:
    """Maps world indices to buffer indices."""
    buffer_idx = np.empty(world_idx.shape[0], dtype=types.int64)
    for i in range(world_idx.shape[0]):
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
        self._node2edges = np.full(
            n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int
        )
        self._world2buffer = typed.Dict.empty(types.int64, types.int64)
        self._buffer2world = np.full(
            n_nodes, fill_value=self._NODE_EMPTY_PTR, dtype=int
        )

        # edge-wise buffers
        self._empty_edge_idx = 0 if n_edges > 0 else _EDGE_EMPTY_PTR
        self._n_edges = 0
        self._edges_buffer = np.full(
            n_edges * self._EDGE_DUPLICATION * self._EDGE_SIZE,
            fill_value=_EDGE_EMPTY_PTR,
            dtype=int,
        )
        self._edges_buffer[
            self._LL_EDGE_POS : -self._EDGE_SIZE : self._EDGE_SIZE
        ] = np.arange(1, self._EDGE_DUPLICATION * n_edges)

    def init_nodes_from_dataframe(
        self,
        nodes_df: pd.DataFrame,
        coordinates_columns: List[str],
    ) -> None:
        """Initialize graph nodes from data frame data.

        Graph nodes will be indexed by data frame indices.

        Parameters
        ----------
        nodes_df : pd.DataFrame
            Data frame containing node's features and coordinates.
        coordinates_columns : List[str]
            Names of coordinate columns.
        """
        if nodes_df.index.dtype != np.int64:
            raise ValueError(
                f"Nodes indices must be int64. Found {nodes_df.index.dtype}."
            )

        n_nodes = len(nodes_df)

        if (
            n_nodes > self._coords.shape[0]
            or len(coordinates_columns) != self._coords.shape[1]
        ):
            self._coords = nodes_df[coordinates_columns].values.astype(
                np.float32, copy=True
            )
            self._node2edges = np.full(
                n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int
            )
            self._buffer2world = nodes_df.index.values.astype(
                np.uint64, copy=True
            )
            self._empty_nodes = []
        else:
            self._coords[:n_nodes] = nodes_df[coordinates_columns].values
            self._node2edges.fill(_EDGE_EMPTY_PTR)
            self._buffer2world[:n_nodes] = nodes_df.index.values
            self._empty_nodes = list(
                reversed(range(n_nodes, len(self._buffer2world)))
            )  # reversed so we add nodes to the end of it

        self._world2buffer = _create_world2buffer_map(
            self._buffer2world[:n_nodes]
        )

        # NOTE:
        #  - feats and buffers arrays length may not match after this
        #  - feats should be indexed by their pandas DataFrame index
        #    (world index)
        self._feats = nodes_df.drop(coordinates_columns, axis=1)

    @property
    def ndim(self) -> int:
        return self._coords.shape[1]

    @property
    def n_allocated_nodes(self) -> int:
        """Number of total allocated nodes."""
        return len(self._buffer2world)

    @property
    def n_empty_nodes(self) -> int:
        """Number of nodes allocated but not used."""
        return len(self._empty_nodes)

    def __len__(self) -> int:
        """Number of nodes in use."""
        return self.n_allocated_nodes - self.n_empty_nodes

    def nodes(self) -> np.ndarray:
        """Indices of graph nodes."""
        return self._buffer2world[self._buffer2world != self._NODE_EMPTY_PTR]

    def coordinates(
        self, node_indices: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """Coordinates of the given nodes.

        If none is provided it returns the coordinates of every node.
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
            raise NotImplementedError(
                "Node buffers size decrease not implemented."
            )

        elif size_diff == 0:
            raise ValueError("Tried to realloc to current buffer size.")

        self._coords.resize((size, self._coords.shape[1]))  # zero-filled
        self._node2edges = np.append(
            self._node2edges,
            np.full(size_diff, fill_value=_EDGE_EMPTY_PTR, dtype=int),
        )
        self._buffer2world = np.append(
            self._buffer2world,
            np.full(size_diff, fill_value=self._NODE_EMPTY_PTR, dtype=int),
        )
        self._empty_nodes = list(reversed(range(prev_size, size)))

        # FIXME: self._feats --- how should it be pre-allocated?

    def add_node(
        self, index: int, coords: np.ndarray, features: Dict = {}
    ) -> None:
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
                int(
                    max(
                        self.n_allocated_nodes * self._ALLOC_MULTIPLIER,
                        self._ALLOC_MIN,
                    )
                )
            )

        buffer_index = self._empty_nodes.pop()
        self._coords[buffer_index, :] = coords
        self._world2buffer[index] = buffer_index
        self._buffer2world[buffer_index] = index
        # FIXME: not efficient, how should we pre alloc the features?
        if len(features) > 0:
            features = pd.DataFrame([features], index=[index])
        else:
            features = pd.DataFrame(
                np.NaN, index=[index], columns=self._feats.keys()
            )

        self._feats = pd.concat((self._feats, features))

    def remove_node(self, index: int, is_buffer_domain: bool = False) -> None:
        """Remove node of given `index`, by default it's the world index.

        Parameters
        ----------
        index : int
            node index
        is_buffer_domain : bool, optional
            indicates if the index in on the buffer domain, by default False
        """
        if is_buffer_domain:
            index = self._buffer2world[index]
        buffer_index = self._world2buffer.pop(index)
        self._remove_node_edges(buffer_index)
        self._buffer2world[buffer_index] = self._NODE_EMPTY_PTR
        self._empty_nodes.append(buffer_index)

    @abstractmethod
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
            raise NotImplementedError(
                "Edge buffer size decrease not implemented."
            )
        elif diff_size == 0:
            raise ValueError("Tried to realloc to current buffer size.")

        prev_buffer_size = len(self._edges_buffer)

        self._edges_buffer = np.append(
            self._edges_buffer,
            np.full(
                diff_size * self._EDGE_SIZE,
                fill_value=_EDGE_EMPTY_PTR,
                dtype=int,
            ),
        )

        # fills empty edges ptr
        self._edges_buffer[
            prev_buffer_size
            + self._LL_EDGE_POS : -self._EDGE_SIZE : self._EDGE_SIZE
        ] = np.arange(prev_size + 1, size)

        # appends existing empty edges linked list to the end of the new list
        self._edges_buffer[
            self._LL_EDGE_POS - self._EDGE_SIZE
        ] = self._empty_edge_idx
        self._empty_edge_idx = prev_size

    @property
    def n_allocated_edges(self) -> int:
        """Number of total allocated edges."""
        return len(self._edges_buffer) // (
            self._EDGE_DUPLICATION * self._EDGE_SIZE
        )

    @property
    def n_empty_edges(self) -> int:
        """Number of allocated edges but not used."""
        return self.n_allocated_edges - self.n_edges

    @property
    def n_edges(self) -> int:
        """Number of edges in use."""
        return self._n_edges

    def _validate_nodes(
        self, node_indices: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """Converts and validates the nodes indices."""

        # NOTE: maybe the nodes could be mappend inside this function
        if node_indices is None:
            return self.nodes()

        node_indices = np.atleast_1d(node_indices)

        if not np.issubdtype(node_indices.dtype, np.integer):
            raise ValueError(
                f"Node indices must be integer. Found {node_indices.dtype}."
            )

        if node_indices.ndim != 1:
            raise ValueError(
                "Node indices must be 1-dimensional. "
                f"Found {node_indices.ndim}-dimensional."
            )

        return node_indices

    def _validate_edges(self, edges: ArrayLike) -> np.ndarray:
        """Converts and validates the edges."""
        edges = np.atleast_2d(edges)

        if not np.issubdtype(edges.dtype, np.integer):
            raise ValueError(f"Edges must be integer. Found {edges.dtype}.")

        if edges.ndim != 2:
            raise ValueError(
                "Edges must be 1- or 2-dimensional. "
                f"Found {edges.ndim}-dimensional."
            )

        if edges.shape[1] != 2:
            raise ValueError(
                f"Edges must be a sequence of length 2 arrays. "
                f"Found length {edges.shape[1]}"
            )

        return edges

    @abstractmethod
    def _add_edges(self, edges: np.ndarray) -> None:
        """Abstract method.

        Requires different implementation for undirected and directed graphs.
        """
        raise NotImplementedError

    def add_edges(self, edges: ArrayLike) -> None:
        """Add edges into the graph.

        TODO : add parameter for edges features

        Parameters
        ----------
        edges : ArrayLike
            A list of 2-dimensional tuples or an Nx2 array with a pair of
            node indices.
        """
        edges = self._validate_edges(edges)

        if self.n_empty_edges < len(edges):
            self._realloc_edges_buffers(len(edges))

        self._add_edges(edges)

    @abstractmethod
    def _remove_edges(self, edges: np.ndarray) -> None:
        raise NotImplementedError

    def remove_edges(self, edges: ArrayLike) -> None:
        """Remove edges from the graph.

        Parameters
        ----------
        edges : ArrayLike
            A list of 2-dimensional tuples or an Nx2 array with a pair of
            node indices.
        """
        edges = self._validate_edges(edges)
        edges = self._map_world2buffer(edges)
        self._remove_edges(edges)

        # FIXME: this can lead to inconsistency in the count if removing edges
        # raises an error
        self._n_edges -= len(edges)

    def _map_world2buffer(self, world_idx: np.ndarray) -> np.ndarray:
        """Flatten the world indices buffer maps into buffer coordinates.

        ... and reshape back to original space.
        """
        shape = world_idx.shape
        buffer_idx = _vmap_world2buffer(
            self._world2buffer, world_idx.reshape(-1)
        )
        return buffer_idx.reshape(shape)

    def _iterate_edges(
        self,
        node_world_indices: ArrayLike,
        node2edges: np.ndarray,
        iterate_edges_func: Callable[
            [np.ndarray, np.ndarray], List[np.ndarray]
        ],
    ) -> List[List]:
        """Helper function to iterate over edges and return buffer indices.

        Parameters
        ----------
        node_world_indices : ArrayLike
            Nodes world indices.
        node2edges : np.ndarray
            Mapping from node indices to edge buffer indices -- head of edges linked list.
        iterate_edges_func : [np.ndarray, np.ndarray] -> List[np.ndarray]
            Function that iterates the edges from `edges_ptr_indices` and
            `edges_buffer`.

        Returns
        -------
        List[List]
            List of Lists of length 2 * N_i, where N_i is the number of edges
            at the ith node.
        """
        node_world_indices = self._validate_nodes(node_world_indices)

        flat_edges = iterate_edges_func(
            node2edges[self._map_world2buffer(node_world_indices)],
            self._edges_buffer,
        )
        return flat_edges

    def _iterate_edges_generic(
        self,
        node_world_indices: ArrayLike,
        node2edges: np.ndarray,
        iterate_edges_func: Callable[
            [np.ndarray, np.ndarray], List[np.ndarray]
        ],
        mode: str,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Iterate over any kind of edges and return their world indices.

        Parameters
        ----------
        node_world_indices : ArrayLike
            Nodes world indices.
        node2edges : np.ndarray
            Mapping from node indices to edge buffer indices -- head of edges linked list.
        iterate_edges_func : [np.ndarray, np.ndarray] -> List[np.ndarray]
            Function that iterates the edges from `edges_ptr_indices` and
            `edges_buffer`.
        mode : str
            Type of data queried from the edges. For example, `indices` or
            `coords`.

        Returns
        -------
        List[np.ndarray]
            List of N_i x 2 x D arrays, where N_i is the number of edges at
            the ith node.  D is the dimensionality of `coords` when
            mode == `coords` and it's ignored when mode == `indices`.
        """
        flat_edges = self._iterate_edges(
            node_world_indices, node2edges, iterate_edges_func
        )

        if mode.lower() == 'indices':
            edges_data = [
                self._buffer2world[e].reshape(-1, 2)
                if len(e) > 0
                else np.empty((0, 2))
                for e in flat_edges
            ]
        elif mode.lower() == 'coords':
            dim = self._coords.shape[1]
            edges_data = [
                self._coords[e].reshape(-1, 2, dim)
                if len(e) > 0
                else np.empty((0, 2, dim))
                for e in flat_edges
            ]
        # NOTE: here `mode` could also query the edges features.
        # Not implemented yet.
        else:
            modes = ('indices', 'coords')
            raise ValueError(
                f"Edge iteration mode not found. Received {mode}, "
                f"expected {modes}."
            )

        if len(edges_data) == 1:
            return edges_data[0]
        else:
            return edges_data

    @abstractmethod
    def edges(
        self, nodes: Optional[ArrayLike] = None, mode: str = 'indices'
    ) -> Union[List[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def edges_buffers(
        self, is_buffer_domain: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return valid edges in buffer or world domain.

        Return the indices (buffer domain) and the (source, target) values
        (world domain) of all valid edges.

        Undirected edges are not duplicated.

        This function is useful for loading the data for visualization.

        Parameters
        ----------
        is_buffer_domain : bool
            flag indicating if it should return `world` or `buffer` domain.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Buffer indices (buffer domain) and (source, target) (world domain by default).
        """
        unique_edge_size = self._EDGE_SIZE * self._EDGE_DUPLICATION
        buffer_size = len(self._edges_buffer)
        indices = np.arange(0, buffer_size, unique_edge_size)

        # reshaping such that each row is (source id, target id, ...)
        buffer = self._edges_buffer.reshape((-1, unique_edge_size))
        edges = buffer[:, :2]  # (source, target)

        valid = edges[:, 0] != _EDGE_EMPTY_PTR

        indices = indices[valid]
        edges = edges[valid]
        if not is_buffer_domain:
            edges = self._buffer2world[edges]

        return indices, edges
