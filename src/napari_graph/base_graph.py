from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from napari_graph.numba import njit, typed, types

"""
_NODE_EMPTY_PTR is used to fill the values of uninitialized/empty/removed nodes
"""
_NODE_EMPTY_PTR = -1

"""
_EDGE_EMPTY_PTR is used to fill the values of uninitialized/empty/removed edges
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
def _contains_keys(
    map: typed.Dict,
    keys: np.ndarray,
) -> bool:
    """Returns true if at least one `key` is present on `map`."""
    for k in keys:
        if k in map:
            return True
    return False


@njit
def _update_world2buffer(
    world2buffer: typed.Dict,
    world_idx: np.ndarray,
    buffer_idx: np.ndarray,
) -> None:
    """Updates `world_idx` (keys) and `buffer_idx` (values) to `world2buffer` mapping."""
    for w, b in zip(world_idx, buffer_idx):
        world2buffer[w] = b


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
    edges : ArrayLike
        Nx2 array of pair of nodes (edges).
    coords :
        Optional array of spatial coordinates of nodes.
    ndim : int
        Number of spatial dimensions of graph.
    n_nodes : int
        Optional number of nodes to pre-allocate in the graph.
    n_edges : int
        Optional number of edges to pre-allocate in the graph.
    """

    # abstract constants
    _EDGE_DUPLICATION: int
    _EDGE_SIZE: int
    _LL_EDGE_POS: int

    # allocation constants
    _ALLOC_MULTIPLIER = 1.1
    _ALLOC_MIN = 25

    def __init__(
        self,
        edges: ArrayLike = (),
        coords: Optional[Union[pd.DataFrame, ArrayLike]] = None,
        ndim: Optional[int] = None,
        n_nodes: Optional[int] = None,
        n_edges: Optional[int] = None,
    ):
        # validate nodes
        if coords is not None:
            if not isinstance(coords, pd.DataFrame):
                coords = pd.DataFrame(coords)
            if not np.issubdtype(coords.index.dtype, np.integer):
                raise ValueError(
                    f"The index of `coords` (data type: {coords.index.dtype}) must be an integer."
                )

            # validate nodes: ndim
            if len(coords.index) > 0:
                if ndim is not None:
                    if ndim != len(coords.columns):
                        raise ValueError(
                            f"`ndim` ({ndim}) does not match the number of columns in `coords` ({len(coords.columns)})."
                        )
                else:
                    ndim = len(coords.columns)

            # validate nodes: n_nodes
            if n_nodes is not None:
                if n_nodes < len(coords.index):
                    raise ValueError(
                        f"`n_nodes` ({n_nodes}) must be greater or equal than `coords` length ({len(coords.index)})."
                    )
            else:
                n_nodes = len(coords.index)

        # initialize nodes
        if n_nodes is None:
            n_nodes = self._ALLOC_MIN
        else:
            n_nodes = max(n_nodes, self._ALLOC_MIN)

        self._init_node_buffers(n_nodes)

        if ndim is not None:
            self._coords = np.empty((n_nodes, ndim), dtype=np.float32)
        else:
            self._coords = None

        if coords is not None:
            assert self._coords is not None
            self.add_nodes(indices=coords.index, coords=coords)

        # validate edges
        edges = np.asarray(edges)
        if len(edges) > 0:
            if edges.ndim != 2:
                raise ValueError(
                    f"`edges` ({edges.ndim} dimensions) must have 2 dimensions."
                )
            if edges.shape[1] != 2:
                raise ValueError(
                    f"`edges` (shape: {edges.shape}) must have shape E x 2."
                )

        # validate edges: n_edges
        if n_edges is not None:
            if n_edges < len(edges):
                raise ValueError(
                    f"`n_edges` ({n_edges}) must be greater or equal than `edges` length ({len(edges)}."
                )
        else:
            n_edges = len(edges)

        n_edges = max(n_edges, self._ALLOC_MIN)

        # initialize edges
        self._init_edge_buffers(n_edges)
        if len(edges) > 0:
            if coords is None:
                self.add_nodes(indices=np.unique(edges))
            self.add_edges(edges)

    def _init_node_buffers(self, n_nodes: int) -> None:
        self._empty_nodes: List[int] = list(reversed(range(n_nodes)))
        self._node2edges = np.full(
            n_nodes, fill_value=_EDGE_EMPTY_PTR, dtype=int
        )
        self._world2buffer = typed.Dict.empty(types.int64, types.int64)
        self._buffer2world = np.full(
            n_nodes, fill_value=_NODE_EMPTY_PTR, dtype=int
        )

    def _init_edge_buffers(self, n_edges: int) -> None:
        # if condition just to be safe, in case MIN_ALLOC is set to 0
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

    @property
    def ndim(self) -> int:
        return self._coords.shape[1]

    @property
    def n_nodes(self) -> int:
        """Number of nodes in use."""
        return self.n_allocated_nodes - self.n_empty_nodes

    @property
    def n_allocated_nodes(self) -> int:
        """Number of total allocated nodes."""
        return len(self._buffer2world)

    @property
    def n_empty_nodes(self) -> int:
        """Number of nodes allocated but not used."""
        return len(self._empty_nodes)

    def get_nodes(self) -> np.ndarray:
        """Indices of graph nodes."""
        return self._buffer2world[self._buffer2world != _NODE_EMPTY_PTR]

    def get_coordinates(
        self, node_indices: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """Coordinates of the given nodes.

        If none is provided it returns the coordinates of every node.
        """
        if self._coords is None:
            raise ValueError(
                "`get_coordinates` is only available for spatial graphs."
            )

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

        if self._coords is not None:
            self._coords.resize(
                (size, self._coords.shape[1]), refcheck=False
            )  # zero-filled

        self._node2edges = np.append(
            self._node2edges,
            np.full(size_diff, fill_value=_EDGE_EMPTY_PTR, dtype=int),
        )
        self._buffer2world = np.append(
            self._buffer2world,
            np.full(size_diff, fill_value=_NODE_EMPTY_PTR, dtype=int),
        )
        self._empty_nodes = list(reversed(range(prev_size, size)))

    def get_next_valid_indices(self, count: int) -> ArrayLike:
        if count <= 0:
            raise ValueError(
                f"`count` must be a positive integer. Found {count}"
            )

        next_indices = self._buffer2world.max() + 1
        return np.arange(next_indices, next_indices + count)

    def add_nodes(
        self,
        *,
        indices: Optional[ArrayLike] = None,
        coords: Optional[ArrayLike] = None,
        count: Optional[int] = None,
    ) -> ArrayLike:
        """
        Add nodes to graph, at least one of the arguments must be supplied.
        `count` cannot be supplied with other arguments.

        Parameters
        ----------
        index : int
            Node index.
        coords : np.ndarray
            Node coordinates, optional for non-spatial graph.
        count : int:
            Number of nodes to be added.

        Returns
        -------
        ArrayLike
            Added nodes indices.
        """
        if count is not None and (indices is not None or coords is not None):
            raise ValueError(
                "`count` cannot be supplied with `indices` and `coords`."
            )

        if count is None and indices is None and coords is None:
            raise ValueError(
                "One of `indices`, `coords` or `count` must be supplied."
            )

        if coords is not None:
            coords = np.atleast_2d(coords)

        if indices is None:
            if count is None:
                count = coords.shape[0]
            indices = self.get_next_valid_indices(count)

        indices = np.atleast_1d(indices)
        if indices.ndim > 1:
            raise ValueError(
                f"`indices` must be 1-dimensional. Found {indices.ndim}."
            )

        if (self._coords is None) != (coords is None):
            if coords is None:
                raise ValueError(
                    "`coords` must be provided for spatial graphs."
                )
            else:
                raise ValueError(
                    "`coords` cannot be provided for non-spatial graphs."
                )

        if _contains_keys(self._world2buffer, indices):
            raise ValueError(
                f"One of the nodes {indices} are already present in the buffer."
            )

        if self.n_empty_nodes < len(indices):
            self._realloc_nodes_buffers(
                self._get_alloc_size(self.n_nodes + len(indices))
            )

        # flipping since _empty_nodes is a stack
        buffer_indices = np.flip(self._empty_nodes[-len(indices) :])

        if coords is not None:
            if indices.shape[0] != coords.shape[0]:
                raise ValueError(
                    f"`indices` and `coords` must be equal. Found {len(indices)} and {len(coords)}."
                )

            self._coords[buffer_indices] = coords

        _update_world2buffer(self._world2buffer, indices, buffer_indices)
        self._empty_nodes = self._empty_nodes[: -len(indices)]
        self._buffer2world[buffer_indices] = indices

        return indices

    def _get_alloc_size(self, size: int) -> int:
        return int(max(size * self._ALLOC_MULTIPLIER, self._ALLOC_MIN))

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
        self._remove_incident_edges(buffer_index)
        self._buffer2world[buffer_index] = _NODE_EMPTY_PTR
        self._empty_nodes.append(buffer_index)

    @abstractmethod
    def _remove_incident_edges(self, node_buffer_index: int) -> None:
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
            raise ValueError(
                f"Tried to realloc to current buffer size ({self.n_allocated_edges})."
            )

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
            return self.get_nodes()

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
            self._realloc_edges_buffers(
                self._get_alloc_size(self.n_edges + len(edges))
            )

        self._add_edges(self._map_world2buffer(edges))

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
        if mode.lower() == 'coords' and self._coords is None:
            raise ValueError(
                "`coords` mode only available for spatial graphs."
            )

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
            ndim = self._coords.shape[1]
            edges_data = [
                self._coords[e].reshape(-1, 2, ndim)
                if len(e) > 0
                else np.empty((0, 2, ndim))
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
    def get_edges(
        self, nodes: Optional[ArrayLike] = None, mode: str = 'indices'
    ) -> Union[List[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def get_edges_buffers(
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

    def __len__(self) -> int:
        """Number of nodes in use."""
        return self.n_nodes

    def initialized_buffer_mask(self) -> np.ndarray:
        """Compute mask of nodes that have already been initialized.

        Returns
        -------
        np.ndarray
            Boolean array of valid node, it has the same length the buffer.
        """
        return self._buffer2world != _NODE_EMPTY_PTR

    @property
    def coords_buffer(self) -> np.ndarray:
        """Returns the actual coordinates buffer. It's not a copy."""
        if self._coords is None:
            raise ValueError(
                "graph does not have a `coords` attribute. "
                "It is not a spatial graph."
            )
        return self._coords

    def is_spatial(self) -> bool:
        """True if self is a spatial graph (has coordinates attribute)."""
        return self._coords is not None

    @staticmethod
    def from_networkx(graph: nx.Graph) -> BaseGraph:
        """Loads a Directed or Undirected napari-graph from a NetworkX graph.

        Parameters
        ----------
        graph : nx.Graph
            The NetworkX graph to be converted.
        """
        from napari_graph.directed_graph import DirectedGraph
        from napari_graph.undirected_graph import UndirectedGraph

        nodes = np.array(list(graph.nodes()))
        if not np.issubdtype(nodes.dtype, np.integer) or nodes.ndim > 1:
            graph_int_nodes = nx.convert_node_labels_to_integers(
                graph, label_attribute='_node_id'
            )
            warnings.warn(
                'Node IDs must be integers. They have been converted '
                'automatically.'
            )
        else:
            graph_int_nodes = graph
        coords_dict = nx.get_node_attributes(graph_int_nodes, "pos")
        if len(coords_dict) > 0:
            coords_df = pd.DataFrame.from_dict(coords_dict, orient="index")
        else:
            coords_df = None

        edges = graph_int_nodes.edges
        if len(edges) > 0:
            edges = np.atleast_2d(edges)

        return (
            DirectedGraph(edges, coords_df)
            if graph_int_nodes.is_directed()
            else UndirectedGraph(edges, coords_df)
        )

    def to_networkx(self) -> nx.Graph:
        """Convert it self into NetworkX graph.

        Parameters
        ----------
        graph : BaseGraph
            napari-graph Graph

        Returns
        -------
        nx.Graph
            An equivalent NetworkX graph.
        """
        from napari_graph.directed_graph import DirectedGraph

        if isinstance(self, DirectedGraph):
            out_graph = nx.DiGraph()
        else:
            out_graph = nx.Graph()

        if self.is_spatial():
            for node_id, pos in zip(self.get_nodes(), self.get_coordinates()):
                # note: some nx functions are unhappy with arrays in node
                # attributes because you can't compare arrays with ==.
                # So one day we might want to cast to tuple.
                out_graph.add_node(node_id, pos=pos)
        else:
            out_graph.add_nodes_from(self.get_nodes())

        edges = self.get_edges()
        if isinstance(edges, list) and len(edges) > 0:
            edges = np.concatenate(edges, axis=0)

        edges_as_tuples = list(map(tuple, edges))
        out_graph.add_edges_from(edges_as_tuples)

        return out_graph

    def subgraph_edges(
        self,
        node_indices: ArrayLike,
        is_buffer_domain: bool = False,
    ) -> ArrayLike:
        """Returns edges (node pair) where both nodes are presents.

        Parameters
        ----------
        nodes_indices : np.ndarray
            Subset of nodes used for selection.
        is_buffer_domain : bool
            When true `node_indices` and returned edges are on buffer domain.

        Returns
        -------
        np.ndarray
            (N x 2) array of nodes indices, where N is the number of valid edges from the induced subgraph.
        """
        _, edges = self.get_edges_buffers(is_buffer_domain)

        if is_buffer_domain:
            mask = np.zeros(self._buffer2world.shape[0], dtype=bool)
            mask[node_indices] = True
            subgraph_edges = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]

        else:
            mask = np.isin(edges, node_indices).all(axis=1)
            assert mask.shape[0] == edges.shape[0]
            subgraph_edges = edges[mask]

        return subgraph_edges
