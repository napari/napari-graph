
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field
from collections import namedtuple

import numpy as np
import pandas as pd
import networkx as nx
import numba as nb


Edge = namedtuple('Edge', ['node', 'weight'])


@dataclass
class Node:
    index: int
    coordinates: np.ndarray
    features: Dict = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)


class UndirectedGraph:
    def __init__(self) -> None:
        self._nodes: Dict[int, Node] = {}

    def from_dataframe(
        self,
        nodes_df: pd.DataFrame,
        coordinates_columns: List[str],
        edges: Union[np.ndarray, List[Tuple[int, int]]],
    ) -> None:

        coordinates = nodes_df[coordinates_columns].values
        indices = nodes_df.index
        features = nodes_df.drop(coordinates_columns, axis=1).to_dict('records')
        if len(features) == 0:
            features = [{}] * len(indices)

        self._nodes = {
            idx: Node(idx, coords, feats)
            for idx, coords, feats in zip(indices, coordinates, features)
        }

        for i, j in edges:
            node_i = self._nodes[i]
            node_j = self._nodes[j]
            node_i.edges.append(Edge(node_j, 0))
            node_j.edges.append(Edge(node_i, 0))

    def to_networkx(self) -> nx.Graph:

        graph = nx.Graph()

        for index, node in self._nodes.items():
            graph.add_node(
                index,
                coordinates=node.coordinates,
                **node.features,
            )

            for edge in node.edges:
                if edge.node.index < index:  # avoid duplicating edges
                    graph.add_edge(edge.node.index, index, weight=edge.weight)

        return graph


class UndirectedGraphBuffer:
    def __init__(self, size: int, ndim: int):
        self._active = np.ones(size, dtype=bool)
        self._coords = np.zeros((size, ndim), dtype=np.float32)
        self._feats: Dict[str, np.ndarray] = {}

    def from_dataframe(
        self,
        nodes_df: pd.DataFrame,
        coordinates_columns: List[str],
        edges: Union[np.ndarray, List[Tuple[int, int]]],
    ) -> None:

        if len(nodes_df) != self._coords.shape[0] or len(coordinates_columns) != self._coords.shape[1]:
            print('Reallocating array')
            self._coords = nodes_df[coordinates_columns].values.astype(np.float32, copy=True)
            self._active = np.ones(len(self._coords), dtype=bool)
        else:
            self._coords[...] = nodes_df[coordinates_columns].values

        self._feats = nodes_df.drop(coordinates_columns, axis=1).to_dict('series')
        self._feats = {k: v.values for k, v in self._feats.items()}

        self._edges = self._create_edges(len(self._active), np.asarray(edges))

    @staticmethod
    @nb.njit()
    def _create_edges(n_nodes: int, edges: np.ndarray) -> List[List[int]]:
        _edges = [nb.typed.List.empty_list(nb.int64) for _ in range(n_nodes)]
        for i, j in edges:
            _edges[i].append(j)
            _edges[j].append(i)
        return _edges


EDGE_SIZE = 3
LL_EDGE_STEP = EDGE_SIZE - 1


@nb.njit()
def _add_edge(buffer: np.ndarray, node2edge: np.ndarray, empty_idx: int, src: int, dst: int) -> int:
    if empty_idx == -1:
        return -2  # buffer is full
    elif empty_idx < 0:
        return -3  # invalid index
    
    next_edge = node2edge[src]
    node2edge[src] = empty_idx

    buffer_index = empty_idx * EDGE_SIZE
    next_empty = buffer[buffer_index + LL_EDGE_STEP]

    buffer[buffer_index] = src
    buffer[buffer_index + 1] = dst
    buffer[buffer_index + LL_EDGE_STEP] = next_edge

    return next_empty


@nb.njit()
def _init_edges(buffer: np.ndarray, edges: np.ndarray, node2edge: np.ndarray) -> bool:
    size = edges.shape[0]
    empty_idx = 0
    for i in range(size):
        if empty_idx == -2:
            return False
        empty_idx = _add_edge(buffer, node2edge, empty_idx, edges[i, 0], edges[i, 1])
        if empty_idx == -2:
            return False
        empty_idx = _add_edge(buffer, node2edge, empty_idx, edges[i, 1], edges[i, 0])
    return True


class UndirectedGraphBufferLLArray(UndirectedGraphBuffer):
    """Undirected graph with buffer and linked lists from array for edges.
        Inspired by: https://github.com/mastodon-sc/mastodon/blob/master/doc/trackmate-graph.pdf

        NOTE: when implementing the function to remove edges both of the pair should be remove too
    """
    def _create_edges(self, n_nodes: int, edges: Union[np.ndarray, List[Tuple[int, int]]]) -> np.ndarray:
        size = len(edges) * 2 * EDGE_SIZE   # 2 (because the edges are duplicated to make it undirected)
        self._edges_buffer = np.full(size, fill_value=-1, dtype=int)
        self._edges_buffer[LL_EDGE_STEP::EDGE_SIZE] = np.arange(1, len(edges) * 2 + 1)
        self._edges = np.full(n_nodes, fill_value=-1, dtype=int)
        self._weights = np.zeros(size // EDGE_SIZE)
        assert _init_edges(self._edges_buffer, edges, self._edges)
        return self._edges
