
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field
from collections import namedtuple

import numpy as np
import pandas as pd
import networkx as nx


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