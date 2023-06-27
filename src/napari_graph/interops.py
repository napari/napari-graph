from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from napari_graph.base_graph import BaseGraph
from napari_graph.undirected_graph import UndirectedGraph

COMPATIBLE_CLASSES = {
    BaseGraph: lambda x: x,
    nx.Graph: BaseGraph.from_networkx,
    pd.DataFrame: lambda x: UndirectedGraph(coords=x),
    np.ndarray: lambda x: UndirectedGraph(coords=x),
}


def to_napari_graph(graph: Any) -> BaseGraph:
    """Generic function to convert "any" graph to a napari-graph.

    Supported formats:
        - napari-graph (itself)
        - NetworkX

    Parameters
    ----------
    graph : Any
        Any kind of graph to a napari-graph. See supported formats above.

    Returns
    -------
    BaseGraph
        A napari-graph graph.
    """
    for cls, conversion_func in COMPATIBLE_CLASSES.items():
        if isinstance(graph, cls):
            return conversion_func(graph)

    raise NotImplementedError(
        f"Conversion from {type(graph)} to napari-graph does not exist."
    )
