import warnings

import networkx as nx
import numpy as np
import pandas as pd

from napari_graph.base_graph import BaseGraph
from napari_graph.directed_graph import DirectedGraph
from napari_graph.undirected_graph import UndirectedGraph


def from_networkx(graph: nx.Graph) -> BaseGraph:
    """Convert a NetworkX graph into a napari-graph UndirectedGraph or DirectedGraph.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph to be converted.

    Returns
    -------
    BaseGraph
        An equivalent napari-graph UndirectedGraph or DirectedGraph.
    """

    coords_dict = nx.get_node_attributes(graph, "pos")
    if len(coords_dict) > 0:
        coords_df = pd.DataFrame.from_dict(coords_dict, orient="index")
    else:
        coords_df = None

    edges = np.asarray(graph.edges)

    if edges.ndim > 1 and edges.shape[1] > 2:
        warnings.warn("Edges weights are not supported yet and were ignored.")
        edges = edges[:, :2]

    if graph.is_directed():
        out_graph = DirectedGraph(edges, coords_df)
    else:
        out_graph = UndirectedGraph(edges, coords_df)

    return out_graph


def to_networkx(graph: BaseGraph) -> nx.Graph:
    """
    Convert a napari-graph UndirectedGraph or DirectedGraph into NetworkX graph.

    Parameters
    ----------
    graph : BaseGraph
        napari-graph Graph

    Returns
    -------
    nx.Graph
        An equivalent NetworkX graph.
    """

    if isinstance(graph, DirectedGraph):
        out_graph = nx.DiGraph()
    else:
        out_graph = nx.Graph()

    if graph.is_spatial():
        for node_id, pos in zip(graph.get_nodes(), graph.get_coordinates()):
            out_graph.add_node(node_id, pos=pos)
    else:
        out_graph.add_nodes_from(graph.get_nodes())

    edges = graph.get_edges()
    if isinstance(edges, list) and len(edges) > 0:
        edges = np.concatenate(edges, axis=0)

    edges_as_tuples = list(map(tuple, edges))
    out_graph.add_edges_from(edges_as_tuples)

    return out_graph
