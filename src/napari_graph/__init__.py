from napari_graph.directed_graph import DirectedGraph
from napari_graph.undirected_graph import UndirectedGraph

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"
