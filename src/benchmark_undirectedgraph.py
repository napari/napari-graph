
from typing import Tuple
import timeit

import pandas as pd
import numpy as np
import networkx as nx

from undirectedgraph import UndirectedGraph, UndirectedGraphBuffer, UndirectedGraphBufferLLArray


def random_graph(size: int, sparsity: float) -> Tuple[pd.DataFrame, np.ndarray]:

    adj_matrix = np.random.uniform(size=(size, size)) < sparsity
    np.fill_diagonal(adj_matrix, 0)

    edges = np.stack(adj_matrix.nonzero()).T
    nodes_df = pd.DataFrame(np.random.randn(size, 3), columns=["z", "y", "x"])

    return nodes_df, edges


def from_networkx(nodes_df, coordinates_columns, edges) -> nx.Graph:
    coordinates = nodes_df[coordinates_columns].values
    indices = nodes_df.index
    features = nodes_df.drop(coordinates_columns, axis=1).to_dict('records')

    if len(features) == 0:
        features = [{}] * len(indices)
    
    graph = nx.Graph()

    for idx, coords, feats in zip(indices, coordinates, features):
        graph.add_node(idx, coordinates=coords, **feats)

    for i, j in edges:
        graph.add_edge(i, j, weight=1.0)

    return graph


def main() -> None:
    size = 5000
    sparsity = 0.005

    setup = f'df, edges = random_graph({size}, {sparsity}); graph = UndirectedGraph()'
    ugraph = 'graph.from_dataframe(df, ["z", "y", "x"], edges)'
    bgraph = 'bgraph = UndirectedGraphBuffer(len(df), 3); bgraph.from_dataframe(df, ["z", "y", "x"], edges)'
    bllgraph = 'bllgraph = UndirectedGraphBufferLLArray(len(df), 3); bllgraph.from_dataframe(df, ["z", "y", "x"], edges)'
    
    timer = timeit.timeit(
        bllgraph, setup=setup, globals=globals(), number=100,
    )
    print('bll: from_dataframe', timer)

    timer = timeit.timeit(
        bgraph, setup=setup, globals=globals(), number=100,
    )
    print('b: from_dataframe', timer)

    timer = timeit.timeit(
        ugraph, setup=setup, globals=globals(), number=100,
    )
    print('from_dataframe', timer)

    timer = timeit.timeit(
        'from_networkx(df, ["z", "y", "x"], edges)', setup=setup, globals=globals(), number=100,
    )
    print('from_networkx', timer)

    timer = timeit.timeit(
        'graph.to_networkx()', setup=f'{setup};{ugraph}', globals=globals(), number=100,
    )
    print('ours_to_networkx', timer)
 

if __name__ == '__main__':
    main()
