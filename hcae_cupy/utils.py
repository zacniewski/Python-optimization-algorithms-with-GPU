# Visualization of the NN created from NDM with NetworkX package Documentation:
# https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html#sphx-glr-auto-examples
# -drawing-plot-weighted-graph-py And: https://networkx.org/documentation/stable/auto_examples/drawing/plot_directed
# .html Stack link #1: https://stackoverflow.com/questions/49319104/drawing-network-with-nodes-and-edges-in-python
# Stack link #2: https://stackoverflow.com/questions/58511546/in-python-is-there-a-way-to-use-networkx-to-display-a
# -neural-network-in-the-sta


import matplotlib.pyplot as plt
import networkx as nx


def draw_neural_network_from_ndm():
    graph_of_ndm = nx.Graph()

    graph_of_ndm.add_edge("a", "b", weight=0.6)
    graph_of_ndm.add_edge("a", "c", weight=0.2)
    graph_of_ndm.add_edge("c", "d", weight=0.1)
    graph_of_ndm.add_edge("c", "e", weight=0.7)
    graph_of_ndm.add_edge("c", "f", weight=0.9)
    graph_of_ndm.add_edge("a", "d", weight=0.3)
    
    elarge = [(u, v) for (u, v, d) in graph_of_ndm.edges(data=True) if d["weight"] > 0.5]
    esmall = [(u, v) for (u, v, d) in graph_of_ndm.edges(data=True) if d["weight"] <= 0.5]
    
    pos = nx.spring_layout(graph_of_ndm, seed=7)  # positions for all nodes - seed for reproducibility
    
    # nodes
    nx.draw_networkx_nodes(graph_of_ndm, pos, node_size=700)
    
    # edges
    nx.draw_networkx_edges(graph_of_ndm, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph_of_ndm, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    
    # node labels
    nx.draw_networkx_labels(graph_of_ndm, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(graph_of_ndm, "weight")
    nx.draw_networkx_edge_labels(graph_of_ndm, pos, edge_labels)
    
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
