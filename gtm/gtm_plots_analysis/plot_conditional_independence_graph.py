import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph_conditional_independencies(
    abs_array,
    gene_names,
    min_abs_mean=0.1,
    storage=None,
    pos_list=None,
    pos_tuple_list=None,
    k=1.5,
    seed_graph=42,
    show_plot=True,
):
    edge_array = np.tril(abs_array, -1)
    edge_array[edge_array < min_abs_mean] = 0

    print("There are", sum(sum(edge_array > 0)), "connections above ", min_abs_mean)

    G = nx.from_numpy_array(edge_array * 10)  # **2 * 10
    keys = list(range(len(gene_names)))
    values = gene_names
    mapping = {k: v for k, v in zip(keys, values)}
    G = nx.relabel_nodes(G, mapping)
    G.edges(data=True)

    widths = nx.get_edge_attributes(G, "weight")
    nodelist = G.nodes()

    values = np.round(list(widths.values()), 2)
    edge_labels = {k: v for k, v in zip(widths.keys(), values)}

    # Old
    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(G, weight="weight", seed=seed_graph, k=k)

    if pos_list is not None:
        for i in range(len(pos_list)):
            pos[pos_list[i]] = pos_tuple_list[i]

    nx.draw_networkx_nodes(
        G, pos, nodelist=nodelist, node_size=5000, node_color="black", alpha=0.7
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=widths.keys(),
        width=list(widths.values()),
        edge_color="lightblue",
        alpha=0.6,
    )
    nx.draw_networkx_labels(
        G, pos=pos, labels=dict(zip(gene_names, nodelist)), font_color="white"
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.box(False)
    if storage is not None:
        plt.savefig(storage)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
