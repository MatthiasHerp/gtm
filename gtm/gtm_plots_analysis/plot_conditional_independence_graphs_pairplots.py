import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_graph_conditional_independencies_with_pairplots(abs_array, gene_names, data, metric, min_abs_mean=0.1, storage=None, lim_axis=[-18, 18],
                                                         pos_list=None, pos_tuple_list=None, k=1.5, seed_graph=42):
    """
    Plots a network graph of conditional independencies and overlays small scatter plots on edges.

    :param abs_array: Adjacency matrix (absolute conditional correlations)
    :param gene_names: List of gene names
    :param data: NumPy array where each column corresponds to a gene
    :param metric: A 3D NumPy array containing pairwise metrics for gene interactions
    :param min_abs_mean: Minimum threshold for edges
    :param storage: Path to save the figure (optional)
    """
    # Extract lower triangular part of adjacency matrix
    edge_array = np.tril(abs_array, -1)
    edge_array[edge_array < min_abs_mean] = 0  # Filter weak edges

    # Count significant edges
    num_edges = np.count_nonzero(edge_array)
    print(f"There are {num_edges} connections above {min_abs_mean}")
    
    if num_edges == 0:
        print("No significant edges found. Exiting function.")
        return

    # Create graph and relabel nodes
    G = nx.from_numpy_array(edge_array * 10)  # Scale edge weights
    mapping = {i: gene for i, gene in enumerate(gene_names)}
    G = nx.relabel_nodes(G, mapping)

    # Get edge attributes
    widths = nx.get_edge_attributes(G, "weight")
    edge_labels = {edge: round(weight, 2) for edge, weight in widths.items()}

    # Initialize figure and layout
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, weight="weight",  seed=seed_graph,
                           k=k
                           )
    if pos_list is not None:
        for i in range(len(pos_list)):
            pos[pos_list[i]] = pos_tuple_list[i]
            

    
    # Find the isolated nodes (nodes with degree 0)
    #isolated_nodes = [node for node, degree in G.degree() if degree == 0]

    # Manually place the isolated nodes on the side (e.g., at x = 2 and vary y)
    #side_x = 2
    #y_place=[-2,2]
    #for i, node in enumerate(isolated_nodes):
    #    pos[node] = [side_x, y_place[i]]  # Place isolated nodes at x = 2, with different y values


    # Draw the graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=5000, node_color="black", alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=widths.keys(), width=list(widths.values()), edge_color="lightblue", alpha=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white")
    # No edge labels needed as we have plots
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=10)

    # Overlay scatter plots on edges
    for (gene1, gene2), weight in widths.items():
        x_mid, y_mid = (pos[gene1][0] + pos[gene2][0]) / 2, (pos[gene1][1] + pos[gene2][1]) / 2
        
        # Apply a small random shift to avoid overlap
        x_shift = -0.12 #np.random.uniform(-0.05, 0.05)
        y_shift = -0.12 #np.random.uniform(-0.05, 0.05)

        try:
            # Create inset axis
            ax_inset = inset_axes(ax, width="100%", height="100%", loc="center",
                                  bbox_to_anchor=(x_mid + x_shift, 
                                                  y_mid + y_shift, 
                                                  0.24, 
                                                  0.24),
                                  bbox_transform=ax.transData, borderpad=0)

            # Scatter plot of gene expressions
            norm = plt.Normalize(-1, 1)
            if data.shape[0] > 0 and metric.shape[0] > 0:
                ax_inset.hexbin(data[:, gene1], data[:, gene2],
                                C=metric[:, gene1, gene2],
                                gridsize=60, cmap="icefire", norm=norm, reduce_C_function=np.mean)
                ax_inset.set_xlim(lim_axis)
                ax_inset.set_ylim(lim_axis)
                
                # Make the inset square
                ax_inset.set_aspect("equal") 
                
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                ax_inset.patch.set_linewidth(1)  # Ensure frame visibility
                ax_inset.patch.set_edgecolor("black")  # Explicitly set frame color
                ax_inset.set_frame_on(True)  # Make sure the frame is enabled
                
                ax_inset.set_zorder(10)  # Bring insets to the front

            else:
                print(f"Skipping ({gene1}, {gene2}): Empty data or metric.")

        except Exception as e:
            print(f"Skipping scatter plot for ({gene1}, {gene2}) due to error: {e}")

    #plt.box(False)
    plt.axis("off")

    if storage:
        plt.savefig(storage, bbox_inches="tight")
    
    plt.draw()
    plt.show()
    
            