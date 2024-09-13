import networkx as nx
import community.community_louvain as community_louvain  # Import the Louvain method
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the dataset (Facebook social network)
def load_facebook_data(file_path):
    G = nx.read_edgelist(file_path, nodetype=int)
    return G

# Apply Louvain Method for community detection
def detect_communities_louvain(G):
    partition = community_louvain.best_partition(G)
    return partition

# Visualize the detected communities with distinct colors and highlight influential nodes
def visualize_communities(G, partition):
    pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    
    # Create a colormap for the communities
    communities = list(set(partition.values()))
    num_communities = len(communities)
    cmap = plt.get_cmap('tab20', num_communities)  # Use tab20 colormap for up to 20 different colors
    
    # Draw nodes for each community with a unique color
    for community in communities:
        nodes = [node for node in partition.keys() if partition[node] == community]
        color = cmap(community / num_communities)  # Get a color from the colormap
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=30, node_color=[color])
    
    # Identify and highlight influential nodes
    degrees = dict(G.degree())
    influential_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]  # Top 10 nodes by degree
    influential_colors = 'orange'
    nx.draw_networkx_nodes(G, pos, nodelist=influential_nodes, node_size=100, node_color=influential_colors, edgecolors='black')

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Add legend for communities
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / num_communities), markersize=10, label=f'Community {i}') for i in range(num_communities)]
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=influential_colors, markersize=15, label='Influential Nodes'))
    plt.legend(handles=handles, title='Communities and Influential Nodes')
    
    plt.show()

if __name__ == "__main__":
    # Path to your uploaded file
    file_path = 'facebook_combined.txt'  # Adjust this path if needed
    
    # Load the graph
    G = load_facebook_data(file_path)
    
    # Detect communities
    partition = detect_communities_louvain(G)
    
    # Visualize the communities and influential nodes
    visualize_communities(G, partition)
