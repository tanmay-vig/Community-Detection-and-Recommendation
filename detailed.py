import networkx as nx
import community.community_louvain as community_louvain  # Import the Louvain method
import matplotlib.pyplot as plt

# Step 1: Load the combined edge list into a graph
def load_combined_data(file_path):
    G = nx.read_edgelist(file_path, nodetype=int)
    return G

# Step 2: Apply Louvain Method for community detection
def detect_communities_louvain(G):
    partition = community_louvain.best_partition(G)
    return partition

# Step 3: Visualize the detected communities with distinct colors and highlight influential nodes
def visualize_communities(G, partition):
    pos = nx.spring_layout(G , seed=42)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    
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

# Step 4: Analyze node centrality
def analyze_centrality(G):
    centrality = nx.degree_centrality(G)
    return centrality

# Step 5: Recommend content based on community detection
def recommend_within_community(user_id, G, partition, content_dict):
    # Get the community the user belongs to
    user_community = partition[user_id]
    
    # Find all users in the same community
    users_in_community = [user for user, community in partition.items() if community == user_community]
    
    # Get the most interacted content in this community
    content_popularity = {}
    
    for user in users_in_community:
        if user in content_dict:
            for content in content_dict[user]:
                if content not in content_popularity:
                    content_popularity[content] = 0
                content_popularity[content] += 1
    
    # Recommend the most popular content within the community
    recommended_content = sorted(content_popularity, key=content_popularity.get, reverse=True)[:5]
    
    return recommended_content

# Step 6: Recommend friends based on centrality within the same community
def recommend_friends(user_id, G, partition):
    user_community = partition[user_id]
    
    # Get users in the same community
    users_in_community = [user for user in partition if partition[user] == user_community]
    
    # Calculate degree centrality for users in the same community
    centrality = nx.degree_centrality(G)
    
    # Exclude the current user and users they are already connected with
    friends = set(G.neighbors(user_id))
    non_friends = [user for user in users_in_community if user != user_id and user not in friends]
    
    # Recommend users with the highest degree centrality
    recommended_friends = sorted(non_friends, key=lambda x: centrality[x], reverse=True)[:5]
    
    return recommended_friends

# Step 7: Link prediction for future friend recommendation (Adamic-Adar Index)
def link_prediction(G, user_id):
    # Using Adamic-Adar index as a link prediction technique
    preds = nx.adamic_adar_index(G, [(user_id, n) for n in G.nodes() if n != user_id])
    
    # Sort predictions based on the highest scores
    recommended_links = sorted(preds, key=lambda x: x[2], reverse=True)[:5]
    
    return [pred[1] for pred in recommended_links]

if __name__ == "__main__":
    # Path to your extracted file
    file_path = 'facebook_combined.txt'  # Adjust this path if needed
    
    # Load the graph
    G = load_combined_data(file_path)
    
    # Print basic information about the graph
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    
    # Detect communities
    partition = detect_communities_louvain(G)
    
    # Visualize the communities and influential nodes
    visualize_communities(G, partition)
    
    # Analyze and print centrality
    centrality = analyze_centrality(G)
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:10]
    print("Top 10 Influential Nodes by Centrality:", top_nodes)
    
    # Example content interactions (user -> content)
    content_dict = {
        1: ["Content_A", "Content_B"],
        2: ["Content_D"],
        3: ["Content_B", "Content_C"],
        4: ["Content_C"],
        5: ["Content_A" , "Content_E"],
        6: ["Content_C,", "Content_D"],
        7: ["Content_D"],
        8: ["Content_F"],
        9: ["Content_A" , "Content_G"],
        10: ["Content_G" , "Content_H"],
    }
    
    # Example recommendations for a user (change user_id accordingly)
    user_id = 1
    recommended_content = recommend_within_community(user_id, G, partition, content_dict)
    print(f"Recommended Content for User {user_id}:", recommended_content)
    
    # Friend recommendations for the same user
    recommended_friends = recommend_friends(user_id, G, partition)
    print(f"Recommended Friends for User {user_id}:", recommended_friends)
    
    # Predict future friends using link prediction
    predicted_friends = link_prediction(G, user_id)
    print(f"Predicted Future Friends for User {user_id}:", predicted_friends)
