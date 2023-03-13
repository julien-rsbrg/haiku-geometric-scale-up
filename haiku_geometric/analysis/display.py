# common librairies
import networkx as nx
import matplotlib.pyplot as plt
import jax.numpy as jnp

# this librairy 
from haiku_geometric.datasets.base import DataGraphTuple,GraphDataset

def convert_datagraph_to_networkx_graph(graph: DataGraphTuple) -> nx.Graph:
    nodes, edges, receivers, senders, _, _,  _, position, _, _ = graph
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(graph.n_node[0]):
            if position is not None:
                nx_graph.add_node(n, node_feature=nodes[n],pos=position[n])
            else:
                nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(graph.n_edge[0]):
            nx_graph.add_edge(
                int(senders[e]), int(receivers[e]), edge_feature=edges[e])
    return nx_graph

def draw_datagraph_structure(graph: DataGraphTuple) -> None:
    print(graph.nodes.shape)
    nx_graph = convert_datagraph_to_networkx_graph(graph)
    pos = nx.get_node_attributes(nx_graph,'pos')
    #pos =  nx.spring_layout(nx_graph)
    print(pos)

    # edge weight labels
    edge_labels = nx.get_edge_attributes(nx_graph, "edge_feature")
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels)

    nx.draw(
        nx_graph, pos=pos, with_labels=True, node_size=500, font_color='yellow')

    plt.show()

def describe_datagraph(graph:DataGraphTuple,in_detail:bool=False)->None:
    print("--describe datagraph--")
    print(f'Number of nodes: {graph.n_node[0]}')
    print(f'Number of edges: {graph.n_edge[0]}')
    print(f"Nodes features size: {graph.nodes.shape[-1]}")
    if graph.train_mask is not None:
        print(f"Number of training nodes: {jnp.count_nonzero(graph.train_mask)}")

    print(f'Average node degree: {graph.n_edge[0] / graph.n_node[0]:.2f}')
    
    if in_detail:
        print("Nodes report")
        for i in range(graph.n_node[0]):
            print(f"node n°{i}: feature {graph.nodes[i]}")

        print("Edges report:")
        for i in range(graph.n_edge[0]):
            sender,receiver = graph.senders[i],graph.receivers[i]
            print(f"edge n°{i}: {sender} --> {receiver} with feature {graph.edges[i]}")
    
    print()   
        

def describe_dataset(dataset:GraphDataset)->None:
    print(f'Dataset: {dataset}:')
    print('======================')
    print("Number of graphs :", len(dataset.data))

    graph = dataset.data[0]  # Get the first graph object.
    print("for first graph:")
    print('===============================================================================')
    describe_datagraph(graph=graph)

    