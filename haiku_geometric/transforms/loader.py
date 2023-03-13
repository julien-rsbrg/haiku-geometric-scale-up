
# common librairies
import jax.numpy as jnp
import numpy as np
import jax.tree_util as tree
import jraph
import networkx as nx
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple

# this library
from haiku_geometric.datasets.base import DataGraphTuple, GraphDataset

######### General #########

def get_converter_to_id_loc(id_glob):
    converter_to_loc = -1*np.ones(np.max(id_glob)+1,dtype=np.int32)
    for i in range(len(id_glob)):
        converter_to_loc[id_glob[i]]=i
    return converter_to_loc



# TODO: check np => jnp
# TODO: what about n-hop?
def sample_neighbours(graph:DataGraphTuple, num_neigbours:List[int], batch_size:int,input_nodes:jnp.ndarray, return_index:bool=False) -> List:
    '''
    Sample subgraphs from a main graph using starting from input nodes to take 1-hop neighbours
    Only made for node classification

    Args:
        graph: graph to sample in jraph.GraphsTuple format
        num_neigbours: [min number of neighbotrain_maskrs to take, max number of neighbours to take] both included
        batch_size: the number of distinct input_nodes' elements to *try* to have in each subgraph. 
                    (nodes from input_nodes in a subgraph could be taken by another subgraph as neighbours 
                     of its chosen nodes from input_nodes)
        input_nodes: jnp.ndarray of flags of nodes driving the sampling (ex: input_nodes=train_mask)
        return_index: bool assuring that the indices of the nodes taken from the original graph by each subgraph is returned

    Returns:
        subgraphs_dataset: GraphDataset made of the subgraphs sampled from graph
        indices_taken_nodes_in_original_graph: List[np.array] useful to reconstruct the original 
                          graph from sampled subgraphs (not returned if return_index=False)
    '''
    assert len(num_neigbours) == 2

    subgraphs_info = []
    input_nodes_indices = jnp.arange(input_nodes.shape[0])[input_nodes]
    available_nodes = input_nodes_indices.copy()
    while len(available_nodes)>0:

        n_nodes_to_take = batch_size if len(available_nodes)-batch_size>=0 else len(available_nodes)
        id2avn_taken_nodes = np.random.choice(len(available_nodes),n_nodes_to_take,replace=False)
        # taken_nodes is a list of the ids of the nodes taken in graph.nodes
        id2n_taken_nodes = [available_nodes[i] for i in id2avn_taken_nodes]
        available_nodes = [id2n for id2avn, id2n in enumerate(available_nodes) if id2avn not in id2avn_taken_nodes]


        all_id2n_taken_nodes = id2n_taken_nodes.copy()
        all_id2edg_taken_sending_neighbours = []

        # add 1-hop neighbours with edges for each taken node
        for node in id2n_taken_nodes:
            n_neighbours_to_take = np.random.randint(num_neigbours[0],num_neigbours[1]+1)
            # nodes where the central node is as receiver as it is the prediction of the central node that matters
            id2edg_sending_neighbours = np.where(graph.receivers==node)[0]
            
            n_neighbours_to_take = min(n_neighbours_to_take,id2edg_sending_neighbours.shape[0])
            id2sneig_taken_neighbours = np.random.choice(id2edg_sending_neighbours.shape[0],n_neighbours_to_take,replace=False)
            id2edg_taken_sending_neighbours = id2edg_sending_neighbours[id2sneig_taken_neighbours]

            all_id2edg_taken_sending_neighbours+=id2edg_taken_sending_neighbours.tolist()
            all_id2n_taken_nodes += graph.senders[id2edg_taken_sending_neighbours].tolist()

        all_id2edg_taken_sending_neighbours = np.array(all_id2edg_taken_sending_neighbours)

        all_id2n_taken_nodes = np.unique(all_id2n_taken_nodes)

        # be sure np.unique has not shuffle the indices
        augmented = np.concatenate([id2n_taken_nodes,all_id2n_taken_nodes],axis=0)
        all_id2n_taken_nodes, indices = np.unique(augmented,return_index=True)
        train_mask = np.full(all_id2n_taken_nodes.shape, False)
        id2mask_to_true = np.where(indices<len(id2n_taken_nodes))[0]
        train_mask[id2mask_to_true] = True
        train_mask = jnp.array(train_mask)

        nodes = graph.nodes[all_id2n_taken_nodes,:]

        converter_to_id_loc = get_converter_to_id_loc(all_id2n_taken_nodes)
        glob_receivers = graph.receivers[all_id2edg_taken_sending_neighbours]
        loc_receivers = converter_to_id_loc[glob_receivers]
        assert np.all(loc_receivers>=0)

        glob_senders = graph.senders[all_id2edg_taken_sending_neighbours]
        loc_senders = converter_to_id_loc[glob_senders]
        assert np.all(loc_senders>=0)

        y = graph.y[all_id2n_taken_nodes,:] if graph.y is not None else None 

        new_graph = DataGraphTuple(
            nodes=nodes,
            edges=graph.edges[all_id2edg_taken_sending_neighbours],
            receivers=loc_receivers,
            senders=loc_senders,
            globals=graph.globals,
            n_node=jnp.array([len(all_id2n_taken_nodes)]),
            n_edge=jnp.array([len(all_id2edg_taken_sending_neighbours)]),
            position=nodes, 
            y = y,
            train_mask=train_mask
        )
        subgraphs_info.append({"subgraph":new_graph,"id2n_taken_nodes":np.array(all_id2n_taken_nodes)})

    subgraphs_dataset = GraphDataset([subgraphs_info[i]["subgraph"] for i in range(len(subgraphs_info))])
    if return_index:
        indices_taken_nodes_in_original_graph = [subgraphs_info[i]["id2n_taken_nodes"] for i in range(len(subgraphs_info))]
        return subgraphs_dataset,indices_taken_nodes_in_original_graph
    
    return subgraphs_dataset

# TODO: account for train_mask and y
def reconstruct_graph_from_subgraphs(subgraphs_dataset:GraphDataset, indices_taken_nodes_in_original_graph:List[np.array])-> DataGraphTuple:
    '''
    Reconstruct the full graph from subgraphs (these subgraphs contains some information about how to reconstruct the graph)
    
    Args:
      subgraphs_dataset: GraphDataset
      indices_taken_nodes_in_original_graph

    Returns: 
      reconstructed_graph: full graph reconstructed from subgraphs.
        train_mask and y should be meant for node classification
    '''

    def _get_converter_after_merging(old_indices,new_indices):
        n_old_indices = old_indices.shape[0]
        converter_after_merging = -np.ones(new_indices.shape[0],dtype=np.int32)
        id2newInd_new_elems = []
        for i in range(new_indices.shape[0]):
            for j in range(n_old_indices):
                if new_indices[i] == old_indices[j] and converter_after_merging[i]<0:
                    converter_after_merging[i] = j
            if converter_after_merging[i]<0:
                converter_after_merging[i] = n_old_indices+len(id2newInd_new_elems)
                id2newInd_new_elems.append(i)

        return converter_after_merging,np.array(id2newInd_new_elems)

    all_nodes = np.array([])
    all_edges = np.array([])
    all_senders = np.array([])
    all_receivers = np.array([])

    id2n_glob_node_added = np.array([],dtype=np.int32)


    for i in range(len(subgraphs_dataset.data)):
        subgraph = subgraphs_dataset.data[i]

        # identify which nodes are added
        id2n_converter_to_glob = indices_taken_nodes_in_original_graph[i]
        id2n_glob_receivers = id2n_converter_to_glob[subgraph.receivers]
        id2n_glob_senders = id2n_converter_to_glob[subgraph.senders]
        id2n_glob_node_to_add = np.concatenate([id2n_glob_receivers,id2n_glob_senders],axis=0)
        id2n_glob_node_to_add = np.unique(id2n_glob_node_to_add)

        id2n_converter_after_merging,id2globToAdd_new_nodes = _get_converter_after_merging(id2n_glob_node_added,id2n_glob_node_to_add)

        # add the new nodes added to the reconstructed graph 
        if id2globToAdd_new_nodes.size:
            id2n_converter_to_loc = get_converter_to_id_loc(id2n_converter_to_glob)
            id2n_loc_new_nodes = id2n_converter_to_loc[id2n_glob_node_to_add[id2globToAdd_new_nodes]]
            all_nodes = np.concatenate([all_nodes,subgraph.nodes[id2n_loc_new_nodes]],axis=0) if all_nodes.size else subgraph.nodes[id2n_loc_new_nodes]

            # update the record of the node added
            id2n_glob_node_added = np.concatenate([id2n_glob_node_added,id2n_glob_node_to_add[id2globToAdd_new_nodes]],axis=0) if id2n_glob_node_added.size else id2n_glob_node_to_add
            
        # add all edges, senders and receivers to the reconstructed graph (redundancy corrected after)
        id2n_merged_receivers = id2n_converter_after_merging[subgraph.receivers]
        id2n_merged_senders =  id2n_converter_after_merging[subgraph.senders]

        all_edges = np.concatenate([all_edges,subgraph.edges],axis=0) if all_edges.size else subgraph.edges
        all_senders = np.concatenate([all_senders,id2n_merged_senders],axis=0) if all_senders.size else id2n_merged_senders
        all_receivers = np.concatenate([all_receivers,id2n_merged_receivers],axis=0) if all_receivers.size else id2n_merged_receivers


    # remove the redundancy within the edges, senders and receivers  
    aug_all_senders = np.expand_dims(all_senders,axis=-1)
    aug_all_receivers = np.expand_dims(all_receivers,axis=-1)
    sender_receiver = np.concatenate([aug_all_senders,aug_all_receivers],axis=-1)

    _,unique_indices = np.unique(sender_receiver,axis=0,return_index=True)
    all_edges=all_edges[unique_indices]
    all_senders = all_senders[unique_indices]
    all_receivers = all_receivers[unique_indices]

    n_node = np.array([all_nodes.shape[0]])
    n_edge = np.array([all_edges.shape[0]])
    # Information assumed to be useless
    global_context = jnp.array([[]])
    
    reconstructed_graph = DataGraphTuple(
      nodes=all_nodes,
      edges=all_edges,
      senders=all_senders,
      receivers=all_receivers,
      n_node=n_node,
      n_edge=n_edge,
      globals=global_context,
      position= all_nodes,
      y = None,
      train_mask=None
      )

    return reconstructed_graph
