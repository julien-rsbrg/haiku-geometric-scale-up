# librairies
import jax.numpy as jnp
import numpy as np
import jraph
import networkx as nx
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple

# this librairy 
import haiku_geometric.datasets.mesh.mesh_generator as mesh_generator
from haiku_geometric.datasets.base import DataGraphTuple,GraphDataset
from haiku_geometric.datasets.utils import check_row_exists_in_matrix


class MeshDataset(GraphDataset):
    r"""A simple mesh converted into a graph
    
    **Attributes:**
        - **data** List[DataGraphTuple]: List of length 1 containing a single graph.
    
    Stats:
        .. list-table::
            :widths: 10 10 10 10 10 10
            :header-rows: 1

            * - #nodes
              - #edges
              - #features
              - nodes features size
              - edge features size
            * - (n_subdivision+1)**2
              - (n_subdivision-1)**2*4+(n_subdivision-1)*4*3+4*2
              - 2
              - 1
              - 0
    """

    def _convert_elem_mesh2undirected_graph(self,node_coords:np.array, p_elem2nodes:np.array, elem2nodes:np.array) -> DataGraphTuple:
        node_features = jnp.array(node_coords)

        edges = None
        for idstart in range(p_elem2nodes.shape[0]-1):
            nodes_elem = elem2nodes[p_elem2nodes[idstart]:p_elem2nodes[idstart+1]]

            for j_node in range(nodes_elem.shape[0]):
                to_add_edges = jnp.array(
                    [[nodes_elem[j_node], nodes_elem[(j_node+1) % nodes_elem.shape[0]]]])
                if edges is not None:
                    # no duplicate
                    is_new_edge = np.full((to_add_edges.shape[0],),True)
                    for i in range(to_add_edges.shape[0]):
                        is_direct_in = check_row_exists_in_matrix(edges,to_add_edges[i,:])
                        is_indirect_in = check_row_exists_in_matrix(edges,to_add_edges[i,::-1])
                        is_new_edge[i] = not(is_direct_in or is_indirect_in)
                    edges = jnp.concatenate([edges,to_add_edges[is_new_edge]],axis=0)
                else:
                    edges = to_add_edges


        # undirected graph
        edges = jnp.concatenate([edges, edges[:, ::-1]], axis=0)
        senders = edges[:, 0]
        receivers = edges[:, 1]

        # optional : add edge features
        edge_features = jnp.zeros((edges.shape[0],))
        for i in range(edges.shape[0]):
            dist = jnp.linalg.norm(
                node_coords[edges[i, 0], :]-node_coords[edges[i, 1], :])
            edge_features = edge_features.at[i].set(dist)

        # We then save the number of nodes and the number of edges.
        # This information is used to make running GNNs over multiple graphs
        # in a GraphsTuple possible.
        n_node = jnp.array([node_coords.shape[0]])
        n_edge = jnp.array([edges.shape[0]])

        graph = DataGraphTuple(
            nodes=node_features,
            edges=edge_features,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            globals=None,
            position=node_features,
            y=None,
            train_mask=np.full((node_features.shape[0],),True)
        )
        return graph

    def _build_mesh_graph(self,n_subdivision:int=2)->DataGraphTuple:
        node_coords, p_elem2nodes, elem2nodes, boundary2nodes = mesh_generator.set_simple_mesh()
        # remove z coord
        node_coords = node_coords[:, :2]
        node_coords, p_elem2nodes, elem2nodes, boundary2nodes = mesh_generator.subdivide_all(
            node_coords, p_elem2nodes, elem2nodes, boundary2nodes, n_subdivision=n_subdivision)
        graph = self._convert_elem_mesh2undirected_graph(
            node_coords, p_elem2nodes, elem2nodes)
        return graph
        
    def __init__(self,n_subdivision:int):
        """"""
        graph = self._build_mesh_graph(n_subdivision)

        super().__init__([graph])
        