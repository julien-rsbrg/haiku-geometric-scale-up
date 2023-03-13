# Common packages
import numpy as np
import scipy
from typing import Any, Callable, Dict, List, Optional, Tuple

def set_quadmesh(xmin:int, xmax:int, ymin:int, ymax:int, nx:int, ny:int)->Tuple[np.array]:
    '''
    Create a mesh of quadrangles

    Args:
        xmin, xmax, ymin, ymax: boundaries of the mesh
        nx,ny: nbre of points per axis
    
    Returns:
        node_coords: node coordinates
        p_elem2nodes: indices in elem2nodes between which there is an element
        elem2nodes: indices in node_coords to take to form an element
    '''

    spacedim = 3
    nnodes = (nx + 1) * (ny + 1)
    node_coords = np.empty((nnodes, spacedim), dtype=np.float64)
    nodes_per_elem = 4
    nelems = nx * ny
    p_elem2nodes = np.empty((nelems + 1,), dtype=np.int64)
    p_elem2nodes[0] = 0
    for i in range(0, nelems):
        p_elem2nodes[i + 1] = p_elem2nodes[i] + nodes_per_elem
    elem2nodes = np.empty((nelems * nodes_per_elem,), dtype=np.int64)

    # elements
    k = 0
    for j in range(0, ny):
        for i in range(0, nx):
            elem2nodes[k + 0] = j * (nx + 1) + i
            elem2nodes[k + 1] = j * (nx + 1) + i + 1
            elem2nodes[k + 2] = (j + 1) * (nx + 1) + i + 1
            elem2nodes[k + 3] = (j + 1) * (nx + 1) + i
            k += nodes_per_elem

    k = 0
    for j in range(0, ny + 1):
        yy = ymin + (j * (ymax - ymin) / ny)
        for i in range(0, nx + 1):
            xx = xmin + (i * (xmax - xmin) / nx)
            node_coords[k, :] = xx, yy, 0.0
            k += 1

    return node_coords, p_elem2nodes, elem2nodes



def add_elem_to_mesh(node_coords:np.array, p_elem2nodes:np.array, elem2nodes:np.array, new_elemid_nodes:np.array)->Tuple[np.array]:
    elem2nodes = np.concatenate(
        (elem2nodes, new_elemid_nodes), axis=-1)
    p_elem2nodes = np.concatenate(
        (p_elem2nodes, [p_elem2nodes[-1]+new_elemid_nodes.shape[-1]]), axis=-1)
    return node_coords, p_elem2nodes, elem2nodes



def get_id_n_nearest_nodes(node_coords:np.array, new_node_coords:np.array, n:int=2)->np.array:
    assert node_coords.shape[0] >= n
    assert new_node_coords.shape[-1] == new_node_coords.shape[-1], "not the same dimension of nodes"
    Mdist = np.ones((node_coords.shape[0], 1))@new_node_coords-node_coords
    Mdist = Mdist**2@np.ones((node_coords.shape[-1], 1))
    return np.argpartition(Mdist[:, 0], n)[:n]


def add_node_to_mesh(node_coords:np.array, p_elem2nodes:np.array, elem2nodes:np.array, new_node_coords:np.array, join_to_new_elem:bool=True)->Tuple[np.array]:
    '''
    link new node at _node_coords to the 2 closest nodes of node_coords
    '''
    n_nodes, space_dim = node_coords.shape
    assert new_node_coords.shape[-1] == space_dim, (
        "wrong dimension of new_node: ", new_node_coords.shape, 'while space_dim =', space_dim)

    if len(new_node_coords.shape) == 1:
        new_node_coords = np.expand_dims(new_node_coords, axis=0)

    node_coords = np.concatenate((node_coords, new_node_coords), axis=0)

    if join_to_new_elem:
        id_2nearest_nodes = get_id_n_nearest_nodes(node_coords, new_node_coords)  # , is_inside_element, elem_included_in
        new_elem_node = np.concatenate((id_2nearest_nodes, np.array([n_nodes])), axis=-1)
        node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(
            node_coords, p_elem2nodes, elem2nodes, new_elem_node)

    return node_coords, p_elem2nodes, elem2nodes



def set_simple_mesh(l_bound:List[int]=[0.0, 1.0, 0.0, 1.0])->Tuple[np.array]:
    '''
    set the mesh constituted of one rectangle with points [l_bound[0],l_bound[2]] and [l_bound[1],l_bound[3]]  
    
    Args:
        l_bound = [xmin, xmax, ymin, ymax]
    
    Returns:
        node_coords, p_elem2nodes, elem2nodes, boundary2nodes: np.array
            forming a mesh of one rectangle
    '''
    xmin, xmax, ymin, ymax = l_bound
    nelemsx, nelemsy = 1, 1
    node_coords, p_elem2nodes, elem2nodes = set_quadmesh(
        xmin, xmax, ymin, ymax, nelemsx, nelemsy)

    # twice the same node in it at the beginning and the end => easier to plot and use
    # always in anti trigonometric order
    boundary2nodes = np.array([0, 2, 3, 1, 0])
    return node_coords, p_elem2nodes, elem2nodes, boundary2nodes


def get_global_ids_nodes(node_coords:np.array, nodes_to_check:np.array)-> np.array:
    global_ids = np.zeros(nodes_to_check.shape[0], dtype=int)
    for local_id in range(nodes_to_check.shape[0]):
        flag_matrix = np.zeros(node_coords.shape)
        for i in range(nodes_to_check.shape[-1]):
            flag_matrix[:, i] = np.where(
                node_coords[:, i] == nodes_to_check[local_id, i], 1, 0)
        # WARNING: next line can output an error if the same node is twice in node_coords
        global_ids[local_id] = int(np.arange(flag_matrix.shape[0])[
            np.prod(flag_matrix, axis=1) == 1])

    return global_ids


def subdivide(node_coords:np.array, p_elem2nodes:np.array, elem2nodes:np.array, id_p_elem2nodes:int, n_subdivision:int=4)->Tuple[np.array]:
    '''
    Subdivide the given element into n_subdivision**2 elements

    Args:
     - node_coords, p_elem2nodes, elem2nodes
     - id_p_elem2nodes : the id in p_elem2nodes of the nodes constituting the element to subdivide
     - n_subdivision : number of elements added on one axis

    Returns:
     - node_coords, p_elem2nodes, elem2nodes with the element composed of elemid nodes subdivided
    '''
    assert id_p_elem2nodes < p_elem2nodes.shape[0] - \
        1, "want to subdivide a non-existing elem"
    if n_subdivision == 1:
        return node_coords, p_elem2nodes, elem2nodes

    # find elemid in p_elem2nodes
    # delete element
    elem = elem2nodes[p_elem2nodes[id_p_elem2nodes]
        :p_elem2nodes[id_p_elem2nodes+1]]
    n_nodes_elem = elem.shape[0]
    elem2nodes = np.concatenate(
        [elem2nodes[:p_elem2nodes[id_p_elem2nodes]], elem2nodes[p_elem2nodes[id_p_elem2nodes+1]:]], axis=-1)
    p_elem2nodes = np.concatenate(
        [p_elem2nodes[:id_p_elem2nodes], p_elem2nodes[id_p_elem2nodes+1:]-n_nodes_elem], axis=-1)

    # --- add n_subdivision**2 elements within previous elemid ---
    vector_edge = (node_coords[elem[1], :] -
                   node_coords[elem[0], :])/n_subdivision

    # remark: could put directly in node_coords_one_row
    new_nodes = np.zeros((n_subdivision-1, node_coords.shape[-1]))
    for i in range(0, n_subdivision-1):
        new_nodes[i, :] = node_coords[elem[0], :]+(i+1)*vector_edge
    node_coords_one_row = np.concatenate(
        [[node_coords[elem[0], :]], new_nodes, [node_coords[elem[1], :]]], axis=0)

    elem_nodes_coords = np.zeros(((n_subdivision+1)**2, node_coords.shape[-1]))
    u_move = (node_coords[elem[-1], :] -
              node_coords[elem[0], :])/n_subdivision
    u_move = np.expand_dims(u_move, axis=0)
    matrix_move = np.ones((node_coords_one_row.shape[0], 1))@u_move
    for i in range(0, n_subdivision+1):
        elem_nodes_coords[(n_subdivision+1)*i:(n_subdivision+1) *
                          (i+1), :] = node_coords_one_row+i*matrix_move


    # add nodes to mesh
    for node_cand_to_add in elem_nodes_coords:
        node_coords, p_elem2nodes, elem2nodes = add_node_to_mesh(node_coords, p_elem2nodes,
                                                                          elem2nodes, node_cand_to_add, join_to_new_elem=False)
    # remove duplicates 
    node_coords = np.unique(node_coords,axis=0)

    # add elements to mesh
    global_ids = get_global_ids_nodes(
        node_coords, elem_nodes_coords)

    for j in range(n_subdivision):
        for i in range(n_subdivision):
            # j handles column movement and i row movement
            # local ids
            new_elemid_nodes = np.array(
                [j+i*(n_subdivision+1), (j+1)+i*(n_subdivision+1),
                 (j+1)+(i+1)*(n_subdivision+1), j+(i+1)*(n_subdivision+1)])
            # global ids
            new_elemid_nodes = global_ids[new_elemid_nodes]
            node_coords, p_elem2nodes, elem2nodes = add_elem_to_mesh(
                node_coords, p_elem2nodes, elem2nodes, new_elemid_nodes)

    return node_coords, p_elem2nodes, elem2nodes



def find_nodeid_from_nodecoord(node_coords:np.array, node_coord_to_find:np.array, tolerance:float=1e-6)->Tuple:
    node_coord_to_find = np.expand_dims(node_coord_to_find, axis=0)
    Mdist = np.ones((node_coords.shape[0], 1))@node_coord_to_find-node_coords
    Mdist = np.sqrt(Mdist**2@np.ones((node_coords.shape[-1], 1)))
    where_coord_correct = np.where(Mdist <= tolerance, 1, 0)
    indexes = np.arange(node_coords.shape[0])
    nodeid = indexes[where_coord_correct[:, 0] == 1]
    assert nodeid.shape[0] <= 1, (' there is several times the same node in node_coords.\nActual result is : ' +
                                  str(nodeid.shape[0]) +
                                  "\nhere is node_coords:\n", node_coords, "\nhere is node_coord_to_find\n:", node_coord_to_find)
    if nodeid.shape[0] == 0:
        return False, None
    return True, nodeid[0]


def update_boundary(node_coords:np.array, boundary2nodes:np.array, n_subdivision:int)->np.array:
    N0_boundary = boundary2nodes.shape[0]
    for i in range(N0_boundary-1):
        i = i*n_subdivision
        u_step = (node_coords[boundary2nodes[i+1]] -
                  node_coords[boundary2nodes[i]])/n_subdivision
        l_new_nodeids = []
        for j in range(1, n_subdivision):
            new_node_for_boundary_coord = node_coords[boundary2nodes[i]]+j*u_step
            _, new_nodeid = find_nodeid_from_nodecoord(
                node_coords,  new_node_for_boundary_coord)
            l_new_nodeids.append(new_nodeid)
        boundary2nodes = np.concatenate(
            [boundary2nodes[:i+1], l_new_nodeids, boundary2nodes[i+1:]], axis=-1)
    return boundary2nodes



def subdivide_all(node_coords:np.array, p_elem2nodes:np.array, elem2nodes:np.array, boundary2nodes:np.array, n_subdivision:int=4)->Tuple[np.array]:
    '''
    Sequentially subdivide all the elements of the given mesh
     
    Args:
     - node_coords, p_elem2nodes, elem2nodes, boundary2nodes
     - n_subdivision : number of elements added on one axis
    returns:
     - node_coords, p_elem2nodes, elem2nodes, boundary2nodes with the element composed of elemid nodes subdivided
    '''
    for _ in range(p_elem2nodes.shape[0]-1):
        node_coords, p_elem2nodes, elem2nodes = subdivide(
            node_coords, p_elem2nodes, elem2nodes, 0, n_subdivision=n_subdivision)

    boundary2nodes = update_boundary(
        node_coords, boundary2nodes, n_subdivision)

    return node_coords, p_elem2nodes, elem2nodes, boundary2nodes