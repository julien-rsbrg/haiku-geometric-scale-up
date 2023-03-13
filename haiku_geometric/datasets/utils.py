import os
import os.path as osp
import ssl
import sys
import urllib
from typing import Optional
from zipfile import ZipFile
import pickle 
import numpy as np


# function adapted from:
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/download.html#download_url
def download_url(url: str, folder: str,
                 filename: Optional[str] = None):

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]
    
    path = osp.join(folder, filename)
    if osp.exists(path):
        print(f'Using existing file {filename}', file=sys.stderr)
        return path
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def extract_zip(zip_file, folder):
    with ZipFile(zip_file, 'r') as zObject:
        zObject.extractall(path=folder)

        
def pickle_save_object(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def check_row_exists_in_matrix(matrix, row, tol=1e-5):
        '''
        Check that all the rows of matrix are distinct from row 

        Args:
        - matrix: np.array of shape (n,dim)
        - row: np.array of shape (dim,)
        Returns:
        - bool of whether the row is in the matrix or not 
        '''
        flag_matrix = np.zeros(matrix.shape)
        for i in range(row.shape[-1]):
            flag_matrix[:, i] = np.where(
                abs(matrix[:, i] - row[i]) < tol, 1, 0)
            
        is_same_row = np.prod(flag_matrix, axis=1)
        N_identic_rows = np.sum(is_same_row)
        return N_identic_rows >= 1