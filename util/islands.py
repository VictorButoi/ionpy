# ionpy imports
import torch
from ionpy.util.validation import validate_arguments_init

# misc imports
from typing import List


@validate_arguments_init
def dfs(
    i: int,
    j: int,
    array: torch.Tensor,
    visited: torch.Tensor
    ) -> torch.Tensor:
    """
    Returns an individual component
    : param array: the array that we are searching
    : param row: the current row index we are checking
    : param col: the current col index we are checking
    : param visited: a torch tensor tracking which 
    : return: A map corresponding to a single connected component.
    """
    num_rows, num_cols = array.shape

    # Check if this is NOT a components pixel.
    if (i < 0) or (j < 0) or (i >= num_rows) or (j >= num_cols) or visited[i,j] or (not array[i,j]):
        return []
    
    # If we passed the if then we ARE a part of the component, thus.
    visited[i,j] = True
    component = [(i,j)]

    # Directions we can go in
    dirs = [(-1,  0), ( 1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Visit neighbors
    for dx, dy in dirs: 
        component += dfs(i + dx, j + dy, array, visited)

    return torch.Tensor(component).long()



@validate_arguments_init
def get_connected_components(
    label: torch.Tensor
) -> List[torch.Tensor]:
    """
    Returns a list of images with the same shape as label, where each image 
    corresponds to a binary mask for each connected commponent in label.
    :param label: A binary 2D torch tensor.
    :return: A list of binary 2D torch tensors, one for each connected component.
    """
    rows, cols = label.shape 
    visited = torch.zeros(label.shape, dtype=bool)
    connected_components = []

    for i in range(rows):
        for j in range(cols):
            if not visited[i, j] and label[i, j]:
                component_idxs = dfs(i, j, label, visited)
                comp_image = torch.zeros_like(label)
                comp_image[component_idxs[:, 0], component_idxs[:, 1]] = 1
                # Make sure the island is a bool tensor so it can be used for indexing.
                connected_components.append(comp_image.bool())
    
    return connected_components

