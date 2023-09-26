# ionpy imports
import torch
from ionpy.util.validation import validate_arguments_init

# misc imports
from typing import List
from scipy.ndimage.measurements import label


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
def slow_get_connected_components(
    array: torch.Tensor,
    visited: torch.Tensor = None,
) -> List[torch.Tensor]:
    """
    Returns a list of images with the same shape as label, where each image 
    corresponds to a binary mask for each connected commponent in label.
    :param label: A binary 2D torch tensor.
    :return: A list of binary 2D torch tensors, one for each connected component.
    """
    rows, cols = array.shape 
    
    # If you haven't explored anything yet, then make a visited tensor. 
    if visited is None:
        visited = torch.zeros(array.shape, dtype=bool)

    # Iterate through the image and get the connected components.
    connected_components = []
    for i in range(rows):
        for j in range(cols):
            if not visited[i, j] and array[i, j]:
                component_idxs = dfs(i, j, array, visited)

                # Create an image corresponding to THIS connected commponent.
                comp_image = torch.zeros_like(array)
                comp_image[component_idxs[:, 0], component_idxs[:, 1]] = 1

                # Make sure the island is a bool tensor so it can be used for indexing.
                connected_components.append(comp_image.bool())
    
    return connected_components


@validate_arguments_init
def get_connected_components(
    binary_tensor: torch.Tensor
) -> List[torch.Tensor]:
    """
    Returns a list of images with the same shape as label, where each image 
    corresponds to a binary mask for each connected commponent in label.
    :param binary_tensor: A binary 2D torch tensor.
    :return: A list of binary 2D torch tensors, one for each connected component.
    """

    # Convert torch tensor to numpy array
    binary_array = binary_tensor.cpu().numpy()

    # Label the connected components
    labeled_array, num_features = label(binary_array)

    if num_features > 0:
        return [torch.from_numpy(labeled_array == i).bool() for i in range(1, num_features + 1)]
    else:
        return []

