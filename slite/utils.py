import torch


def get_most_free_gpu(gpu_list):
    free_memory_per_gpu = {} 
    for gpu_idx, gpu in enumerate(gpu_list):
        try:
            properties = torch.cuda.get_device_properties(gpu_idx)
            free_memory_per_gpu[gpu] = properties.total_memory - torch.cuda.memory_allocated(gpu_idx)
        except Exception as e:
            print(f"Error processing GPU {gpu}: {e}")
    if not free_memory_per_gpu:
        raise ValueError("No valid GPUs found in the provided list.")
    # Return the gpu with the most free memory
    return max(free_memory_per_gpu, key=free_memory_per_gpu.get)
    