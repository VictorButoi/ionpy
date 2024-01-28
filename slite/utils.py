import torch


def get_most_free_gpu(gpu_list):
    free_memory_per_gpu = {} 
    for gpu in gpu_list:
        free_memory_per_gpu[gpu] = torch.cuda.get_device_properties(gpu).total_memory - torch.cuda.memory_allocated(gpu)
    # Return the gpu with the most free memory
    return max(free_memory_per_gpu, key=free_memory_per_gpu.get)
    