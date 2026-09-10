import os

from global_utils import seed_everything as _seed_everything
from global_variables import CUDA_VISIBLE_DEVICES, DEVICE


def get_absolute_path(relative_path:str):
    basePath = os.path.dirname(os.path.abspath(__file__))
    return basePath + relative_path

def print_cuda_info():
    import torch

    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    if torch.cuda.is_available():
        print("CUDA is available")
    print("CUDA_VISIBLE_DEVICES: ", CUDA_VISIBLE_DEVICES)

def write_pickle(data,filename,path):
    import pickle

    with open(path+filename + ".pkl", "wb") as file:
        pickle.dump(data, file)

def load_pickle(filename,path):
    import pickle

    with open(path+filename + ".pkl", "rb") as file:
        my_list = pickle.load(file)
    return my_list

def clear_unused_gpu_memory():
    import torch
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
def print_number_of_parameters(model):
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

def get_device():
    return DEVICE

def seed_everything(seed: int):
    _seed_everything(seed)

def init_wandb(cfg):
    import wandb
    import os

    if cfg.wandb.use_wandb:
        os.environ["WANDB_MODE"] = "dryrun"
        default_name = cfg.model.name 
        wandb.init(
            project=cfg.wandb.project_name + "_" + cfg.benchmark.name if cfg.wandb.project_name != "None" else None,
            name=  cfg.wandb.name + default_name   if cfg.wandb.name else default_name,
            entity = "thomas-klassert",
            group=cfg.wandb.group_name if cfg.wandb.group_name != "None" else None,
            config=dict(cfg), 
            mode="online")
