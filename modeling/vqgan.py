from typing import Optional
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.distributed as dist

import importlib
from typing import Mapping, Optional

def set_train(*models):
    for model in models:
        model.train()

def set_eval(*models):
    for model in models:
        model.eval()

def check_rank_zero():
    if not dist.is_available() or not dist.is_initialized():
        return True
    else:
        if dist.get_rank() == 0:
            return True
        else:
            return False

def seed_everything(seed: Optional[int] = None):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def torch_pca(data):
    """
    Args:
        data: (N, D)
    Returns:
        eigenvectors: (D, D)
    """
    # center the data
    data_centered = data - data.mean(dim=0, keepdim=True)

    # compute the covariance matrix
    cov_matrix = (data_centered.t() @ data_centered) / (data_centered.shape[0] - 1)

    # compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # sort the eigenvalues and eigenvectors
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvectors

def sinkhorn(cost, n_iters: int = 3, epsilon: float = 1, is_distributed: bool = False):
    """
    Sinkhorn algorithm.
    Args:
        cost (Tensor): shape with (B, K)
    """
    Q = torch.exp(- cost * epsilon).t() # (K, B)
    B = Q.size(1)
    K = Q.size(0)

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= (sum_Q + 1e-8)

    for _ in range(n_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= (sum_of_rows + 1e-8)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
        Q /= B
    
    Q *= B # the columns must sum to 1 so that Q is an assignment
    return Q.t() # (B, K)

def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{(i+1):06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 1,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def to_numpy(data):
    return data.detach().cpu().numpy()
def clamp_log(x, eps: float = 1e-6):
    return torch.log(torch.clamp(x, eps))

def feature_to_index(x: torch.Tensor, quantize_levels: torch.Tensor):
    assert x.size(-1) == len(quantize_levels)
    x = x.clamp(0, 1 - 1e-6)
    index = (x * quantize_levels).floor().long()
    return index

def index_to_id(index, quantize_levels, tokens_per_data: int = 1):
    """
    Args
        index (tensor): (BS, NG)
    """
    if isinstance(quantize_levels, int):
        quantize_levels = torch.tensor([quantize_levels] * tokens_per_data,
                                        device=index.device)
    base_vector = torch.cat([
        torch.tensor([1.0], device=quantize_levels.device), 
        torch.cumprod(quantize_levels, dim=0)[:-1]
    ])
    id = torch.einsum("ij,j->i", index.float(), base_vector)
    return id.long()

def reconstruct_sample(device, loader, vae_decode, vq_infer):
    with torch.no_grad():
        # visualize the results
        sample = next(iter(loader))["feature"]
        sample = sample[:8].to(device)
        # GT images
        rec_GT = vae_decode(sample)
        # quantized images
        rec_ours = vae_decode(vq_infer(sample))
        # combine
        combined = torch.cat([rec_GT, rec_ours], dim=0)
        combined = torchvision.utils.make_grid(combined, nrow=8, padding=2)
        combined = combined * 0.5 + 0.5
        combined = combined.clamp(0, 1).permute(1, 2, 0).detach().cpu()
    return combined

def allocate_index(num_sample: int, num_code: int, ndim: int = 2):
    assert num_code ** ndim >= num_sample, "num_code ** ndim must be greater than num_sample"
    candidate_index = torch.arange(num_code)
    candidate_index = torch.cartesian_prod(*[candidate_index for _ in range(ndim)])
    # shuffle the candidate_index
    candidate_index = candidate_index[torch.randperm(candidate_index.size(0))]
    candidate_index = candidate_index[:num_sample]
    if ndim == 1:
        candidate_index = candidate_index.unsqueeze(1)
    return candidate_index

def to_ddp_model(rank: int, *models):
    output_list = []
    if dist.is_available() and dist.is_initialized():
        for model in models:
            if len(list(model.parameters())) == 0:
                output_list.append(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                    module=model, 
                    device_ids=[rank],
                    output_device=rank,
                    find_unused_parameters=True
                )
                output_list.append(model)
        print(f"Setup DDP quantizer on rank {rank}")
    else:
        for model in models:
            output_list.append(model)
        print("Setup quantizer without DDP")
    if len(output_list) == 1:
        return output_list[0]
    else:
        return tuple(output_list)
    
def initiate_from_config(config: Mapping):
    assert "target" in config, f"Expected key `target` to initialize!"
    module, cls = config["target"].rsplit(".", 1)
    meta_class = getattr(importlib.import_module(module, package=None), cls)
    return meta_class(**config.get("params", dict()))

def initiate_from_config_recursively(config: Mapping):
    assert "target" in config, f"Expected key `target` to initialize!"
    update_config = {"target": config["target"], "params": {}}
    for k, v in config["params"].items():
        if isinstance(v, Mapping) and "target" in v:
            sub_instance = initiate_from_config_recursively(v)
            update_config["params"][k] = sub_instance
        else:
            update_config["params"][k] = v
    return initiate_from_config(update_config)

if __name__ == "__main__":
    index = torch.cartesian_prod(torch.arange(30), torch.arange(20), torch.arange(100))
    ids = index_to_id(index, num_code=torch.tensor([30, 20, 100]))