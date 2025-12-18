from typing import Mapping, Text, Tuple
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from accelerate.utils.operations import gather

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

def find_indices(x, y, use_sinkhorn: bool = True):
    d = torch.cdist(x, y, p=2)
    if use_sinkhorn:
        d - (d - d.mean()) / (d.std() + 1e-8)
        d = d - d.min()
        q = sinkhorn(d, n_iters=5, epsilon=10)
        indices = torch.argmax(q, dim=-1)
    else:
        indices = torch.argmin(d, dim=-1)
    return indices

class OptVQ(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = False,
        use_shared_linear: bool = True,
        use_sinkhorn: bool = True,
        num_group: int = 1,
    ):
        super(OptVQ, self).__init__()
        self.commitment_cost = commitment_cost
        self.use_sinkhorn = use_sinkhorn
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.num_group = num_group

        if self.num_group == 1:
            self.find_indices = partial(find_indices, use_sinkhorn=self.use_sinkhorn)
        else:
            self.find_indices = torch.vmap(
                partial(find_indices, use_sinkhorn=self.use_sinkhorn),
                in_dims=0, out_dims=0, chunk_size=2
            )

        # self.find_indices = torch.vmap(
        #     partial(find_indices, use_sinkhorn=self.use_sinkhorn),
        #     in_dims=0, out_dims=0, chunk_size=2
        # )

        self.embedding = nn.Parameter(
            torch.randn(self.num_group, self.codebook_size, self.token_size)
        )
        self.embedding.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

        # normalization
        self.use_l2_norm = use_l2_norm
        if self.use_l2_norm:
            self.normalize = partial(torch.nn.functional.normalize, dim=-1, p=2)
        else:
            self.normalize = lambda x: x
        # elif self.use_l2_norm == "layer_norm":
        #     self.normalize = nn.LayerNorm(self.token_size)

        # shard linear
        self.use_shared_linear = use_shared_linear
        if self.use_shared_linear:
            self.shared_linear = nn.Linear(self.token_size, self.token_size, bias=True)
            self.shared_linear.weight.data.copy_(torch.eye(self.token_size))
            self.shared_linear.bias.data.zero_()
        else:
            self.shared_linear = nn.Identity()

        index_offset = torch.tensor(
            [i * self.codebook_size for i in range(self.num_group)]
        ).long()
        self.register_buffer("index_offset", index_offset)
    
    @torch.no_grad()
    def prepare_codebook(self, data, method: str = "random"):
        """
        Args:
            data (tensor): size (N, C, H, W)
        """
        data = data.permute(0, 2, 3, 1).contiguous()
        data = data.view(-1, self.num_group, self.token_size) # (B*S, nG, D)
        BS = data.shape[0]

        for i in range(self.num_group):
            if method == "random":
                # sample num_code samples from the data
                indices = torch.randint(0, BS, (self.codebook_size,))
                # sample the codebook from the data
                self.embedding.data[i] = data[indices][:, i].clone().detach()
            else:
                raise ValueError(f"Unknown method {method}.")

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """
        Args:
            z (torch.Tensor): Input tensor of shape (B, C, H, W)
        """
        z = z.float()
        B = z.shape[0]
        z = rearrange(z, "b c h w -> b h w c").contiguous() # (B, H, W, D)
        z_flattened = z.view(-1, self.num_group, self.token_size) # (B*S, nG, D)
        z_flattened = z_flattened.permute(1, 0, 2).contiguous() # (nG, B*S, D)

        # normalize the input
        z_flattened = self.normalize(z_flattened)
        embedding = self.normalize(self.shared_linear(self.embedding))

        with torch.no_grad():
            # find the quantized indices
            if self.num_group == 1:
                # 单组情况：z_flattened shape: (1, B*S, D), embedding shape: (1, K, D)
                indices = self.find_indices(
                    z_flattened.squeeze(0),  # (B*S, D)
                    embedding.squeeze(0)     # (K, D)
                ).unsqueeze(0)  # 恢复到 (1, B*S)
            else:
                # 多组情况
                indices = self.find_indices(z_flattened, embedding) # (nG, B*S)
            # indices = self.find_indices(z_flattened, embedding) # (nG, B*S)
            indices = indices.permute(1, 0) # (B*S, nG)
            indices = indices.reshape(B, -1) # (B, S*nG)
        
        # get the quantized vectors
        z_quantized = self.get_codebook_entry(indices).view(z.shape)

        z = self.normalize(z)
        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)
        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices (torch.Tensor): Indices of shape (B, S*nG)
        """
        # get quantized latent vectors
        indices = indices.reshape(-1, self.num_group) # (B*S, nG)
        indices_offset = indices + self.index_offset # (B*S, nG)
        indices_offset = indices_offset.flatten()
        z_quantized = self.shared_linear(self.embedding).view(-1, self.token_size)[indices_offset.int()] # (B*S*nG, D)
        z_quantized = self.normalize(z_quantized)
        return z_quantized