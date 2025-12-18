import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class MultiStepDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.decoder.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.decoder.vit_dec_model_size
        self.num_latent_tokens = config.model.decoder.num_latent_tokens
        self.token_size = config.model.decoder.token_size
        self.mask_token_id = config.model.decoder.codebook_size
        self.codebook_size = config.model.decoder.codebook_size
        self.num_proxy_codes = config.model.decoder.num_proxy_codes
        self.is_legacy = config.model.decoder.get("is_legacy", True)
        self.embedding_init = config.model.decoder.get("embedding_init", False)
        self.replace_prob = config.model.decoder.get("replace_prob", 0.8)
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", False)
        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.embedding_width = self.width  # 默认和width相同
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.condition_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))

        self.learnable_embedding = nn.Embedding(
            self.codebook_size + 1, self.embedding_width)
        self.learnable_embedding.weight.data = scale * torch.randn(
            self.codebook_size + 1, self.embedding_width
        )

        if self.embedding_init:
            self.load_pretrained_embedding("/path/to/pretrained/pixel_quantizer_embedding.bin")

        if self.embedding_width != self.width:
            self.embedding_transform = nn.Linear(self.embedding_width, self.width)
        else:
            self.embedding_transform = nn.Identity()
        
        self.learnable_embedding_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_proxy_codes + 1, self.width))
        
        # add mask token and query pos embed
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy:
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
                Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                    p1 = self.patch_size, p2 = self.patch_size),)
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)

    def load_pretrained_embedding(self, pretrained_weight_path):
        checkpoint = torch.load(pretrained_weight_path, map_location='cpu')
        pixel_quantizer_embedding = None
        
        possible_keys = [
            'quantize.embedding.weight',
        ]
        
        for key in possible_keys:
            if key in checkpoint:
                pixel_quantizer_embedding = checkpoint[key]
                del checkpoint
                break
        
        if pixel_quantizer_embedding is None:
            print(f"Warning: Could not find pixel quantizer embedding in {pretrained_weight_path}")
            print(f"Available keys: {list(checkpoint.keys())}")
            return
        
        pretrained_shape = pixel_quantizer_embedding.shape
        expected_shape = (self.codebook_size, self.embedding_width)  # (1024, width)
        
        
        if pretrained_shape[1] != expected_shape[1]:
            print(f"Warning: Pretrained embedding dim {pretrained_shape[1]} != expected {expected_shape[1]}")
            if pretrained_shape[1] < expected_shape[1]:
                padded_embedding = torch.zeros(expected_shape)
                padded_embedding[:, :pretrained_shape[1]] = pixel_quantizer_embedding
                pixel_quantizer_embedding = padded_embedding
            else:
                pixel_quantizer_embedding = pixel_quantizer_embedding[:, :expected_shape[1]]
        
        with torch.no_grad():
            self.learnable_embedding.weight.data[:self.codebook_size] = pixel_quantizer_embedding.clone()
        
        del pixel_quantizer_embedding
        print(f"Successfully loaded pretrained embedding from {pretrained_weight_path}")

    
    def forward(self, proxy_codes, z_quantized, mask_all=False, cond_drop_prob=0.1, use_mask=True, guided_mask=None,
                temperature=1.0, mask_schedule="arccos", mask_power=1.0,
                min_mask_ratio=0.0, max_mask_ratio=1.0):
        N, C, H, W = z_quantized.shape
        assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        if use_mask:
            mask_tokens, masks = self.masking_tokens(proxy_codes, 
                mask_all=mask_all, guided_mask=guided_mask,
                temperature=temperature, mask_schedule=mask_schedule, mask_power=mask_power,
                min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio)
        else:
            mask_tokens = proxy_codes
            masks = None
        mask_tokens = self.learnable_embedding(mask_tokens)
        mask_tokens = self.embedding_transform(mask_tokens)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.learnable_embedding_positional_embedding.to(mask_tokens.dtype)

        x = x + self.condition_positional_embedding.to(x.dtype)
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x) # B, 1024, 16, 16
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # B, 1024, 16, 16 -> B, 256, 1024
        return x, masks
    
    
    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale=3.0,
                 guidance_decay="constant",
                 guidance_scale_pow=3.0,
                 randomize_temperature=4.5,
                 softmax_temperature_annealing=False,
                 num_sample_steps=8):
        if guidance_decay not in ["constant", "linear", "power-cosine"]:
            # contstant: constant guidance scale
            # linear: linear increasing the guidance scale as in MUSE
            # power-cosine: the guidance schedule from MDT
            raise ValueError(f"Unsupported guidance decay {guidance_decay}")
        device = condition.device
        ids = torch.full((condition.shape[0], self.num_proxy_codes),
                          self.mask_token_id, device=device)
        prev_masking = torch.ones_like(ids, dtype=torch.bool, device=device)
        cfg_scale = guidance_scale if guidance_decay == "constant" else 0.
        bs = condition.shape[0]

        for step in range(num_sample_steps):
            ratio = 1. * (step + 1) / num_sample_steps
            annealed_temp = randomize_temperature * (1.0 - ratio)
            is_mask = (ids == self.mask_token_id)

            if guidance_decay == "power-cosine":
                guidance_scale_pow = torch.ones((1), device=device) * guidance_scale_pow
                scale_step = (1 - torch.cos(((step / num_sample_steps) ** guidance_scale_pow) * torch.pi)) * 1/2
                cfg_scale = (guidance_scale - 1) * scale_step + 1

            if cfg_scale != 0:
                cond_logits, _ = self.forward(
                    ids, condition, cond_drop_prob=0.0, use_mask=False
                )
                uncond_logits, _ = self.forward(
                    ids, condition, cond_drop_prob=1.0, use_mask=False
                )
                if guidance_decay == "power-cosine":
                    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                else:
                    logits = cond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits, _ = self.forward(
                    ids, condition, cond_drop_prob=0.0, use_mask=False
                )

            if softmax_temperature_annealing:
                softmax_temperature = 0.5 + 0.8 * (1 - ratio)
                logits = logits / softmax_temperature

            # Add gumbel noise
            def log(t, eps=1e-20):
                return torch.log(t.clamp(min=eps))
            def gumbel_noise(t):
                noise = torch.zeros_like(t).uniform_(0, 1)
                return -log(-log(noise))
            def add_gumbel_noise(t, temperature):
                return t + temperature * gumbel_noise(t)

            sampled_ids = add_gumbel_noise(logits, annealed_temp).argmax(dim=-1)
            # sampled_ids = logits.argmax(dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            # originally we use arccos schedule as in MUSE
            mask_ratio = np.arccos(ratio) / (math.pi * 0.5)

            mask_len = torch.Tensor([np.floor(self.num_proxy_codes * mask_ratio)]).to(device)
            mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                     torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1,
                                                   mask_len))[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp)
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
            masking = (confidence <= cut_off)

            if step == num_sample_steps - 1:
                ids = sampled_ids
                new_unmasked = prev_masking
            else:
                ids = torch.where(masking, self.mask_token_id, sampled_ids)
                new_unmasked = prev_masking & (~(ids == self.mask_token_id))
                prev_masking = (ids == self.mask_token_id)
          
        logits_all = logits  
        return ids, logits_all

    def masking_tokens(self, input_tokens, timesteps=None, 
                       mask_all=False, guided_mask=None,
                       temperature=1.0, mask_schedule="arccos", mask_power=1.0,
                       min_mask_ratio=0.1, max_mask_ratio=0.5):
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        if mask_all:
            # Mask all tokens
            masks = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
            masked_tokens = torch.full_like(input_tokens, self.mask_token_id)
            return masked_tokens, masks

        if timesteps is None:
            timesteps = torch.zeros((batch_size,), device=device).float().uniform_(0, 1.0)

        if mask_schedule == "arccos":
            base_ratio = torch.acos(timesteps) / (math.pi * 0.5)
            mask_ratio = base_ratio ** mask_power
        elif mask_schedule == "linear":
            mask_ratio = 1.0 - timesteps
        elif mask_schedule == "quadratic":
            mask_ratio = (1.0 - timesteps) ** mask_power
        elif mask_schedule == "cubic":
            mask_ratio = (1.0 - timesteps) ** 3
        elif mask_schedule == "exponential":
            mask_ratio = torch.exp(-mask_power * timesteps)
        elif mask_schedule == "cosine":
            mask_ratio = 0.5 * (1 + torch.cos(math.pi * timesteps))
        else:
            raise ValueError(f"Unsupported mask schedule {mask_schedule}")

        mask_ratio = torch.clamp(mask_ratio, min=0., max=1.)
        mask_ratio = min_mask_ratio + (max_mask_ratio - min_mask_ratio) * mask_ratio

        min_tokens = max(0, int(seq_len * min_mask_ratio))  # at least mask 0
        max_tokens = min(seq_len, int(seq_len * max_mask_ratio))  # at most mask max_mask_ratio
        num_token_masked = (seq_len * mask_ratio).round().long()
        num_token_masked = torch.clamp(num_token_masked, min=min_tokens, max=max_tokens)

        batch_randperm = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        masks = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        if guided_mask is not None:
            guided_tokens = torch.argmax(guided_mask, dim=-1)
            if self.finetune_decoder:
                input_tokens = guided_tokens
            else:
                replacement_prob = torch.rand(batch_size, seq_len, device=device)
                should_replace = (replacement_prob < self.replace_prob)
                input_tokens = torch.where(should_replace, guided_tokens, input_tokens)

        masked_tokens = torch.where(masks, self.mask_token_id, input_tokens)

        return masked_tokens, masks
