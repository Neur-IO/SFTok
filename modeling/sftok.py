import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from timm.models import create_model

from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import TiTokEncoder
from modeling.modules.blocks_multistep import MultiStepDecoder
from modeling.quantizer.quantizer import DiagonalGaussianDistribution
from modeling.quantizer.optvq import OptVQ as VectorQuantizer
from modeling.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from modeling.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from modeling.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
from modeling.modules.clip_loss import ClipLoss
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin

class Normalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Normalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class Denormalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Denormalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return x * self.std + self.mean

class PretrainedTokenizer(nn.Module):
    def __init__(self, pretrained_weight):
        super().__init__()
        conf = OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 256})
        self.encoder = Pixel_Eecoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        # Load pretrained weights
        self.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states)
        return codebook_indices.detach()
    
    @torch.no_grad()
    def decode(self, codes):
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
    
    @torch.no_grad()
    def decode_tokens(self, codes):
        return self.decode(codes)


class SFTok(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-tokenization"], repo_url=None, license="apache-2.0"):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1, stage2 and True for stage3.1
        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")
        
        if self.finetune_decoder and self.quantize_mode not in ["vq"]:
            raise ValueError("Only supprot finetune_decoder with vq quantization for now.")

        self.encoder = TiTokEncoder(config)
        self.decoder = MultiStepDecoder(config)
        
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,
                use_shared_linear=True,
                use_sinkhorn=True,
                num_group=config.model.vq_model.num_group)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        self.semantic_guide = config.model.vq_model.get("semantic_guide", None)
        if self.finetune_decoder:
            # Freeze encoder/quantizer/latent tokens
            self.latent_tokens.requires_grad_(False)
            self.encoder.eval()
            self.encoder.requires_grad_(False)
            self.quantize.eval()
            self.quantize.requires_grad_(False)

            # Include MaskGiT-VQGAN's quantizer and decoder
            self.pixel_quantize = Pixel_Quantizer(
                num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
            self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
                {"channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "resolution": 256,
                "z_channels": 256}))
            
            # semantic guidance
            self.semantic_guide = config.model.vq_model.get("semantic_guide", None)
            if self.semantic_guide == "dinov2":
                # build semantic model
                semantic_model = create_model(
                    model_name="vit_base_patch14_dinov2.lvd142m",
                    pretrained=True, img_size=256, patch_size=16, 
                    drop_path_rate=0.0
                )
                semantic_model.eval()
                for param in semantic_model.parameters():
                    param.requires_grad = False
                self.semantic_model = semantic_model
                # build semantic loss
                self.semantic_loss = ClipLoss(
                    local_loss=True,
                    gather_with_grad=True,
                    cache_labels=True,
                    rank=dist.get_rank(),
                    world_size=dist.get_world_size()
                )
                self.sem_loss_weight = config.model.vq_model.get("sem_loss_weight", 0.0)
                # build semantic layer
                embed_dim = self.pixel_quantize.embedding.weight.size(-1)
                self.sem_norm = nn.LayerNorm(embed_dim, eps=1e-6)
                self.sem_linear = nn.Linear(embed_dim, 768)
                self.sem_scale = 1 # nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                
                # sem_normalize
                self.sem_denormalize = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                self.sem_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        if self.finetune_decoder:
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_quantized, result_dict = self.quantize(z)
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        return z_quantized, result_dict
    
    def decode(self, z_quantized, proxy_codes, return_latent: bool = False, drop_prob: float = None, mask_all: bool = False, guided_mask=None):
        cond_drop_prob = drop_prob if drop_prob is not None else self.config.model.generator.class_label_dropout
        decoded, masks = self.decoder(proxy_codes=proxy_codes, 
                            z_quantized=z_quantized, 
                            cond_drop_prob=cond_drop_prob,
                            mask_all=mask_all,
                            guided_mask=guided_mask,
                            mask_schedule=self.config.training.get("mask_schedule", "arccos"),
                            mask_power=self.config.training.get("mask_power", 1.0),
                            min_mask_ratio=self.config.training.get("min_mask_ratio", 0.0),
                            max_mask_ratio=self.config.training.get("max_mask_ratio", 1.0))
        quantized_states = None
        decoded_img = None
        if self.finetune_decoder:
            decoded_reshape = decoded.permute(0, 2, 1).reshape(decoded.size(0), 1024, 16, 16).contiguous() # B x C x H x W
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', decoded_reshape.softmax(1),
                self.pixel_quantize.embedding.weight)
            decoded_img = self.pixel_decoder(quantized_states)
        if return_latent:
            return dict(
                decoded=decoded,
                latent=quantized_states
            )
        else:
            return decoded, masks, decoded_img
    
    def decode_tokens(self, tokens):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded_img = self.generate_codes(condition=z_quantized, config=self.config)
        return decoded_img

    @torch.no_grad()
    def generate_codes(self, condition, config):
        """Generate codes from input images."""
        guidance_scale = config.model.generator.get("guidance_scale", 3.0)
        guidance_decay = config.model.generator.get("guidance_decay", "constant")
        guidance_scale_pow=config.model.generator.get("guidance_scale_pow", 3.0)
        randomize_temperature=config.model.generator.get("randomize_temperature", 2.0)
        softmax_temperature_annealing=config.model.generator.get("softmax_temperature_annealing", False)
        num_sample_steps=config.model.generator.get("num_steps", 8)

        rec_codes, rec_embeddings = self.decoder.generate(condition=condition,
                                          guidance_scale=guidance_scale,
                                          guidance_decay=guidance_decay,
                                          guidance_scale_pow=guidance_scale_pow,
                                          randomize_temperature=randomize_temperature,
                                          softmax_temperature_annealing=softmax_temperature_annealing,
                                          num_sample_steps=num_sample_steps)
        if self.finetune_decoder:
            rec_embeddings_reshape = rec_embeddings.permute(0, 2, 1).reshape(rec_embeddings.size(0), 1024, 16, 16).contiguous() # B x C x H x W
            quantized_states = torch.einsum(
                'nchw,cd->ndhw', rec_embeddings_reshape.softmax(1),
                self.pixel_quantize.embedding.weight)
            rec_codes = self.pixel_decoder(quantized_states)

        return rec_codes

    def generate_guided_mask(self, x, proxy_codes, use_semantic: bool = False, mask_all: bool = True, guided_mask=None):
        z_quantized, _ = self.encode(x)
        decoded, _ = self.decoder(proxy_codes=proxy_codes, 
                z_quantized=z_quantized, 
                cond_drop_prob=0.0,
                mask_all=mask_all,
                guided_mask=guided_mask,
                mask_schedule=self.config.training.get("mask_schedule", "arccos"),
                mask_power=self.config.training.get("mask_power", 1.0),
                min_mask_ratio=self.config.training.get("min_mask_ratio", 0.0),
                max_mask_ratio=self.config.training.get("max_mask_ratio", 1.0))
        return decoded
    
    def forward(self, x, proxy_codes, use_semantic: bool = False, mask_all: bool = False, guided_mask=None):
        z_quantized, result_dict = self.encode(x)
        out, masks, decoded_img = self.decode(z_quantized=z_quantized,
                          proxy_codes=proxy_codes,
                          mask_all=mask_all,
                          guided_mask=guided_mask)
        if isinstance(out, dict):
            latent = out["latent"]
            decoded = out["decoded"]
        else:
            decoded = out
            latent = None
        
        # semantic guidance
        if self.semantic_guide is not None and use_semantic:
            # x is in the range [0, 1]
            x_copy = self.sem_normalize(x)
            # compute the semantic reference
            with torch.no_grad():
                clip_ref = self.semantic_model(x_copy)
                clip_ref = F.normalize(clip_ref, dim=-1, p=2)
            # compute the projected latent
            clip_vis = torch.mean(latent, dim=(2, 3))
            clip_vis = self.sem_norm(clip_vis)
            clip_vis = self.sem_linear(clip_vis)
            clip_vis = F.normalize(clip_vis, dim=-1, p=2)

            with torch.amp.autocast('cuda', enabled=False):
                sem_loss = self.semantic_loss(
                    image_features=clip_vis.float(),
                    text_features=clip_ref.float(),
                    logit_scale=self.sem_scale.exp(),
                ) * self.sem_loss_weight
                result_dict["sem_loss"] = sem_loss
                result_dict["quantizer_loss"] = result_dict["quantizer_loss"] + sem_loss
        else:
            result_dict["sem_loss"] = torch.tensor(0.0, device=decoded.device)

        if self.finetune_decoder:
            return decoded_img, result_dict, masks, decoded
        else:
            return decoded, result_dict, masks, decoded_img