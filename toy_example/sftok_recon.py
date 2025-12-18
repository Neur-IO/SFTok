from utils.train_utils import (
    get_config, create_pretrained_tokenizer,
    create_model_and_loss_module,
    create_dataloader
)
from data import SimpleImageDataset
import torch
from omegaconf import OmegaConf
from modeling.titok import TiTok, PretrainedTokenizer
from modeling.sftok import SFTok
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Times New Roman', 'Times', 'Palatino', 'serif']
import math
import os
import torchvision.utils as vutils
from PIL import Image

def main(device, config_path=None, model_ckpt_path=None, image_ids=None, ori_path="", recon_path=""):
    os.makedirs(ori_path, exist_ok=True)
    os.makedirs(recon_path, exist_ok=True)

    config = OmegaConf.load(config_path)

    # pretrained_tokenizer = create_pretrained_tokenizer(config).to(device)
    model = SFTok(config)

    if model_ckpt_path is not None:
        print(f"Loading model from {model_ckpt_path}")
        state_dict = torch.load(model_ckpt_path, map_location='cpu')
        if "decoder.embedding_positional_embedding" in state_dict:
            state_dict["decoder.learnable_embedding_positional_embedding"] = state_dict.pop("decoder.embedding_positional_embedding")
            print("Renamed decoder.embedding_positional_embedding to decoder.learnable_embedding_positional_embedding")
    
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded.")

    dataset = SimpleImageDataset(
        train_shards_path="/path/to/data/train",
        eval_shards_path="/path/to/data/val",
        num_train_examples=1281167,
        per_gpu_batch_size=4,
        global_batch_size=1,
        num_workers_per_gpu=8,
        resize_shorter_edge=256,
        crop_size=256,
        random_crop=True,
        random_flip=True,
        dataset_with_class_label=True,
        dataset_with_text_label=False,
        res_ratio_filtering=False
    )
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader

    dataloader = eval_dataloader
    
    model = model.to(device)
    model.eval()


    if isinstance(image_ids, int):
        image_ids = [image_ids]
    assert isinstance(image_ids, list) and len(image_ids) > 0, "image_ids must be a non-empty list or an integer."

    global_idx = 0

    over = False
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        batch_size = images.size(0)

        for j in range(batch_size):
            if global_idx in image_ids:
                with torch.no_grad():
                    encoded_tokens = model.encode(images)[1]["min_encoding_indices"]
                    reconstructed_images = model.decode_tokens(encoded_tokens)

                original_image = images[j].clamp(0, 1)
                reconstructed_image = reconstructed_images[j].clamp(0, 1)

                ori_path_full = os.path.join(ori_path, f"ori_{global_idx}.png")
                recon_path_full = os.path.join(recon_path, f"sftok-b_{global_idx}.png")

                vutils.save_image(original_image, ori_path_full)
                vutils.save_image(reconstructed_image, recon_path_full)

                print(f"Saved image ID {global_idx}:")

            global_idx += 1
            if global_idx > max(image_ids):
                over = True
                break

        if over:
            break

    print("Done!")

if __name__ == "__main__":
    device = 'cuda:0'
    config_path = '/path/to/config.yaml'
    model_ckpt_path = '/path/to/model-checkpoint'
    original_images_dir = "/path/to/ori"
    reconstructed_images_dir = "/path/to/sftok-b"

    selected_ids = [1, 2, 3, 4, 5]  # Specify the image IDs you want to reconstruct

    main(
        device=device,
        config_path=config_path,
        model_ckpt_path=model_ckpt_path,
        image_ids=selected_ids,
        ori_path=original_images_dir,
        recon_path=reconstructed_images_dir
    )
