# Reproducing sftok-b
torchrun --nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:9999 sample_imagenet_sftok.py \
    config=configs/infer_sftok.yaml \
    experiment.output_dir="sftok-b" \
    experiment.generator_checkpoint="/path/to/generator-checkpoint/ema_model/pytorch_model.bin"

python3 guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz sftok-b.npz