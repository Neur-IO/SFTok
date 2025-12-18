# generator 
WANDB_MODE=offline PYTHONPATH=./ HF_ENDPOINT=https://hf-mirror.com /path/to/accelerate launch --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_maskgit.py config=configs/sftok_maskgit.yaml \
    experiment.project="sftok_generation" \
    experiment.name="sfok-b_run1" \
    experiment.output_dir="outputs/sfok-b_run1" \
    experiment.tokenizer_checkpoint='/path/to/sftok_tokenizer' \