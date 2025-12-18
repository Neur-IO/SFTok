MODEL_TYPE="sftok"
BATCH_SIZE=32
NGPUS=8
PORT=9999
SINGLE_STEP_GENERATION=False
NAME="sftok-b_stage2_run1"
MODEL_SIZE="base"
REPLACE_PROB=1.0
GUIDED_MASK=True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_MODE=offline PYTHONPATH=./ HF_ENDPOINT=https://hf-mirror.com /path/to/accelerate launch \
    --num_machines=1 --num_processes=${NGPUS} --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=${PORT} \
    --same_network scripts/train_sftok.py config=configs/sftok_b64_stage1.yaml \
    experiment.project=sftok-b \
    experiment.name=${NAME} \
    experiment.output_dir=outputs/${NAME} \
    model.model_type=${MODEL_TYPE} \
    training.per_gpu_batch_size=${BATCH_SIZE} \
    training.single_step_generation=${SINGLE_STEP_GENERATION} \
    model.decoder.vit_dec_model_size=${MODEL_SIZE} \
    model.decoder.replace_prob=${REPLACE_PROB} \
    training.guided_mask=${GUIDED_MASK}

