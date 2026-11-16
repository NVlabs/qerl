#!/bin/bash
# QeRL on several GPUs of one node (any GPU supported by the single-GPU scripts: H100, B200, GB300, RTX 5090, ...).
# Every GPU runs one DDP process with its own vLLM engine on the packed NVFP4 weights (+ the live LoRA) and its
# own NVFP4 training replica; only the LoRA / RMSNorm gradients are averaged across GPUs.
export WANDB_PROJECT=QeRL
export MASTER_ADDR=localhost
export MASTER_PORT=29539
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="qwen2.5-7B-multi-gpus-quantized"
MODEL_PATH=${MODEL_PATH:-""} # your NVFP4 model
NUM_GPUS=${NUM_GPUS:-4}

max_prompt_length=1024
max_completion_length=8192
max_seq_length=10000

# Completions per optimizer step = perdevice_train_batch_size * NUM_GPUS * gradient_accumulation_steps,
# which must be a multiple of num_generations (completions per prompt).
# Completions of one optimizer step are grouped by length across the accumulation steps, so prefer a smaller
# perdevice_train_batch_size with more accumulation steps when GPU memory is limited.
perdevice_train_batch_size=${PER_DEVICE_BATCH:-8}
gradient_accumulation_steps=${GRAD_ACCUM:-4}
num_generations=8
gradient_checkpointing=${GRAD_CKPT:-True}  # keep it on: without it one very long completion can exhaust the GPU memory
vllm_gpu_memory_utilization=${VLLM_MEM:-0.3}
lora_rank=16
lora_alpha=16

rl_mode="grpo"
beta=0.04
num_iterations=1
epsilon_high=0.2
loss_type="bnpo"  # bnpo equals to DAPO, the default is grpo (sample level and token level)
DATA_NAME=${DATA_NAME:-"gsm8k"}  # dapomath, gsm8k, codeforces
optim="adamw_8bit"  # Optimizer type, adamw_8bit or adamw

sigma_start=1e-2
sigma_end=1e-4
num_stages=10

if [ "$DATA_NAME" == "gsm8k" ]; then
    max_prompt_length=256
    max_completion_length=2048
    max_seq_length=2500
elif [ "$DATA_NAME" == "bigmath" ]; then
    max_prompt_length=1024
    max_completion_length=4096
    max_seq_length=5500
fi

if [ "$loss_type" == "bnpo" ]; then
    rl_mode="dapo"
    beta=0.0
    epsilon_high=0.28
    num_iterations=1
fi

lr=${LR:-3e-5}
ln=True

runname=${MODEL_NAME}_${lr}_${epsilon_high}_${DATA_NAME}_n${NUM_GPUS}_b${perdevice_train_batch_size}_g${gradient_accumulation_steps}_r${lora_rank}_${ln}

CKPTS_DIR=${CKPTS_DIR:-"./ckpt/${runname}"}

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/ddp.yaml --num_processes=$NUM_GPUS --main_process_port $MASTER_PORT \
    qerl.py  \
    --dataset $DATA_NAME \
    --model-name $MODEL_PATH \
    --output-dir $CKPTS_DIR \
    --use-vllm True \
    --learning-rate $lr \
    --adam-beta1 0.9 \
    --adam-beta2 0.99 \
    --weight-decay 0.1 \
    --warmup-ratio 0.1 \
    --lr-scheduler-type "cosine" \
    --optim $optim \
    --logging-steps 1 \
    --per-device-train-batch-size $perdevice_train_batch_size \
    --gradient-accumulation-steps $gradient_accumulation_steps \
    --num-generations $num_generations \
    --max-prompt-length $max_prompt_length \
    --max-completion-length $max_completion_length \
    --num-train-epochs 1 \
    --max-steps ${MAX_STEPS:--1} \
    --save-steps 50 \
    --save-strategy "steps" \
    --save-total-limit 5 \
    --max-grad-norm 0.2 \
    --max-seq-length $max_seq_length \
    --lora-rank $lora_rank \
    --lora-alpha $lora_alpha \
    --fast-inference True \
    --vllm-gpu-memory-utilization $vllm_gpu_memory_utilization \
    --vllm-enable-sleep-mode False \
    --gradient-checkpointing $gradient_checkpointing \
    --ddp-find-unused-parameters False \
    --random-state 2025 \
    --loss-type $loss_type \
    --beta $beta \
    --epsilon-high $epsilon_high \
    --num-iterations $num_iterations \
    --mask-truncated-completions True \
    --report-to ${REPORT_TO:-wandb} \
    --run-name $runname \
    --ln $ln \
    --sigma-start $sigma_start \
    --sigma-end $sigma_end \
    --num-stages $num_stages
