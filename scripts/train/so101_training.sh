#!/bin/bash
# DreamZero SO-101 Training Script
#
# Usage:
#   # 1. Download & convert dataset:
#   hf download imstevenpmwork/thanos_picking_power_gem --repo-type dataset --local-dir ./data/so101_thanos
#   python scripts/data/convert_lerobot_to_gear.py \
#     --dataset-path ./data/so101_thanos \
#     --embodiment-tag so101 \
#     --state-keys '{"joint_pos": [0, 5], "gripper_pos": [5, 6]}' \
#     --action-keys '{"joint_pos": [0, 5], "gripper_pos": [5, 6]}' \
#     --relative-action-keys joint_pos gripper_pos \
#     --force
#
#   # 2. Train:
#   SO101_DATA_ROOT=./data/so101_thanos bash scripts/train/so101_training.sh
#
# Prerequisites:
#   - SO-101 dataset converted to GEAR format (see above)
#   - Wan2.1-I2V-14B-480P weights (auto-downloaded if missing)
#   - umt5-xxl tokenizer (auto-downloaded if missing)
#   - DreamZero-AgiBot pretrained checkpoint:
#     hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir ./checkpoints/DreamZero-AgiBot

export HYDRA_FULL_ERROR=1

# ============ CONFIGURATION ============
SO101_DATA_ROOT=${SO101_DATA_ROOT:-"./data/so101_thanos"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_so101_lora"}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-2}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
# =======================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    hf download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

# Validate dataset exists
if [ ! -d "$SO101_DATA_ROOT" ]; then
    echo "ERROR: SO-101 dataset not found at $SO101_DATA_ROOT"
    echo "Download it first: hf download imstevenpmwork/thanos_picking_power_gem --repo-type dataset --local-dir $SO101_DATA_ROOT"
    exit 1
fi

if [ ! -f "$SO101_DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing — run convert_lerobot_to_gear.py first (see header of this script)"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/so101_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=10000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=4 \
    max_steps=100000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    so101_data_root=$SO101_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=./checkpoints/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
