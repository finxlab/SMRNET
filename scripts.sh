#!/bin/bash

# ==========================================
# 1. Hyperparameter Settings
# ==========================================
MODEL_NAME="SMRNET"
SEED=2026
TEST_DATE="2025-12-31"

LR=0.0001    # learning rate
HD=64        # hidden dimension (32 64 128)
DR=0.3       # dropout (0.1 0.3 0.5)

# ==========================================
# 2. Execute Single Run
# ==========================================
echo "=========================================================="
echo "  STARTING SINGLE RUN: TEST END DATE = $TEST_DATE"
echo "=========================================================="
echo "Running: lr=$LR, hd=$HD, dr=$DR"

python run.py \
    --model_type $MODEL_NAME \
    --random_seed $SEED \
    --learning_rate $LR \
    --hidden_dim $HD \
    --dropout $DR \
    --train_epochs 100 \
    --patience 10 \
    --loss IC \
    --test_date "$TEST_DATE" 
