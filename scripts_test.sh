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
# 2. Execute Test Run
# ==========================================
SETTING="seed${SEED}_lr${LR}_hd${HD}_gh1_nh1_dr${DR}"

echo "=========================================================="
echo "  STARTING TEST RUN: TEST END DATE = $TEST_DATE"
echo "=========================================================="
echo "Setting: $SETTING"

python test.py \
    --model_type $MODEL_NAME \
    --random_seed $SEED \
    --test_date "$TEST_DATE" \
    --setting "$SETTING"

echo "=========================================================="
echo "  TEST DONE"
echo "=========================================================="
