#!/bin/bash
set -e
cd /home/ubuntu/Workspaces/gSASRec-pytorch

PYTHON=".venv/bin/python"
SCRIPT="train_gsasrec_torchrec_tuned.py"
CONFIG="config_ml1m.py"
EPOCHS=20

echo "========== Grid Search: 4 configs x $EPOCHS epochs =========="

# Config 1: Baseline
echo ""
echo "[1/4] Baseline (default settings)"
$PYTHON $SCRIPT --config $CONFIG --max_epochs_tune $EPOCHS --label baseline

# Config 2: CUDA init + Xavier init
echo ""
echo "[2/4] CUDA init + Xavier init"
$PYTHON $SCRIPT --config $CONFIG --use_cuda_init --init_type xavier --max_epochs_tune $EPOCHS --label cuda_xavier

# Config 3: CUDA init + Xavier + Separate LRs
echo ""
echo "[3/4] CUDA init + Xavier + Separate LRs (emb=3e-4, dense=1e-3)"
$PYTHON $SCRIPT --config $CONFIG --use_cuda_init --init_type xavier --emb_lr 0.0003 --dense_lr 0.001 --max_epochs_tune $EPOCHS --label cuda_xavier_sep_lr

# Config 4: CUDA init + Xavier + AdamW + Grad Clip
echo ""
echo "[4/4] CUDA init + Xavier + AdamW (wd=1e-4) + Grad Clip=1.0"
$PYTHON $SCRIPT --config $CONFIG --use_cuda_init --init_type xavier --grad_clip 1.0 --weight_decay 1e-4 --max_epochs_tune $EPOCHS --label cuda_xavier_adamw_clip

echo ""
echo "========== Grid Search Complete =========="
# Summarize results
echo ""
echo "Best metrics per config:"
grep "FINAL BEST METRIC" logs/train_torchrec_tuned_*.log
