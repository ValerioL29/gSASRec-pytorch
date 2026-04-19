# Training Preparation Guide for gSASRec

This guide explains how to adapt a **custom dataset** to the gSASRec pipeline, configure hyperparameters, and run training with either the original PyTorch or the TorchRec implementation.

---

## 1. Custom Dataset Format

The pipeline expects data under `datasets/<dataset_name>/` with the following layout:

```
datasets/
└── mydataset/
    ├── dataset_stats.json          # {"num_items": N}
    ├── train/
    │   └── input.txt               # one user sequence per line (space-separated item IDs)
    ├── val/
    │   ├── input.txt               # user history sequence
    │   └── output.txt              # single ground-truth next item per line
    └── test/
        ├── input.txt
        └── output.txt
```

### File details

| File | Format | Example |
|------|--------|---------|
| `train/input.txt` | One sequence per line. Items enumerated from **1**. | `1 42 7 3 99` |
| `val/input.txt` | Same as train (history for each user). | `1 42 7 3` |
| `val/output.txt` | One integer per line (next item to predict). | `99` |
| `dataset_stats.json` | `{"num_items": 3706}` — must match the highest item ID. | — |

> **Note:** Item IDs start at **1**. The pipeline reserves ID `num_items + 1` as the **padding token** internally. Do not use `0` or `num_items + 1` in your data.

---

## 2. Create a Config File

Copy `config_ml1m.py` to `config_mydataset.py` and tune the fields:

```python
from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='mydataset',        # must match directory name under datasets/
    sequence_length=200,             # max sequence length (truncate / pad)
    embedding_dim=128,               # hidden dim (128-256 typical)
    num_heads=1,                     # attention heads (1-4)
    num_blocks=2,                    # transformer blocks (2-4)
    dropout_rate=0.5,                # 0.2-0.5 typical
    negs_per_pos=256,                # negatives per positive for gBCE (128-512)
    gbce_t=0.75,                     # temperature for gBCE loss (0.5-1.0)
    max_batches_per_epoch=100,       # cap batches per epoch
    train_batch_size=128,
    eval_batch_size=512,
    early_stopping_patience=200,     # stop if no improvement for N eval steps
)
```

### Key hyperparameters explained

| Parameter | Effect | Tuning Guidance |
|-----------|--------|-----------------|
| `embedding_dim` | Model capacity | 128 for small/medium catalogs; 256+ for large |
| `num_blocks` | Depth | 2 for ML1M; 3-4 for larger datasets |
| `num_heads` | Attention heads | 1 works well for 128-dim; use 2-4 for 256-dim |
| `dropout_rate` | Regularization | 0.5 for small data; 0.2-0.3 for large data |
| `negs_per_pos` | Sampling density | 256 is standard; lower for faster training |
| `gbce_t` | Loss temperature | 0.75 default; lower = harder negatives |
| `max_batches_per_epoch` | Epoch granularity | 50-200; smaller = more frequent eval |

---

## 3. Choose a Model Variant

Three model implementations are available:

| Variant | File | Best For |
|---------|------|----------|
| **Original PyTorch** | `gsasrec.py` | Best accuracy; simple `nn.Embedding` |
| **TorchRec** | `gsasrec_torchrec.py` | TorchRec integration baseline |
| **TorchRec Tuned** ⭐ | `gsasrec_torchrec_tuned.py` | **Best speed/accuracy trade-off** |

### Variant comparison (ML1M benchmark)

| Variant | Best nDCG@10 | Training Speed | Notes |
|---------|-------------|----------------|-------|
| Original PyTorch | **0.02291** | ~4.0 it/s | Highest accuracy |
| TorchRec (baseline) | 0.02175 | ~3.3 it/s | Slower, CPU init overhead |
| TorchRec Tuned | **0.02178** | **~5.4 it/s** | **+35% faster**, matches TorchRec baseline |

> The tuned TorchRec variant is **recommended** for production use because it trains significantly faster while retaining ~95% of the original accuracy.

---

## 4. Training

### 4.1 Original PyTorch

```bash
source .venv/bin/activate
python train_gsasrec.py --config config_mydataset.py
```

### 4.2 TorchRec Tuned (recommended)

```bash
source .venv/bin/activate
python train_gsasrec_torchrec_tuned.py --config config_mydataset.py \
    --use_cuda_init --init_type xavier \
    --emb_lr 0.0003 --dense_lr 0.001 \
    --max_epochs_tune 50
```

### CLI flags for `train_gsasrec_torchrec_tuned.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config_ml1m.py` | Path to config file |
| `--use_cuda_init` | `False` | Create `EmbeddingCollection` directly on CUDA |
| `--init_type` | `default` | `default` / `xavier` / `he` |
| `--emb_lr` | `0.001` | Learning rate for embedding parameters |
| `--dense_lr` | `0.001` | Learning rate for transformer / dense parameters |
| `--grad_clip` | `0.0` | Gradient clipping max norm (e.g. `1.0`) |
| `--weight_decay` | `0.0` | AdamW weight decay (e.g. `1e-4`) |
| `--use_sparse_adam` | `False` | Use `SparseAdam` for embeddings |
| `--max_epochs_tune` | `20` | Training epochs |
| `--label` | `''` | Experiment tag appended to log/model names |

---

## 5. Hyperparameter Tuning Strategy

Based on the grid search performed on ML1M, here is a recommended tuning workflow:

### Step 1: Baseline run
```bash
python train_gsasrec_torchrec_tuned.py --config config_mydataset.py --label baseline
```

### Step 2: Apply CUDA init + Xavier initialization
```bash
python train_gsasrec_torchrec_tuned.py --config config_mydataset.py \
    --use_cuda_init --init_type xavier --label cuda_xavier
```
> This single change provided the **largest accuracy improvement** (+0.0038 nDCG@10 on ML1M).

### Step 3: Try separate learning rates
```bash
python train_gsasrec_torchrec_tuned.py --config config_mydataset.py \
    --use_cuda_init --init_type xavier \
    --emb_lr 0.0003 --dense_lr 0.001 --label sep_lr
```
> Embeddings benefit from a slower LR than dense transformer layers.

### Step 4: Optional — AdamW + gradient clipping
```bash
python train_gsasrec_torchrec_tuned.py --config config_mydataset.py \
    --use_cuda_init --init_type xavier \
    --grad_clip 1.0 --weight_decay 1e-4 --label adamw_clip
```
> This stabilized training but did not further improve peak nDCG@10 on ML1M.

### Tuning summary for new datasets

1. **Always use `--use_cuda_init --init_type xavier`** — it is faster and more accurate.
2. **Start with `emb_lr=0.0003, dense_lr=0.001`** — separate LRs consistently helped.
3. **If overfitting** (metric drops after early peak), add `--grad_clip 1.0`.
4. **If still underfitting**, increase `embedding_dim` or `num_blocks` in the config.
5. **Run 20-50 epochs** with early stopping; the best metric usually appears within the first 10 epochs.

---

## 6. Outputs & Checkpoints

Training produces:

```
logs/train_torchrec_tuned_<label>.log    # per-epoch metrics & loss
models/torchrec-tuned-<dataset>-<label>-step:...-metric:<value>.pt
```

The best model is saved each time validation `nDCG@10` improves. Older best models are auto-deleted.

### Monitor progress
```bash
tail -f logs/train_torchrec_tuned_mydataset.log
grep "evaluation result" logs/train_torchrec_tuned_mydataset.log
```

---

## 7. Evaluation on Test Set

After training, evaluate the saved checkpoint:

```bash
python evaluate_gsasrec.py \
    --config config_mydataset.py \
    --model_path models/torchrec-tuned-mydataset-best-step:...-metric:....pt \
    --split test
```

---

## 8. Quick Reference: Minimal Custom Dataset Setup

```bash
# 1. Prepare data
mkdir -p datasets/mydataset/{train,val,test}
cp my_train_sequences.txt datasets/mydataset/train/input.txt
cp my_val_histories.txt  datasets/mydataset/val/input.txt
cp my_val_targets.txt    datasets/mydataset/val/output.txt
cp my_test_histories.txt datasets/mydataset/test/input.txt
cp my_test_targets.txt   datasets/mydataset/test/output.txt
echo '{"num_items": 5000}' > datasets/mydataset/dataset_stats.json

# 2. Create config
cp config_ml1m.py config_mydataset.py
# edit dataset_name='mydataset' and tune hyperparameters

# 3. Train
python train_gsasrec_torchrec_tuned.py --config config_mydataset.py \
    --use_cuda_init --init_type xavier --emb_lr 0.0003 --dense_lr 0.001

# 4. Check results
grep "FINAL BEST METRIC" logs/train_torchrec_tuned_*.log
```
