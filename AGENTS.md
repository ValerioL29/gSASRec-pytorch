# gSASRec-pytorch

This is an official PyTorch port of the **gSASRec** sequential recommendation model from the RecSys '23 paper *"gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling"*. The repository also supports vanilla SASRec training as a special case.

## Technology Stack

- **Language**: Python 3
- **Deep Learning Framework**: PyTorch
- **Key Dependencies**:
  - `torch` – model implementation and training
  - `ir_measures` – ranking metrics (nDCG@10, Recall@1, Recall@10)
  - `tqdm` – progress bars
  - `torchinfo` – model summaries
  - `numpy`, `requests` – data preprocessing utilities

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

The project uses a flat, script-based layout. There is no `pyproject.toml`, `setup.py`, or package build step.

| File / Directory | Purpose |
|------------------|---------|
| `config.py` | Base configuration class (`GSASRecExperimentConfig`) with all hyperparameters and defaults. |
| `config_ml1m.py` | gSASRec config for MovieLens-1M (256 negatives per positive, `gbce_t=0.75`). |
| `config_ml1m_sasrec.py` | Vanilla SASRec config for MovieLens-1M (`negs_per_pos=1`, `gbce_t=0.0`). |
| `gsasrec.py` | Main model class (`GSASRec`). Transformer-decoder-based sequential recommender. |
| `transformer_decoder.py` | Custom `MultiHeadAttention` and `TransformerBlock` adapted from the original SASRec TensorFlow code. |
| `dataset_utils.py` | `SequenceDataset`, PyTorch `DataLoader` builders, and collate functions for training/validation/test. |
| `eval_utils.py` | `evaluate()` function that runs inference and computes metrics via `ir_measures`. |
| `utils.py` | Helper utilities: `load_config()`, `build_model()`, `get_device()`. |
| `train_gsasrec.py` | Training loop with Adam optimizer, gBCE loss computation, and early stopping. |
| `evaluate_gsasrec.py` | Standalone evaluation script that loads a checkpoint and reports test metrics. |
| `datasets/ml1m/` | Pre-split MovieLens-1M dataset (`train/`, `val/`, `test/`, `dataset_stats.json`). |
| `pre_trained/` | Provided checkpoints for both SASRec and gSASRec on MovieLens-1M. |
| `models/` | Checkpoints saved during training (gitignored, created automatically). |

## Build and Run Commands

There is no formal build system. The workflow is pure Python script execution.

### Training

```bash
python3 train_gsasrec.py --config=config_ml1m.py
```

During training, validation metrics are printed each epoch. The best checkpoint (by the configured `val_metric`) is saved to `models/`. Training stops automatically after `early_stopping_patience` epochs without improvement.

### Evaluation

```bash
python3 evaluate_gsasrec.py --config=config_ml1m.py --checkpoint <checkpoint_path>
```

Example with the provided pre-trained gSASRec checkpoint:

```bash
python3 evaluate_gsasrec.py --config=config_ml1m.py --checkpoint pre_trained/gsasrec-ml1m-step:86064-t:0.75-negs:256-emb:128-dropout:0.5-metric:0.1974453226738962.pt
```

Example with the provided pre-trained SASRec checkpoint:

```bash
python3 evaluate_gsasrec.py --config=config_ml1m_sasrec.py --checkpoint pre_trained/gsasrec-ml1m-step:47520-t:0.0-negs:1-emb:128-dropout:0.5-metric:0.1428058429831465.pt
```

## Code Organization and Module Divisions

- **Configuration**: Python files act as configs. `load_config()` dynamically imports the specified file and returns a `GSASRecExperimentConfig` instance.
- **Model** (`gsasrec.py` + `transformer_decoder.py`):
  - `GSASRec` embeds items and positions, passes them through stacked `TransformerBlock`s, and outputs sequence representations.
  - `get_output_embeddings()` supports both shared and separate item embedding tables (`reuse_item_embeddings` flag).
  - `get_predictions()` scores all items for the last position in a sequence and returns top-`k` recommendations.
- **Data** (`dataset_utils.py`):
  - Training data: sequences read from `datasets/<name>/train/input.txt`. Random negatives are sampled on the fly in the collate function.
  - Validation / Test data: read from `input.txt` (history) and `output.txt` (ground-truth next item). Already-rated items can be filtered at evaluation time.
  - Item IDs are assumed to range from `1` to `num_items`. Padding token is `num_items + 1`.
- **Training** (`train_gsasrec.py`):
  - Uses Adam optimizer.
  - Applies a numerically-stable gBCE loss via positive-logit transformation (uses `float64` internally for stability).
  - Early stopping based on validation metric improvement.
- **Evaluation** (`eval_utils.py`):
  - Generates ranked lists with `get_predictions()`.
  - Computes metrics via `ir_measures.calc_aggregate()`.

## Development Conventions

- **Language for comments/docs**: English.
- **Config-driven**: All experiments are controlled by a Python config file that instantiates `GSASRecExperimentConfig`. Do not hard-code hyperparameters in training scripts.
- **Dataset layout convention**: If adding a new dataset, create a folder under `datasets/<name>/` containing:
  - `dataset_stats.json` with `num_items`
  - `train/input.txt`
  - `val/input.txt` and `val/output.txt`
  - `test/input.txt` and `test/output.txt`
  - Files contain space-separated integers (one sequence per line in `input.txt`, one item ID per line in `output.txt`).
- **Checkpoints**: Saved automatically in `models/` with descriptive filenames that encode dataset, step, `gbce_t`, negatives per positive, embedding dim, dropout, and best metric.
- **No automated tests**: The repository does not contain a test suite (e.g., `pytest`). Validation is done by comparing evaluation metrics against reported paper results.
- **No linting/formatting config**: There is no `.pre-commit-config.yaml`, `setup.cfg`, `pyproject.toml`, or similar. Follow the existing Python style (PEP-8-ish, 4-space indentation).

## Security Considerations

- The dataset preprocessor (`datasets/ml1m/preprocess_ml1m.py`) downloads data from an external URL (`https://raw.githubusercontent.com/asash/ml1m-sas-mapping/main/ml-1m_sas.txt`). If modifying or running this script, verify the source.
- There is no input validation or sanitization on config file paths beyond basic `argparse`. Config files are executed via `importlib`, so only load trusted config files.
- No secrets, API keys, or credentials are present in the repository.

## Notes for Agents

- When modifying model logic, keep the custom `TransformerBlock` structure intact; it intentionally replicates the original SASRec/gSASRec TensorFlow behavior (e.g., shared linear projections across attention heads).
- If adding a new dataset, mirror the `ml1m` directory structure and update/create a config file accordingly.
- The `reuse_item_embeddings` flag determines whether the output layer reuses the input item embedding table. gSASRec experiments in the paper used a separate output embedding table (`reuse_item_embeddings=False`), while vanilla SASRec experiments may set it to `True`.
- Numerical stability is critical for the gBCE loss; the training script casts positive logits to `torch.float64` before sigmoid and power operations. Preserve this behavior when editing loss computation.
