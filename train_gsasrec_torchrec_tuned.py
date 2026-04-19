from argparse import ArgumentParser
import os

import torch
from utils import load_config, get_device
from gsasrec_torchrec_tuned import TorchRecGSASRecTuned
from dataset_utils import get_train_dataloader, get_num_items, get_val_dataloader
from tqdm import tqdm
from eval_utils import evaluate
from torchinfo import summary

models_dir = "models"
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config_ml1m.py')
parser.add_argument('--use_cuda_init', action='store_true')
parser.add_argument('--init_type', type=str, default='default', choices=['default', 'xavier', 'he'])
parser.add_argument('--emb_lr', type=float, default=0.001)
parser.add_argument('--dense_lr', type=float, default=0.001)
parser.add_argument('--grad_clip', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--use_sparse_adam', action='store_true')
parser.add_argument('--label', type=str, default='')
parser.add_argument('--max_epochs_tune', type=int, default=20)
args = parser.parse_args()

config = load_config(args.config)

num_items = get_num_items(config.dataset_name)
device = get_device()
model = TorchRecGSASRecTuned(
    num_items,
    sequence_length=config.sequence_length,
    embedding_dim=config.embedding_dim,
    num_heads=config.num_heads,
    num_blocks=config.num_blocks,
    dropout_rate=config.dropout_rate,
    use_cuda_init=args.use_cuda_init,
    init_type=args.init_type,
)

train_dataloader = get_train_dataloader(
    config.dataset_name,
    batch_size=config.train_batch_size,
    max_length=config.sequence_length,
    train_neg_per_positive=config.negs_per_pos,
)
val_dataloader = get_val_dataloader(
    config.dataset_name,
    batch_size=config.eval_batch_size,
    max_length=config.sequence_length,
)

# Build optimizer(s)
emb_params = []
for ec in [model.item_embedding_collection]:
    emb_params.extend(list(ec.parameters()))
if not model.reuse_item_embeddings:
    emb_params.extend(list(model.output_embedding_collection.parameters()))

dense_params = [p for n, p in model.named_parameters()
                if 'embedding_collection' not in n]

optimizers = []
if args.use_sparse_adam:
    optimizers.append(torch.optim.SparseAdam(emb_params, lr=args.emb_lr))
else:
    if args.weight_decay > 0:
        emb_wd = 0.0  # no weight decay on embeddings
        optimizers.append(torch.optim.AdamW(emb_params, lr=args.emb_lr, weight_decay=emb_wd))
    else:
        optimizers.append(torch.optim.Adam(emb_params, lr=args.emb_lr))

if args.weight_decay > 0:
    optimizers.append(torch.optim.AdamW(dense_params, lr=args.dense_lr, weight_decay=args.weight_decay))
else:
    optimizers.append(torch.optim.Adam(dense_params, lr=args.dense_lr))

def zero_grad():
    for opt in optimizers:
        opt.zero_grad()

def step():
    for opt in optimizers:
        opt.step()

batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))

best_metric = float("-inf")
best_model_name = None
step_counter = 0
steps_not_improved = 0

model = model.to(device)
summary(model, (config.train_batch_size, config.sequence_length), batch_dim=None)

log_file = f"logs/train_torchrec_tuned_{args.label}.log" if args.label else "logs/train_torchrec_tuned.log"
os.makedirs("logs", exist_ok=True)

for epoch in range(args.max_epochs_tune):
    model.train()
    batch_iter = iter(train_dataloader)
    pbar = tqdm(range(batches_per_epoch))
    loss_sum = 0
    for batch_idx in pbar:
        step_counter += 1
        positives, negatives = [tensor.to(device) for tensor in next(batch_iter)]
        model_input = positives[:, :-1]
        last_hidden_state, attentions = model(model_input)
        labels = positives[:, 1:]
        negatives = negatives[:, 1:, :]
        pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives], dim=-1)
        pos_neg_embeddings = model.get_output_embeddings_dense(pos_neg_concat)
        mask = (model_input != num_items + 1).float()
        logits = torch.einsum('bse,bsne->bsn', last_hidden_state, pos_neg_embeddings)
        gt = torch.zeros_like(logits)
        gt[:, :, 0] = 1

        alpha = config.negs_per_pos / (num_items - 1)
        t = config.gbce_t
        beta = alpha * ((1 - 1 / alpha) * t + 1 / alpha)

        positive_logits = logits[:, :, 0:1].to(torch.float64)
        negative_logits = logits[:, :, 1:].to(torch.float64)
        eps = 1e-10
        positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1 - eps)
        positive_probs_adjusted = torch.clamp(
            positive_probs.pow(-beta), 1 + eps, torch.finfo(torch.float64).max
        )
        to_log = torch.clamp(
            torch.div(1.0, (positive_probs_adjusted - 1)), eps, torch.finfo(torch.float64).max
        )
        positive_logits_transformed = to_log.log()
        logits = torch.cat([positive_logits_transformed, negative_logits], -1)
        loss_per_element = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none')
            .mean(-1)
            * mask
        )
        loss = loss_per_element.sum() / mask.sum()
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        step()
        zero_grad()
        loss_sum += loss.item()
        pbar.set_description(f"Epoch {epoch} loss: {loss_sum / (batch_idx + 1)}")

    evaluation_result = evaluate(
        model,
        val_dataloader,
        config.metrics,
        config.recommendation_limit,
        config.filter_rated,
        device=device,
    )
    msg = f"Epoch {epoch} evaluation result: {evaluation_result}"
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')

    if evaluation_result[config.val_metric] > best_metric:
        best_metric = evaluation_result[config.val_metric]
        model_name = (
            f"models/torchrec-tuned-{config.dataset_name}-{args.label}-step:{step_counter}"
            f"-t:{config.gbce_t}-negs:{config.negs_per_pos}-emb:{config.embedding_dim}"
            f"-dropout:{config.dropout_rate}-metric:{best_metric}.pt"
        )
        print(f"Saving new best model to {model_name}")
        with open(log_file, 'a') as f:
            f.write(f"Saving new best model to {model_name}\n")
        if best_model_name is not None:
            os.remove(best_model_name)
        best_model_name = model_name
        steps_not_improved = 0
        torch.save(model.state_dict(), model_name)
    else:
        steps_not_improved += 1
        msg2 = f"Validation metric did not improve for {steps_not_improved} steps"
        print(msg2)
        with open(log_file, 'a') as f:
            f.write(msg2 + '\n')
        if steps_not_improved >= config.early_stopping_patience:
            msg3 = f"Stopping training, best model was saved to {best_model_name}"
            print(msg3)
            with open(log_file, 'a') as f:
                f.write(msg3 + '\n')
            break

print(f"\nFINAL BEST METRIC: {best_metric}")
with open(log_file, 'a') as f:
    f.write(f"\nFINAL BEST METRIC: {best_metric}\n")
