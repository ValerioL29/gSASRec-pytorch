import torch
from transformer_decoder import TransformerBlock
from torchrec import EmbeddingCollection, EmbeddingConfig, KeyedJaggedTensor


class TorchRecGSASRec(torch.nn.Module):
    """gSASRec model with TorchRec EmbeddingCollection for item embeddings."""

    def __init__(
        self,
        num_items,
        sequence_length=200,
        embedding_dim=256,
        num_heads=4,
        num_blocks=3,
        dropout_rate=0.5,
        reuse_item_embeddings=False,
    ):
        super().__init__()
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.embeddings_dropout = torch.nn.Dropout(dropout_rate)
        self.num_heads = num_heads
        self.reuse_item_embeddings = reuse_item_embeddings

        # TorchRec EmbeddingCollection for input item embeddings
        self.item_embedding_collection = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name="item_table",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_items + 2,
                    feature_names=["item_id"],
                )
            ],
            device=torch.device("cpu"),
        )

        self.position_embedding = torch.nn.Embedding(sequence_length, embedding_dim)

        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, embedding_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.seq_norm = torch.nn.LayerNorm(embedding_dim)

        if not self.reuse_item_embeddings:
            self.output_embedding_collection = EmbeddingCollection(
                tables=[
                    EmbeddingConfig(
                        name="item_table",
                        embedding_dim=embedding_dim,
                        num_embeddings=num_items + 2,
                        feature_names=["item_id"],
                    )
                ],
                device=torch.device("cpu"),
            )

    def _dense_to_kjt(self, dense_ids):
        """Convert a dense tensor of IDs to a KeyedJaggedTensor."""
        flat = dense_ids.flatten()
        lengths = torch.ones(flat.shape[0], dtype=torch.int32, device=flat.device)
        return KeyedJaggedTensor(
            keys=["item_id"],
            values=flat,
            lengths=lengths,
        )

    def _lookup_embeddings(self, dense_ids, embedding_collection):
        """Lookup embeddings via EmbeddingCollection and reshape back."""
        kjt = self._dense_to_kjt(dense_ids)
        jt = embedding_collection(kjt)["item_id"]
        values = jt.values()
        return values.view(*dense_ids.shape, self.embedding_dim)

    def get_output_embeddings(self):
        """Return the output embedding collection (used in training)."""
        if self.reuse_item_embeddings:
            return self.item_embedding_collection
        else:
            return self.output_embedding_collection

    def get_output_embeddings_dense(self, pos_neg_concat):
        """Batched embedding lookup for positive/negative candidates.

        Args:
            pos_neg_concat: (B, L, 1+N) tensor of item IDs.
        Returns:
            (B, L, 1+N, D) tensor of embeddings.
        """
        ec = self.get_output_embeddings()
        return self._lookup_embeddings(pos_neg_concat, ec)

    def get_all_output_embeddings(self, device):
        """Lookup embeddings for all items 0..num_items+1 for inference scoring."""
        all_ids = torch.arange(0, self.num_items + 2, device=device)
        ec = self.get_output_embeddings()
        return self._lookup_embeddings(all_ids, ec)

    def forward(self, input):
        seq = self._lookup_embeddings(input.long(), self.item_embedding_collection)
        mask = (input != self.num_items + 1).float().unsqueeze(-1)

        bs = seq.size(0)
        positions = torch.arange(seq.shape[1]).unsqueeze(0).repeat(bs, 1).to(input.device)
        pos_embeddings = self.position_embedding(positions)[:input.size(0)]
        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask

        attentions = []
        for block in self.transformer_blocks:
            seq, attention = block(seq, mask)
            attentions.append(attention)

        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions

    def get_predictions(self, input, limit, rated=None):
        with torch.no_grad():
            model_out, _ = self.forward(input)
            seq_emb = model_out[:, -1, :]
            output_embeddings = self.get_all_output_embeddings(input.device)
            scores = torch.einsum("bd,nd->bn", seq_emb, output_embeddings)
            scores[:, 0] = float("-inf")
            scores[:, self.num_items + 1 :] = float("-inf")
            if rated is not None:
                for i in range(len(input)):
                    for j in rated[i]:
                        scores[i, j] = float("-inf")
            result = torch.topk(scores, limit, dim=1)
            return result.indices, result.values
