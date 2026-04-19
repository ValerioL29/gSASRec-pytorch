from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='criteo',
    sequence_length=200,
    embedding_dim=64,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.2,
    negs_per_pos=64,
    gbce_t=0.75,
    early_stopping_patience=20,
)
