
# model parameters
arch="transformer"
task="translation"
share_embeddings="" # if want to use this option, add "--share-all-embeddings" in the train script.
train_num_workers=40
decode_num_workers=10
save_interval=5
#keep_last_epochs=30

# encoder
encoder_layers=6
encoder_embed_dim=2048
encoder_ffn_embed_dim=256
encoder_attention_heads=4

# decoder
decoder_layers=6
decoder_embed_dim=2048
decoder_ffn_embed_dim=256
decoder_attention_heads=4

# training parameters
optimizer="adam"
lr_scheduler="inverse_sqrt"
update_freq=1 # gradient accumulation for every N_i batches
clip_norm=5
patience=0
dropout=0.1

max_epoch=100
lr=0.0005
init_lr=1e-5
min_lr=1e-07
warmup_updates=8000
weight_decay=0.0001
#length_norm_loss=
#batch_size=96 # for BPE
max_tokens=4000
curriculum=0 # don't shuffle the first N epochs

criterion="label_smoothed_cross_entropy"
label_smoothing=0.1

# Transformer related
transformer_attn_dropout=0.1
#transformer_init=