# model parameters
arch="transformer"
task="translation"

share_embeddings="" # if want to use this option, add "--share-all-embeddings" in the train script.
train_num_workers=4
decode_num_workers=4
save_interval=2
#keep_last_epochs=30

# encoder
encoder_layers=6
encoder_embed_dim=512
encoder_ffn_embed_dim=2048
encoder_attention_heads=8

# decoder
decoder_layers=6
decoder_embed_dim=512
decoder_ffn_embed_dim=2048
decoder_attention_heads=8

# training parameters
optimizer="adam"
#adam_betas='(0.9 0.98)'
adam_eps="1e-9"
update_freq=1 # gradient accumulation for every N_i batches
clip_norm=0.1
patience=0

max_epoch=50
lr_scheduler="inverse_sqrt"
lr=5e-4
init_lr=1e-7
min_lr=1e-05
warmup_updates=4000
dropout=0.3
activation_dropout=0.1
attention_dropout=0.1
weight_decay=0.0001
#length_norm_loss=
#batch_size=128
max_tokens=20000
update_freq=2
criterion="label_smoothed_cross_entropy"
label_smoothing=0.1

curriculum=0 # don't shuffle the first N epochs

