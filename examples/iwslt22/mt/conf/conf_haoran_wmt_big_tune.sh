# model parameters
arch="transformer_vaswani_wmt_en_de_big"
task="translation"

share_embeddings="" # if want to use this option, add "--share-all-embeddings" in the train script.
train_num_workers=4
decode_num_workers=4
save_interval=5
#keep_last_epochs=30

# encoder
encoder_layers=6
encoder_embed_dim=1024
encoder_ffn_embed_dim=4096
encoder_attention_heads=16

# decoder
decoder_layers=6
decoder_embed_dim=1024
decoder_ffn_embed_dim=4096
decoder_attention_heads=16

# training parameters
optimizer="adam"
#adam_betas='(0.9 0.98)'
adam_eps="1e-9"
update_freq=1 # gradient accumulation for every N_i batches
clip_norm=0.1
patience=0

max_epoch=75
lr_scheduler="inverse_sqrt"
lr=1e-4
init_lr=1e-7
min_lr=1e-05
warmup_updates=2000
dropout=0.3
activation_dropout=0.1
attention_dropout=0.1
weight_decay=0.00002
#length_norm_loss=
#batch_size=128
max_tokens=2048
update_freq=4
criterion="label_smoothed_cross_entropy"
label_smoothing=0.1

curriculum=0 # don't shuffle the first N epochs

