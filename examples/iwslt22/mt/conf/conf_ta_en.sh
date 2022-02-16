# model parameters
arch="transformer_iwslt_de_en"
task="translation"

share_embeddings="" # if want to use this option, add "--share-all-embeddings" in the train script.
train_num_workers=4
decode_num_workers=4
save_interval=1
#keep_last_epochs=30

# encoder
#encoder_layers=2
#encoder_embed_dim=256
#encoder_hidden_size=512

# decoder
#decoder_layers=2
#decoder_embed_dim=256
#decoder_hidden_size=512

# training parameters
optimizer="adam"
adam_betas="(0.9 0.98)"
update_freq=1 # gradient accumulation for every N_i batches
clip_norm=0.0
patience=0

max_epoch=30
lr_scheduler="inverse_sqrt"
lr=5e-4
init_lr=1e-7
min_lr=1e-05
warmup_updates=5000
dropout=0.3
weight_decay=0.0001
#length_norm_loss=
#batch_size=64
max_tokens=4096
criterion="label_smoothed_cross_entropy"
label_smoothing=0.1

curriculum=0 # don't shuffle the first N epochs

