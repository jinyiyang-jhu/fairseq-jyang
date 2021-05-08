# Data and model paths
src="zh"
tgt="en"
bpe_type="sentencepiece"
detok=false
data_dir=/exp/jyang1/exp/gale_man_mt/data
bpe_dir=/exp/jyang1/exp/gale_man_mt/data/preprocessed/bpe
exp_dir=/exp/jyang1/exp/gale_man_mt/exp/zh-en-spm-30k

# Model parameters
arch="transformer"
task="translation"

share_embeddings="" # if want to use this option, add "--share-all-embeddings" in the train script.
train_num_workers=40
decode_num_workers=10
save_interval=2
#keep_last_epochs=30

# Encoder
encoder_layers=4
encoder_embed_dim=512
encoder_ffn_embed_dim=1024
encoder_attention_heads=8

# Decoder
decoder_layers=4
decoder_embed_dim=512
decoder_ffn_embed_dim=1024
decoder_attention_heads=8

# Regularization
dropout=0.3
activation_dropout=0.0
label_smoothing=0.1
weight_decay=0.0001
clip_norm=0.0


# Training parameters
#batch_size=256
update_freq=4 # gradient accumulation for every N_i batches
data_buffer_size=4
max_tokens=1024
optimizer="adam"
lr_scheduler="inverse_sqrt"
warmup_init_lr=0.0002
warmup_updates=8000
lr=0.001
min_lr=1e-5

# Others
adam_betas="(0.9 0.98)"
patience=0
max_epoch=100
criterion="label_smoothed_cross_entropy"
curriculum=0 # don't shuffle the first N epochs

