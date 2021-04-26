#!/bin/bash

stage=-1
ngpus=4
preprocess_num_workers=40
decode_mdl="checkpoint_best"
#bpe_type="@@ "
#generate_bsz=128

. path.sh
. parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
    echo "Usage: $0 <train-configuration>"
    exit 1;
fi

conf=$1
source $conf
bin_dir=$exp_dir/bpe_bin

if [ $stage -le 0 ]; then
  # fairseq-preprocess on gold transcript and translations
  echo "$(date '+%Y-%m-%d %H:%M:%S') fairseq-preprocess for $src-$tgt"
  fairseq-preprocess --source-lang $src --target-lang $tgt \
      --trainpref $bpe_dir/train.bpe \
      --validpref $bpe_dir/dev.bpe \
      --testpref $bpe_dir/test.bpe,$bpe_dir/sharedeval-bc.bpe,$bpe_dir/sharedeval-bn.bpe \
      --destdir $bin_dir \
      --workers $preprocess_num_workers || exit 1;
fi

if [ $stage -le 1 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') fairseq-train for $src-$tgt"
    mkdir -p $exp_dir/logs || exit 1
    cp $conf $exp_dir
    qsub -v PATH -S /bin/bash -b y -q gpu.q -cwd -j y -N fairseq_train \
        -l gpu=$ngpus,num_proc=$train_num_workers,mem_free=64G,h_rt=600:00:00 \
        -o $exp_dir/logs/train.log -sync y -m ea -M jyang126@jhu.edu \
        fairseq-train $bin_dir \
        --source-lang $src \
        --target-lang $tgt \
        --num-workers $train_num_workers \
        --task $task \
        --arch $arch \
        --encoder-layers $encoder_layers \
        --encoder-embed-dim $encoder_embed_dim \
        --encoder-ffn-embed-dim $encoder_ffn_embed_dim \
        --encoder-attention-heads $encoder_attention_heads \
        --decoder-layers $decoder_layers \
        --decoder-embed-dim $decoder_embed_dim \
        --decoder-ffn-embed-dim $decoder_ffn_embed_dim \
        --decoder-attention-heads $decoder_attention_heads \
        --max-tokens $max_tokens \
        --data-buffer-size $data_buffer_size \
        --optimizer $optimizer \
        --warmup-init-lr $warmup_init_lr \
        --warmup-updates $warmup_updates \
        --lr $lr \
        --lr-scheduler $lr_scheduler \
        --update-freq $update_freq \
        --clip-norm $clip_norm \
        --dropout $dropout \
        --activation-dropout $activation_dropout \
        --label-smoothing $label_smoothing \
        --weight-decay $weight_decay \
        --curriculum $curriculum \
        --max-epoch $max_epoch \
        --criterion $criterion \
        --save-dir $exp_dir/checkpoints \
        --save-interval $save_interval \
        --log-format json \
        --tensorboard-logdir $exp_dir/tensorboard-log || exit 1
fi

# if [ $stage -le 2 ]; then
#     decode_dir=$exp_dir/decode_dev_${decode_mdl}
#     awk '{print $1}' $text_dir/$dset_name.en/text > $bin_dir/$dset.uttid || exit 1;
#     echo "$(date '+%Y-%m-%d %H:%M:%S') fairseq-interactive for $src-$tgt:$src"
#     mkdir -p $decode_dir || exit 1
#     qsub -v PATH -S /bin/bash -b y -q gpu.q -cwd -j y -N fairseq_interactive \
#     -l gpu=1,num_proc=10,mem_free=16G,h_rt=600:00:00 \
#     -o $decode_dir/logs/decode.log -sync y -m ea -M jyang126@jhu.edu \
#     cat $bpe_dir/dev."$src" | fairseq-interactive \
#         --lang-pairs "$src-$tgt" \
#         --source-lang "$src" --target-lang "$tgt" \
#         --task $task \
#         --path $exp_dir/checkpoints/${decode_mdl}.pt \
#         --buffer-size 2000
#         --batch-size $generate_bsz \
#         --beam 5 \
#         --remove-bpe "$bpe_type" \
#         --num-workers $decode_num_workers \
#         > $decode_dir/results_${decode_mdl}.txt || exit 1
# fi