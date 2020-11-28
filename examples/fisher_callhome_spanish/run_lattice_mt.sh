#!/bin/bash
# This script trains a MT model with transformer structure on lattice translation pairs

stage=1
preprocess_stage=1
ngpus=4

conf="conf/lat_transformer_bpe_Nov16.sh"
exp_dir=exp/lat_mt_subword_nmt_Nov16
bin_dir=exp/lat_mt_subword_nmt/bpe_bin

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

conf=$1
exp_dir=$2
. $conf

# Dataset dir/names
original_dsets=("fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest")
dsets=("test" "test1" "test2" "test3" "test4")
#bpe_train_text=exp/espnet_bpe_model/input.txt
#bpe_code_dir=exp/bpe_es_en_lc_subword_nmt
#non_lan_syms=data/lang/en_es_non_lang_syms_lc.txt

if [ $stage -le 0 ]; then
    echo "$(date) => preprocessing source lattices and target words"
    bash local/preprocess_subword_nmt.sh --stage $preprocess_stage || exit 1;
fi

if [ $stage -le 1 ]; then
    echo "$(date) => training transfromer model with configuration: $conf"
    mkdir -p $exp_dir/log || exit 1
    cp $conf $exp_dir
    $cuda_cmd --gpu $ngpus $exp_dir/log/train.log \
        fairseq-train $bin_dir \
        --prepend-bos-tgt \
        --num-workers $train_num_workers \
        --skip-invalid-size-inputs-valid-test \
        --empty-cache-freq 1 \
        --task $task \
        --arch $arch \
        --encoder-layers $encoder_layers \
        --encoder-embed-dim $encoder_embed_dim \
        --encoder-ffn-embed-dim $encoder_ffn_embed_dim \
        --encoder-attention-heads $encoder_attention_heads \
        --encoder-attn-mask-scale $encoder_mask_scale \
        --decoder-layers $decoder_layers \
        --decoder-embed-dim $decoder_embed_dim \
        --decoder-ffn-embed-dim $decoder_ffn_embed_dim \
        --decoder-attention-heads $decoder_attention_heads \
        --tensorboard-logdir $exp_dir/tensorboard-log \
        --activation-fn relu \
        --optimizer $optimizer --adam-betas '(0.9, 0.98)' \
        --lr-scheduler $lr_scheduler \
        --update-freq $update_freq \
        --clip-norm $clip_norm \
        --patience $patience \
        --dropout $dropout \
        --max-epoch $max_epoch \
        --lr $lr \
        --warmup-init-lr $init_lr \
        --min-lr $min_lr \
        --warmup-updates $warmup_updates \
        --weight-decay $weight_decay \
        --max-tokens $max_tokens \
        --curriculum $curriculum \
        --criterion $criterion \
        --label-smoothing $label_smoothing \
        --attention-dropout $transformer_attn_dropout \
        --save-dir $exp_dir/checkpoints \
        --save-interval $save_interval \
        --log-format json || exit 1
fi

if [ $stage -le 2 ]; then
    for idx in $(seq 0 $((${#dsets[@]}-1))); do
        dset=${dsets[$idx]}
        dset_name=${original_dsets[idx]}
        decode_dir=$exp_dir/decode_${dset_name}_${decode_mdl}
        echo "$(date) => decoding $dset_name with $exp_dir/checkpoints/${decode_mdl}.pt"
        mkdir -p $decode_dir || exit 1
        $cuda_cmd --gpu 1 --mem 16G $decode_dir/log/decode.log \
         fairseq-generate-from-lattice $bin_dir \
            --task $task \
            --skip-invalid-size-inputs-valid-test \
            --gen-subset $dset \
            --path $exp_dir/checkpoints/${decode_mdl}.pt \
            --batch-size $generate_bsz \
            --remove-bpe "$bpe_type" \
            --num-workers $decode_num_workers \
            > $decode_dir/results_${decode_mdl}.txt || exit 1
        # echo "$(date) => scoring BLEU for $dset_name with MOSES tools"
        # TODO
    done
fi
