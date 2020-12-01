#!/bin/bash
# This script trains a MT model with transformer structure on gold translation pairs

stage=2
ngpus=4
decode_mdl="checkpoint_best"
bpe_type="@@ "
generate_bsz=8

conf="conf/mt_gold_transformer_bpe.sh"
exp_dir=exp/kaldi_1best_mt_subword_nmt
bin_dir=exp/kaldi_1best_mt_subword_nmt/bpe_bin

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

conf=$1
exp_dir=$2
bin_dir=$exp_dir/bpe_bin
. $conf

# Dataset dir/names
src_lan="es"
tgt_lan="en"
original_datadir=data/espnet_prepared
orginal_bpedir=data/gold_mt/bpe
original_dsets=("fisher_dev" "fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest")
dsets=("valid" "test" "test1" "test2" "test3" "test4")
# BPE related path
nbpe=1000
case="lc.rm"
bpe_train_text=exp/espnet_bpe_model/input.txt
bpe_code_dir=exp/bpe_es_en_lc_subword_nmt
non_lan_syms=data/lang/en_es_non_lang_syms_lc.txt


#if [ $stage -le 0 ]; then
#    echo "$(date) => training BPE model"
#    bash local/train_bpe_subword_nmt.sh $bpe_train_text $non_lan_syms $bpe_model_dir || exit 1;
#fi
#
#if [ $stage -le 1 ]; then
#    echo "$(date) => preprocessing datasets"
#    bash local/preprocess_subword_nmt_text.sh $bpe_code_dir $original_datadir $orginal_bpedir $exp_dir
#fi

if [ $stage -le 2 ]; then
    echo "$(date) => training transfromer model"
    mkdir -p $exp_dir/log || exit 1
    cp $conf $exp_dir
    $cuda_cmd --gpu $ngpus $exp_dir/log/train.log \
        fairseq-train $bin_dir \
        -s $src_lan \
        -t $tgt_lan \
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

if [ $stage -le 3 ]; then
    for idx in $(seq 0 $((${#dsets[@]}-1))); do
        dset=${dsets[$idx]}
        dset_name=${original_dsets[idx]}
        decode_dir=$exp_dir/decode_${dset_name}_${decode_mdl}
        echo "$(date) => decoding $dset_name with $exp_dir/checkpoints/${decode_mdl}.pt"
        mkdir -p $decode_dir || exit 1
        $cuda_cmd --gpu 1 --mem 4G $decode_dir/log/decode.log \
         fairseq-generate $bin_dir \
            -s $src_lan \
            -t $tgt_lan \
            --task $task \
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
