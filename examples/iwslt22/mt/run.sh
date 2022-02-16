#!/bin/bash

stage=-1
nj_preprocess=4
ngpus=4

src_lan="ta"
tgt_lan="en"
tgt_case="tc"

bpedir=/exp/jyang1/data/IWSLT22/TA_AST/st/spm_bpe
destdir=exp
datadir_ori=/exp/jyang1/data/IWSLT22/TA_AST/st
trainpref=${bpedir}/train.bpe.${src_lan}-${tgt_lan}
validpref=${bpedir}/dev.bpe.${src_lan}-${tgt_lan}
testpref=${bpedir}/test1.bpe.${src_lan}-${tgt_lan}
bindir=$destdir/bin_${src_lan}2${tgt_lan}
testset_name="test1"
conf=conf/conf_ta_en.sh

. path.sh
. cmd.sh
. parse_options.sh

if [ $stage -le 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Preprocessing data"
    fairseq-preprocess \
        --source-lang $src_lan --target-lang $tgt_lan \
        --trainpref $trainpref --validpref $validpref --testpref $testpref \
        --destdir ${bindir} --thresholdtgt 0 --thresholdsrc 0 \
        --workers $nj_preprocess || exit 1;
fi

if [ $stage -le 1 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') fairseq-train for ${src_lan}-${tgt_lan}"
    mkdir -p $destdir/log || exit 1
    cp $conf $destdir
    $cuda_cmd --gpu $ngpus ${destdir}/log/train.log \
        fairseq-train ${bindir} \
        -s ${src_lan} \
        -t ${tgt_lan} \
        --num-workers 4 \
        --task translation \
        --arch $arch \
        --encoder-layers $encoder_layers \
        --encoder-embed-dim $encoder_embed_dim \
        --encoder-hidden-size $encoder_hidden_size \
        --encoder-bidirectional \
        --decoder-layers $decoder_layers \
        --decoder-embed-dim $decoder_embed_dim \
        --decoder-hidden-size $decoder_hidden_size \
        --tensorboard-logdir ${destdir}/tensorboard-log \
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
        --warmup-updates $warmup_updates \
        --weight-decay $weight_decay \
        --batch-size $batch_size \
        --curriculum $curriculum \
        --criterion $criterion \
        --label-smoothing $label_smoothing \
        --save-dir ${destdir}/checkpoints \
        --save-interval $save_interval \
        --log-format json || exit 1
    echo "$(date '+%Y-%m-%d %H:%M:%S') Training succeed !"
fi

decode_mdl="checkpoint_best"
if [ $stage -le 2 ]; then
    decode_dir=${destdir}/decode_${testset_name}_${decode_mdl}
    awk '{print $1}' ${datadir_ori}/${testset_name}/text.${tgt_case}.${tgt_lan} > ${bindir}/${testset_name}.uttid || exit 1;
    echo "$(date '+%Y-%m-%d %H:%M:%S') fairseq-interactive for ${src_lan}-${tgt_lan}:${src_lan}"
    mkdir -p $decode_dir || exit 1
    $cuda_cmd --gpu 1 --mem 8G $decode_dir/log/decode.log \
    cat ${bpedir}/${testset_name}.bpe.${src_lan}-${tgt_lan}.${src_lan} | fairseq-interactive \
        --lang-pairs "${src_lan}-${tgt_lan}" \
        --source-lang "${src_lan}" --target-lang "${tgt_lan}" \
        --task translation \
        --path ${destdir}/checkpoints/${decode_mdl}.pt \
        --buffer-size 2000
        --beam 5 \
        --remove-bpe=sentencepiece \
        > $decode_dir/results_${decode_mdl}.txt || exit 1
    echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding done !"
fi
    