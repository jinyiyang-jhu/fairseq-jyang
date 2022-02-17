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
testset_name="test" # "valid" for dev; "test" for test1
conf=conf/conf_ta_en.sh

. path.sh
. cmd.sh
. parse_options.sh
source $conf

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
    #$cuda_cmd --gpu $ngpus ${destdir}/log/train.log \
    qsub -v PATH -S /bin/bash -b y -q gpu.q -cwd -j y -N fairseq_train \
        -l gpu=$ngpus,num_proc=4,mem_free=64G,h_rt=600:00:00 \
        -o ${destdir}/log/train.log -sync y -m ea -M jyang126@jhu.edu \
        fairseq-train ${bindir} \
        -s ${src_lan} \
        -t ${tgt_lan} \
        --num-workers 4 \
        --task translation \
        --arch $arch \
        --encoder-layers $encoder_layers \
        --encoder-embed-dim $encoder_embed_dim \
        --encoder-ffn-embed-dim $encoder_hidden_size \
        --encoder-attention-heads $encoder_attention_heads \
        --decoder-layers $decoder_layers \
        --decoder-embed-dim $decoder_embed_dim \
        --decoder-ffn-embed-dim $decoder_hidden_size \
        --decoder-attention-heads $decoder_attention_heads \
        --tensorboard-logdir ${destdir}/tensorboard-log \
        --activation-fn relu \
        --optimizer $optimizer \
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
        --adam-betas '(0.9, 0.98)' \
        --label-smoothing $label_smoothing \
        --save-dir ${destdir}/checkpoints \
        --save-interval $save_interval \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --log-format json || exit 1
    echo "$(date '+%Y-%m-%d %H:%M:%S') Training succeed !"
fi

decode_mdl="checkpoint_best"
if [ $stage -le 2 ]; then
    decode_dir=${destdir}/decode_${testset_name}_${decode_mdl}
    echo "$(date '+%Y-%m-%d %H:%M:%S') fairseq-interactive for ${src_lan}-${tgt_lan}:${src_lan}"
    mkdir -p $decode_dir || exit 1
    #$cuda_cmd --gpu 1 --mem 8G $decode_dir/log/decode.log \
   # cat ${bpedir}/${testset_name}.bpe.${src_lan}-${tgt_lan}.${src_lan} \
   [ -f ${decode_dir}/logs/decode.log ] && rm ${decode_dir}/logs/decode.log
    qsub -v PATH -S /bin/bash -b y -q gpu.q -cwd -j y -N fairseq_interactive \
        -l gpu=1,num_proc=10,mem_free=16G,h_rt=600:00:00 \
        -o ${decode_dir}/logs/decode.log -sync y -m ea -M jyang126@jhu.edu \
        fairseq-generate ${bindir} \
            --source-lang "${src_lan}" --target-lang "${tgt_lan}" \
            --task translation \
            --tokenizer moses \
            --path ${destdir}/checkpoints/${decode_mdl}.pt \
            --batch-size 128 \
            --beam 5 \
            --gen-subset $testset_name \
            --remove-bpe=sentencepiece || exit 1
    grep ^D $decode_dir/logs/decode.log | LC_ALL=C sort -V | cut -f3 > $decode_dir/hyp.txt || exit 1
    grep ^T $decode_dir/logs/decode.log | LC_ALL=C sort -V | cut -f2- > $decode_dir/ref.txt || exit 1
    sacrebleu $decode_dir/ref.txt -i $decode_dir/hyp.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
    echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding done !"
fi
    