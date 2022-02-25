#!/bin/bash

stage=-1
nj_preprocess=16
ngpus=4

src_lan="ar"
tgt_lan="en"
# tgt_case="tc"

bpedir=data/msa-en_processed/spm2000
#bpedir=/home/hltcoe/jyang1/tools/espnet/egs2/iwslt22_dialect/st1/data_clean/spm_bpe
destdir=exp_msa-en_bpe2000

trainpref=${bpedir}/train.bpe.${src_lan}-${tgt_lan}
validpref=${bpedir}/dev.bpe.${src_lan}-${tgt_lan}
testpref=${bpedir}/test.bpe.${src_lan}-${tgt_lan}
bindir=$destdir/bin_${src_lan}2${tgt_lan}
testset_name="valid" # "valid" for dev; "test" for test1
conf=conf/conf_msa_en.sh

. path.sh
. cmd.sh
. parse_options.sh
source $conf

mkdir -p $destdir || exit 1;

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
    ${rtx_cuda_cmd} --gpu ${ngpus} $destdir/log/train.log \
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
        --adam-eps $adam_eps \
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
        --max-tokens $max_tokens \
        --update-freq $update_freq \
        --curriculum $curriculum \
        --criterion $criterion \
        --label-smoothing $label_smoothing \
        --save-dir ${destdir}/checkpoints \
        --save-interval $save_interval \
        --memory-efficient-fp16 \
        --log-format json || exit 1
    echo "$(date '+%Y-%m-%d %H:%M:%S') Training succeed !"
fi

decode_mdl="checkpoint_best"
if [ $stage -le 2 ]; then
    decode_dir=${destdir}/decode_${testset_name}_${decode_mdl}
    echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding for ${src_lan}-${tgt_lan}:${src_lan}"
    mkdir -p $decode_dir || exit 1
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
    