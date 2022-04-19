#!/bin/bash

stage=-1
nj_preprocess=16
ngpus=4

src_lan="ta"
tgt_lan="en"
src_case="tc.rm"
tgt_case="tc"
nbpe=32000
tokenized="True"
# tgt_case="tc"

#bpedir=data/msa-en_processed/spm2000
data_feats="data/ta-en_ta_translated_en_true"
#data_feats="data/msa-ta_msa_translated_ta_true"
destdir="exp_ta-en_ta_translated_en_true_spm32k_ta_en_both_42M"
bpemodel_dir="data/ta-en_ta_translated_en_true/spm_32000_ta_en_both_42M"
bpedir=${bpemodel_dir}
bindir=$destdir/bin_${src_lan}2${tgt_lan}

train_set="train"
dev_set="dev"
test_set="test1"
trainpref=${bpedir}/${train_set}.bpe.${src_lan}-${tgt_lan}
validpref=${bpedir}/${dev_set}.bpe.${src_lan}-${tgt_lan}
testpref=${bpedir}/${test_set}.bpe.${src_lan}-${tgt_lan}
testset_name="test" # "valid" for dev; "test" for test1
conf=conf/conf_haoran_wmt_big.sh

. path.sh
. cmd.sh
. parse_options.sh
source $conf

for d in $destdir $bpedir; do
    [ -d $d ] && mkdir -p $d 
done

if [ $stage -le 0 ]; then
    for lan in "${src_lan}" "${tgt_lan}"; do
        if [ "${lan}" == "${src_lan}" ]; then
            lan_case=$src_case
        else
            lan_case=$tgt_case
        fi
        bpemodel=${bpemodel_dir}/${lan}/bpe.model

        for dset in "${train_set}" "${dev_set}" "${test_set}"; do
            text=${data_feats}/${dset}/text.${lan_case}.${lan}
            tok_text=${data_feats}/${dset}/text.${lan_case}.${lan}.tok

            if [ $tokenized == "False" ]; then
                echo "$(date -u) Tokenizing : ${dset}-${lan}"
                tokenizer.perl -l en -q -no-escape < $text > $tok_text || exit 1;
            else
                cut -d " " -f2- $text > $tok_text
            fi

            #text="${data_feats}/${dset}/text.${lan_case}.${lan}.tok"
            text=$tok_text
            echo "$(date -u) spm-encoding for ${dset}-${lan}"
            spm_encode \
                --model="${bpemodel}" \
                --output_format=piece \
                < $text \
                > ${bpedir}/${dset}.bpe.${src_lan}-${tgt_lan}.${lan} || exit 1;
        done
    done
fi

if [ $stage -le 1 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Preprocessing data"
    fairseq-preprocess \
        --source-lang ${src_lan} --target-lang ${tgt_lan} \
        --trainpref $trainpref --validpref $validpref --testpref $testpref \
        --destdir ${bindir} --thresholdtgt 0 --thresholdsrc 0 \
        --workers $nj_preprocess || exit 1;
fi

if [ $stage -le 2 ]; then
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
        --encoder-ffn-embed-dim $encoder_ffn_embed_dim \
        --encoder-attention-heads $encoder_attention_heads \
        --decoder-layers $decoder_layers \
        --decoder-embed-dim $decoder_embed_dim \
        --decoder-ffn-embed-dim $decoder_ffn_embed_dim \
        --decoder-attention-heads $decoder_attention_heads \
        --attention-dropout ${attention_dropout} \
        --tensorboard-logdir ${destdir}/tensorboard-log \
        --activation-fn relu \
        --activation-dropout ${activation_dropout} \
        --optimizer $optimizer \
        --adam-betas '(0.9, 0.98)' \
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
        --save-interval-updates 1000 \
        --keep-interval-updates 3 \
        --memory-efficient-fp16 \
        --skip-invalid-size-inputs-valid-test \
        --log-format json || exit 1
    echo "$(date '+%Y-%m-%d %H:%M:%S') Training succeed !"
fi

decode_mdl="checkpoint_best"
if [ $stage -le 3 ]; then
    decode_dir=${destdir}/decode_${testset_name}_${decode_mdl}
    echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding for ${src_lan}-${tgt_lan}:${src_lan}"
    mkdir -p $decode_dir || exit 1
   [ -f ${decode_dir}/logs/decode.log ] && rm ${decode_dir}/logs/decode.log
    # qsub -v PATH -S /bin/bash -b y -q gpu.q -cwd -j y -N fairseq_interactive \
    #     -l gpu=1,num_proc=10,mem_free=16G,h_rt=600:00:00 \
    #     -o ${decode_dir}/logs/decode.log -sync y -m ea -M jyang126@jhu.edu \
    ${decode_cmd} --gpu 1 ${decode_dir}/logs/decode.log \
        fairseq-generate ${bindir} \
            --source-lang "${src_lan}" --target-lang "${tgt_lan}" \
            --task translation \
            --tokenizer moses \
            --path ${destdir}/checkpoints/${decode_mdl}.pt \
            --batch-size 128 \
            --beam 5 \
            --gen-subset $testset_name \
            --skip-invalid-size-inputs-valid-test \
            --remove-bpe=sentencepiece || exit 1
    grep ^D $decode_dir/logs/decode.log | LC_ALL=C sort -V | cut -f3 > $decode_dir/hyp.txt || exit 1
    grep ^T $decode_dir/logs/decode.log | LC_ALL=C sort -V | cut -f2- > $decode_dir/ref.txt || exit 1
    sacrebleu $decode_dir/ref.txt -i $decode_dir/hyp.txt -m bleu -lc > ${decode_dir}/results.txt || exit 1
    echo "$(date '+%Y-%m-%d %H:%M:%S') Decoding done !"
fi
    