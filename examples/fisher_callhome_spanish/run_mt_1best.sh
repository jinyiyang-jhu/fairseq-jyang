#!/bin/bash

stage=1
ngpus=4

src="es"
tgt="en"
preprocess_src="true"
preprocess_tgt="true"
preprocess_num_workers=40
#conf=conf/lat_transformer_bpe_Nov22.sh
#exp_dir="exp/mt_gold"
#bin_dir=$exp_dir/bpe_bin

dsets=("fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest" "train")
fairseq_dsets=("valid" "test" "test1" "test2" "test3" "train")

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

espnet_ref_dir="espnet_data/data/"
train_text="espnet_data/data/lang_1spm/input.txt"  # for training BPE models, using both es and en text
glossaries="local/glossaries.txt" # a list of words to skip the BPE the tokenizations
bpe_dict_dir="exp/bpe_subword_nmt" # where the BPE dictionary and vocab are
bpe_dir="exp/bpe_subword_nmt/1best" # a directory containing trained BPE vocabs and codes
espnet_datadir="espnet_data/data" # original data prepared by ESPNET
data_utt_dir="data/uttids" # a directory containing manually select utterance lists
src_text_dir="data/1best"
tgt_text_dir="data/text" # sequential data (transcripts, 1best, translation) used for training Transformers

decode_mdl="checkpoint_best"
bpe_type="@@ "
generate_bsz=10

conf=$1
exp_dir=$2
bin_dir=$exp_dir/bpe_bin
. $conf

mkdir -p $bin_dir || exit 1;
if [ $stage -le 0 ]; then
  # Train BPE model with train text
  echo "$(date) Getting 1best paths"
  bash local/lattice_preprocess/get_lattice_1best.sh || exit 1;
fi

if [ $stage -le 1 ]; then
  # Perform BPE tokenization of 1best  and translations
  [ -d $bpe_dir ] || mkdir -p $bpe_dir || exit 1;
  for idx in $(seq 0 $((${#dsets[@]}-1))); do
    d=${dsets[$idx]}
    fd=${fairseq_dsets[idx]}
    if [ $preprocess_src == "true" ]; then
      echo "$(date) BPE tokenization for $d.$src"
      [ -d $tgt_text_dir/$d.$src ] || mkdir -p $text_dir/$d.$src || exit 1;
      source_file=$src_text_dir/$d.$src/text
      sed -i 's/\&apos\;/ \&apos\; /g' $source_file
      cut -f 2- -d " " $source_file |\
        subword-nmt apply-bpe -c $bpe_dict_dir/code.txt \
            --vocabulary $bpe_dict_dir/vocab.all.txt \
            --glossaries $(cat ${bpe_dict_dir}/glossaries.txt | tr -s '\n' ' ') \
            --vocabulary-threshold 1 > $bpe_dir/$d.$src
    fi

    if [ $preprocess_tgt == "true" ]; then
      echo "$(date) BPE tokenization for $d.$tgt"
      [ -d $tgt_text_dir/$d.$tgt ] || mkdir -p $tgt_text_dir/$d.$tgt || exit 1;
      local/filter_scp.sh $data_utt_dir/$d.uttids $espnet_datadir/$d.$tgt/text.lc.rm $tgt_text_dir/$d.$tgt/text
      target_file=$tgt_text_dir/$d.$tgt/text
      sed -i 's/\&apos\;/ \&apos\; /g' $target_file
      cut -f 2- -d " " $target_file |\
        subword-nmt apply-bpe -c $bpe_dict_dir/code.txt \
            --vocabulary $bpe_dict_dir/vocab.all.txt \
            --glossaries $(cat ${bpe_dict_dir}/glossaries.txt | tr -s '\n' ' ') \
            --vocabulary-threshold 1 > $bpe_dir/$d.$tgt
    fi
    num_src=$(wc -l $bpe_dir/$d.$src | cut -d " " -f1)
    num_tgt=$(wc -l $bpe_dir/$d.$tgt | cut -d " "  -f1)
    if [ $num_src -ne $num_tgt ]; then
        echo "Number of src utterances and number of target utterances differ !" && exit 1;
    fi
    awk '{print $1" "NR-1}' $src_text_dir/$d.$src/text > $bin_dir/$fd.uttid || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  # fairseq-preprocess on gold transcript and translations
  echo "$(date) fairseq-preprocess for gold transcripts and translations"
  fairseq-preprocess --source-lang $src --target-lang $tgt \
      --append-eos-src --append-eos-tgt \
      --joined-dictionary \
      --workers $preprocess_num_workers \
      --trainpref $bpe_dir/train \
      --validpref $bpe_dir/fisher_dev \
      --testpref $bpe_dir/fisher_dev,$bpe_dir/fisher_dev2,$bpe_dir/fisher_test,$bpe_dir/callhome_devtest,$bpe_dir/callhome_evltest \
      --destdir $bin_dir || exit 1;
fi

if [ $stage -le 3 ]; then
    echo "$(date) => training transfromer model"
    mkdir -p $exp_dir/log || exit 1
    cp $conf $exp_dir
    $cuda_cmd --gpu $ngpus $exp_dir/log/train.log \
        fairseq-train $bin_dir \
        -s $src \
        -t $tgt \
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

if [ $stage -le 4 ]; then
    for idx in $(seq 0 $((${#dsets[@]}-2))); do
        dset=${fairseq_dsets[$idx]}
        dset_name=${dsets[idx]}
        decode_dir=$exp_dir/decode_${dset_name}_${decode_mdl}
        echo "$(date) => decoding $dset_name with $exp_dir/checkpoints/${decode_mdl}.pt"
        mkdir -p $decode_dir || exit 1
        $cuda_cmd --gpu 1 --mem 8G $decode_dir/log/decode.log \
         fairseq-generate $bin_dir \
            -s $src \
            -t $tgt \
            --task $task \
            --gen-subset $dset \
            --path $exp_dir/checkpoints/${decode_mdl}.pt \
            --batch-size $generate_bsz \
            --remove-bpe "$bpe_type" \
            --num-workers $decode_num_workers \
            > $decode_dir/results_${decode_mdl}.txt || exit 1
        # echo "$(date) => scoring BLEU for $dset_name with MOSES tools"
        bash local/score_bleu_espnet.sh $decode_dir $bin_dir/$dset.uttid \
            $espnet_ref_dir/$dset_name.$tgt/ref.wrd.trn.detok.lc.rm \
            $espnet_ref_dir/$dset_name.$tgt/text.lc.rm
    done
fi