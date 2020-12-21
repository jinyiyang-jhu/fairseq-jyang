#!/bin/bash

stage=-1
ngpus=4

lat_process_stage=2
lat_prune="true"
src="es"
tgt="en"
preprocess_num_workers=40
lat_depth_thres=85
beam=4
acwt=0.1
lmwt=1.0
lat_conf=conf/lat_transformer_bpe_Nov22.sh
lat_trans_mdl_dir="exp/mt_lat_conf_Dec12"
fairseq_bin_lat=$lat_trans_mdl_dir/bpe_bin

dsets=("fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest" "train")
fairseq_dsets=("valid" "test" "test1" "test2" "test3" "train")

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

train_text="espnet_data/data/lang_1spm/input.txt"  # for training BPE models, using both es and en text
glossaries="local/glossaries.txt" # a list of words to skip the BPE the tokenizations
bpe_dict_dir="exp/bpe_subword_nmt" # where the BPE dictionary and vocab are
bpe_dir="exp/bpe_subword_nmt/gold" # a directory containing trained BPE vocabs and codes
espnet_datadir="espnet_data/data" # original data prepared by ESPNET
data_utt_dir="data/uttids" # a directory containing manually select utterance lists
text_dir="data/text" # sequential data (transcripts, 1best, translation) used for training Transformers
plf_dir="data/plf_pruned_beam_${beam}_acwt_${acwt}_lmwt_${lmwt}_depth_thres_${lat_depth_thres}"
fairseq_bin_gold="exp/mt_gold/bpe_bin"

if [ $stage -le 0 ]; then
  # Train BPE model with train text
  echo "$(date) Training BPE model with subword-nmt"
  mkdir -p $bpe_dict_dir || exit 1;
  cut -d " " -f2- $train_text > $bpe_dict_dir/input.txt
  local/train_bpe_subword_nmt.sh $bpe_dict_dir/input.txt $glossaries $bpe_dict_dir
fi

if [ $stage -le 1 ]; then
  # Perform BPE tokenization of gold transcripts and translations
  [ -d $bpe_dir ] || mkdir -p $bpe_dir || exit 1;
  for idx in $(seq 0 $((${#dsets[@]}-1))); do
    d=${dsets[$idx]}
    fd=${fairseq_dsets[idx]}
    [ -d $text_dir/$d.$src ] || mkdir -p $text_dir/$d.$src || exit 1;
    [ -d $text_dir/$d.$tgt ] || mkdir -p $text_dir/$d.$tgt || exit 1;

    echo "$(date) Processing BPE tokenization for dataset (gold): $d.$src"
    local/filter_scp.sh $data_utt_dir/$d.uttids $espnet_datadir/$d.$src/text.lc.rm $text_dir/$d.$src/text
    local/filter_scp.sh $data_utt_dir/$d.uttids $espnet_datadir/$d.$tgt/text.lc.rm $text_dir/$d.$tgt/text

    source_file=$text_dir/$d.$src/text
    target_file=$text_dir/$d.$tgt/text
    sed -i 's/\&apos\;/ \&apos\; /g' $source_file
    sed -i 's/\&apos\;/ \&apos\; /g' $target_file
    # For 1best paths which may contain <unk> or apostrophe
    # sed -i "s/<unk>//g; s/\&apos\;/ \&apos\; /g; s/'/ \&apos\; /g;" $1best

    cut -f 2- -d " " $source_file |\
        subword-nmt apply-bpe -c $bpe_dict_dir/code.txt \
            --vocabulary $bpe_dict_dir/vocab.all.txt \
            --glossaries $(cat ${bpe_dict_dir}/glossaries.txt | tr -s '\n' ' ') \
            --vocabulary-threshold 1 > $bpe_dir/$d.$src
    
    cut -f 2- -d " " $target_file |\
        subword-nmt apply-bpe -c $bpe_dict_dir/code.txt \
            --vocabulary $bpe_dict_dir/vocab.all.txt \
            --glossaries $(cat ${bpe_dict_dir}/glossaries.txt | tr -s '\n' ' ') \
            --vocabulary-threshold 1 > $bpe_dir/$d.$tgt

    num_src=$(wc -l $bpe_dir/$d.$src | cut -d " " -f1)
    num_tgt=$(wc -l $bpe_dir/$d.$tgt | cut -d " "  -f1)
    if [ $num_src -ne $num_tgt ]; then
        echo "Number of src utterances and number of target utterances differ !" && exit 1;
    fi
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
      --destdir $fairseq_bin_gold || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "$(date) lattice preprocess"
  local/lattice_preprocess/run_lattice_preprocess.sh --stage $lat_process_stage $fairseq_bin_lat || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$(date) fairseq-preprocess for lattice transformer targets"
  tgt_dir=$plf_dir/sorted_text
  mkdir -p $tgt_dir/text
  mkdir -p $tgt_dir/bpe
  for idx in $(seq 0 $((${#dsets[@]}-1))); do
    d=${dsets[$idx]}
    fd=${fairseq_dsets[idx]}
    local/filter_scp.sh $fairseq_bin_lat/$fd.uttid $espnet_datadir/$d.$tgt/text.lc.rm $tgt_dir/text/$d.$tgt || exit 1;
    sed -i 's/\&apos\;/ \&apos\; /g' $tgt_dir/text/$d.$tgt
    cut -f 2- -d " " $tgt_dir/text/$d.$tgt | \
      subword-nmt apply-bpe -c $bpe_dict_dir/code.txt \
            --vocabulary $bpe_dict_dir/vocab.all.txt \
            --glossaries $(cat ${bpe_dict_dir}/glossaries.txt | tr -s '\n' ' ') \
            --vocabulary-threshold 1 > $tgt_dir/bpe/$d.$tgt
  done

  fairseq-preprocess --source-lang $src --target-lang $tgt \
      --only-target \
      --append-eos-tgt \
      --joined-dictionary \
      --workers $preprocess_num_workers \
      --tgtdict $fairseq_bin_gold/dict.$tgt.txt \
      --trainpref $tgt_dir/bpe/train \
      --validpref $tgt_dir/bpe/fisher_dev \
      --testpref $tgt_dir/bpe/fisher_dev2,$tgt_dir/bpe/fisher_test,$tgt_dir/bpe/callhome_devtest,$tgt_dir/bpe/callhome_evltest \
      --destdir $fairseq_bin_lat || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$(date) Train lattice transformer"
  local/train_lattice_transformer.sh $lat_conf $lat_trans_mdl_dir
fi
