#!/bin/bash

# This script perform BPE tokenization for train/dev/eval data.
# The datasets are provided from ESPNET recipe.

stage=-1
source_lang="es"
target_lang="en"
case=".lc.rm"
preprocess_num_workers=40

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

target_text_dir=data/espnet_prepared
target_bpe_dir=data/common_espnet_kaldi
kaldi_lat_dir=data/kaldi_prepared/unselected
processed_lat_dir=data/common_espnet_kaldi
common_uttid_dir=data/common_espnet_kaldi
word_map=data/kaldi_prepared/lang/words.txt
bpe_code_dir=exp/bpe_${source_lang}_${target_lang}_lc_subword_nmt
fairseq_bpe_dict=exp/lat_mt_subword_nmt/bpe_bin/dict.es.txt
fairseq_bin_dir=exp/lat_mt_subword_nmt/bpe_bin

dsets=("fisher_dev" "fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest" "train")
fairseq_dsets=("valid" "test" "test1" "test2" "test3" "test4" "train")


# if [ $# -ne 4 ]; then
#     echo "Usage: $0 <bpe-code-dir> <input-data-dir> <output-bpe-dir> <exp-dir>"
#     echo "E.g.: $0 exp/bpe_es_en_lc_subword_nmt data/espnet_prepared data/gold_mt/bpe_subword_nmt exp/gold_mt_subword_nmt"
#     exit 1
# fi


if [ $stage -le 0 ]; then
    echo "$(date) Preprocess lattices for source dataset"
    bash local/lattice_preprocess/run_lattice_preprocess.sh --stage -1 \
        $kaldi_lat_dir $processed_lat_dir $common_uttid_dir $word_map \
        $bpe_code_dir $fairseq_bpe_dict $fairseq_bin_dir
fi

if [ $stage -le 1 ]; then
    for d in train fisher_dev fisher_dev fisher_dev2 fisher_test callhome_devtest callhome_evltest; do
        bpe_dir=$target_bpe_dir/$d.$target_lang/bpe_subword_nmt
        [ -d $bpe_dir ] || mkdir -p $bpe_dir || exit 1
        if [ -d $target_text_dir/$d.$target_lang ]; then
            echo "$(date) Processing BPE tokenization for target  text dataset: $d.$target_lang"
            input_file=$target_text_dir/$d.$target_lang/text${case}
            awk 'NR==FNR {a[$1]; next} $1 in a {print $0}' $fairseq_bin_dir/$d.uttid $input_file |\
            cut -f 2- -d " " $input_file | sed -e 's/\&apos\;/ \&apos\; /g' |\
                subword-nmt apply-bpe -c $bpe_code_dir/code.txt \
                    --vocabulary $bpe_code_dir/vocab.all.txt \
                    --glossaries "$(cat ${bpe_code_dir}/glossaries.txt)" \
                    --vocabulary-threshold 1 > $bpe_dir/$d.$target_lang || exit 1
        else
            echo "$target_text_dir/$d.$target_lang does not exist !" && exit 1;
        fi
        num_src=$(wc -l $fairseq_bin_dir/$d.uttid)
        num_tgt=$(wc -l $bpe_dir/$d.$target_lang)
        if [ $num_src -ne $num_tgt ]; then
            echo "Number of src files not equal to number of target files" && exit 1;
    done
fi

if [ $stage -le 2 ]; then
    echo "$(date) Fairseq preprocess for target dataset"
    for idx in $(seq 0 $((${#dsets[@]}-1))); do
    echo "$(date) Filtering target data"

    fairseq-preprocess --source-lang $source_lang --target-lang $target_lang \
        --trainpref $bpe_dir/train --validpref $bpe_dir/fisher_dev \
        --testpref $bpe_dir/fisher_dev,$bpe_dir/fisher_dev2,$bpe_dir/fisher_test,$bpe_dir/callhome_devtest,$bpe_dir/callhome_evltest \
        --destdir $fairseq_bin_dir --joined-dictionary --append-eos-src --append-eos-tgt \
        --workers $preprocess_num_workers
fi
