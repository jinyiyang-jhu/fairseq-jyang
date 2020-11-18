#!/bin/bash

# This script perform BPE tokenization for train/dev/eval data.
# The datasets are provided from ESPNET recipe.

stage=-1
source_lang="es"
target_lang="en"

case=".lc.rm"
preprocess_num_workers=40
perform_bpe="true"

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

target_text_dir=data/espnet_prepared
target_bpe_dir=data/common_espnet_kaldi
kaldi_lat_dir=data/kaldi_prepared/unselected
processed_lat_dir=data/common_espnet_kaldi
common_uttid_dir=data/common_espnet_kaldi
word_map=data/kaldi_prepared/lang/words.txt
bpe_code_dir=exp/bpe_es_en_lc_subword_nmt
fairseq_bpe_dict_tgt=exp/lat_mt_subword_nmt/bpe_bin/dict.es.txt
fairseq_bpe_dict_src=exp/lat_mt_subword_nmt/bpe_bin/dict.es.txt

uttid_dir=exp/lat_mt_subword_nmt/bpe_bin
fairseq_bin_dir=exp/gold_mt_subword_nmt/bpe_bin

dsets=("fisher_dev" "fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest" "train")
fairseq_dsets=("valid" "test" "test1" "test2" "test3" "test4" "train")

if [ $stage -le 1 ]; then
    for idx in $(seq 0 $((${#dsets[@]}-1))); do
        d=${dsets[$idx]}
        fd=${fairseq_dsets[idx]}
        bpe_dir=$target_bpe_dir/bpe_subword_nmt
        [ -d $bpe_dir ] || mkdir -p $bpe_dir || exit 1
        if [ $perform_bpe == "true" ]; then
            for lan in $source_lang $target_lang; do
                if [ -d $target_text_dir/$d.$lan ]; then
                    echo "$(date) Processing BPE tokenization for target  text dataset: $d.$lan"
                    input_file=$target_text_dir/$d.$lan/text${case}
                    awk 'NR==FNR {a[$1]; next} $1 in a {print $0}' $uttid_dir/$fd.uttid $input_file |\
                        cut -f 2- -d " " | sed -e 's/\&apos\;/ \&apos\; /g' |\
                        subword-nmt apply-bpe -c $bpe_code_dir/code.txt \
                            --vocabulary $bpe_code_dir/vocab.all.txt \
                            --glossaries "$(cat ${bpe_code_dir}/glossaries.txt)" \
                            --vocabulary-threshold 1 > $bpe_dir/$d.$lan || exit 1
                else
                    echo "$target_text_dir/$d.$lan does not exist !" && exit 1;
                fi
                num_src=$(wc -l $uttid_dir/$fd.uttid | cut -d " " -f1)
                num_tgt=$(wc -l $bpe_dir/$d.$lan | cut -d " "  -f1)
                if [ $num_src -ne $num_tgt ]; then
                    echo "Number of src files not equal to number of target files" && exit 1;
                fi
            done
        fi
    done
        echo "$(date) Fairseq preprocess for target dataset: $d"
        bpe_dir=$target_bpe_dir/bpe_subword_nmt
        fairseq-preprocess --source-lang $source_lang --target-lang $target_lang \
            --append-eos-tgt \
            --joined-dictionary \
            --workers $preprocess_num_workers \
            --tgtdict $fairseq_bpe_dict_tgt \
            --trainpref $bpe_dir/train \
            --validpref $bpe_dir/fisher_dev \
            --testpref $bpe_dir/fisher_dev,$bpe_dir/fisher_dev2,$bpe_dir/fisher_test,$bpe_dir/callhome_devtest,$bpe_dir/callhome_evltest \
            --destdir $fairseq_bin_dir || exit 1;
fi
