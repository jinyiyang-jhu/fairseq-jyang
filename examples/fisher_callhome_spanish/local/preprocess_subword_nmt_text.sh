#!/bin/bash

# This script perform BPE tokenization for train/dev/eval data.
# The datasets are provided from ESPNET recipe.

stage=-1
source_lang="es"
target_lang="en"
case=".lc.rm"
perform_bpe="true"
filter="false"
source_type="kaldi_1best" # or "gold"
preprocess_num_workers=40

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

# Input data
target_text_dir="data/espnet_prepared"
source_text_dir="data/kaldi_prepared/unselected"
common_uttid_dir="data/common_espnet_kaldi"
bpe_code_dir=exp/bpe_es_en_lc_subword_nmt
fairseq_bpe_dict_tgt=exp/lat_mt_subword_nmt/bpe_bin/dict.es.txt # same dict for both src and tgt
dsets=("fisher_dev" "fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest" "train")
fairseq_dsets=("valid" "test" "test1" "test2" "test3" "test4" "train")

# Output dir
filtered_dir=data/common_espnet_kaldi
output_bpe_dir=data/common_espnet_kaldi
fairseq_bin_dir=exp/${source_type}_mt_subword_nmt/bpe_bin

if [ $stage -le 1 ]; then
    for idx in $(seq 0 $((${#dsets[@]}-1))); do
        d=${dsets[$idx]}
        fd=${fairseq_dsets[idx]}
        bpe_dir=$output_bpe_dir/bpe_subword_nmt_${source_type}
        [ -d $bpe_dir ] || mkdir -p $bpe_dir || exit 1;
        if [ $filter == "true" ]; then
            mkdir -p $filtered_dir/$d.$source_lang/$source_type || exit 1;
            bash local/lattice_preprocess/filter_plf.sh $source_text_dir/$d.$source_lang/$source_type.text \
                $common_uttid_dir/$d.$source_lang/overlap.uttid $filtered_dir/$d.$source_lang/$source_type/text || exit 1;

            bash local/lattice_preprocess/filter_plf.sh $target_text_dir/$d.$target_lang/text${case} \
                $common_uttid_dir/$d.$source_lang/overlap.uttid $filtered_dir/$d.$target_lang/text
        fi

        if [ $perform_bpe == "true" ]; then
            echo "$(date) Processing BPE tokenization for dataset: $d.$source_lang"
            source_file=$filtered_dir/$d.$source_lang/$source_type/text
            target_file=$filtered_dir/$d.$target_lang/text
            if diff <(cut -d " "  -f1 $source_file) <(cut -d " " -f1 $target_file) >& /dev/null ;
            then
                echo "Utterance numbers (or order) differ between $source_file and $target_file !" && exit 1;
            fi

            cut -f 2- -d " " $source_file | sed -e 's/\&apos\;/ \&apos\; /g' |\
                subword-nmt apply-bpe -c $bpe_code_dir/code.txt \
                    --vocabulary $bpe_code_dir/vocab.all.txt \
                    --glossaries $(cat ${bpe_code_dir}/glossaries.txt | tr -s '\n' ' ') \
                    --vocabulary-threshold 1 > $bpe_dir/$d.$source_lang
            
            cut -f 2- -d " " $target_file | sed -e 's/\&apos\;/ \&apos\; /g' |\
                subword-nmt apply-bpe -c $bpe_code_dir/code.txt \
                    --vocabulary $bpe_code_dir/vocab.all.txt \
                    --glossaries $(cat ${bpe_code_dir}/glossaries.txt | tr -s '\n' ' ') \
                    --vocabulary-threshold 1 > $bpe_dir/$d.$target_lang
            num_src=$(wc -l $bpe_dir/$d.$source_lang | cut -d " " -f1)
            num_tgt=$(wc -l $bpe_dir/$d.$target_lang | cut -d " "  -f1)
            if [ $num_src -ne $num_tgt ]; then
                echo "Number of src utterances and number of target utterances differ !" && exit 1;
            fi
        fi
    done
        echo "$(date) Fairseq preprocess for target dataset: $d"
        bpe_dir=$output_bpe_dir/bpe_subword_nmt_${source_type}
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
