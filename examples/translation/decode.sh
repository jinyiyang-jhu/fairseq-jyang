#!/bin/bash

# This script decodes given test sets, with a trained model, using Fairseq.

skip_decode=false
preprocess_num_workers=40
decode_mdl="checkpoint_best"
bpe_type="sentencepiece"
generate_bsz=32
ori_sets=("dev" "test" "sharedeval-bc" "sharedeval-bn")
bin_sets=("valid" "test" "test1" "test2")

. path.sh
. parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
    echo "Usage: $0 <train-configuration>"
    echo "E.g. , $0 conf_zh_en.sh"
    exit 1;
fi

conf=$1
source $conf

bin_dir=$exp_dir/bpe_bin

for i in $(seq 0 $((${#ori_sets[@]}-1))); do
(   
    set_name=${ori_sets[$i]}
    bin_name=${bin_sets[$i]}
    decode_dir=$exp_dir/decode_${set_name}_${decode_mdl}
    #awk '{print $1}' $text_dir/$dset_name.en/text > $bin_dir/$dset.uttid || exit 1;
    if ! $skip_decode ; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') fairseq-interactive for $src-$tgt:$src"
        mkdir -p $decode_dir || exit 1
        qsub -v PATH -S /bin/bash -b y -q gpu.q -cwd -j y -N fairseq_interactive \
        -l gpu=1,num_proc=10,mem_free=16G,h_rt=600:00:00 \
        -o $decode_dir/results_${decode_mdl}.txt -sync y -m ea -M jyang126@jhu.edu \
        decode_interactive.sh $conf $bpe_dir/$set_name.bpe."$src" || exit 1;
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') scoring for $src-$tgt:$src"
    grep "S-" $decode_dir/results_${decode_mdl}.txt | cut -d$'\t' -f3- > $decode_dir/text.ref.$tgt
    grep "H-" $decode_dir/results_${decode_mdl}.txt | cut -d$'\t' -f3- > $decode_dir/text.hyp.$tgt
    fairseq-score -s $decode_dir/text.hyp.$tgt \
        -r $data_dir/$set_name/text.$tgt \
        --ignore-case \
        --sacrebleu \
        > $decode_dir/score.txt || exit 1;
) &
done
wait 