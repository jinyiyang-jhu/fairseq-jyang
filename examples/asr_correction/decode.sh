#!/bin/bash

# This script is used for decoding a binarized datset, given model. It is applicable if 
# you have multiple GPUs on the current machine; if only one GPU available, remove Line 23 & Line 71

stage=-1
exp_dir=$1
bpe_dir=$2
data_dir=$3
decode_mdl="checkpoint_best"

src="lattice"
tgt="asr_oracle"
task="translation_lattice"
valid_sets_names=("dev" "test")
valid_sets=("valid" "valid1")

constrained_softmax_fill_value=0
generate_bsz=256
train_num_workers=8

for cuda_index in $(seq 0 $((${#valid_sets[@]}-1))); do # In case you have one more GPU on this machine
(
    if [ $cuda_index == 0 ]; then
        s="valid"
    else
        s="valid${cuda_index}"
    fi
    echo "`date` => Decode $s with $exp_dir/checkpoints/${decode_mdl}.pt"
    decode_dir=$exp_dir/decode_${s}
    if [ $stage -le 0 ]; then
        mkdir -p $decode_dir || exit 1
        if [ $src == "lattice" ]; then
            CUDA_VISIBLE_DEVICES=$cuda_index fairseq-generate-from-lattice $bpe_dir/bpe_data_bin \
                --task $task \
                --gen-subset $s \
                --path $exp_dir/checkpoints/${decode_mdl}.pt \
                --batch-size $generate_bsz \
                --remove-bpe \
                --num-workers $train_num_workers \
                --constrained-softmax-fill-value $constrained_softmax_fill_value \
                > $decode_dir/results_${decode_mdl}.txt || exit 1
        else
            CUDA_VISIBLE_DEVICES=$cuda_index fairseq-generate $bpe_dir/bpe_data_bin \
                --task $task \
                --gen-subset $s \
                --path $exp_dir/checkpoints/${decode_mdl}.pt \
                --batch-size $generate_bsz \
                --remove-bpe \
                --num-workers $train_num_workers \
                > $decode_dir/results_${decode_mdl}.txt || exit 1
        fi
    fi

    if [ $stage -le 1 ]; then
        echo "`date` => Scoring $s with $decode_mdl"
        d=${valid_sets_names[$cuda_index]}
        cp $data_dir/$d/transcript/selected_utt.index $decode_dir/selected_utt.index
        cp $data_dir/$d/transcript/transcript.csv $decode_dir
        python $(dirname $0)/local/reorder_generate_sequence.py $decode_dir/results_${decode_mdl}.txt \
            $decode_dir/selected_utt.index \
            $decode_dir/${src}.csv \
            $decode_dir/hyp_${decode_mdl}.csv \
            $decode_dir/$tgt.csv 

        if [ $tgt != "transcript" ]; then
            python $(dirname $0)/local/compute_wer.py \
                $decode_dir/transcript.csv $decode_dir/hyp_${decode_mdl}.csv $decode_dir/wer_details/
        fi
    fi
) &
done 
wait
