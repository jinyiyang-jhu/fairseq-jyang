#!/bin/bash
# This script trains a MT model with transformer structure on gold translation pairs

stage=1
ngpus=1
conf="conf/lat_transformer_bpe_Nov16.sh"
exp_dir=exp/mt_lat_conf_Dec12
bin_dir=exp/mt_lat_conf_Dec12/bpe_bin

#conf=$1
#exp_dir=$2
#bin_dir=$3

. cmd.sh
. path.sh
. parse_options.sh || exit 1;

#conf=$1
#exp_dir=$2
#bin_bin=$3
. $conf


src="es"
tgt="en"
dsets=("valid" "test" "test1" "test2" "test3")
original_dsets=("fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest")
espnet_ref_dir="espnet_data/data/"
decode_mdl="checkpoint_best"
bpe="subword_nmt"
bpe_type="@@ "
generate_bsz=4


for idx in $(seq 0 $((${#dsets[@]}-1))); do
    dset=${dsets[$idx]}
    dset_name=${original_dsets[idx]}
    decode_dir=$exp_dir/decode_${dset_name}_${decode_mdl}_decode_lat_remove_bpe
    if [ $stage -le 1 ]; then
        echo "$(date) => Decode $dset_name with $exp_dir/checkpoints/${decode_mdl}.pt"
        mkdir -p $decode_dir || exit 1
        $cuda_cmd --gpu 1 --mem 16G $decode_dir/log/decode.log \
         fairseq-generate-from-lattice $bin_dir \
            --task $task \
            --skip-invalid-size-inputs-valid-test \
            --gen-subset $dset \
            --path $exp_dir/checkpoints/${decode_mdl}.pt \
            --batch-size $generate_bsz \
            --remove-bpe "$bpe_type" \
            --num-workers $decode_num_workers \
            > $decode_dir/results_${decode_mdl}.txt || exit 1
    fi
    bash local/score_bleu_espnet.sh $decode_dir $bin_dir/$dset.uttid \
        $espnet_ref_dir/$dset_name.$tgt/ref.wrd.trn.detok.lc.rm \
        $espnet_ref_dir/$dset_name.$tgt/text.lc.rm
done
