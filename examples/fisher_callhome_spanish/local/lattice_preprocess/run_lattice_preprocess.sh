#!/bin/bash
# This script will convert lattice to PLF node format, calculate the positional matrices and
# probabilistic mask matrices for transformer.
# To be run from examples/asr_correction.
# Make sure to run in pytorch_p36 env.

stage=-1
cmd=run.pl
mask_direction="None" # "fwd", "bwd" or "None"

[ -f ./path.sh ] && . ./path.sh;
. cmd.sh
. parse_options.sh || exit 1;

# Use Kaldi tools to process lattice
if [ $# -ne 4 ]; then
    echo "Usage : $0 <old-lat-dir> <new-plf-dir> <word2int> <bpe-model>"
    echo "E.g.: $0 data/lattice data/plf data/lang/words.txt data/bpe/train.en.model"
    exit 1
fi

lat_dir=$1
lat_plf_dir=$2
word_map=$3
bpe_model=$4
#bpe_code=$3 #/home/ec2-user/workspace/data/data_filtered_long_utts/bpe/code_file.txt
#bpe_vocab=$4 #/home/ec2-user/workspace/data/data_filtered_long_utts/bpe/vocab.all.txt

lat_processed=$new_lat_dir/plf_processed/

if [ $stage -le 0 ]; then
    echo "$(date -u): converting lattices to FSTs"
    bash local/lattice_preprocess/lattice2FST.sh --cmd "$train_cmd" \
    $lat_dir $lat_plf_dir/plf_edge $word_map || exit 1;
fi

if [ $stage -le 1 ]; then
    # Edge lattice to node lattice
    echo "$(date -u): converting edge PLF to node PLFs"
    nj=$(ls $lat_plf_dir/plf_edge/plf.*.txt | wc -l)
    mkdir -p $lat_plf_dir/plf_node || exit 1;
    $cmd JOB=1:$nj $lat_plf_dir/plf_node/log/edge2node.JOB.log \
        python local/lattice_preprocess/preproc-lattice.py \
        $lat_plf_dir/plf_edge/plf.JOB.txt $lat_plf_dir/plf_node/plf.node.JOB.txt
fi

if [ $stage -le 2 ]; then
# Compute pos and mask
    echo "$(date -u): getting BPE tokens, pos indice and probability matrices from PLFs"
    mkdir -p $lat_processed || exit 1
    $cmd JOB=1:$nj $lat_processed/log/plfinfo.JOB.log \
    python $SCRIPT_DIR/compute_attn_pos_enc.py --lat_ifile $lat_plf_dir/plf.node.JOB.txt \
        --lat_bpe_file $lat_processed/plf.BPE.JOB.txt \
        --lat_pos_file $lat_processed/plf.pos.JOB.npz \
        --lat_prob_mask $lat_processed/plf.mask.JOB.npz \
        --prob_mask_direction $mask_direction \
        --bpe_code $bpe_code \
        --bpe_vocab $bpe_vocab || exit 1