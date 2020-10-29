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
    echo "Usage : $0 <old-lat-dir> <new-plf-dir> <word2int> <bpe-code-dir>"
    echo "E.g.: $0 data/lattice data/plf data/lang/words.txt exp/bpe_es_en_lc_subword_nmt"
    exit 1
fi

lat_dir=$1
lat_plf_dir=$2
word_map=$3
bpe_code_dir=$4

lat_processed=$lat_plf_dir/plf_processed

if [ $stage -le 0 ]; then
    echo "$(date): converting lattices to FSTs"
    bash local/lattice_preprocess/lattice2FST.sh --cmd "$train_cmd" \
    $lat_dir $lat_plf_dir/plf_edge $word_map || exit 1;
fi

if [ $stage -le 1 ]; then
    # Edge lattice to node lattice
    echo "$(date): converting edge PLF to node PLFs"
    nj=$(ls $lat_plf_dir/plf_edge/plf.*.txt | wc -l)
    mkdir -p $lat_plf_dir/plf_node || exit 1;
    $train_cmd JOB=1:$nj $lat_plf_dir/plf_node/log/edge2node.JOB.log \
        python local/lattice_preprocess/preproc-lattice.py \
        $lat_plf_dir/plf_edge/plf.JOB.txt $lat_plf_dir/plf_node/plf.node.JOB.txt || exit 1;
fi

if [ $stage -le 2 ]; then
# Compute pos and mask
    echo "$(date): getting BPE tokens, pos indice and probability matrices from PLFs"
    mkdir -p $lat_processed || exit 1;
    nj=$(ls $lat_plf_dir/plf_node/plf.node.*.txt | wc -l)
    $train_cmd JOB=1:$nj $lat_processed/log/plfinfo.JOB.log \
        python local/lattice_preprocess/compute_attn_pos_enc.py \
            --lat_ifile $lat_plf_dir/plf_node/plf.node.JOB.txt \
            --output_dir $lat_processed/JOB \
            --prob_mask_direction $mask_direction \
            --bpe_code $bpe_code_dir/code.txt \
            --bpe_vocab $bpe_code_dir/vocab.all.txt \
            --bpe_gloss $bpe_code_dir/glossaries.txt || exit 1
fi

