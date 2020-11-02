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
if [ $# -ne 6 ]; then
    echo "Usage : $0 <old-lat-dir> <new-plf-dir> <dset-name> <word2int> <bpe-code-dir> <fairseq-bpe-dict>"
    echo "E.g.: $0 data/lattice data/plf \"train\" data/lang/words.txt exp/bpe_es_en_lc_subword_nmt exp/lat_mt_subword_nmt/bpe_bin/dict.es.txt"
    exit 1
fi

lat_dir=$1
lat_plf_dir=$2
dset_name=$3
word_map=$4
bpe_code_dir=$5
fairseq_bpe_dict=$6

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
    nj=$(ls $lat_plf_dir/plf_node/plf.node.*.txt | wc -l)
    # Create multiple storage links to store the processed files over different disks to avoid heavy I/O
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $lat_processed ]; then
        create_split_dir.pl \
            /export/b0{4,5,6,7,8,9}/${USER}/fairseq-data/plf_processed/fisher_callhome-$(date '+%m_%d_%H_%M')/$dset_name/storage \
            $lat_processed/storage || { echo "create multiple storages failed !" 1>&2 ; exit 1; }
    
        for n in $(seq $nj); do
            for f in "lat.uttid" "bin" "idx" "pos.idx" "pos.bin" "mask.idx" "mask.bin"; do
                create_data_link.pl $lat_processed/${dset_name}.$n.$f || { echo "create multiple storage links for $f failed !" 1>&2 ; exit 1; }
            done
        done
    fi

    $train_cmd JOB=1:$nj $lat_processed/log/binarize-lat.JOB.log \
        python local/lattice_preprocess/compute_attn_pos_enc.py \
            --lat_ifile $lat_plf_dir/plf_node/plf.node.JOB.txt \
            --dset_name $dset_name.JOB \
            --output_dir $lat_processed \
            --prob_mask_direction $mask_direction \
            --dict $fairseq_bpe_dict \
            --bpe_code $bpe_code_dir/code.txt \
            --bpe_vocab $bpe_code_dir/vocab.all.txt \
            --bpe_gloss $bpe_code_dir/glossaries.txt || exit 1
fi

