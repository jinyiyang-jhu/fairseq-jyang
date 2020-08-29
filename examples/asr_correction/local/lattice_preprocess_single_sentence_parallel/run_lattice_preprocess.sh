#!/bin/bash
# This script will convert lattice to PLF node format, calculate the positional matrices and
# probabilistic mask matrices for transformer.
# This script takes a single lat.ark file as input, and produces corresponding files into the output dir.
# To be run from examples/asr_correction.
# Make sure to run in pytorch_p36 env.
 
mask_direction="None" # "fwd", "bwd" or "None"
lat_weight_scale=1.0
 
lat_file=$1
bpe_code=$2
bpe_vocab=$3
output_dir=$4
plf_file=$output_dir/plf.txt
plf_node_file=$output_dir/plf.node.txt
plf_bpe_file=$output_dir/plf.BPE.txt
plf_pos_file=$output_dir/plf.pos.npz
plf_mask_file=$output_dir/plf.mask.npz
 
[ -d $output_dir ] || mkdir $output_dir || exit 1
# Lattice to FSTs
bash local/lattice_preprocess_parallel/lattice2FST.sh $lat_file $plf_file
 
##################### The codes in this block are specifically set for my machine ##################
source .bashrc
source deactivate pytorch_p36
source activate pytorch_p36
##################### The codes in this block are specifically set for my machine ##################
 
# Edge lattice to node lattice
python3 local/lattice_preprocess_parallel/preproc-lattice.py --lat_weight_scale $lat_weight_scale $plf_file $plf_node_file
 
# If the node plf lattice is empty, then add this uttid to delete_utts.index, and later remove it
# from transcripts and asr 1best.
if  grep -wq "\[\],\[\]" $plf_node_file # This node plf is empty
then
 touch $output_dir/selected_lat.index
else
# Compute pos and mask
 python3 local/lattice_preprocess_parallel/compute_attn_pos_enc.py --lat_ifile $plf_node_file \
   --lat_bpe_file $plf_bpe_file \
   --lat_pos_file $plf_pos_file \
   --lat_prob_mask $plf_mask_file \
   --prob_mask_direction $mask_direction \
   --bpe_code $bpe_code \
   --bpe_vocab $bpe_vocab || exit 1
 echo $lat_file > $output_dir/selected_lat.index
fi
