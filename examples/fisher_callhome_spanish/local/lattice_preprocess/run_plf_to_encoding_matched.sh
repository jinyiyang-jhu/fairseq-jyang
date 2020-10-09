#!/bin/bash
# This script will convert lattice to PLF node format, calculate the positional matrices and
# probabilistic mask matrices for transformer.
# To be run from examples/asr_correction.
# Make sure to run in pytorch_p36 env.
 
lattice_plf=$1
selected_utt_index=$2
bpe_code=$3 #/home/ec2-user/workspace/project_2020/data/data_filtered_long_utts/bpe/code_file.txt
bpe_vocab=$4 #/home/ec2-user/workspace/project_2020/data/data_filtered_long_utts/bpe/vocab.all.txt
lat_plf_dir=$5
mask_direction=$6  # "fwd", "bwd" or "None"
 
lat_processed=$lat_plf_dir
 
# Lattice to FSTs
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
mkdir -p $lat_plf_dir || exit 1
 
##################### The codes in this block are specifically set for my machine ##################
##################### Comment it if you have your path set properly already ########################
#source .bashrc
#source deactivate pytorch_p36
#source activate pytorch_p36
##################### The codes in this block are specifically set for my machine ##################
 
# Edge lattice to node lattice
echo "`date`: converting PLF to node PLFs"
/home/ec2-user/anaconda3/envs/pytorch_p36/bin/python3.6 $SCRIPT_DIR/preproc-lattice.py $lattice_plf $lat_plf_dir/plf.node.tmp.txt
 
# Filter the empty lattices in plf.node.tmp.txt, files not in the new selected_utt.index will be later removed
# from transcripts and asr 1best
grep -v "\[\]" $lat_plf_dir/plf.node.tmp.txt > $lat_plf_dir/plf.node.txt
awk '{if ($0 == "[],[]"){print NR}}' $lat_plf_dir/plf.node.tmp.txt \
| awk '(NR==FNR){arr[$1]=1}(NR != FNR && (!(arr[FNR]))){print $0}'  - $selected_utt_index \
> $lat_plf_dir/selected_utt.index
rm $lat_plf_dir/plf.node.tmp.txt
 
# Compute pos and mask
echo "`date`: computing pos indice and prob masks from node PLFs"
mkdir -p $lat_processed || exit 1
/home/ec2-user/anaconda3/envs/pytorch_p36/bin/python3.6 $SCRIPT_DIR/compute_attn_pos_enc.py --lat_ifile $lat_plf_dir/plf.node.txt \
 --lat_bpe_file $lat_processed/plf.BPE.txt \
 --lat_pos_file $lat_processed/plf.pos.npz \
 --lat_prob_mask $lat_processed/plf.mask.npz \
 --prob_mask_direction $mask_direction \
 --bpe_code $bpe_code \
 --bpe_vocab $bpe_vocab || exit 1
 