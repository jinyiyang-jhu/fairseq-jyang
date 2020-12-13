#!/bin/bash
# This script will convert lattice to PLF node format, calculate the positional matrices and
# probabilistic mask matrices for transformer.
# To be run from examples/asr_correction.
# Make sure to run in pytorch_p36 env.

stage=-1
cmd=run.pl
acwt=0.1
lmwt=1.0
beam=4
lat_depth_thres=85
src="es"
tgt="en"
mask_direction="None" # "fwd", "bwd" or "None"
dsets=("fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest" "train")
fairseq_dsets=("valid" "test" "test1" "test2" "test3" "train")


[ -f ./path.sh ] && . ./path.sh;
. cmd.sh
. parse_options.sh || exit 1;

# Use Kaldi tools to process lattice
# if [ $# -ne 7 ]; then
#     echo "Usage : $0 <old-lat-dir> <output-dir> <common-uttid-dir> <word2int> <bpe-code-dir> <fairseq-bpe-dict> <fairseq-bin-dir>"
#     echo "E.g.: $0 data/kaldi/train_lattice data/common_espnet_kaldi/ data/common_espnet_kaldi data/lang/words.txt exp/bpe_es_en_lc_subword_nmt exp/lat_mt_subword_nmt/bpe_bin/dict.es.txt exp/lat_mt/bin"
#     exit 1
# fi

kaldi_lat_dir="kaldi_data/kaldi_lat_dir"
output_dir="data/plf_pruned_beam_${beam}_acwt_${acwt}_lmwt_${lmwt}_depth_thres_${lat_depth_thres}"
uttid_dir="data/uttids"
word_map="kaldi_data/data/lang_test/words.txt"
bpe_code_dir="exp/bpe_subword_nmt"
fairseq_bpe_dict="exp/mt_gold/bpe_bin/dict.es.txt"
fairseq_bin_dir="exp/mt_lat_transformer/bpe_bin"
mkdir -p $fairseq_bin_dir || exit 1;

for idx in $(seq 0 $((${#dsets[@]}-1))); do
    dset=${dsets[$idx]}
    lat_dir=$kaldi_lat_dir/${dset}.$src
    lat_pruned_dir=$kaldi_lat_dir/${dset}_lats_pruned_beam_${beam}_acwt_${acwt}_lmwt_${lmwt}_depth_thres_${lat_depth_thres}.$src
    dset_output_dir=$output_dir/$dset.$src
    lat_processed=$dset_output_dir/plf_binarized

    if [ $stage -le 0 ]; then
        echo "$(date): converting lattices to FSTs for $dset"

        # Analysis on lattices for the lattice depth (avergae number of arcs per frame)
        bash local/lattice_preprocess/lattice_prune.sh --acwt $acwt --beam $beam --depth_thres $lat_depth_thres $lat_dir $lat_pruned_dir
        bash local/lattice_preprocess/lattice2FST.sh --cmd "$train_cmd" --acwt $acwt --lmwt $lmwt \
            $lat_pruned_dir $lat_pruned_dir/plf_edge_unfiltered $word_map || exit 1;
            
        nj=$(ls $lat_pruned_dir/plf_edge_unfiltered/plf.*.txt | wc -l)
        [ -d $dset_output_dir/plf_edge ] || mkdir -p $dset_output_dir/plf_edge || exit 1;

        # We select the lattices with depth below the lat_depth_thres
        comm -12 <(awk '{print $1}' $lat_pruned_dir/analyze/lattice_below_thres.depth | sort -u) <(sort $uttid_dir/$dset.uttids) \
            > $dset_output_dir/plf_edge/lattice_below_thres.uttid
        comm -12 <(awk '{print $1}' $lat_pruned_dir/analyze/lattice_above_thres.depth | sort -u) <(sort $uttid_dir/$dset.uttids) \
            > $dset_output_dir/plf_edge/lattice_above_thres.uttid
        num_skip=$(wc -l $dset_output_dir/plf_edge/lattice_above_thres.uttid | cut -d " " -f1)

        tot_lat=$(cat $dset_output_dir/plf_edge/lattice_below_thres.uttid $dset_output_dir/plf_edge/lattice_above_thres.uttid | wc -l | cut -d " " -f1)
        tot_utts=$(wc -l $uttid_dir/$dset.uttids | cut -d " " -f1)
        if [ $tot_lat != $tot_utts ]; then
            echo "The lattice to be processed is less than given utterance list: $tot_lat v.s. $tot_utts"
            exit 1;
        fi

        if [ $num_skip != 0 ]; then
            echo "$num_skip lattices skipped due to lattice depth above threshold; using 1best path instead"
            one_best=$lat_dir/scoring_kaldi/1best.txt
            [ ! -f $one_best ] && echo "No such file: $one_best, pleas run local/lattice_preprocess/get_1best.sh first !" && exit 1;
            awk 'NR==FNR{a[$1]; next} $1 in a{print $0}' $dset_output_dir/plf_edge/lattice_above_thres.uttid \
                $one_best > $dset_output_dir/plf_edge/lattice_above_thres.1best
            python local/lattice_preprocess/text_to_plf_edge.py \
                --add_bos --add_eos \
                --text $dset_output_dir/plf_edge/lattice_above_thres.1best \
                --plf $dset_output_dir/plf_edge/lattice_above_thres.plf.edge.txt || exit 1;
            new_nj=$(($nj+1))
            cp $dset_output_dir/plf_edge/lattice_above_thres.plf.edge.txt $dset_output_dir/plf_edge/plf.$new_nj.txt || exit 1;
            echo "Converted lattice 1best to PLF edge: $dset_output_dir/plf_edge/plf.$new_nj.txt"
        fi
        
        # Filter out the empty plf files, and in the ESPNET overlap utterance lists.
        run.pl JOB=1:$nj $dset_output_dir/plf_edge/log/filter_plf.JOB.log \
            bash local/filter_scp.sh $dset_output_dir/plf_edge/lattice_below_thres.uttid \
                $lat_pruned_dir/plf_edge_unfiltered/plf.JOB.txt \
                $dset_output_dir/plf_edge/plf.JOB.txt
        [ -f $dset_output_dir/plf_edge/plf.uttid ] && rm $dset_output_dir/plf_edge/plf.uttid
        for n in $(seq 1 $nj); do
            awk '{print $1}' $dset_output_dir/plf_edge/plf.$n.txt >> $dset_output_dir/plf_edge/plf.uttid
        done
    fi

    if [ $stage -le 1 ]; then
        # Edge lattice to node lattice
        echo "$(date): converting edge PLF to node PLFs for $dset"
        nj=$(ls $dset_output_dir/plf_edge/plf.*.txt | wc -l)
        mkdir -p $dset_output_dir/plf_node || exit 1;
        $train_cmd JOB=1:$nj $dset_output_dir/plf_node/log/edge2node.JOB.log \
            python local/lattice_preprocess/preproc-lattice.py \
            $dset_output_dir/plf_edge/plf.JOB.txt $dset_output_dir/plf_node/plf.node.JOB.txt || exit 1;
        cp $dset_output_dir/plf_edge/plf.uttid $dset_output_dir/plf_node
    fi

    if [ $stage -le 2 ]; then
    # Compute pos and mask
        echo "$(date): getting BPE tokens, pos indice and probability matrices from PLFs for $dset"
        nj=$(ls $dset_output_dir/plf_node/plf.node.*.txt | wc -l)
        # Create multiple storage links to store the processed files over different disks to avoid heavy I/O
        if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $lat_processed ]; then
            create_split_dir.pl \
                /export/b0{4,5,6,7,8,9}/${USER}/fairseq-data/plf_processed/fisher_callhome-$(date '+%m_%d_%H_%M')/$dset/storage \
                $lat_processed/storage || { echo "create multiple storages failed !" 1>&2 ; exit 1; }
        
            for n in $(seq $nj); do
                for f in "lat.uttid" "bin" "idx" "pos.idx" "pos.bin" "mask.idx" "mask.bin"; do
                    create_data_link.pl $lat_processed/${dset}.$n.$f || { echo "create multiple storage links for $f failed !" 1>&2 ; exit 1; }
                done
            done
        fi

        $train_cmd JOB=1:$nj $lat_processed/log/binarize-lat.JOB.log \
            python local/lattice_preprocess/compute_attn_pos_enc.py \
                --lat_ifile $dset_output_dir/plf_node/plf.node.JOB.txt \
                --dset_name $dset.JOB \
                --output_dir $lat_processed \
                --prob_mask_direction $mask_direction \
                --dict $fairseq_bpe_dict \
                --bpe_code $bpe_code_dir/code.txt \
                --bpe_vocab $bpe_code_dir/vocab.all.txt \
                --bpe_gloss $bpe_code_dir/glossaries.txt || exit 1
    fi

    if [ $stage -le 3 ]; then
        echo "$(date) Combininb binarized files for $dset"
        dset_name=${fairseq_dsets[idx]}
        nj=$(ls $dset_output_dir/plf_binarized/$dset.*.pos.idx | wc -l)
        python local/lattice_preprocess/merge_binaries.py \
            --input_prefix $dset_output_dir/plf_binarized/$dset \
            --num_splits $nj \
            --output_prefix $fairseq_bin_dir/$dset_name.$src"-"$tgt.$src || exit 1;
        
        uttid_file=$fairseq_bin_dir/$dset_name.uttid
        [ -f $uttid_file ] && rm $uttid_file
        for n in $(seq 1 $nj); do
            awk '{print $1}' $dset_output_dir/plf_binarized/$dset.$n.lat.uttid  >> $uttid_file.tmp
        done
        awk '{print $1" "NR-1}' $uttid_file.tmp > $uttid_file
        rm $uttid_file.tmp
    fi
done