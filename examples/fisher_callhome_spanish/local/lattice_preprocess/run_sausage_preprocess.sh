#!/bin/bash

cmd=run.pl
stage=1
acwt=0.1
lmwt=1.0
beam=4
lat_depth_thres=85
src="es"
tgt="en"
mask_direction="None" # "fwd", "bwd" or "None"
skip_token="\<eps\>" # or "None" if you don't want to skip any token for model training
sau_prune_threshold=0.85

[ -f ./path.sh ] && . ./path.sh;
. cmd.sh
. parse_options.sh || exit 1;



dsets=("fisher_dev" "fisher_dev2" "fisher_test" "callhome_devtest" "callhome_evltest" "train")
fairseq_dsets=("valid" "test" "test1" "test2" "test3" "train")

[ -f ./path.sh ] && . ./path.sh;
. cmd.sh
. parse_options.sh || exit 1;

# Use Kaldi tools to process confusion networks
#if [ $# -ne 7 ]; then
#    echo "Usage : $0 <old-lat-dir> <output-dir> <common-uttid-dir> <word2int> <bpe-code-dir> <fairseq-bpe-dict> <fairseq-bin-dir>"
#    echo "E.g.: $0 data/kaldi/train_lattice data/kaldi/train_sausage_plf data/common_espnet_kaldi data/lang/words.txt exp/bpe_es_en_lc_subword_nmt exp/lat_mt_subword_nmt/bpe_bin/dict.es.txt exp/lat_mt/bin"
#    exit 1
#fi
fairseq_bin_dir=$1

kaldi_lat_dir="kaldi_data/kaldi_lat_dir"
output_dir="data/plf_sau_pruned_beam_${beam}_acwt_${acwt}_lmwt_${lmwt}_depth_thres_${lat_depth_thres}"
uttid_dir="data/uttids"
word_map="kaldi_data/data/lang_test/words.txt"
bpe_code_dir="exp/bpe_subword_nmt"
fairseq_bpe_dict="exp/mt_gold/bpe_bin/dict.es.txt"
#fairseq_bin_dir=$7
mkdir -p $fairseq_bin_dir || exit 1;

sausage_dir=$kaldi_lat_dir

for idx in $(seq 0 $((${#dsets[@]}-1))); do
    dset=${dsets[$idx]}
    lat_dir=$kaldi_lat_dir/${dset}_lats.$src
    lat_pruned_dir=$kaldi_lat_dir/${dset}_lats_pruned_beam_${beam}_acwt_${acwt}_lmwt_${lmwt}_depth_thres_${lat_depth_thres}.$src
    sau_dir=$sausage_dir/${dset}_sausages_pruned_beam_${beam}_acwt_${acwt}_lmwt_${lmwt}_depth_thres_${lat_depth_thres}.$src
    dset_output_dir=$output_dir/$dset.$src
    plf_edge_unfiltered=$sau_dir/plf_sau_edge_unfiltered_prune_${sau_prune_threshold}
    plf_edge_filtered=$dset_output_dir/plf_sau_edge_prune_${sau_prune_threshold}
    plf_node_dir=$dset_output_dir/plf_sau_node_prune_${sau_prune_threshold}
    lat_processed=$dset_output_dir/plf_sau_binarized_${sau_prune_threshold}
    #utt_list=$uttid_dir/${dset}.$tgt/overlap.uttid

    if [ $stage -le 0 ]; then
        echo "$(date): pruning lattices for $dset"
        bash local/lattice_preprocess/lattice_prune.sh --acwt $acwt --beam $beam --depth_thres $lat_depth_thres $lat_dir $lat_pruned_dir
    fi

    if [ $stage -le 1 ]; then
        echo "$(date): converting lattices to confusion networks for $dset"
        bash local/lattice_preprocess/generate_confusion_network.sh $lat_pruned_dir $sau_dir
    fi

    if [ $stage -le 2 ]; then
        echo "$(date): converting confusion networks to edge PLF for $dset"
        nj=$(ls $sau_dir/*.sau | wc -l)
        [ -d $plf_edge_unfiltered ] || mkdir -p $plf_edge_unfiltered || exit 1;
        $train_cmd JOB=1:$nj $plf_edge_unfiltered/log/sau2plf.JOB.log \
            python local/lattice_preprocess/sausage_to_plf.py \
                --add_bos --add_eos \
                --remove_eps \
                --prune_threshold $sau_prune_threshold \
                --sausage $sau_dir/JOB.sau \
                --plf $plf_edge_unfiltered/plf.JOB.txt \
                --word_sym_table $word_map || exit 1;

        # Filter out the empty plf files, and in the ESPNET overlap utterance lists.
        mkdir -p $plf_edge_filtered
        echo "$(date): PLF (edge) files filtering for $dset"
        $train_cmd JOB=1:$nj $plf_edge_filtered/log/filter_plf.JOB.log \
            bash local/filter_scp.sh $uttid_dir/$dset.uttids \
                $plf_edge_unfiltered/plf.JOB.txt \
                $plf_edge_filtered/plf.JOB.txt || exit 1;
        [ -f $plf_edge_filtered/plf.uttid ] && rm $plf_edge_filtered/plf.uttid
        for n in $(seq 1 $nj); do
            awk '{print $1}' $plf_edge_filtered/plf.$n.txt >> $plf_edge_filtered/plf.uttid
        done
        num_src=$(wc -l $uttid_dir/$dset.uttids | cut -d " " -f1)
        num_current=$(wc -l $plf_edge_filtered/plf.uttid | cut -d " "  -f1)
        if [ $num_src -ne $num_current ]; then
            echo "Number of filtered utterances differs from given utt list: $num_src vs $num_current !"
        fi
    fi

        # Need to add a filter scripts to filter 1) empty plf files 2) not in the overlap uttid list (joint of ESPNET & Kaldi)
    if [ $stage -le 3 ]; then
        # Edge lattice to node lattice
        echo "$(date): converting edge PLF to node PLFs for $dset"
        nj=$(ls $plf_edge_filtered/plf.*.txt | wc -l)
        mkdir -p $plf_node_dir || exit 1;
        $train_cmd JOB=1:$nj $plf_node_dir/log/edge2node.JOB.log \
            python local/lattice_preprocess/preproc-lattice.py \
            $plf_edge_filtered/plf.JOB.txt $plf_node_dir/plf.node.JOB.txt || exit 1;
        cp $plf_edge_filtered/plf.uttid $plf_node_dir
    fi

    if [ $stage -le 4 ]; then
    # Compute pos and mask
        echo "$(date): getting BPE tokens, pos indice and probability matrices from PLFs for $dset"
        nj=$(ls $plf_node_dir/plf.node.*.txt | wc -l)
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
                --lat_ifile $plf_node_dir/plf.node.JOB.txt \
                --dset_name $dset.JOB \
                --output_dir $lat_processed \
                --prob_mask_direction $mask_direction \
                --dict $fairseq_bpe_dict \
                --bpe_code $bpe_code_dir/code.txt \
                --bpe_vocab $bpe_code_dir/vocab.all.txt \
                --bpe_gloss $bpe_code_dir/glossaries.txt || exit 1
    fi

    if [ $stage -le 5 ]; then
        echo "$(date) Combininb binarized files for $dset"
        dset_name=${fairseq_dsets[idx]}
        nj=$(ls $lat_processed/$dset.*.pos.idx | wc -l)

        python local/lattice_preprocess/merge_binaries.py \
            --input_prefix $lat_processed/$dset \
            --num_splits $nj \
            --output_prefix $fairseq_bin_dir/$dset_name.$tgt"-"$src.$tgt || exit 1;
        
        uttid_file=$fairseq_bin_dir/$dset_name.uttid
        [ -f $uttid_file ] && rm $uttid_file
        for n in $(seq 1 $nj); do
            awk '{print $1}' $lat_processed/$dset.$n.lat.uttid  >> $uttid_file.tmp
        done
        awk '{print $1" "NR-1}' $uttid_file.tmp > $uttid_file
        rm $uttid_file.tmp
    fi
    echo -e " prune beam: ${beam}\n prune acwt: ${acwt}\n prune lmwt: ${lmwt}\n depth thres: ${lat_depth_thres}\n sausage prune threshold: ${sau_prune_threshold}" > $fairseq_bin_dir/lat.info
done