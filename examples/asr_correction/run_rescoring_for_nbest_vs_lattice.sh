#!/bin/bash
 
stage=-1
src="lattice"
tgt="transcript"
data_dir=/storage/home/ec2-user/workspace/data
bpe_code=/storage/home/ec2-user/workspace/data/1month/bpe_transcript_vs_1best/bpe/code_file.txt
bpe_vocab=/storage/home/ec2-user/workspace/data/1month/bpe_transcript_vs_1best/bpe/vocab.all.txt
src_dict=/home/ec2-user/workspace/data/1month/bpe_${tgt}_vs_${src}/bpe/bpe_data_bin/dict.transcript.txt
mdl_dir=/home/ec2-user/workspace/exp/train_transcript_lattice_Aug_17_scale_5.0_no_cross_mask_max_tokens_8192_lr_0.0005/
 
valid_sets=("dev" "test")
decode_mdl="checkpoint_best" # This model gives best WER
task="translation_lattice"
preprocess_num_workers=64
backup="True"
date=`date | awk '{print $2"_"$3}'`
 
for idx in $(seq 0 9); do
 exp_dir=$mdl_dir/rescoring_${date}/rescoring_lat_vs_${idx}best
 bpe_dir=$exp_dir/bpe
 mkdir -p $bpe_dir || exit 1
 src_name=asr_${idx}best
 
 if [ $stage -le 0 ];then
   for d in ${valid_sets[@]}; do
     echo "`date` => Data preparation: applying BPE"
     ref=$data_dir/$d/nbest/${idx}_best.txt
     ref_bpe=$bpe_dir/$d.$src_name
     subword-nmt apply-bpe -c $bpe_code --vocabulary $bpe_vocab \
       --vocabulary-threshold 1 < $ref > $ref_bpe || exit 1
 
     # Lattices are already BPE converted.
     echo "`date` => Copying lattice for ${idx}-best"
     hyp=$data_dir/$d/plf_processed/plf.BPE.txt
     hyp_bpe=$bpe_dir/$d.lat
     cp $hyp $hyp_bpe || exit 1
     cp $data_dir/$d/plf_processed/plf.pos.npz $bpe_dir/$d.lat.pos.npz
     cp $data_dir/$d/plf_processed/plf.mask.npz $bpe_dir/$d.lat.mask.npz
   done
 fi
 
 if [ $stage -le 1 ];then
   echo "`date` => Fairseq preprocessing ${idx}-best"
     src_name=asr_${idx}best
     fairseq-preprocess-lattice --source-lang lat --target-lang $src_name --srcdict $src_dict \
       --validpref $bpe_dir/${valid_sets[0]},$bpe_dir/${valid_sets[1]} \
       --destdir $bpe_dir/bpe_data_bin --joined-dictionary --append-eos-tgt \
       --workers $preprocess_num_workers
 fi
 
   echo "`date` => Fairseq rescoring ${idx}-best"
   for cuda_index in $(seq 0 $((${#valid_sets[@]}-1))); do # This machine is p3.16large and it has 8 GPUs.
     (
       if [ $cuda_index == 0 ]; then
         s="valid"
       else
         s="valid${cuda_index}"
       fi
       decode_dir=$exp_dir/decode_${s}
       if [ $stage -le 2 ]; then
         mkdir -p $decode_dir || exit 1
         CUDA_VISIBLE_DEVICES=$cuda_index fairseq-generate-from-lattice $bpe_dir/bpe_data_bin \
           --task $task \
           --prepend-bos-tgt \
           --gen-subset $s \
           --path $mdl_dir/checkpoints/${decode_mdl}.pt \
           --batch-size 128 \
           --remove-bpe \
           --num-workers 8 \
           --score-reference \
           > $decode_dir/results_${decode_mdl}.txt || exit 1
       fi
 
       if [ $stage -le 3 ]; then
         echo "`date` => Reordering the rescored ${idx}-best"
         d=${valid_sets[$cuda_index]}
         cp $data_dir/$d/nbest/selected_utt.index $decode_dir/${d}_selected_utt.index || { echo "copy $data_dir/$d/bpe/selected_utt.index failed"; exit1;}
         python $(dirname $0)/local/reorder_rescored_results.py $decode_dir/results_${decode_mdl}.txt \
           $decode_dir/${d}_selected_utt.index \
           $decode_dir/src.csv \
           $decode_dir/hyp.csv \
           $decode_dir/rescored_scores.csv
       fi
     ) &
   done
   wait
done
 
if [ $backup == "True" ]; then
 echo "`date` => Backup exp folder to s3 buckets"
 mdl_name=`echo $mdl_dir | sed 's@^.*exp/@@g'`
 /apollo/bin/env -e HoverboardDefaultMLPS3Tool mlps3 cp -r \
   $mdl_dir/rescoring_${date} \
   s3://bluetrain-workspaces/jinyiyan/workspace/project_2020/exp/$mdl_name/ || exit 1
 echo "Data backup succeed"
fi
