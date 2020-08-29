#!/bin/bash
 
# This script trains a lattice transformer with asr_oracle and lattice. By default, it uses constrained softmax on the decoer output.
# If you do not want to use constrained softmax, remove the "--constrained-softmax-mask" in stage 3 & 4 and change model name in Line 53.
 
stage=0
backup="True"
 
# Data args. BPE code and vocab should be the same used for lattice BPT tokenization.
data_dir=/storage/home/ec2-user/workspace/data
#bpe_dir=/home/ec2-user/workspace/data/1month/bpe_asr_oracle_vs_lattice/bpe
bpe_code=/storage/home/ec2-user/workspace/data/1month/bpe_transcript_vs_1best/bpe/code_file.txt # This should be learned with training data transcription
bpe_vocab=/storage/home/ec2-user/workspace/data/1month/bpe_transcript_vs_1best/bpe/vocab.all.txt  # This should be learned with training data transcription
 
ref_name="asr_oracle" # transcript or asr_oracle
sets=("train" "dev" "test")
valid_sets=("${sets[@]:1}")
 
# The "valid" referes to the first term in valid_sets, "valid1" referes to the second term in valid_sets, ...
 
# Train params
init_lr=1e-05
min_lr=1e-07
lr=0.0005
warmup_updates=4000
weight_decay=0.0001
max_epoch=50
max_tokens=8192
generate_bsz=256
dropout=0.3
preprocess_num_workers=64
 
# Model params
task="translation_lattice"
train_num_workers=8
model_name="transformer_lattice"
encoder_layers=6
decoder_layers=6
encoder_attention_heads=8
encoder_mask_scale=5.0
cross_mask_scale=1.0
decoder_attention_heads=8
encoder_embed_dim=512 # default: 1024
encoder_ffn_embed_dim=2048 # default: 4096
decoder_embed_dim=512 # default: 1024
decoder_ffn_embed_dim=2048 # default: 4096
constrained_softmax_fill_value=-100000000
 
# Model dir args
decode_mdl="checkpoint_best" # This model gives best WER
date=`date | awk '{print $2"_"$3}'`
mdl_name="train_${ref_name}_lattice_${date}_scale_${encoder_mask_scale}_${cross_mask_scale}_max_tokens_${max_tokens}_warmup_${warmup_updates}_lr_${lr}_layers_${encoder_layers}_${decoder_layers}_dim_${encoder_embed_dim}_${encoder_ffn_embed_dim}_heads_${encoder_attention_heads}_${decoder_attention_heads}_max_epoch_${max_epoch}_dropout_${dropout}_constrained_softmax_${constrained_softmax_fill_value}"
exp_dir=exp/$mdl_name
 
if [ $stage -le 1 ]; then
 bpe_dir=$exp_dir/bpe
 mkdir -p $bpe_dir || exit 1
elif [ $stage == 3 ]; then
 [ ! -d $bpe_dir ] && echo "stage == 3 and bpe_dir needs to be specified " && exit 1
fi
 
if [ $stage -le 1 ]; then
 echo "`date` => Data preparation: applying BPE"
 for d in ${sets[@]}; do
   # Apply BPE to references
   ref=$data_dir/$d/$ref_name/${ref_name}.txt
   ref_bpe=$bpe_dir/$d.$ref_name
   subword-nmt apply-bpe -c $bpe_code --vocabulary $bpe_vocab \
     --vocabulary-threshold 1 < $ref > $ref_bpe || exit 1
  
   # Lattices are already BPE converted.
   hyp=$data_dir/$d/plf_processed/plf.BPE.txt
   hyp_bpe=$bpe_dir/$d.lat
   cp $hyp $hyp_bpe || exit 1
   cp $data_dir/$d/plf_processed/plf.pos.npz $bpe_dir/$d.lat.pos.npz
   cp $data_dir/$d/plf_processed/plf.mask.npz $bpe_dir/$d.lat.mask.npz
 done
fi
 
if [ $stage -le 2 ]; then
 echo "`date` => Fairseq preprocessing"
   fairseq-preprocess-lattice --source-lang lat --target-lang $ref_name \
     --trainpref $bpe_dir/train --validpref $bpe_dir/${valid_sets[0]},$bpe_dir/${valid_sets[1]} \
     --destdir $bpe_dir/bpe_data_bin --joined-dictionary --append-eos-tgt \
     --workers $preprocess_num_workers
fi
 
if [ $stage -le 3 ]; then
 echo "`date` => training transfromer model"
 mkdir -p $exp_dir/log || exit 1
 # This machine is p3.16large and it has 8 GPUs.
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MKL_THREADING_LAYER=GNU fairseq-train $bpe_dir/bpe_data_bin \
   --task $task \
   --prepend-bos-tgt \
   --max-tokens $max_tokens \
   --num-workers $train_num_workers \
   --arch $model_name \
   --encoder-layers $encoder_layers \
   --encoder-embed-dim $encoder_embed_dim \
   --encoder-ffn-embed-dim $encoder_ffn_embed_dim \
   --encoder-attention-heads $encoder_attention_heads \
   --encoder-attn-mask-scale $encoder_mask_scale \
   --decoder-layers $decoder_layers \
   --decoder-embed-dim $decoder_embed_dim \
   --decoder-ffn-embed-dim $decoder_ffn_embed_dim \
   --decoder-attention-heads $decoder_attention_heads \
   --cross-attn-mask-scale $cross_mask_scale \
   --tensorboard-logdir $exp_dir/tensorboard-log \
   --share-all-embeddings \
   --activation-fn relu \
   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
   --lr-scheduler inverse_sqrt \
   --warmup-init-lr $init_lr --warmup-updates $warmup_updates --lr $lr --min-lr $min_lr \
   --dropout $dropout --weight-decay $weight_decay \
   --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
   --save-dir $exp_dir/checkpoints \
   --save-interval 5 \
   --max-epoch $max_epoch \
   --constrained-softmax-fill-value $constrained_softmax_fill_value \
   --log-format json > $exp_dir/log/train.log || exit 1
fi
 
if [ $stage -le 4 ]; then
 for cuda_index in $(seq 0 $((${#valid_sets[@]}-1))); do # This machine is p3.16large and it has 8 GPUs.
 (
   if [ $cuda_index == 0 ]; then
     s="valid"
   else
     s="valid${cuda_index}"
   fi
   decode_dir=$exp_dir/decode_${s}
   echo "`date` => Decode $s with $exp_dir/checkpoints/${decode_mdl}.pt"
   mkdir -p $decode_dir || exit 1
   CUDA_VISIBLE_DEVICES=$cuda_index fairseq-generate-from-lattice $bpe_dir/bpe_data_bin \
     --task $task \
     --gen-subset $s \
     --path $exp_dir/checkpoints/${decode_mdl}.pt \
     --batch-size $generate_bsz \
     --remove-bpe \
     --num-workers $train_num_workers \
     --constrained-softmax-fill-value $constrained_softmax_fill_value \
     > $decode_dir/results_${decode_mdl}.txt || exit 1
  
   echo "`date` => Scoring $s with $decode_mdl"
   d=${valid_sets[$cuda_index]}
   cp $data_dir/$d/transcript/selected_utt.index $decode_dir/${d}_selected_utt.index || { echo "copy $data_dir/$d/transcript/selected_utt.index failed"; exit1;}
   python $(dirname $0)/local/reorder_generate_sequence.py $decode_dir/results_${decode_mdl}.txt \
     $decode_dir/${d}_selected_utt.index \
     $decode_dir/asr-lat.csv \
     $decode_dir/hyp_${decode_mdl}.csv \
     $decode_dir/$ref_name.csv || exit 1
 
   /apollo/env/BlueshiftModelingTools/bin/YapStats \
     /apollo/env/BlueshiftModelingTools/etc/yapstatsPropertyFiles/Yapstats-en_US_Echo.properties \
     -truthFile $decode_dir/$ref_name.csv \
     -hypothesisDirectories $decode_dir/hyp_${decode_mdl}.csv \
     -outputFileDirectory $decode_dir/yapstats-hyp-vs-${ref_name}-${decode_mdl} || exit 1
  
   if [ $ref_name != "transcript" ]; then
     /apollo/env/BlueshiftModelingTools/bin/YapStats \
       /apollo/env/BlueshiftModelingTools/etc/yapstatsPropertyFiles/Yapstats-en_US_Echo.properties \
       -truthFile $data_dir/$d/transcript/transcript.csv \
       -hypothesisDirectories $decode_dir/hyp_${decode_mdl}.csv \
       -outputFileDirectory $decode_dir/yapstats-hyp-vs-transcript-${decode_mdl} || exit 1
   fi
 ) &
 done
wait
fi
 
if [ $backup == "True" ]; then
 echo "`date` => Backup exp folder to s3 buckets"
 if [ -f run.log ]; then
   cp $0 $exp_dir
   cp run.log $exp_dir
 fi
 /apollo/bin/env -e HoverboardDefaultMLPS3Tool mlps3 cp -r $exp_dir s3://bluetrain-workspaces/jinyiyan/workspace/project_2020/exp/ || exit 1
 echo "Data backup succeed"
fi
