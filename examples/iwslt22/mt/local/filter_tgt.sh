#!/bin/bash

nj=8
tgt_dir="data/msa-en_processed"
src_dir="data/ta-en_ta_translated_en_true/train"

tgt_name="train.ar-en.en.lc.rm"
src_name="text.bpe.lc.rm.en"
tgt_sorted_name="text.tc.en.tok"

[ -f $src_dir/$tgt_sorted_name ] && rm ${src_dir}/${tgt_sorted_name}

for i in $(seq ${nj}); do
    tgt=${tgt_dir}/split${nj}/${i}/${tgt_name}
    src=${src_dir}/split${i}/${src_name}
    new_tgt=${tgt_dir}/split${nj}/${i}/${tgt_sorted_name}

    tgt_tmp=$tgt.tmp
    # awk '{print (NR-1)" "$0}' $tgt > $tgt_tmp

    awk 'NR==FNR{a[$1];next}$1 in a{print $0}' $src $tgt_tmp | cut -d " " -f2- > $new_tgt
    cat ${new_tgt} >> ${src_dir}/${tgt_sorted_name}
done



