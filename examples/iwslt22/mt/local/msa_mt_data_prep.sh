#!/bin/bash

. path.sh

data_dir=data/msa-en
out_dir=data/msa-en_processed
src="ar"
src_case="rm"
tgt="en"
tgt_case="lc.rm"

#sets=("GlobalVoices" "News-Commentary" "OpenSubtitles2018" "QED" "TED2020" "UNv1.0.6way")
#sets=("UNv1.0.6way")
sets=("GlobalVoices")
mkdir -p ${out_dir} || exit 1

for name in "${sets[@]}"; do
    # # normalize punctuation
    # echo "$(date) Processing set ${name}"
    
    # # English
    # # remove punctuations, lower case
    # echo "$(date) Set ${name}: normalizing source ${src}"
    # python local/normalize_en.py ${data_dir}/${name}.${src}-${tgt}.${tgt} > ${out_dir}/${name}.${src}-${tgt}.${tgt}.${tgt_case} || exit 1;
    # # Tokenization
    # tokenizer.perl -l en -q -no-escape < ${out_dir}/${name}.${src}-${tgt}.${tgt}.${tgt_case} > ${out_dir}/${name}.${src}-${tgt}.${tgt}.${tgt_case}.tok || exit 1;
    
    # # Arabic
    # echo "$(date) Set ${name}: normalizing target ${tgt}"
    # # remove punctuations etc., with Amir cleaning script
    # python local/transcript_cleaning_arabic_text.py ${data_dir}/${name}.${src}-${tgt}.${src} ${out_dir}/${name}.${src}-${tgt}.${src}.${src_case} || exit 1;

    # num_src=$(wc -l "${out_dir}/${name}.${src}-${tgt}.${src}.${src_case}" | cut -d " " -f1)
    # num_tgt=$(wc -l "${out_dir}/${name}.${src}-${tgt}.${tgt}.${tgt_case}.tok" | cut -d " " -f1)
    # if [ $num_src != $num_tgt ]; then
    #     echo "Warning: Set [ ${name} ] src-tgt mismatch: ${num_src} vs ${num_tgt}"
    # fi

    # Remove empty lines
    echo "$(date) Set ${name}: removing empty lines"
    awk 'NR==FNR{if (NF>0) {a[NR]=NF}}{if (NF >0){if (NR in a){print NR}}}' \
        ${out_dir}/${name}.${src}-${tgt}.${src}.${src_case} \
        ${out_dir}/${name}.${src}-${tgt}.${tgt}.${tgt_case}.tok \
        > ${out_dir}/${name}.${src}-${tgt}.non_empty.uttids

    awk 'NR==FNR{a[$0]; next}{if (NR in a){print $0}}' \
        ${out_dir}/${name}.${src}-${tgt}.non_empty.uttids \
        ${out_dir}/${name}.${src}-${tgt}.${src}.${src_case} \
        > ${out_dir}/${name}.${src}-${tgt}.${src}.text.${src_case} 

    echo "$(date) Set ${name}: selecting non empty ${src} and ${tgt}"
    awk 'NR==FNR{a[$0]; next}{if (NR in a){print $0}}' \
        ${out_dir}/${name}.${src}-${tgt}.non_empty.uttids \
        ${out_dir}/${name}.${src}-${tgt}.${tgt}.${tgt_case}.tok \
        > ${out_dir}/${name}.${src}-${tgt}.${tgt}.text.${tgt_case} 
    
    num_src=$(wc -l "${out_dir}/${name}.${src}-${tgt}.${src}.text.${src_case}" | cut -d " " -f1)
    num_tgt=$(wc -l "${out_dir}/${name}.${src}-${tgt}.${tgt}.text.${tgt_case}" | cut -d " " -f1)
    if [ $num_src -ne $num_tgt ]; then
        echo "Warning: After cleaning empty lines, set [ ${name} ] src-tgt mismatch: ${num_src} vs ${num_tgt}"
    fi

    # Creat train/dev/test: 80:10:10
    for lan in $src $tgt; do
        if [ $lan == $src ]; then
            lan_case=$src_case
        else
            lan_case=$tgt_case
        fi
        echo "$(date) Set ${name}: creating train/dev/test for ${src}"
        train_size=$(python -c 'import sys; print(round(float(sys.argv[1]) * 0.8))' "${num_src}")
        dev_size=$(python -c 'import sys; print(round(float(sys.argv[1]) * 0.1))' "${num_src}")
        awk -v var="$train_size" 'NR<=var' \
            ${out_dir}/${name}.${src}-${tgt}.${lan}.text.${lan_case} \
            > ${out_dir}/${name}.${src}-${tgt}.${lan}.train.${lan_case}
        awk -v var="$train_size" -v dur=$(($train_size+$dev_size)) 'NR>var && NR<=dur' \
            ${out_dir}/${name}.${src}-${tgt}.${lan}.text.${lan_case} \
            > ${out_dir}/${name}.${src}-${tgt}.${lan}.dev.${lan_case}
        awk -v var=$(($train_size+$dev_size)) 'NR>=var' \
            > ${out_dir}/${name}.${src}-${tgt}.${lan}.test.${lan_case}
    done
done