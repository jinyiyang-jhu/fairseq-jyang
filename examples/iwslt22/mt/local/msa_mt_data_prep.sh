#!/bin/bash

. path.sh

data_dir=data/msa-en
out_dir=data/msa-en_processed
src="ar"
tgt="en"
sets=("GlobalVoices" "News-Commentary" "OpenSubtitles2018" "QED" "TED2020" "UNv1.0.6way")

mkdir -p ${out_dir} || exit 1

for name in "${sets[@]}"; do
    # normalize punctuation
    echo "$(date) Processing set ${name}"
    
    # English
    # remove punctuations, lower case
    python local/normalize_en.py ${data_dir}/${name}.${src}-${tgt}.${tgt} > ${out_dir}/${name}.${src}-${tgt}.${tgt}.lc.rm || exit 1;
    # Tokenization
    tokenizer.perl -l en -q -no-escape < ${out_dir}/${name}.${src}-${tgt}.${tgt}.lc.rm > ${out_dir}/${name}.${src}-${tgt}.${tgt}.lc.rm.tok || exit 1;
    
    # Arabic
    # remove punctuations etc., with Amir cleaning script
    python local/transcript_cleaning_arabic_text.py ${data_dir}/${name}.${src}-${tgt}.${src} ${out_dir}/${name}.${src}-${tgt}.${src}.rm || exit 1;

    num_src=$(wc -l "${out_dir}/${name}.${src}-${tgt}.${src}.rm" | cut -d " " -f1)
    num_tgt=$(wc -l "${out_dir}/${name}.${src}-${tgt}.${tgt}.lc.rm.tok" | cut -d " " -f1)
    if [ $num_src != $num_tgt ]; then
        echo "Warning: Set [ ${name} ] src-tgt mismatch: ${num_src} vs ${num_tgt}"
    fi
done