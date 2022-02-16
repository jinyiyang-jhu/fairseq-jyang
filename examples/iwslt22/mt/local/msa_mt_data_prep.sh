#!/bin/bash

. path.sh

data_dir=
out_dir=
lang="en"
sets=("GlobalVoices", "News-Commentary", "OpenSubtitles2018", "QED", "TED2020", "UNv1.0.6way")

for name in "${sets[@]}"; do
    # normalize punctuation
    normalize-punctuation.perl -l ${lang} < ${dst}/${lang}.org > ${dst}/${lang}.norm

    # lowercasing
    lowercase.perl < ${dst}/${lang}.norm > ${dst}/${lang}.norm.lc
    cp ${dst}/${lang}.norm ${dst}/${lang}.norm.tc

    # remove punctuation
    remove_punctuation.pl < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.rm

    # tokenization
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.tc > ${dst}/${lang}.norm.tc.tok
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc > ${dst}/${lang}.norm.lc.tok
    tokenizer.perl -l ${lang} -q < ${dst}/${lang}.norm.lc.rm > ${dst}/${lang}.norm.lc.rm.tok

    paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.tc.tok | sort > ${dst}/text.tc.${lang}
    paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.lc.tok | sort > ${dst}/text.lc.${lang}
    paste -d " " ${dst}/.yaml2 ${dst}/${lang}.norm.lc.rm.tok | sort > ${dst}/text.lc.rm.${lang}

    # save original and cleaned punctuation
    lowercase.perl < ${dst}/${lang}.org | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.${lang}
    lowercase.perl < ${dst}/${lang}.norm.tc | text2token.py -s 0 -n 1 | tr " " "\n" \
        | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' > ${dst}/punctuation.clean.${lang}
done