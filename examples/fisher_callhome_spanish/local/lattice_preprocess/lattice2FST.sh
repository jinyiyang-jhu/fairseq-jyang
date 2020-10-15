#!/bin/bash

# Derived from the Kaldi recipe for Fisher Spanish (KALDI_TRUNK/egs/fisher_callhome_spanish/s5/local)

cmd=run.pl
[ -f ./path.sh ] && . ./path.sh;
. cmd.sh
. parse_options.sh || exit 1;

# Use Kaldi tools to process lattice
if [ $# -ne 3 ]; then
  echo "Usage : $0 <lat-dir>  <output-file> <word2int-file>"
  echo "E.g.: $0 exp/decode_test exp/decode_test_plf data/lang/words.txt"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
lat_dir=$1
plf_dir=$2
symtable=$3

! which fstcompile \
  && echo "OpenFST path not found! Please add openfst path to ./path.sh and run this script again !" && exit 1;
! which lattice-to-fst \
  && echo "Kaldi path not found! Please add kaldi path to ./path.sh and run this script again !" && exit 1;

[ -f $symtable ] || (echo "$symtable does not exist !" && exit 1)
acoustic_scale=1.0
lat_dir=$(realpath $lat_dir)

nj=$(ls $lat_dir/lat.*.gz | wc -l)
sos_exists=$(grep "<s>" $symtable)
eos_exists=$(grep "</s>" $symtable)
len_word_map=$(sort -n -k2 $symtable | tail -n1 | awk '{print $2}')

if [ "$sos_exists" == "" ]; then # No "<s>" in the word to int mapping file
  echo "Adding <s> to the word2int mapping file"
  sos_index=$((len_word_map + 1))
  echo -e "<s>\t${sos_index}" >> $symtable
fi

if [ "$sos_exists" == "" ]; then # No "<s>" in the word to int mapping file
  echo "Adding </s> to the word2int mapping file"
  eos_index=$((len_word_map + 2))
  echo -e "<s>\t${eos_index}" >> $symtable
fi

mkdir -p $plf_dir || exit 1;

cat > $plf_dir/sos.txt <<EOF
1 2 <s> <s>
2
EOF
cat > $plf_dir/eos.txt <<EOF
1 2 </s> </s>
2
EOF

for prefix in sos eos; do
  cat $plf_dir/$prefix.txt |
    fstcompile --isymbols=$symtable --osymbols=$symtable --arc_type=log > $plf_dir/$prefix.fst || exit 1
done
$cmd JOB=1:$nj $plf_dir/log/lat2fst.JOB.log \
  lattice-to-fst --lm-scale=1.0 --acoustic-scale=$acoustic_scale ark:"gunzip -c $lat_dir/lat.JOB.gz|" ark,t:- \| \
    bash $SCRIPT_DIR/fst2plf.sh $symtable $plf_dir/sos.fst $plf_dir/eos.fst '>' $plf_dir/plf.JOB.txt || exit 1;
echo "Succeeded in converting lattice to PLF format"
