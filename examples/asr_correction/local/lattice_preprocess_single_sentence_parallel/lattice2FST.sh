#!/bin/bash
 
# Derived from the Kaldi recipe for Fisher Spanish (KALDI_TRUNK/egs/fisher_callhome_spanish/s5/local)
 
# Use Kaldi tools to process lattice
if [ $# -lt 2 ]; then
 echo "Usage : $0 <lat-file>  <output-file>"
 echo "E.g., $0 1.lat.ark 1.plf.txt"
 exit 1
fi
 
. $(dirname $0)/path.sh
 
lat_file=$1
plf_file=$2
acoustic_scale=1.0
 
lat_name=$(basename $lat_file .lat.ark)
plf_dir=$(dirname $plf_file)
[ -d $plf_dir ] || mkdir $plf_dir || exit 1
 
[ -f $plf_file ] && rm $plf_file && echo "$plf_file exists, deleting it and rewriting it!"
cat > $plf_dir/sos.txt <<EOF
1 2 <s> <s>
2
EOF
cat > $plf_dir/eos.txt <<EOF
1 2 </s> </s>
2
EOF
 
# Get word2int map. Append <s> and </s>
symtable=$plf_dir/word2int.map
/apollo/bin/env -e Pryon lattice_symbol_table_tool $lat_file $symtable
max_symtable=`sort -n -k2 $symtable | tail -n1 | awk '{print $2}'`
sos_index=$((max_symtable+1))
eos_index=$((max_symtable+2))
echo -e "<s>\t${sos_index}" >> $symtable
echo -e "</s>\t${eos_index}" >> $symtable
for prefix in sos eos; do
 cat $plf_dir/$prefix.txt | \
   fstcompile --isymbols=$symtable --osymbols=$symtable --arc_type=log > $plf_dir/$prefix.fst || exit 1
done
 
# Convert Kaldi lattice to PLF format
lattice-to-fst --lm-scale=1.0 --acoustic-scale=$acoustic_scale ark:$lat_file ark,t:- \
 | awk 'NR>1' | fstcompile --arc_type=log \
 | fstpush --push_weights --remove_total_weight \
 | fstconcat $plf_dir/sos.fst - | fstconcat - $plf_dir/eos.fst \
 | fstrmepsilon | fstminimize | fsttopsort \
 | fstprint --isymbols=$symtable --osymbols=$symtable \
 | $(dirname $0)/txt2plf.pl > $plf_file || exit 1
 
# Remove temporary files
rm $symtable
for prefix in sos eos; do
 rm $plf_dir/$prefix.fst
 rm $plf_dir/$prefix.txt
done
echo "Succeeded in converting lattice to PLF format"
