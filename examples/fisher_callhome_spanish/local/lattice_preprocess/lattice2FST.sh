#!/bin/bash
 
# Derived from the Kaldi recipe for Fisher Spanish (KALDI_TRUNK/egs/fisher_callhome_spanish/s5/local)
 
# Use Kaldi tools to process lattice
if [ $# -lt 2 ]; then
 echo "Usage : $0 <lat-dir>  <output-file>"
 exit 1
fi
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
 
. $SCRIPT_DIR/path.sh
 
lat_dir=$1
plf_dir=$2
acoustic_scale=1.0
plf_file=$plf_dir/plf.txt
 
lat_dir=`realpath $lat_dir`
 
if [ ! -f $lat_dir/lat_file.index ]; then
 find $lat_dir -name "*.lat.ark" > $lat_dir/lat_file.index || exit 1
fi
 
[ -f $plf_file ] && rm $plf_file && echo "$plf_file exists, deleting it and rewriting it!"
cat > $plf_dir/sos.txt <<EOF
1 2 <s> <s>
2
EOF
cat > $plf_dir/eos.txt <<EOF
1 2 </s> </s>
2
EOF
 
while read line; do
 dir=`dirname $line`
 lat_name=$(basename $line .lat.ark)
 
 # We need to create a word2int map from given lattice.
 symtable=$plf_dir/$lat_name.map
 /apollo/bin/env -e Pryon lattice_symbol_table_tool $line $symtable
 max_symtable=`sort -n -k2 $symtable | tail -n1 | awk '{print $2}'`
 sos_index=$((max_symtable+1))
 eos_index=$((max_symtable+2))
 echo -e "<s>\t${sos_index}" >> $symtable
 echo -e "</s>\t${eos_index}" >> $symtable
 for prefix in sos eos; do
   cat $plf_dir/$prefix.txt | \
     fstcompile --isymbols=$symtable --osymbols=$symtable --arc_type=log > $plf_dir/$prefix.fst || exit 1
 done
 #fname=`echo $lat_name | sed 's/^\.\///g' | sed 's/\(.*\)-/\1\//g'`
 lattice-to-fst --lm-scale=1.0 --acoustic-scale=$acoustic_scale ark:$line ark,t:- \
   | awk 'NR>1' | fstcompile --arc_type=log \
   | fstpush --push_weights --remove_total_weight \
   | fstconcat $plf_dir/sos.fst - | fstconcat - $plf_dir/eos.fst \
   | fstrmepsilon | fstminimize | fsttopsort \
   | fstprint --isymbols=$symtable --osymbols=$symtable \
   | $SCRIPT_DIR/txt2plf.pl >> $plf_file || exit 1
 rm $symtable
 #rm $plf_dir/sos.fst
 #rm $plf_dir/eos.fst
done < $lat_dir/lat_file.index
 
for prefix in sos eos; do
 rm $plf_dir/$prefix.fst
 rm $plf_dir/$prefix.txt
done
echo "Succeeded in converting lattice to PLF format"