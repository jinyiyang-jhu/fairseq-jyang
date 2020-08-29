lat_plf_file=$1
selected_utt=$2
output_selected_utt=$3
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
 
# Edge lattice to node lattice
echo "`date`: converting PLF to node PLFs"
/home/ec2-user/anaconda3/envs/pytorch_p36/bin/python3.6 $SCRIPT_DIR/preproc-lattice.py --lat_weight_scale $lat_weight_scale  $lat_plf_file $lat_plf_file.node.tmp.txt
 
# Filter the empty lattices in plf.node.tmp.txt, files not in the new selected_utt.index will be later removed
# from transcripts and asr 1best
grep -v "\[\]" $lat_plf_file.node.tmp.txt > $lat_plf_file.node.txt
awk 'BEGIN{print "-1"}{if ($0 == "[],[]"){print NR}}' $lat_plf_file.node.tmp.txt \
| awk '(NR==FNR){arr[$1]=1}(NR != FNR && (!(arr[FNR]))){print $0}'  - $selected_utt \
> $output_selected_utt
#rm $lat_plf_file.node.tmp.txt
#rm $lat_plf_file.node.txt