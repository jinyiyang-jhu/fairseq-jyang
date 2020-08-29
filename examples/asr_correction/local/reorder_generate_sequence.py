#!/usr/bin/env python3 -u
 
from __future__ import print_function
import argparse
import re
import sys
 
def main():
 parser = argparse.ArgumentParser(add_help=True, allow_abbrev=False)
 parser.add_argument('results', help='Input: results file from fairseq-generate')
 parser.add_argument('utt_list', help='Inpug: original utternace id list. Same order for transcript/1best/nbest/oracle/lattice')
 parser.add_argument('output_src', help='Output file for source tokens')
 parser.add_argument('output_hyp', help='Output file for hypotheis tokens')
 parser.add_argument('output_tgt', metavar='TARGET', default=None,
   help='output file for target (transcription) tokens')
 args = parser.parse_args()
 
 fp_src = open(args.output_src, 'w', encoding='utf-8')
 fp_hyp = open(args.output_hyp, 'w', encoding='utf-8')
 fp_tgt = open(args.output_tgt, 'w', encoding='utf-8')
  dict_src = {}
 dict_hyp = {}
 dict_tgt = {}
 
 patterns = {'^S-':dict_src, '^T-':dict_tgt}
 with open(args.results, encoding='utf-8') as fp_res:
   for line in fp_res:
     line = line.strip()
     pat = '^D-'
     if re.match(pat, line):
       tokens = line.split()
       id_str = tokens.pop(0)
       tokens.pop(0)
       id_int = int(re.sub(r'{}'.format(pat), '', id_str))
       dict_hyp[id_int] = ' '.join(tokens)
       continue
     for pat, pat_dict in patterns.items():
       if re.match(pat, line):
         tokens = line.split()
         id_str = tokens.pop(0)
         id_int = int(re.sub(r'{}'.format(pat), '', id_str))
         pat_dict[id_int] = ' '.join(tokens)
         break
 
 idx = 0
 with open(args.utt_list, 'r', encoding='utf-8') as fp_utt_list:
   for line in fp_utt_list:
     line = line.strip()
     if idx in dict_src.keys():
       print(line + ', ' + dict_src[idx], file=fp_src)
     else:
       print('utterances {}: S-{} is missing in {}'.format(line, idx, args.results), file=sys.stderr)
 
     if idx in dict_hyp.keys():
       print(line + ', ' + dict_hyp[idx], file=fp_hyp)
     else:
       print('utterances {}: D-{} is missing in {}'.format(line, idx, args.results), file=sys.stderr)
    
     if idx in dict_tgt.keys():
       print(line + ', ' + dict_tgt[idx], file=fp_tgt)
     else:
       print('utterances {}: T-{} is missing in {}'.format(line, idx, args.results), file=sys.stderr)
     idx +=1
    
 fp_src.close()
 fp_hyp.close()
 fp_tgt.close()
 
if __name__ == '__main__':
   main()
