# coding=utf-8
# Copyright (TBD)
# License (TBD)
 
"""
This script calculates the attention masks and positional encodings from lattices
for the transformer encoder.
"""
from __future__ import print_function
import argparse
import codecs
import numpy as np
from numpy import savez_compressed, asarray
from apply_bpe import read_vocabulary, BPE
from tqdm import tqdm, trange
from lattice_utils import LatticeReader, LatticeNode, Lattice
 
 
def tokenization_lat(word_seq, bpe_codes, bpe_vocab, bpe_gloss):
 """
 Tokenization for lattice text. Return the tokenized strings and id maps.
 """
 bpe = BPE(codes=bpe_codes, vocab=bpe_vocab, glossaries=bpe_gloss)
 split_tokens = []
 split_map = []
 word_seq = word_seq.strip().split()
 for i, word in enumerate(word_seq):
   tokens_str = bpe.process_line(line=word)
   tokens = tokens_str.split(' ')
   split_tokens.extend([t for t in tokens])
   split_map.extend([(i, n) for n in range(len(tokens))])
 return split_tokens, split_map
 
 
def load_dataset(lattice_file, lattice_bpe_file, lat_pos_file, lat_prob_mask_file, bpe_codes, bpe_vocab, bpe_gloss,
                probabilistic_masks=True, mask_direction=None, linearize=False):
 """
 Compute the attention masks and positional encoding from lattice.
 """
 with open(lattice_file) as lat_file, open(lattice_bpe_file, 'w') as lat_bpe:
   output_pos = []
   output_mask = []
   lat_reader = LatticeReader()
   for i, line in enumerate(tqdm(lat_file)):
     line = line.strip()
     lattice = lat_reader.read_sent(line, i)
    
     #lattice.plot(f"lattice_fwd", show_log_probs=["fwd_log_prob"])
     #lattice.plot(f"lattice_bwd", show_log_probs=["bwd_log_prob"])
     #lattice.plot(f"lattice_margin", show_log_probs=["marginal_log_prob"])
 
     utt = " ".join(lattice.str_tokens())
     tokens, mapping = tokenization_lat(utt, bpe_codes, bpe_vocab, bpe_gloss)
     nodes = []
     node_map = {}
     for j, (node_idx, n) in enumerate(mapping):
       if node_idx not in node_map:
         node_map[node_idx] = []
       node_map[node_idx].append(j)
      
     for token, (node_idx, n) in zip(tokens, mapping):
       orig_node = lattice.nodes[node_idx]
       if n != 0:
           nodes_prev = [len(nodes)-1]
       else:
           nodes_prev = [node_map[node][-1] for node in orig_node.nodes_prev]
       if n != len(node_map[node_idx])-1:
           nodes_next = [len(nodes)+1]
       else:
           nodes_next = [node_map[node][0] for node in orig_node.nodes_next]
       node = LatticeNode(nodes_prev=nodes_prev, nodes_next=nodes_next, value=token,
                          fwd_log_prob=orig_node.fwd_log_prob if n == 0 else 0,
                          marginal_log_prob=orig_node.marginal_log_prob,
                          bwd_log_prob=orig_node.bwd_log_prob if n == len(node_map[node_idx])-1 else 0)
       nodes.append(node)
     lattice_split = Lattice(idx=i, nodes=nodes)
     if linearize:
       pos = [i for i in range(len(lattice_split.nodes))]
     else:
       pos = lattice_split.longest_distances()
       pos = [p+1 for p in pos]
       log_conditional = lattice_split.compute_pairwise_log_conditionals(mask_direction, probabilistic_masks)[0]
       print (' '.join(tokens), file=lat_bpe)
       output_pos.append(pos)
       output_mask.append(log_conditional)
   final_pos = {f'{i}': np.array(output_pos[i]) for i in range(len(output_pos))}
   final_mask = {f'{i}': np.array(output_mask[i]) for i in range(len(output_mask))}
   savez_compressed(lat_pos_file, **final_pos)
   savez_compressed(lat_prob_mask_file, **final_mask)
 
def main():
 parser = argparse.ArgumentParser()
 parser.add_argument('--lat_ifile', type=str, required=True, help='Input lattice PLF file name')
 parser.add_argument('--lat_bpe_file', type=str, required=True, help='Output lattice bpe file name')
 parser.add_argument('--lat_pos_file', type=str, required=True, help='Output lattice postion indexes')
 parser.add_argument('--lat_prob_mask', type=str, required=True, help='Output lattice probabilistic mask')
 parser.add_argument('--prob_mask_direction', type=str, default=None, choices=('None', 'fwd', 'bwd'),
                     help='Output lattice mask direction')
 parser.add_argument('--bpe_code', type=str, required=True, help='Code file for BPE')
 parser.add_argument('--bpe_vocab', type=str, required=True,  help='BPE vocabulary')
 parser.add_argument('--bpe_gloss', type=str, default="<s> </s>",  help='BPE glossaries terms')
 parser.add_argument('--bpe_vocab_thres', type=str, default=1,help='BPE vocabulary frequency threshod')
 
 
 args = parser.parse_args()
 bpe_codes = codecs.open(args.bpe_code, encoding='utf-8')
 bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
 bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thres)
 bpe_gloss = args.bpe_gloss.split()
 if args.prob_mask_direction == "None":
   mask_direction = None
 else:
   mask_direction = args.prob_mask_direction
  
 load_dataset(args.lat_ifile, args.lat_bpe_file, args.lat_pos_file, args.lat_prob_mask,
             bpe_codes, bpe_vocab, bpe_gloss, probabilistic_masks=True,
             mask_direction=mask_direction, linearize=False)
if __name__ == '__main__':
 main()

