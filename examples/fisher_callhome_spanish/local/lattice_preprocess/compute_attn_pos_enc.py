# coding=utf-8
# Copyright (TBD)
# License (TBD)

"""
This script calculates the attention masks and positional encodings from lattices
for the transformer encoder.
"""
from __future__ import print_function
import os
import argparse
import codecs
import numpy as np
from apply_bpe import read_vocabulary, BPE
from lattice_utils import LatticeReader, LatticeNode, Lattice

def tokenization_lat(word_seq, bpe_codes, bpe_vocab, bpe_gloss):
    """
    Tokenization for lattice text. Return the tokenized strings and id maps.
    Args:
    word_seq (list): a list of words (str)
    bpe_codes (codecs.StreamReaderWriter): bpe code file
    bpe_vocab (codecs.StreamReaderWriter): bpe vocabulary file
    bpe_gloss (str): a file contains bpe glossaries
    """
    split_tokens = []
    split_map = []

    with open(bpe_gloss, 'r') as f:
        glossaries = f.readlines()
    glossaries = [x.strip() for x in glossaries]
    bpe = BPE(codes=bpe_codes, vocab=bpe_vocab, glossaries=glossaries)

    for i, word in enumerate(word_seq):
        tokens_str = bpe.process_line(line=word)
        tokens = tokens_str.split(' ')
        split_tokens.extend([t for t in tokens])
        split_map.extend([(i, n) for n in range(len(tokens))])
    return split_tokens, split_map

def load_dataset(lattice_file, lattice_bpe_file, lat_utt_id, npz_dir,
                bpe_codes, bpe_vocab, bpe_gloss, probabilistic_masks=True, 
                mask_direction=None, linearize=False):
    """
    Compute the attention masks and positional encoding from lattice.
    """
    with open(lattice_file) as lat_file, open(lattice_bpe_file, 'w') as f_bpe, open(lat_utt_id, 'w') as f_uttid:
        lat_reader = LatticeReader()
        for i, line in enumerate(lat_file.readlines()):
            line = line.strip()
            tokens = line.split('\t')
            uttid = tokens.pop(0)
            lattice = lat_reader.read_sent(tokens[0], i)
            tokens, mapping = tokenization_lat(
                lattice.str_tokens(), bpe_codes, bpe_vocab, bpe_gloss)
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
                    nodes_prev = [node_map[node][-1]
                                  for node in orig_node.nodes_prev]
                if n != len(node_map[node_idx])-1:
                    nodes_next = [len(nodes)+1]
                else:
                    nodes_next = [node_map[node][0]
                                  for node in orig_node.nodes_next]
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
            print(uttid + '\t' + ' '.join(tokens), file=f_bpe)
            print(uttid, file=f_uttid)
            np.savez(os.path.join(npz_dir, uttid), pos=np.array(pos), mask=np.array(log_conditional))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat_ifile', type=str, required=True, help='Input lattice PLF file name')
    parser.add_argument('--output_dir', type=str, required=True, 
        help='Output directory for encoded PLF, positional indice and probability matrice')
    parser.add_argument('--prob_mask_direction', type=str, default=None, choices=('None', 'fwd', 'bwd'),
                        help='Output lattice mask direction')
    parser.add_argument('--bpe_code', type=str, required=True, help='Code file for BPE')
    parser.add_argument('--bpe_vocab', type=str, required=True,  help='BPE vocabulary')
    parser.add_argument('--bpe_gloss', type=str, required=True, help='BPE glossaries terms')
    parser.add_argument('--bpe_vocab_thres', type=str, default=1,help='BPE vocabulary frequency threshod')
    
    args = parser.parse_args()
    bpe_codes = codecs.open(args.bpe_code, encoding='utf-8')
    bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thres)
    bpe_gloss = args.bpe_gloss
    output_dir= args.output_dir

    lat_bpe_file = os.path.join(output_dir, "plf.bpe.txt")
    lat_utt_id = os.path.join(output_dir, "uttids.txt")
    npz_dir = os.path.join(output_dir, "matrices")

    if not os.path.exists(npz_dir):
        os.makedirs(npz_dir)

    if args.prob_mask_direction == "None":
        mask_direction = None
    else:
        mask_direction = args.prob_mask_direction

    load_dataset(args.lat_ifile, lat_bpe_file, lat_utt_id, npz_dir,
        bpe_codes, bpe_vocab, bpe_gloss, probabilistic_masks=True,
        mask_direction=mask_direction, linearize=False)

if __name__ == '__main__':
    main()
